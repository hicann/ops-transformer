# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FFN-only runtime adapters for TTK revisions without replay/GEIR hooks.

No TTK source files are changed. These adapters deliberately depend on private
TTK entry points; unsupported signatures/templates fail instead of skipping
state restoration. Torch registration is provided by the installed extension package.
"""

import functools
import inspect
import json
import logging
from pathlib import Path

_GRAPH_STATES = {}


def _require_parameters(function, names):
    if not set(names).issubset(inspect.signature(function).parameters):
        raise RuntimeError(f"FFN TTK compatibility: unsupported signature: {function}")


def remember_graph_state(tensor, before, after):
    # One device case executes per worker. Discard closures retaining old tensors.
    _GRAPH_STATES.clear()
    _GRAPH_STATES[tensor.data_ptr()] = (before, after)


def load_graph_execution():
    """Resolve the aclgraph compile entry across TTK revisions.

    TTK >= 2026-10 moved it to framework_api.torch.graph_runner (extra
    fullgraph/super_kernel/online_compile parameters); older revisions expose
    framework_api.graph_execution. Returns None when neither exists.
    """
    try:
        from ttk.core_modules.framework_api.torch import graph_runner

        return graph_runner
    except ImportError:
        pass
    try:
        from ttk.core_modules.framework_api import graph_execution

        return graph_execution
    except ImportError:
        return None


def install_graph():
    graph = load_graph_execution()
    if graph is None:
        raise RuntimeError(
            "FFN TTK compatibility: aclgraph module not found "
            "(framework_api.torch.graph_runner / framework_api.graph_execution)"
        )
    if getattr(graph, "SUPPORTS_NPU_GRAPH_BEFORE_RUN", False):
        return  # Newer frameworks own the hooks; never execute them twice.
    original = graph._compile_model_aclgraph
    if getattr(original, "_ffn_assets_adapter", False):
        return
    _require_parameters(original, ("model", "backend"))

    @functools.wraps(original)
    def compile_ffn(model, backend, *args, **kwargs):
        import torch

        target = torch.ops.cann_ops_transformer.ffn_worker_batching
        api = getattr(model, "api_func", None)
        if api is None:
            # TTK may wrap the model (SuperKernel scope / core limit); unwrap once.
            api = getattr(getattr(model, "_model", None), "api_func", None)
        compiled = original(model, backend, *args, **kwargs)
        if api is not target and api is not target.default:
            return compiled

        @functools.wraps(compiled)
        def call(*args, **kwargs):
            import torch_npu

            context = args[0] if args else kwargs["schedule_context"]
            # The official FFN schema puts need_schedule among keyword-only attrs.
            if kwargs.get("need_schedule", 0) != 1:
                return compiled(*args, **kwargs)
            hooks = _GRAPH_STATES.get(context.data_ptr())
            if hooks is None:
                raise RuntimeError(
                    "FFN graph replay has no snapshot for this context address"
                )
            before, after = hooks
            torch_npu.npu.synchronize(context.device)
            before()
            torch_npu.npu.synchronize(context.device)
            result = compiled(*args, **kwargs)
            torch_npu.npu.synchronize(context.device)
            after()
            return result

        return call

    compile_ffn._ffn_assets_adapter = True
    graph._compile_model_aclgraph = compile_ffn
    logging.info("FFN assets adapter: ACL graph replay enabled (TTK source unchanged)")


def adapt_geir_source(source):
    """Extend only the generated FFN executable, with checked unique anchors."""
    if "static bool RelocateInput(" in source:
        return source  # Framework already supplies the complete relocation path.
    header = str(Path(__file__).with_name("ffn_geir_relocations.h").resolve())
    # Keep the generated cache sensitive to helper content, not just its path.
    import hashlib

    digest = hashlib.sha256(Path(header).read_bytes()).hexdigest()
    # TTK 2026-10-12 起 GEIR 模板 serve-mode 化：config 移入常驻 while 循环内
    # 逐例解析（GEInitialize 之前不再有 config）。两代模板均在
    # inputs_cfg 行处拥有 config/input_prefix/device_id 作用域，以它为锚。
    additions = (
        (
            "static int LoadInputFromFile(const string& path, DataType dt,\n",
            f'// FFN relocation helper SHA256: {digest}\n#include "{header}"\n\n',
        ),
        (
            "    TensorDesc tdesc(Shape(data_shape), FORMAT_ND, dt);\n",
            "    if (!RelocateInput(path, buf, expected)) {\n"
            '        fprintf(stderr, "[TTK-GEIR] FFN auxiliary relocation failed: %s\\n", path.c_str());\n'
            "        delete[] buf;\n        return FAILED;\n    }\n",
        ),
        (
            '    const JsonValue& inputs_cfg  = config.at("inputs");\n',
            '    input_relocations = config.at("input_relocations");\n'
            '    relocation_input_prefix = input_prefix ? input_prefix : "";\n'
            "    if (input_relocations.size() && aclrtSetDevice(device_id) != ACL_SUCCESS) {\n"
            "        GEFinalize();\n        return FAILED;\n    }\n",
        ),
        (
            "        GEFinalize();\n    };\n",
            "        for (void* buffer : auxiliary_buffers) aclrtFree(buffer);\n"
            "        auxiliary_buffers.clear();\n",
        ),
    )
    for anchor, insertion in additions:
        if source.count(anchor) != 1:
            raise RuntimeError(
                f"Unsupported TTK GEIR template; missing/ambiguous anchor: {anchor!r}"
            )
        source = source.replace(anchor, insertion + anchor, 1)
    # serve-mode 模板：多个用例共享一个常驻进程，辅助 HBM 缓冲必须逐例释放，
    # 否则跨用例累积设备内存。一次性模板没有该逐例收尾行，跳过即可。
    per_round = "    if (ttk_serve_mode) prof_teardown();"
    if source.count(per_round) == 1:
        line_end = source.index("\n", source.index(per_round)) + 1
        free_round = (
            "    for (void* buffer : auxiliary_buffers) aclrtFree(buffer);\n"
            "    auxiliary_buffers.clear();\n"
        )
        source = source[:line_end] + free_round + source[line_end:]
    return source


def install_geir(export_inputs):
    from ttk.core_modules.geir import graph_builder
    from ttk.core_modules.geir.compiler import GeirCompiler

    builder = graph_builder.GeirGraphBuilder
    if getattr(builder.write_case_config, "_ffn_assets_adapter", False):
        return
    render = graph_builder._render_template
    write_config = builder.write_case_config
    compile_cmd = GeirCompiler._build_compile_cmd
    _require_parameters(render, ("name",))
    _require_parameters(write_config, ("self", "testcase", "mode", "work_dir"))
    _require_parameters(
        compile_cmd,
        ("self", "source_path", "binary_path", "include_dirs", "lib_dirs", "libs"),
    )

    @functools.wraps(render)
    def render_ffn(name, **context):
        source = render(name, **context)
        if (
            name == "geir_op_template.cpp.j2"
            and context.get("op_class") == "FfnWorkerBatching"
        ):
            return adapt_geir_source(source)
        return source

    @functools.wraps(write_config)
    def write_ffn_config(self, testcase, mode="const", work_dir=None):
        path = write_config(self, testcase, mode=mode, work_dir=work_dir)
        if path is not None and testcase.op_name == "ffn_worker_batching":
            source = (Path(self.op_dir) / "ffn_worker_batching.cpp").read_text()
            if "RelocateInput(path, buf, expected)" not in source:
                raise RuntimeError(
                    "FFN GEIR executable has no auxiliary pointer relocation support"
                )
            prefix = str(Path(self.work_dir) / f"{testcase.testcase_name}_input")
            relocations = export_inputs(
                testcase.input_arrays, testcase.attributes, prefix
            )
            config = json.loads(Path(path).read_text())
            config["input_relocations"] = relocations
            Path(path).write_text(json.dumps(config))
        return path

    @functools.wraps(compile_cmd)
    def compile_ffn(self, source_path, binary_path, include_dirs, lib_dirs, libs):
        if Path(source_path).stem == "ffn_worker_batching" and "ascendcl" not in libs:
            libs = [*libs, "ascendcl"]
        return compile_cmd(self, source_path, binary_path, include_dirs, lib_dirs, libs)

    write_ffn_config._ffn_assets_adapter = True
    graph_builder._render_template = render_ffn
    builder.write_case_config = write_ffn_config
    GeirCompiler._build_compile_cmd = compile_ffn
    logging.info(
        "FFN assets adapter: GEIR auxiliary relocation enabled (TTK source unchanged)"
    )
