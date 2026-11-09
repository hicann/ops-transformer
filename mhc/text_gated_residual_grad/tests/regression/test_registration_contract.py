# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check registration agreement across host, kernel and binary-build metadata."""

import json
from pathlib import Path
import re
import unittest


OPERATOR_ROOT = Path(__file__).resolve().parents[2]
DTYPE_NAMES = {"DT_FLOAT16": "float16", "DT_BF16": "bfloat16", "DT_FLOAT": "float32"}


def read_source(relative):
    return (OPERATOR_ROOT / relative).read_text()


def registered_ports(kind):
    source = read_source("op_host/text_gated_residual_grad_def.cpp")
    pattern = rf'this->{kind}\("(\w+)"\)(.*?);'
    ports = []
    for name, block in re.findall(pattern, source, re.DOTALL):
        param_type = re.search(r"\.ParamType\((\w+)\)", block).group(1).lower()
        dtype_list = re.search(r"\.DataType\(\{([^}]+)\}\)", block).group(1)
        dtypes = [
            DTYPE_NAMES[item.strip().removeprefix("ge::")]
            for item in dtype_list.split(",")
        ]
        ports.append({"name": name, "paramType": param_type, "dtypes": dtypes})
    return ports


def snake_case(name):
    return re.sub(r"([A-Z])", r"_\1", name).lower()


class RegistrationContractTest(unittest.TestCase):
    def setUp(self):
        self.inputs = registered_ports("Input")
        self.outputs = registered_ports("Output")
        self.input_names = [port["name"] for port in self.inputs]
        self.output_names = [port["name"] for port in self.outputs]

    def test_required_ports_precede_optional_ports(self):
        for ports in (self.inputs, self.outputs):
            optional_seen = False
            for port in ports:
                if port["paramType"] == "optional":
                    optional_seen = True
                else:
                    self.assertFalse(optional_seen, port["name"])

    def test_graph_launcher_and_kernel_use_registered_order(self):
        proto = read_source("op_graph/text_gated_residual_grad_proto.h")
        graph_inputs = re.findall(r"\.(OPTIONAL_INPUT|INPUT)\((\w+),", proto)
        self.assertEqual([name for _, name in graph_inputs], self.input_names)
        self.assertEqual(
            [
                "optional" if kind == "OPTIONAL_INPUT" else "required"
                for kind, _ in graph_inputs
            ],
            [port["paramType"] for port in self.inputs],
        )
        self.assertEqual(re.findall(r"\.OUTPUT\((\w+),", proto), self.output_names)
        launcher = read_source("op_host/op_api/text_gated_residual_grad.cpp")
        for kind, expected in (
            ("INPUT", self.input_names),
            ("OUTPUT", self.output_names),
        ):
            arguments = re.search(rf"OP_{kind}\(([^)]+)\)", launcher).group(1)
            self.assertEqual(
                [snake_case(name.strip()) for name in arguments.split(",")], expected
            )
        kernel = read_source("op_kernel/text_gated_residual_grad.cpp")
        signature = re.search(
            r"void text_gated_residual_grad\((.*?)\)\s*\{", kernel, re.DOTALL
        ).group(1)
        self.assertEqual(
            re.findall(r"GM_ADDR\s+(\w+)", signature),
            self.input_names + self.output_names + ["workspace", "tiling"],
        )

    def test_api_and_internal_signatures_match_registration(self):
        signatures = (
            (
                "op_host/op_api/aclnn_text_gated_residual_grad.h",
                "aclnnTextGatedResidualGradGetWorkspaceSize",
            ),
            (
                "op_host/op_api/aclnn_text_gated_residual_grad.cpp",
                "aclnnTextGatedResidualGradGetWorkspaceSize",
            ),
            ("op_host/op_api/text_gated_residual_grad.h", "TextGatedResidualGrad"),
            ("op_host/op_api/text_gated_residual_grad.cpp", "TextGatedResidualGrad"),
            (
                "op_host/op_api/text_gated_residual_grad.cpp",
                "TextGatedResidualGradAICore",
            ),
            ("op_kernel/arch35/text_gated_residual_grad_kernel_common.h", "Init"),
            ("op_kernel/arch35/text_gated_residual_grad_vector_arch35.h", "Init"),
        )
        for path, function in signatures:
            pattern = rf"\b{function}\((.*?)\)\s*(?:;|\{{)"
            signature = re.search(pattern, read_source(path), re.DOTALL).group(1)
            names = []
            for argument in signature.split(","):
                name = re.search(r"(\w+)\s*$", argument).group(1)
                names.append(snake_case(re.sub(r"Optional$", "", name)))
            self.assertEqual(names[: len(self.inputs)], self.input_names, path)
            tail = names[len(self.inputs) :]
            if function == "Init":
                self.assertEqual(tail[: len(self.outputs)], self.output_names, path)
            else:
                self.assertEqual(tail[0], "channels", path)
                if function != "TextGatedResidualGrad":
                    self.assertEqual(
                        tail[1 : 1 + len(self.outputs)], self.output_names, path
                    )

    def test_host_indices_and_template_binding_match_registration(self):
        index_names = {"down_pre_act": "DOWN_PRE"}
        for path in (
            "op_host/op_tiling/text_gated_residual_grad_infershape.cpp",
            "op_host/op_tiling/arch35/text_gated_residual_grad_tiling_arch35.cpp",
        ):
            indices = dict(
                re.findall(r"constexpr size_t IDX_(\w+) = (\d+);", read_source(path))
            )
            for index, name in enumerate(self.input_names):
                key = index_names.get(name, name.upper())
                self.assertEqual(int(indices[key]), index, f"{path}: {name}")
        template = read_source(
            "op_kernel/arch35/text_gated_residual_grad_tiling_key_arch35.h"
        )
        x_index = int(
            re.search(r"#define HC_GRAD_TPL_INPUT_X (\d+)", template).group(1)
        )
        self.assertEqual(self.input_names[x_index], "x")
        self.assertIn("ASCENDC_TPL_INPUT(HC_GRAD_TPL_INPUT_X)", template)

    def test_binary_variants_match_registered_order_and_types(self):
        config = json.loads(
            read_source("op_host/config/ascend950/text_gated_residual_grad_binary.json")
        )
        self.assertEqual(config["op_type"], "TextGatedResidualGrad")
        seen_dtypes = set()
        x_dtypes = next(port["dtypes"] for port in self.inputs if port["name"] == "x")
        for variant in config["op_list"]:
            inputs = {item["name"]: item for item in variant["inputs"]}
            dtype = inputs["x"]["dtype"]
            seen_dtypes.add(dtype)
            column = x_dtypes.index(dtype)
            for group, expected in (("inputs", self.inputs), ("outputs", self.outputs)):
                actual = variant[group]
                self.assertEqual(
                    [item["name"] for item in actual],
                    [port["name"] for port in expected],
                )
                for index, (item, port) in enumerate(zip(actual, expected)):
                    self.assertEqual(item["index"], index)
                    self.assertEqual(
                        item["dtype"], port["dtypes"][column], item["name"]
                    )
                    self.assertEqual(item["paramType"], port["paramType"], item["name"])
                    self.assertEqual(item["format"], "ND")
        self.assertEqual(seen_dtypes, set(x_dtypes))


if __name__ == "__main__":
    unittest.main()
