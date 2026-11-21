#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ATK short-sequence smoke cases using the shared SMLA test assets."""

import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PYTEST_DIR = Path(__file__).resolve().parents[2] / "pytest"
GOLDEN = load_module("smla_st_golden", PYTEST_DIR / "sparse_flash_mla_golden.py")


@ACCURACY_REGISTRY.register("smla_pytest")
class SparseFlashMlaCompare(BaseAccuracyCompare):
    def compute_accuracy_result(self, local_output, remote_output, data_file):
        if local_output.shape != remote_output.shape:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info="SMLA output shape mismatch",
            )
        if (
            not torch.isfinite(local_output).all()
            or not torch.isfinite(remote_output).all()
        ):
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info="SMLA short case has nonfinite output",
            )
        # Use the same criterion as the operator's existing pytest/TTK tests.
        import logging

        original = logging.basicConfig
        try:
            logging.basicConfig = lambda *args, **kwargs: None
            compare = load_module(
                "smla_st_compare", PYTEST_DIR / "result_compare_method.py"
            )
        finally:
            logging.basicConfig = original
        status, percent = compare.check_result(remote_output, local_output)
        return AccuracyConfig(
            result=status == "Pass",
            filename=data_file,
            error_info=f"SMLA pytest: {status}, fulfillment={percent}%",
        )


def run_npu(data):
    import torch_npu  # noqa: F401

    # Load only SMLA: importing all operators also compiles unrelated extensions.
    if not hasattr(torch.ops.cann_ops_transformer, "sparse_flash_mla"):
        spec = importlib.util.find_spec("cann_ops_transformer")
        if spec is None:
            raise ImportError("Install the cann_ops_transformer Torch extension first")
        root = Path(next(iter(spec.submodule_search_locations)))
        for name, path in (
            ("cann_ops_transformer", root),
            ("cann_ops_transformer.ops", root / "ops"),
        ):
            if name not in sys.modules:
                package = types.ModuleType(name)
                package.__path__ = [str(path)]
                package.__file__ = str(path / "__init__.py")
                sys.modules[name] = package
        candidates = (
            root / "ops/attention/sparse_flash_mla/sparse_flash_mla.py",
            root / "ops/sparse_flash_mla.py",
        )
        wrapper = next((p for p in candidates if p.is_file()), None)
        if wrapper is None:
            raise ImportError("SparseFlashMla Torch wrapper is missing")
        load_module("cann_ops_transformer.ops.sparse_flash_mla", wrapper)

    def move(value):
        return value.npu() if isinstance(value, torch.Tensor) else value

    meta = {
        k: move(v)
        for k, v in data["metadata_input"].items()
        if k not in ("N1", "N2", "D", "B", "K1", "K")
    }
    meta["cmp_ratio"] = meta["cmp_ratio"] or 1
    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(**meta)
    args = {k: move(v) for k, v in data["input"].items()}
    args.update(
        metadata=metadata,
        cmp_ratio=meta["cmp_ratio"],
        ori_topk_length=meta["ori_topk_length"],
        cmp_topk_length=meta["cmp_topk_length"],
        return_softmax_lse=False,
    )
    if args["layout_q"] == "BSND":
        args["cu_seqlens_q"] = None
    output, _ = torch.ops.cann_ops_transformer.sparse_flash_mla(**args)
    torch.npu.synchronize()
    return output


def build_params(mode, layout_q, layout_kv, heads, query_length=4):
    """One batch, a short query and 256 original KV tokens."""
    compressed = mode in ("CSA", "HCA")
    ratio = 128 if mode == "HCA" else 4
    cmp_length = 256 // ratio
    paged = layout_kv == "PA_BBND"
    return dict(
        testcase_name=f"st_{mode}_{layout_q}_{layout_kv}_n{heads}_s{query_length}",
        template_mode="ORI_SPARSE" if mode == "DSPARK" else mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        q_type=torch.bfloat16,
        ori_kv_type=torch.bfloat16,
        cmp_kv_type=torch.bfloat16 if compressed else None,
        B=1,
        S1=query_length,
        S2=256,
        T1=query_length,
        T2=256,
        T3=cmp_length if compressed else None,
        N1=heads,
        N2=1,
        D=512,
        K1=32 if mode == "DSPARK" else None,
        K=32 if mode == "CSA" else None,
        block_num1=2 if paged else None,
        block_num2=1 if paged and compressed else None,
        block_size1=128 if paged else None,
        block_size2=128 if paged and compressed else None,
        cu_seqlens_q=[0, query_length] if layout_q == "TND" else None,
        cu_seqlens_ori_kv=None if paged else [0, 256],
        cu_seqlens_cmp_kv=[0, cmp_length] if compressed and not paged else None,
        seqused_q=[query_length],
        seqused_ori_kv=[256],
        seqused_cmp_kv=[cmp_length] if compressed else None,
        cmp_residual_kv=[0] if compressed else None,
        softmax_scale=512**-0.5,
        cmp_ratio=ratio if compressed else None,
        ori_mask_mode=0 if mode == "DSPARK" else 4,
        cmp_mask_mode=3 if compressed else 0,
        ori_win_left=127,
        ori_win_right=0,
        topk_value_mode=1,
        return_softmax_lse=False,
        batch_consistency=False,
        ori_kv_topk_mode="fullK" if mode == "DSPARK" else "no",
        cmp_kv_topk_mode="no",
        ori_sparse_indices_mode="full",
        cmp_sparse_indices_mode="full",
        q_datarange=[-1, 1],
        ori_kv_datarange=[-1, 1],
        cmp_kv_datarange=[-1, 1],
    )


@register("smla_st")
class SparseFlashMlaST(BaseApi):
    def __call__(self, input_data, with_output=False):
        attrs = input_data.kwargs
        params = build_params(
            attrs["mode"],
            attrs["layout_q"],
            attrs["layout_kv"],
            int(attrs["heads"]),
            int(attrs.get("query_length", 4)),
        )
        # Regenerate identical CPU inputs on both ATK nodes, independently of
        # framework RNG state. Restore host RNG state after generation.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(20261116)
            np.random.seed(20261116)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261116)
                data = GOLDEN.gen_data(
                    params,
                    prepare_device_storage=False,
                    generate_golden=self.device == "cpu",
                )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        if self.device == "cpu":
            # PA golden retains FP32 while BSND golden is already cast. ATK
            # requires matching output dtypes before invoking the comparator.
            output = data["cpu_output"].to(params["q_type"])
        else:
            output = run_npu(data)
        return output
