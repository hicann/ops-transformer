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

"""Input customization for LightningIndexer V2 TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch


class LightningIndexerV2InputAdapter:
    """Translate a TTK case and reuse the pytest input/golden generator."""

    def __init__(self):
        self.pytest_golden = None

    @staticmethod
    def load_golden_store():
        name = "li_v2_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "lightning_indexer_v2_golden.py"
        name = "li_v2_pytest_golden"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(name, None)
                    raise
            self.pytest_golden = module
            return module
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[index + 1]) - int(value[index])
                for index in range(len(value) - 1)]

    @staticmethod
    def data_range(input_ranges, index, default=(-1, 1)):
        if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
            return repr(list(input_ranges[index]))
        return repr(list(default))

    @staticmethod
    def qk_dtype_name(tensor):
        if tensor.dtype == torch.float16:
            return "FP16"
        if tensor.dtype == torch.bfloat16:
            return "BF16"
        raise ValueError(f"unsupported LI_V2 q/k dtype: {tensor.dtype}")

    def geometry(self, q, k, layout_q, layout_k, cu_q, cu_k, seq_q, seq_k):
        q_lengths = self.prefix_lengths(cu_q) or (seq_q or [])
        k_lengths = self.prefix_lengths(cu_k) or (seq_k or [])
        if layout_q == "BSND":
            batch_size, q_seq, q_head_num, head_dim = [int(item) for item in q.shape]
            q_t_size = 0
        elif layout_q == "TND":
            q_t_size, q_head_num, head_dim = [int(item) for item in q.shape]
            batch_size = len(q_lengths)
            q_seq = max(q_lengths, default=q_t_size)
        else:
            raise ValueError(f"unsupported LI_V2 query layout: {layout_q}")

        if layout_k == "BSND":
            _, k_seq, k_head_num, _ = [int(item) for item in k.shape]
            k_t_size = 0
            block_size = 0
            block_num = 0
        elif layout_k == "TND":
            k_t_size, k_head_num, _ = [int(item) for item in k.shape]
            k_seq = max(k_lengths, default=k_t_size)
            block_size = 0
            block_num = 0
        elif layout_k == "PA_BBND":
            block_num, block_size, k_head_num, _ = [int(item) for item in k.shape]
            k_seq = max(k_lengths, default=block_size)
            k_t_size = 0
        else:
            raise ValueError(f"unsupported LI_V2 key layout: {layout_k}")
        return (
            batch_size, q_seq, k_seq, q_t_size, k_t_size,
            q_head_num, k_head_num, head_dim, block_size, block_num,
        )

    def build_case_params(self, q, k, layout_q, layout_k, kwargs):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_k = self.list_value(kwargs, "cu_seqlens_k")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_k = self.list_value(kwargs, "seqused_k")
        residual = self.list_value(kwargs, "cmp_residual_k")
        geometry = self.geometry(q, k, layout_q, layout_k, cu_q, cu_k, seq_q, seq_k)
        input_ranges = kwargs.get("input_ranges") or ()
        output_range = kwargs.get("output_idx_offset_range")
        output_range = None if output_range is None else repr(list(output_range))
        max_seqlen_q = kwargs.get("max_seqlen_q", -1)
        max_seqlen_q = -1 if max_seqlen_q is None else int(max_seqlen_q)
        return geometry + (
            self.qk_dtype_name(q),
            cu_q,
            cu_k,
            seq_q,
            seq_k,
            residual,
            output_range,
            layout_q,
            layout_k,
            int(kwargs.get("topk", 0)),
            int(kwargs.get("mask_mode", 0)),
            self.data_range(input_ranges, 0),
            self.data_range(input_ranges, 1),
            self.data_range(input_ranges, 2),
            int(kwargs.get("cmp_ratio", 1)),
            int(kwargs.get("return_value", 0)),
            max_seqlen_q,
        )

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            return
        if src is None:
            raise ValueError(f"{name} is present in CSV but pytest generator returned None")
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} "
                f"pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    @staticmethod
    def golden_data(data, layout_q, cu_seqlens_q):
        cu_q = None
        if cu_seqlens_q is not None:
            cu_q = cu_seqlens_q.detach().cpu().clone()
        return {
            "cpu_result": data["cpu_result"],
            "topk_value": data["topk_value"],
            "cpu_topk_value": data["cpu_topk_value"],
            "output_idx_offset": data.get("output_idx_offset"),
            "score_layout": layout_q,
            "cu_seqlens_q": cu_q,
        }

    def customize(self, q, k, w, cu_seqlens_q, cu_seqlens_k,
                  seqused_q, seqused_k, cmp_residual_k, block_table,
                  output_idx_offset, layout_q, layout_k, kwargs):
        params = self.build_case_params(q, k, layout_q, layout_k, kwargs)
        golden_store = self.load_golden_store().CASE_DATA
        golden_store.clear()
        data = self.load_pytest_golden().generate_liv2_test_data(params)
        for name, dst, src_name in (
            ("q", q, "query"),
            ("k", k, "key"),
            ("w", w, "weights"),
            ("cu_seqlens_q", cu_seqlens_q, "cu_seqlens_q"),
            ("cu_seqlens_k", cu_seqlens_k, "cu_seqlens_k"),
            ("seqused_q", seqused_q, "seqused_q"),
            ("seqused_k", seqused_k, "seqused_k"),
            ("cmp_residual_k", cmp_residual_k, "cmp_residual_k_for_npu"),
            ("block_table", block_table, "block_table"),
            ("output_idx_offset", output_idx_offset, "output_idx_offset"),
        ):
            self.copy_tensor(dst, data.get(src_name), name)
        testcase_name = kwargs.get("testcase_name")
        golden_store.put(
            testcase_name,
            self.golden_data(data, layout_q, cu_seqlens_q),
        )
        return data


INPUT_ADAPTER = LightningIndexerV2InputAdapter()


def generate_li_v2_inputs(q, k, w, *, cu_seqlens_q=None, cu_seqlens_k=None,
                          seqused_q=None, seqused_k=None, cmp_residual_k=None,
                          block_table=None, output_idx_offset=None,
                          layout_q="BSND", layout_k="BSND", **kwargs):
    """Generate the exact pytest inputs and CPU golden for a TTK case."""
    INPUT_ADAPTER.customize(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_q,
        layout_k,
        kwargs,
    )
