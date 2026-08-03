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

"""Input customization for QuantSparseFlashMla TTK cases."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


class QuantSparseFlashMlaInputAdapter:
    """Translate a TTK case to pytest parameters and reuse pytest generation."""

    TEMPLATE_MODES = frozenset(("SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"))

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load QuantSparseFlashMla module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_modules = {}

    @staticmethod
    def load_golden_store():
        name = "qsmla_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise QuantSparseFlashMlaInputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    def load_pytest_module(self, stem, filename):
        if stem in self.pytest_modules:
            return self.pytest_modules[stem]

        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        module_path = pytest_dir / filename
        name = f"qsmla_pytest_{stem}"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.pytest_modules[stem] = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error(f"pytest {stem}", module_path, exc) from exc
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
    def tensor_dtype(tensor):
        if torch.is_tensor(tensor):
            return tensor.dtype
        if "hifloat8" in str(tensor.dtype):
            return torch.uint8
        return torch.from_numpy(np.asarray(tensor)).dtype

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[i + 1]) - int(value[i]) for i in range(len(value) - 1)]

    @staticmethod
    def data_range(input_ranges, index):
        if not input_ranges or index >= len(input_ranges):
            return None
        value = input_ranges[index]
        if value is None or any(item is None for item in value):
            return None
        return list(value)

    @classmethod
    def select_template_mode(cls, kwargs):
        mode = kwargs.get("template_run_mode") or kwargs.get("template_mode")
        if mode is None:
            return None
        mode = str(mode).strip()
        if mode not in cls.TEMPLATE_MODES:
            raise ValueError(f"unsupported explicit template_run_mode: {mode!r}")
        return mode

    def build_case_params(self, q, ori_kv, cmp_kv, ori_sparse_indices,
                          cmp_sparse_indices, sinks, layout_q, layout_kv, kwargs):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_ori = self.list_value(kwargs, "cu_seqlens_ori_kv")
        cu_cmp = self.list_value(kwargs, "cu_seqlens_cmp_kv")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_ori = self.list_value(kwargs, "seqused_ori_kv")
        seq_cmp = self.list_value(kwargs, "seqused_cmp_kv")
        residual = self.list_value(kwargs, "cmp_residual_kv")

        if layout_q == "BSND":
            batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
        else:
            _, q_heads, head_dim = [int(x) for x in q.shape]
            batch_size = len(seq_q or self.prefix_lengths(cu_q))
            q_seq = max(self.prefix_lengths(cu_q) or seq_q or [int(q.shape[0])])
        if kwargs.get("S1") is not None:
            q_seq = int(kwargs["S1"])

        if layout_kv == "BSND":
            _, kv_seq, kv_heads, _ = [int(x) for x in ori_kv.shape]
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
        elif layout_kv == "TND":
            _, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [int(ori_kv.shape[0])])
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
        else:
            shape_block_num, shape_block_size, kv_heads, _ = [int(x) for x in ori_kv.shape]
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
            block_num1 = shape_block_num if block_num1 is None else int(block_num1)
            block_size1 = shape_block_size if block_size1 is None else int(block_size1)
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [block_size1])

        metadata_cmp_topk = kwargs.get("metadata_cmp_topk")
        requested_mode = self.select_template_mode(kwargs)
        if cmp_kv is None:
            block_num2 = kwargs.get("block_num2")
            block_size2 = kwargs.get("block_size2")
        elif layout_kv == "PA_BBND":
            block_num2 = kwargs.get("block_num2")
            block_size2 = kwargs.get("block_size2")
            block_num2 = int(cmp_kv.shape[0]) if block_num2 is None else int(block_num2)
            block_size2 = int(cmp_kv.shape[1]) if block_size2 is None else int(block_size2)
        else:
            block_num2 = kwargs.get("block_num2")
            block_size2 = kwargs.get("block_size2")

        params = dict(kwargs)
        input_ranges = kwargs.get("input_ranges") or ()
        data_ranges = {
            "q_datarange": self.data_range(input_ranges, 0),
            "ori_kv_datarange": self.data_range(input_ranges, 1),
            "cmp_kv_datarange": self.data_range(input_ranges, 2),
        }
        params.update({
            "Testcase_Name": kwargs.get("testcase_name"),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "q_type": self.tensor_dtype(q),
            "ori_kv_type": self.tensor_dtype(ori_kv),
            "cmp_kv_type": self.tensor_dtype(cmp_kv) if cmp_kv is not None else None,
            "B": batch_size,
            "S1": q_seq,
            "S2": kwargs.get("S2") if kwargs.get("S2") is not None else kv_seq,
            "N1": q_heads,
            "N2": kv_heads,
            "D": head_dim,
            "K1": (
                int(ori_sparse_indices.shape[-1])
                if ori_sparse_indices is not None
                else kwargs.get("K1")
            ),
            "K": (
                metadata_cmp_topk
                if metadata_cmp_topk is not None
                else (
                    int(cmp_sparse_indices.shape[-1])
                    if cmp_sparse_indices is not None
                    else None
                )
            ),
            "block_num1": block_num1,
            "block_num2": block_num2,
            "block_size1": block_size1,
            "block_size2": block_size2,
            "seqused_q": seq_q,
            "cu_seqlens_q": cu_q,
            "seqused_ori_kv": seq_ori,
            "seqused_cmp_kv": seq_cmp,
            "cu_seqlens_ori_kv": cu_ori,
            "cu_seqlens_cmp_kv": cu_cmp,
            "cmp_residual_kv": residual,
            "cmp_ratio": kwargs.get("cmp_ratio"),
            "cmp_mask_mode": kwargs.get("cmp_mask_mode"),
            "isSink": sinks is not None,
        })
        pytest_cmp_mask_mode = kwargs.get("pytest_cmp_mask_mode")
        if pytest_cmp_mask_mode is not None:
            params["cmp_mask_mode"] = int(pytest_cmp_mask_mode)
        if requested_mode is not None:
            params["template_run_mode"] = requested_mode
        for name, value in data_ranges.items():
            if params.get(name) is None and value is not None:
                params[name] = value
        return params

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            if src is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest generator produced a tensor"
                )
            return
        if src is None:
            message = f"{name} is present in CSV but pytest generator returned None"
            if name in {"cu_seqlens_q", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv"}:
                message += (
                    "; check the CSV tensor slot, the corresponding *_values or "
                    "derivable seqused_* values, and the layout/template_run_mode branch"
                )
            raise ValueError(message)
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else src
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} pytest={tuple(src_cpu.shape)}"
            )
        if torch.is_tensor(dst):
            src_tensor = torch.as_tensor(src_cpu)
            dst.copy_(src_tensor.to(dtype=dst.dtype, device=dst.device))
            return

        dst_array = np.asarray(dst)
        src_array = np.asarray(src_cpu)
        if "hifloat8" in str(dst_array.dtype):
            np.copyto(dst_array.view(np.uint8), src_array.view(np.uint8))
        else:
            np.copyto(dst_array, src_array.astype(dst_array.dtype, copy=False))

    def generate_case(self, params):
        pytest_utils = self.load_pytest_module("utils", "utils.py")
        pytest_check = self.load_pytest_module("check", "check_valid_param.py")
        pytest_golden = self.load_pytest_module(
            "golden", "quant_sparse_flash_mla_golden.py"
        )
        filled = pytest_utils.fill_none_params(params)
        for name in ("q_datarange", "ori_kv_datarange", "cmp_kv_datarange"):
            if params.get(name) is not None:
                filled[name] = params[name]
        if filled.get("Testcase_Name") is None:
            filled["Testcase_Name"] = params.get("testcase_name")
        pytest_check.check_valid_param(filled)
        data = pytest_golden.generate_and_save_testdata(filled, save_pt=False)
        testcase_name = params.get("testcase_name") or filled.get("Testcase_Name")
        self.load_golden_store().CASE_DATA.put(testcase_name, data)
        return data


INPUT_ADAPTER = QuantSparseFlashMlaInputAdapter()


def generate_quant_sparse_flash_mla_inputs(
        q, *, ori_kv=None, cmp_kv=None, q_descale=None,
        ori_kv_descale=None, cmp_kv_descale=None, ori_sparse_indices=None,
        cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None,
        cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
        seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
        cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
        sinks=None, **kwargs):
    """Reuse the pytest parameter validation and input processing for a TTK case."""
    params = INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        kwargs.get("layout_q"),
        kwargs.get("layout_kv"),
        kwargs,
    )
    data = INPUT_ADAPTER.generate_case(params)
    op_input = data["op_input"]
    for name, tensor in (
        ("q", q),
        ("ori_kv", ori_kv),
        ("cmp_kv", cmp_kv),
        ("q_descale", q_descale),
        ("ori_kv_descale", ori_kv_descale),
        ("cmp_kv_descale", cmp_kv_descale),
        ("ori_sparse_indices", ori_sparse_indices),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_block_table", ori_block_table),
        ("cmp_block_table", cmp_block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_ori_kv", cu_seqlens_ori_kv),
        ("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv),
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("cmp_residual_kv", cmp_residual_kv),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("sinks", sinks),
    ):
        INPUT_ADAPTER.copy_tensor(tensor, op_input.get(name), name)
