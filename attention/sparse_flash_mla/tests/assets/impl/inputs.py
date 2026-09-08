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

"""Input customization for SparseFlashMla TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch


class SparseFlashMlaInputAdapter:
    """Translate a TTK case to pytest parameters and reuse pytest generation."""

    TEMPLATE_MODES = frozenset(("SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"))

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load SparseFlashMla module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_modules = {}
        self.batch_consistency = None

    def load_batch_consistency(self):
        if self.batch_consistency is not None:
            return self.batch_consistency
        name = "smla_ttk_batch_consistency"
        path = Path(__file__).with_name("batch_consistency.py")
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.batch_consistency = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("assets batch consistency", path, exc) from exc

    @staticmethod
    def load_golden_store():
        name = "smla_ttk_golden"
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
            raise SparseFlashMlaInputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    @staticmethod
    def load_metadata_protocol():
        name = "smla_ttk_metadata_protocol"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("metadata_protocol.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise SparseFlashMlaInputAdapter.module_load_error(
                "assets metadata protocol", path, exc
            ) from exc
        return module

    def load_pytest_module(self, stem, filename):
        if stem in self.pytest_modules:
            return self.pytest_modules[stem]

        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        module_path = pytest_dir / filename
        name = f"smla_pytest_{stem}"
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
        return repr(list(value))

    @staticmethod
    def select_topk_override(name, tensor, mask_mode, attributes):
        """Use an explicit CSV override, otherwise keep pytest's generated value."""
        if attributes is None:
            return tensor
        if hasattr(attributes, "attributes"):
            attributes = attributes.attributes
        attributes = attributes or {}
        if name in attributes:
            return attributes[name]
        if mask_mode != 0:
            return tensor
        return None

    @classmethod
    def select_template_mode(cls, kwargs):
        mode = kwargs.get("template_run_mode") or kwargs.get("template_mode")
        if mode is None:
            return None
        mode = str(mode).strip()
        if mode not in cls.TEMPLATE_MODES:
            raise ValueError(f"unsupported explicit template_run_mode: {mode!r}")
        return mode

    @staticmethod
    def verify_template_inputs(mode, ori_sparse_indices, cmp_sparse_indices, cmp_kv):
        if mode in ("ORI_SPARSE", "ORI_CMP_SPARSE") and ori_sparse_indices is None:
            raise ValueError(f"{mode} requires ori_sparse_indices in the CSV")
        if mode in ("CSA", "ORI_CMP_SPARSE") and cmp_sparse_indices is None:
            raise ValueError(f"{mode} requires cmp_sparse_indices in the CSV")
        if mode in ("SWA", "ORI_SPARSE") and cmp_kv is not None:
            raise ValueError(f"{mode} must not provide cmp_kv in the CSV")
        if mode in ("HCA", "CSA", "ORI_CMP_SPARSE") and cmp_kv is None:
            raise ValueError(f"{mode} requires cmp_kv in the CSV")

    def build_case_params(
        self,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        layout_q,
        layout_kv,
        kwargs,
    ):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_ori = self.list_value(kwargs, "cu_seqlens_ori_kv")
        cu_cmp = self.list_value(kwargs, "cu_seqlens_cmp_kv")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_ori = self.list_value(kwargs, "seqused_ori_kv")
        seq_cmp = self.list_value(kwargs, "seqused_cmp_kv")
        residual = self.list_value(kwargs, "cmp_residual_kv")

        if layout_q == "BSND":
            batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
            q_total = batch_size * q_seq
        else:
            q_total, q_heads, head_dim = [int(x) for x in q.shape]
            batch_size = len(seq_q or self.prefix_lengths(cu_q))
            q_seq = max(self.prefix_lengths(cu_q) or seq_q or [q_total])
        if kwargs.get("S1") is not None:
            q_seq = int(kwargs["S1"])

        if layout_kv == "BSND":
            _, kv_seq, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_total = batch_size * kv_seq
            block_num1, block_size1 = None, None
        elif layout_kv == "TND":
            kv_total, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [kv_total])
            block_num1, block_size1 = None, None
        else:
            block_num1, block_size1, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(seq_ori or [block_size1])
            kv_total = sum(seq_ori or [])
        if kwargs.get("S2") is not None:
            kv_seq = int(kwargs["S2"])

        if cmp_kv is None:
            cmp_total = 0
            block_num2, block_size2 = None, None
        elif layout_kv == "BSND":
            cmp_total = batch_size * int(cmp_kv.shape[1])
            block_num2, block_size2 = None, None
        elif layout_kv == "TND":
            cmp_total = int(cmp_kv.shape[0])
            block_num2, block_size2 = None, None
        else:
            block_num2, block_size2 = int(cmp_kv.shape[0]), int(cmp_kv.shape[1])
            cmp_total = sum(seq_cmp or [])
        mode = self.select_template_mode(kwargs)
        if mode is not None:
            self.verify_template_inputs(
                mode, ori_sparse_indices, cmp_sparse_indices, cmp_kv
            )

        input_ranges = kwargs.get("input_ranges") or ()
        params = dict(kwargs)
        data_ranges = {
            "q_datarange": self.data_range(input_ranges, 0),
            "ori_kv_datarange": self.data_range(input_ranges, 1),
            "cmp_kv_datarange": self.data_range(input_ranges, 2),
        }
        params.update(
            {
                "testcase_name": kwargs.get("testcase_name"),
                "layout_q": layout_q,
                "layout_kv": layout_kv,
                "q_type": q.dtype,
                "ori_kv_type": ori_kv.dtype,
                "cmp_kv_type": cmp_kv.dtype if cmp_kv is not None else None,
                "B": batch_size,
                "S1": q_seq,
                "S2": kv_seq,
                "T1": q_total,
                "T2": kv_total,
                "T3": cmp_total,
                "N1": q_heads,
                "N2": kv_heads,
                "D": head_dim,
                "K1": int(ori_sparse_indices.shape[-1])
                if ori_sparse_indices is not None
                else None,
                "K": int(cmp_sparse_indices.shape[-1])
                if cmp_sparse_indices is not None
                else None,
                "block_num1": block_num1,
                "block_num2": block_num2,
                "block_size1": block_size1,
                "block_size2": block_size2,
                "cu_seqlens_q": cu_q,
                "cu_seqlens_ori_kv": cu_ori,
                "cu_seqlens_cmp_kv": cu_cmp,
                "seqused_q": seq_q,
                "seqused_ori_kv": seq_ori,
                "seqused_cmp_kv": seq_cmp,
                "cmp_residual_kv": residual,
                "template_mode": mode,
                "cmp_ratio": (
                    kwargs.get("pytest_cmp_ratio")
                    if "pytest_cmp_ratio" in kwargs
                    else kwargs.get("cmp_ratio")
                ),
                "cmp_mask_mode": kwargs.get("cmp_mask_mode"),
            }
        )
        params.update(
            {name: value for name, value in data_ranges.items() if value is not None}
        )
        return params

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            if src is not None:
                import logging

                logging.warning(
                    f"{name} is absent from CSV but pytest generator produced a tensor. Skipping copy."
                )
            return
        if src is None:
            raise ValueError(
                f"{name} is present in CSV but pytest generator returned None"
            )
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    def generate_case(
        self, params, batch_random=None, legacy_empty_actual_length_slots=()
    ):
        pytest_utils = self.load_pytest_module("utils", "utils.py")
        pytest_golden = self.load_pytest_module("golden", "sparse_flash_mla_golden.py")
        pytest_input = {name: [value] for name, value in params.items()}
        if params.get("actlen_mode") in ("等长", "不等长"):
            # Historical TTK labels describe the saved vectors, which remain authoritative.
            pytest_input["actlen_mode"] = ["full"]
        param_combinations = pytest_utils.generate_param_combinations([pytest_input])
        if len(param_combinations) != 1:
            raise ValueError(
                f"expected one pytest parameter combination, got {len(param_combinations)}"
            )
        case_params = pytest_utils.generate_case_with_default_param(
            param_combinations[0]
        )
        # Select the batch reference without enabling pytest's own relation generation.
        case_params["batch_consistency"] = params.get(
            "batch_consistency", batch_random is not None
        )
        for name in legacy_empty_actual_length_slots:
            case_params[name] = None
        if batch_random is None:
            data = pytest_golden.gen_data(
                case_params, prepare_device_storage=False, generate_golden=False
            )
        else:
            batch_random.validate_params(case_params)
            with batch_random:
                data = pytest_golden.gen_data(
                    case_params, prepare_device_storage=False, generate_golden=False
                )
            batch_random.normalize_block_tables(data)
        return data

    def customize(
        self,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        layout_q,
        layout_kv,
        kwargs,
        batch_random=None,
        legacy_empty_actual_length_slots=(),
    ):
        params = self.build_case_params(
            q,
            ori_kv,
            cmp_kv,
            ori_sparse_indices,
            cmp_sparse_indices,
            layout_q,
            layout_kv,
            kwargs,
        )
        return self.generate_case(
            params, batch_random, legacy_empty_actual_length_slots
        )


INPUT_ADAPTER = SparseFlashMlaInputAdapter()


def is_negative_case(kwargs):
    value = kwargs.get("is_negative_case", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def zero_metadata(metadata):
    if metadata is None:
        return
    if torch.is_tensor(metadata):
        metadata.zero_()
    else:
        metadata[...] = 0


def generate_sparse_flash_mla_inputs(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    **kwargs,
):
    """Populate pytest-derived inputs; metadata is filled by npu_preprocess."""
    if is_negative_case(kwargs):
        return None

    params = dict(kwargs)
    params.update(
        {
            "softmax_scale": softmax_scale,
            "cmp_ratio": cmp_ratio,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "topk_value_mode": topk_value_mode,
            "return_softmax_lse": return_softmax_lse,
            "ori_topk_length": INPUT_ADAPTER.select_topk_override(
                "ori_topk_length", ori_topk_length, ori_mask_mode, params
            ),
            "cmp_topk_length": INPUT_ADAPTER.select_topk_override(
                "cmp_topk_length", cmp_topk_length, cmp_mask_mode, params
            ),
        }
    )
    batch_module = INPUT_ADAPTER.load_batch_consistency()
    batch_random = batch_module.TorchBatchRandomContext.from_case(
        q, ori_kv, cmp_kv, params
    )
    legacy_empty_actual_length_slots = []
    if (
        layout_q == "TND"
        and seqused_q is None
        and INPUT_ADAPTER.list_value(params, "seqused_q") is None
    ):
        legacy_empty_actual_length_slots.append("seqused_q")
    if (
        layout_kv == "TND"
        and INPUT_ADAPTER.list_value(params, "cu_seqlens_cmp_kv") is not None
        and seqused_ori_kv is None
        and INPUT_ADAPTER.list_value(params, "seqused_ori_kv") is None
    ):
        for name, tensor in (
            ("seqused_ori_kv", seqused_ori_kv),
            ("seqused_cmp_kv", seqused_cmp_kv),
            ("cmp_residual_kv", cmp_residual_kv),
        ):
            if tensor is None and INPUT_ADAPTER.list_value(params, name) is None:
                legacy_empty_actual_length_slots.append(name)
    data = INPUT_ADAPTER.customize(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        layout_q,
        layout_kv,
        params,
        batch_random,
        tuple(legacy_empty_actual_length_slots),
    )
    op_input = data["input"]
    metadata_input = data.get("metadata_input", {})
    for name, tensor in (
        ("q", q),
        ("ori_kv", ori_kv),
        ("cmp_kv", cmp_kv),
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
        source = op_input.get(name)
        if name == "cu_seqlens_q" and layout_q != "TND":
            source = None
        if name in ("ori_topk_length", "cmp_topk_length") and source is None:
            source = metadata_input.get(name)
        INPUT_ADAPTER.copy_tensor(tensor, source, name)
    zero_metadata(metadata)
    case_data = INPUT_ADAPTER.load_golden_store().CASE_DATA
    testcase_name = params.get("testcase_name")
    case_data.put(testcase_name, data)
    INPUT_ADAPTER.load_metadata_protocol().save_metadata_inputs(
        "sparse_flash_mla", testcase_name, metadata_input
    )
    return data


def generate_aclnn_sparse_flash_mla_inputs(
    q,
    ori_kv,
    cmp_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_topk_length,
    cmp_topk_length,
    sinks,
    metadata,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
    topk_value_mode,
    return_softmax_lse,
    attn_out,
    softmax_lse_out,
    **kwargs,
):
    """Map the ACLNN C signature to the canonical pytest input adapter."""
    del attn_out, softmax_lse_out
    return generate_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=topk_value_mode,
        return_softmax_lse=return_softmax_lse,
        **kwargs,
    )
