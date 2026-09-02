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

__input__ = {"e2e": {"torch_npu.npu_quant_lightning_indexer": "generate_qli_inputs"}}

import importlib.util
import sys
from pathlib import Path

import torch


class QuantLightningIndexerInputAdapter:
    """Translate a TTK case and reuse the canonical pytest case generator."""

    PYTEST_FIELDS = (
        "batch_size",
        "q_seq",
        "k_seq",
        "q_t_size",
        "k_t_size",
        "q_head_num",
        "k_head_num",
        "head_dim",
        "block_size",
        "block_num",
        "qk_dtype",
        "weight_dtype",
        "dequant_dtype",
        "actual_seq_dtype",
        "act_seq_q",
        "act_seq_k",
        "query_quant_mode",
        "key_quant_mode",
        "layout_query",
        "layout_key",
        "sparse_count",
        "sparse_mode",
        "query_datarange",
        "key_datarange",
        "weights_datarange",
        "q_scale_datarange",
        "k_scale_datarange",
    )

    INTEGER_FIELDS = (
        "batch_size",
        "q_seq",
        "k_seq",
        "q_t_size",
        "k_t_size",
        "q_head_num",
        "k_head_num",
        "head_dim",
        "block_size",
        "block_num",
        "query_quant_mode",
        "key_quant_mode",
        "sparse_count",
        "sparse_mode",
    )

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load QuantLightningIndexer module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_golden = None

    @staticmethod
    def load_golden_store():
        name = "qli_ttk_golden"
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
            raise QuantLightningIndexerInputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "quant_lightning_indexer_golden.py"
        name = "qli_pytest_golden"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
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
            self.pytest_golden = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("pytest Golden", path, exc) from exc
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    @staticmethod
    def list_value(value):
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif not isinstance(value, (list, tuple)):
            value = [value]
        return [int(item) for item in value]

    @staticmethod
    def torch_dtype(value, field):
        if isinstance(value, torch.dtype):
            return value
        normalized = str(value).strip().lower().removeprefix("torch.")
        mapping = {
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int32": torch.int32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float8_e4m3fn": torch.float8_e4m3fn,
        }
        if normalized not in mapping:
            raise ValueError(f"unsupported QuantLightningIndexer {field}: {value!r}")
        return mapping[normalized]

    def build_case_params(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        query_quant_mode,
        key_quant_mode,
        layout_query,
        layout_key,
        kwargs,
    ):
        missing = [
            f"pytest_{name}"
            for name in self.PYTEST_FIELDS
            if f"pytest_{name}" not in kwargs
        ]
        if missing:
            raise ValueError(
                f"QuantLightningIndexer CSV is missing explicit pytest fields: {missing}"
            )
        params = {name: kwargs[f"pytest_{name}"] for name in self.PYTEST_FIELDS}
        for name in self.INTEGER_FIELDS:
            if params[name] is not None:
                params[name] = int(params[name])
        for name in ("qk_dtype", "weight_dtype", "dequant_dtype", "actual_seq_dtype"):
            params[name] = self.torch_dtype(params[name], name)
        params["act_seq_q"] = self.list_value(params["act_seq_q"])
        params["act_seq_k"] = self.list_value(params["act_seq_k"])
        for name in (
            "query_datarange",
            "key_datarange",
            "weights_datarange",
            "q_scale_datarange",
            "k_scale_datarange",
        ):
            value = params[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"QuantLightningIndexer {name} must contain two values"
                )
            params[name] = list(value)

        expected_api = {
            "query_quant_mode": int(query_quant_mode),
            "key_quant_mode": int(key_quant_mode),
            "layout_query": layout_query,
            "layout_key": layout_key,
            "sparse_count": int(kwargs["sparse_count"]),
            "sparse_mode": int(kwargs["sparse_mode"]),
        }
        mismatches = {
            name: (params[name], expected)
            for name, expected in expected_api.items()
            if params[name] != expected
        }
        if mismatches:
            raise ValueError(
                "QuantLightningIndexer API attributes differ from pytest fields: "
                f"{mismatches}"
            )
        expected_dtypes = (
            (query, params["qk_dtype"], "query"),
            (key, params["qk_dtype"], "key"),
            (weights, params["weight_dtype"], "weights"),
            (query_dequant_scale, params["dequant_dtype"], "query_dequant_scale"),
            (key_dequant_scale, params["dequant_dtype"], "key_dequant_scale"),
            (
                actual_seq_lengths_query,
                params["actual_seq_dtype"],
                "actual_seq_lengths_query",
            ),
            (
                actual_seq_lengths_key,
                params["actual_seq_dtype"],
                "actual_seq_lengths_key",
            ),
        )
        for tensor, expected_dtype, name in expected_dtypes:
            if tensor is not None and tensor.dtype != expected_dtype:
                raise ValueError(
                    f"{name} dtype differs from pytest field: "
                    f"TTK={tensor.dtype}, pytest={expected_dtype}"
                )
        return tuple(params[name] for name in self.PYTEST_FIELDS)

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
                f"{name} shape mismatch: TTK={tuple(dst.shape)} "
                f"pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    @staticmethod
    def golden_data(data):
        return {
            "cpu_result": data["cpu_result"].detach().cpu(),
            "score_values": data["score_values"].detach().cpu(),
            "params": data["params"],
            "topk_value": data["topk_value"].detach().cpu(),
        }

    @staticmethod
    def restore_paged_tensor(tensor, block_table, batch_size, sequence_length):
        """Restore a paged key or scale tensor to its logical BNS(D) layout."""
        physical = tensor.detach().cpu()
        table = block_table.detach().cpu().to(torch.int64)
        if physical.ndim not in (3, 4):
            raise ValueError(
                f"paged tensor must be 3-D or 4-D, got shape {tuple(physical.shape)}"
            )
        block_size, head_num = int(physical.shape[1]), int(physical.shape[2])
        trailing = (int(physical.shape[3]),) if physical.ndim == 4 else ()
        logical = torch.zeros(
            (batch_size, head_num, sequence_length, *trailing), dtype=physical.dtype
        )
        for batch_idx in range(batch_size):
            for logical_block, block_id_value in enumerate(table[batch_idx].tolist()):
                if block_id_value < 0:
                    continue
                if block_id_value >= physical.shape[0]:
                    raise ValueError(
                        f"block id {block_id_value} exceeds paged block count"
                    )
                start = logical_block * block_size
                if start >= sequence_length:
                    break
                count = min(block_size, sequence_length - start)
                block = physical[block_id_value, :count]
                permutation = (1, 0, 2) if physical.ndim == 4 else (1, 0)
                logical[batch_idx, :, start : start + count] = block.permute(
                    *permutation
                )
        return logical

    def rebuild_compare_data(self, compare_context):
        """Rebuild score context from replayed inputs without random generation."""
        tensors = tuple(compare_context.input_tensors or ())
        if len(tensors) < 8:
            raise ValueError(
                "QuantLightningIndexer compare context requires eight tensor slots"
            )
        query, key, weights, query_scale, key_scale = tensors[:5]
        actual_q, actual_k, block_table = tensors[5:8]
        attrs = dict(compare_context.attributes)
        attrs["actual_seq_lengths_query_values"] = self.list_value(actual_q)
        attrs["actual_seq_lengths_key_values"] = self.list_value(actual_k)
        layout_query = attrs.get("layout_query", "BSND")
        layout_key = attrs.get("layout_key", "BSND")
        query_quant_mode = int(attrs.get("query_quant_mode", 0))
        key_quant_mode = int(attrs.get("key_quant_mode", 0))
        params = self.build_case_params(
            query,
            key,
            weights,
            query_scale,
            key_scale,
            actual_q,
            actual_k,
            query_quant_mode,
            key_quant_mode,
            layout_query,
            layout_key,
            attrs,
        )
        model = self.load_pytest_golden().GeneralizedQLI(*params[:22])
        key_for_cpu = key
        key_scale_for_cpu = key_scale
        if layout_key == "PA_BSND":
            if block_table is None:
                raise ValueError("PA_BSND compare context requires block_table")
            sequence_length = max(params[15])
            key_for_cpu = self.restore_paged_tensor(
                key, block_table, params[0], sequence_length
            )
            key_scale_for_cpu = self.restore_paged_tensor(
                key_scale, block_table, params[0], sequence_length
            )
        _, scores_bnsd = model.forward(
            query,
            key_for_cpu,
            weights,
            query_scale,
            key_scale_for_cpu,
            actual_q,
            actual_k,
            block_table,
        )
        scores = model.trans_bnsd_to_layout(
            scores_bnsd,
            list(scores_bnsd.shape),
            layout_query,
            model.actual_seq_lengths_query,
        )
        return {
            "params": params,
            "topk_value": scores_bnsd.detach().cpu(),
            "score_values": scores.detach().cpu(),
        }

    def customize(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        layout_query,
        layout_key,
        kwargs,
    ):
        params = self.build_case_params(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            query_quant_mode,
            key_quant_mode,
            layout_query,
            layout_key,
            kwargs,
        )
        golden_store = self.load_golden_store().CASE_DATA
        golden_store.clear()
        data = self.load_pytest_golden().generate_qli_test_data(params)
        for name, dst in (
            ("query", query),
            ("key", key),
            ("weights", weights),
            ("query_dequant_scale", query_dequant_scale),
            ("key_dequant_scale", key_dequant_scale),
            ("actual_seq_lengths_query", actual_seq_lengths_query),
            ("actual_seq_lengths_key", actual_seq_lengths_key),
            ("block_table", block_table),
        ):
            self.copy_tensor(dst, data.get(name), name)
        golden_store.put(kwargs.get("testcase_name"), self.golden_data(data))


INPUT_ADAPTER = QuantLightningIndexerInputAdapter()


def rebuild_qli_compare_data(compare_context):
    return INPUT_ADAPTER.rebuild_compare_data(compare_context)


def generate_qli_inputs(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    query_quant_mode,
    key_quant_mode,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="BSND",
    **kwargs,
):
    """Generate the exact canonical pytest inputs and CPU golden for a TTK case."""
    INPUT_ADAPTER.customize(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        layout_query,
        layout_key,
        kwargs,
    )
