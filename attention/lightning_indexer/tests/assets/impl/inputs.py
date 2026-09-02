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

__input__ = {
    "e2e": {
        "torch_npu.npu_lightning_indexer": "generate_li_inputs",
    },
    "aclnn": {
        "aclnnLightningIndexer": "generate_li_inputs_aclnn",
    },
}

import importlib.util
import sys
from pathlib import Path

import torch


class LightningIndexerInputAdapter:
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
        "actual_seq_dtype",
        "act_seq_q",
        "act_seq_k",
        "layout_query",
        "layout_key",
        "sparse_count",
        "sparse_mode",
        "query_datarange",
        "key_datarange",
        "weights_datarange",
        "return_value",
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
        "sparse_count",
        "sparse_mode",
    )

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load LightningIndexer module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_golden = None

    @staticmethod
    def load_golden_store():
        name = "li_ttk_golden"
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
            raise LightningIndexerInputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "lightning_indexer_golden.py"
        name = "li_pytest_golden"
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
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "int32": torch.int32,
        }
        if normalized not in mapping:
            raise ValueError(f"unsupported LightningIndexer {field}: {value!r}")
        return mapping[normalized]

    def build_case_params(
        self,
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        return_value,
        kwargs,
    ):
        missing = [
            f"pytest_{name}"
            for name in self.PYTEST_FIELDS
            if f"pytest_{name}" not in kwargs
        ]
        if missing:
            raise ValueError(
                f"LightningIndexer CSV is missing explicit pytest fields: {missing}"
            )
        params = {name: kwargs[f"pytest_{name}"] for name in self.PYTEST_FIELDS}
        for name in self.INTEGER_FIELDS:
            if params[name] is not None:
                params[name] = int(params[name])
        params["qk_dtype"] = self.torch_dtype(params["qk_dtype"], "qk_dtype")
        params["weight_dtype"] = self.torch_dtype(
            params["weight_dtype"], "weight_dtype"
        )
        params["actual_seq_dtype"] = self.torch_dtype(
            params["actual_seq_dtype"], "actual_seq_dtype"
        )
        params["act_seq_q"] = self.list_value(params["act_seq_q"])
        params["act_seq_k"] = self.list_value(params["act_seq_k"])
        for name in ("query_datarange", "key_datarange", "weights_datarange"):
            value = params[name]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"LightningIndexer {name} must contain two values")
            params[name] = list(value)
        params["return_value"] = bool(params["return_value"])

        expected_api = {
            "layout_query": layout_query,
            "layout_key": layout_key,
            "sparse_count": sparse_count,
            "sparse_mode": sparse_mode,
            "return_value": return_value,
        }
        mismatches = {
            name: (params[name], expected)
            for name, expected in expected_api.items()
            if params[name] != expected
        }
        if mismatches:
            raise ValueError(
                f"LightningIndexer API attributes differ from pytest fields: {mismatches}"
            )
        expected_dtypes = (
            (query, params["qk_dtype"], "query"),
            (key, params["qk_dtype"], "key"),
            (weights, params["weight_dtype"], "weights"),
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
        return (
            params["batch_size"],
            params["q_seq"],
            params["k_seq"],
            params["q_t_size"],
            params["k_t_size"],
            params["q_head_num"],
            params["k_head_num"],
            params["head_dim"],
            params["block_size"],
            params["block_num"],
            params["qk_dtype"],
            params["weight_dtype"],
            params["actual_seq_dtype"],
            params["act_seq_q"],
            params["act_seq_k"],
            params["layout_query"],
            params["layout_key"],
            params["sparse_count"],
            params["sparse_mode"],
            params["query_datarange"],
            params["key_datarange"],
            params["weights_datarange"],
            params["return_value"],
        )

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
            "value_dtype": data["query"].dtype,
        }

    @staticmethod
    def restore_paged_tensor(tensor, block_table, batch_size, sequence_length):
        """Restore the logical BNSD tensor consumed by the pytest CPU model."""
        physical = tensor.detach().cpu()
        table = block_table.detach().cpu().to(torch.int64)
        if physical.ndim != 4:
            raise ValueError(
                f"paged key must be 4-D, got shape {tuple(physical.shape)}"
            )
        block_size, head_num, head_dim = (
            int(physical.shape[1]),
            int(physical.shape[2]),
            int(physical.shape[3]),
        )
        logical = torch.zeros(
            (batch_size, head_num, sequence_length, head_dim), dtype=physical.dtype
        )
        for batch_idx in range(batch_size):
            for logical_block, block_id_value in enumerate(table[batch_idx].tolist()):
                if block_id_value < 0:
                    continue
                if block_id_value >= physical.shape[0]:
                    raise ValueError(
                        f"block id {block_id_value} exceeds key block count"
                    )
                start = logical_block * block_size
                if start >= sequence_length:
                    break
                count = min(block_size, sequence_length - start)
                logical[batch_idx, :, start : start + count, :] = physical[
                    block_id_value, :count
                ].permute(1, 0, 2)
        return logical

    def rebuild_compare_data(self, compare_context):
        """Rebuild score context from replayed inputs without regenerating a case."""
        tensors = tuple(compare_context.input_tensors or ())
        if len(tensors) < 6:
            raise ValueError(
                "LightningIndexer compare context requires six tensor slots"
            )
        query, key, weights, actual_q, actual_k, block_table = tensors[:6]
        attrs = dict(compare_context.attributes)
        attrs["pytest_act_seq_q"] = self.list_value(actual_q)
        attrs["pytest_act_seq_k"] = self.list_value(actual_k)
        layout_query = attrs["layout_query"]
        layout_key = attrs["layout_key"]
        params = self.build_case_params(
            query,
            key,
            weights,
            actual_q,
            actual_k,
            layout_query,
            layout_key,
            attrs,
        )
        model = self.load_pytest_golden().GeneralizedLI(*params[:19])
        key_for_cpu = key
        if layout_key == "PA_BSND":
            if block_table is None:
                raise ValueError("PA_BSND compare context requires block_table")
            key_for_cpu = self.restore_paged_tensor(
                key, block_table, params[0], max(params[14])
            )
        _, scores_bnsd = model.forward(
            query, key_for_cpu, weights, actual_q, actual_k, block_table
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
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        return_value,
        kwargs,
    ):
        params = self.build_case_params(
            query,
            key,
            weights,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            layout_query,
            layout_key,
            sparse_count,
            sparse_mode,
            return_value,
            kwargs,
        )
        golden_store = self.load_golden_store().CASE_DATA
        golden_store.clear()
        data = self.load_pytest_golden().generate_li_test_data(params)
        for name, dst in (
            ("query", query),
            ("key", key),
            ("weights", weights),
            ("actual_seq_lengths_query", actual_seq_lengths_query),
            ("actual_seq_lengths_key", actual_seq_lengths_key),
            ("block_table", block_table),
        ):
            self.copy_tensor(dst, data.get(name), name)
        golden_store.put(kwargs.get("testcase_name"), self.golden_data(data))


INPUT_ADAPTER = LightningIndexerInputAdapter()


def rebuild_li_compare_data(compare_context):
    return INPUT_ADAPTER.rebuild_compare_data(compare_context)


def generate_li_inputs(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=2048,
    sparse_mode=0,
    return_value=False,
    **kwargs,
):
    """Generate the exact canonical pytest inputs and CPU golden for a TTK case."""
    INPUT_ADAPTER.customize(
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        return_value,
        kwargs,
    )


def generate_li_inputs_aclnn(
    query,
    key,
    weights,
    actualSeqLengthsQueryOptional=None,
    actualSeqLengthsKeyOptional=None,
    blockTableOptional=None,
    layoutQueryOptional=None,
    layoutKeyOptional=None,
    sparseCount=None,
    sparseMode=None,
    preTokens=None,
    nextTokens=None,
    returnValues=None,
    sparseIndicesOut=None,
    sparseValuesOut=None,
    **kwargs,
):
    """Full aclnn param order. Generate the exact canonical pytest inputs and CPU golden for a TTK case."""
    del sparseIndicesOut, sparseValuesOut, preTokens, nextTokens
    INPUT_ADAPTER.customize(
        query,
        key,
        weights,
        actualSeqLengthsQueryOptional,
        actualSeqLengthsKeyOptional,
        blockTableOptional,
        layoutQueryOptional,
        layoutKeyOptional,
        sparseCount,
        sparseMode,
        returnValues,
        kwargs,
    )
