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

"""Input adapter that delegates SparseFlashAttention data construction to pytest."""

import importlib.util
import sys
from pathlib import Path

import torch


class SparseFlashAttentionInputAdapter:
    """Delegate input construction to pytest while keeping adapter state local."""

    @staticmethod
    def module_load_error(path, exc):
        return RuntimeError(
            "Failed to load SparseFlashAttention module; "
            f"stage=assets Golden store; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def golden_module(self):
        name = "sfa_ttk_golden"
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
            raise self.module_load_error(path, exc) from exc
        return module

    @staticmethod
    def copy_tensor(destination, source, name):
        if destination is None:
            if source is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest generator produced a tensor"
                )
            return
        if source is None:
            raise ValueError(f"{name} is declared by CSV but pytest did not produce it")
        source_cpu = source.detach().cpu() if torch.is_tensor(source) else torch.as_tensor(source)
        if tuple(destination.shape) != tuple(source_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(destination.shape)}, pytest={tuple(source_cpu.shape)}"
            )
        if destination.dtype != source_cpu.dtype:
            raise ValueError(
                f"{name} dtype mismatch: TTK={destination.dtype}, pytest={source_cpu.dtype}"
            )
        destination.copy_(source_cpu.to(device=destination.device))

    @staticmethod
    def copy_sequence(destination, values, name):
        if destination is None:
            if values is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest parameter conversion "
                    "produced values"
                )
            return
        if values is None:
            raise ValueError(
                f"{name} is declared by CSV but pytest parameter conversion returned None"
            )
        source = torch.tensor(values, dtype=destination.dtype, device=destination.device)
        if source.numel() != destination.numel():
            raise ValueError(
                f"{name} size mismatch: TTK={destination.numel()}, pytest={source.numel()}"
            )
        destination.copy_(source.reshape(destination.shape))

    @staticmethod
    def validate_api_attributes(params, scale_value, sparse_block_size,
                                layout_query, layout_kv, sparse_mode,
                                attention_mode, return_softmax_lse,
                                rope_head_dim):
        expected = {
            "scalevalue": scale_value,
            "sparse_blocksize": sparse_block_size,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparsemode": sparse_mode,
            "attention_mode": attention_mode,
            "return_softmax_lse": return_softmax_lse,
            "rope_head_dim": rope_head_dim,
        }
        mismatches = {
            name: (params.get(name), value)
            for name, value in expected.items()
            if params.get(name) != value
        }
        if mismatches:
            raise ValueError(
                "SparseFlashAttention API attributes differ from explicit pytest "
                f"fields: {mismatches}"
            )

    def generate(self, query, key, value, sparse_indices, scale_value, *, block_table=None,
                 actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                 query_rope=None, key_rope=None, sparse_block_size=1,
                 layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                 attention_mode=0, return_softmax_lse=False, testcase_name=None,
                 input_ranges=None, **kwargs):
        """Generate final API tensors and the CPU golden through pytest."""
        del input_ranges
        golden_module = self.golden_module()
        params, pytest_golden = (
            golden_module.SparseFlashAttentionPytestAdapter.convert_params(
                kwargs, testcase_name
            )
        )
        self.validate_api_attributes(
            params,
            scale_value,
            sparse_block_size,
            layout_query,
            layout_kv,
            sparse_mode,
            attention_mode,
            return_softmax_lse,
            int(kwargs["rope_head_dim"]),
        )
        generated = pytest_golden.generate_input_tensors(params)
        outputs, generated, _ = pytest_golden.compute_cpu(generated, params)
        if outputs is None:
            raise RuntimeError("SparseFlashAttention pytest compute_cpu failed")
        if hasattr(torch, "npu"):
            torch.npu.synchronize()

        self.copy_tensor(query, generated.get("query"), "query")
        self.copy_tensor(key, generated.get("key_cache", generated.get("key")), "key")
        self.copy_tensor(value, generated.get("value_cache", generated.get("value")), "value")
        self.copy_tensor(sparse_indices, generated.get("sparse_indices"), "sparse_indices")
        # Pytest creates an internal block table for its CPU model even for non-PA layouts.
        if block_table is not None:
            self.copy_tensor(block_table, generated.get("block_table"), "block_table")
        elif layout_kv == "PA_BSND":
            raise ValueError("PA_BSND requires block_table to be declared by the CSV")
        self.copy_sequence(actual_seq_lengths_query, params["actualseqlengths"], "actual_seq_lengths_query")
        self.copy_sequence(actual_seq_lengths_kv, params["actualseqlengthskv"], "actual_seq_lengths_kv")
        self.copy_tensor(query_rope, generated.get("query_rope"), "query_rope")
        self.copy_tensor(key_rope, generated.get("key_rope_cache", generated.get("key_rope")), "key_rope")

        golden_module.CASE_DATA.put(
            testcase_name,
            {
                "golden": golden_module.normalize_pytest_outputs(
                    outputs, query, params["return_softmax_lse"]
                )
            },
        )


def generate_sfa_inputs(query, key, value, sparse_indices, scale_value, *, block_table=None,
                        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                        query_rope=None, key_rope=None, sparse_block_size=1,
                        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                        attention_mode=0, return_softmax_lse=False, testcase_name=None,
                        input_ranges=None, **kwargs):
    return INPUT_ADAPTER.generate(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
        testcase_name=testcase_name,
        input_ranges=input_ranges,
        **kwargs,
    )


INPUT_ADAPTER = SparseFlashAttentionInputAdapter()


__input__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "generate_sfa_inputs",
    }
}
