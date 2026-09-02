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

"""Input adapter that delegates KvQuantSparseFlashAttention construction to pytest."""

import importlib.util
import sys
from pathlib import Path

import torch


class KvQuantSparseFlashAttentionInputAdapter:
    """Delegate packed input construction to pytest while keeping adapter state local."""

    @staticmethod
    def module_load_error(path, exc):
        return RuntimeError(
            "Failed to load KvQuantSparseFlashAttention module; "
            f"stage=assets Golden store; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def golden_module(self):
        name = "qsfa_ttk_golden"
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
                import logging

                logging.warning(
                    f"{name} is absent from CSV but pytest generator produced a tensor. Skipping copy."
                )
            return
        if source is None:
            raise ValueError(f"{name} is declared by CSV but pytest did not produce it")
        source_cpu = (
            source.detach().cpu()
            if torch.is_tensor(source)
            else torch.as_tensor(source)
        )
        if tuple(destination.shape) != tuple(source_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(destination.shape)}, pytest={tuple(source_cpu.shape)}"
            )
        destination.copy_(
            source_cpu.to(dtype=destination.dtype, device=destination.device)
        )

    @staticmethod
    def copy_sequence(destination, values, name):
        if destination is None:
            if values is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest parameter conversion produced values"
                )
            return
        if values is None:
            raise ValueError(
                f"{name} is declared by CSV but pytest parameter conversion returned None"
            )
        source = torch.tensor(
            values, dtype=destination.dtype, device=destination.device
        )
        if source.numel() != destination.numel():
            raise ValueError(
                f"{name} size mismatch: TTK={destination.numel()}, pytest={source.numel()}"
            )
        destination.copy_(source.reshape(destination.shape))

    @staticmethod
    def validate_api_attributes(
        params,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
    ):
        expected = {
            "scalevalue": scale_value,
            "key_quant_mode": key_quant_mode,
            "value_quant_mode": value_quant_mode,
            "sparse_blocksize": sparse_block_size,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparsemode": sparse_mode,
            "attention_mode": attention_mode,
            "quant_scale_repo_mode": quant_scale_repo_mode,
            "tile_size": tile_size,
            "rope_head_dim": rope_head_dim,
        }
        mismatches = {
            name: (params.get(name), value)
            for name, value in expected.items()
            if params.get(name) != value
        }
        if mismatches:
            raise ValueError(
                f"QSFA API attributes differ from explicit pytest fields: {mismatches}"
            )

    def generate(
        self,
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        *,
        key_dequant_scale=None,
        value_dequant_scale=None,
        block_table=None,
        actual_seq_lengths_query=None,
        actual_seq_lengths_kv=None,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=0,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        testcase_name=None,
        **kwargs,
    ):
        """Generate final packed API tensors and the CPU golden through pytest."""
        golden_module = self.golden_module()
        params, pytest_golden = (
            golden_module.KvQuantSparseFlashAttentionPytestAdapter.convert_params(
                kwargs, testcase_name
            )
        )
        if key_dequant_scale is not None or value_dequant_scale is not None:
            raise ValueError(
                "QSFA CSV key_dequant_scale and value_dequant_scale slots must be None; "
                "pytest stores per-tile scales inside the packed KV tensors"
            )
        self.validate_api_attributes(
            params,
            scale_value,
            key_quant_mode,
            value_quant_mode,
            sparse_block_size,
            layout_query,
            layout_kv,
            sparse_mode,
            attention_mode,
            quant_scale_repo_mode,
            tile_size,
            rope_head_dim,
        )
        generated = pytest_golden.generate_input_tensors(params)
        output, generated, _ = pytest_golden.compute_cpu(generated, params)
        if output is None:
            raise RuntimeError("KvQuantSparseFlashAttention pytest compute_cpu failed")

        self.copy_tensor(query, generated.get("query_cache"), "query")
        self.copy_tensor(key, generated.get("key_cache"), "key")
        self.copy_tensor(value, generated.get("value_cache"), "value")
        self.copy_tensor(
            sparse_indices, generated.get("sparse_indices"), "sparse_indices"
        )
        # Pytest creates an internal block table for its CPU model even for non-PA layouts.
        if block_table is not None:
            self.copy_tensor(block_table, generated.get("block_table"), "block_table")
        elif layout_kv == "PA_BSND":
            raise ValueError("PA_BSND requires block_table to be declared by the CSV")
        self.copy_sequence(
            actual_seq_lengths_query,
            params["actualseqlengths"],
            "actual_seq_lengths_query",
        )
        self.copy_sequence(
            actual_seq_lengths_kv, params["actualseqlengthskv"], "actual_seq_lengths_kv"
        )
        golden_module.CASE_DATA.put(
            testcase_name,
            {"golden": output.detach().cpu()},
        )


def generate_qsfa_inputs(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    attention_mode=0,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    testcase_name=None,
    **kwargs,
):
    return INPUT_ADAPTER.generate(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        key_dequant_scale=key_dequant_scale,
        value_dequant_scale=value_dequant_scale,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        attention_mode=attention_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
        testcase_name=testcase_name,
        **kwargs,
    )


INPUT_ADAPTER = KvQuantSparseFlashAttentionInputAdapter()


__input__ = {
    "e2e": {
        "torch_npu.npu_kv_quant_sparse_flash_attention": "generate_qsfa_inputs",
        "qsfa_ttk_ops.kv_quant_sparse_flash_attention_ttk": "generate_qsfa_inputs",
    }
}
