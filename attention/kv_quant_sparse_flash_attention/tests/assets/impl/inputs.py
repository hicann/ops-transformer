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

import logging

import torch


class KvQuantSparseFlashAttentionInputAdapter:
    """Delegate packed input construction to pytest while keeping adapter state local."""

    def golden_module(self):
        from . import golden

        return golden

    @staticmethod
    def copy_tensor(destination, source, name):
        if destination is None:
            if source is not None:
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
        generated = pytest_golden.generate_input_tensors(params, use_tq4_ranges=True)
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
        if actual_seq_lengths_query is not None or layout_query == "TND":
            self.copy_sequence(
                actual_seq_lengths_query,
                params["actualseqlengths"],
                "actual_seq_lengths_query",
            )
        if actual_seq_lengths_kv is not None or layout_kv != "BSND":
            self.copy_sequence(
                actual_seq_lengths_kv,
                params["actualseqlengthskv"],
                "actual_seq_lengths_kv",
            )
        reference_inputs = {
            name: item.detach().cpu().clone() if torch.is_tensor(item) else item
            for name, item in locals().items()
            if name
            in (
                "query",
                "key",
                "value",
                "sparse_indices",
                "scale_value",
                "key_quant_mode",
                "value_quant_mode",
                "key_dequant_scale",
                "value_dequant_scale",
                "block_table",
                "actual_seq_lengths_query",
                "actual_seq_lengths_kv",
                "sparse_block_size",
                "layout_query",
                "layout_kv",
                "sparse_mode",
                "attention_mode",
                "quant_scale_repo_mode",
                "tile_size",
                "rope_head_dim",
                "sinks",
                "return_softmax_lse",
            )
        }
        reference_inputs.update(
            {
                name: kwargs[name]
                for name in ("pre_tokens", "next_tokens", "key_dtype", "value_dtype")
                if name in kwargs
            }
        )
        golden_module.CASE_DATA.put(
            testcase_name, {"reference_inputs": reference_inputs}
        )
        logging.info(
            "[%s] QSFA CPU inputs ready from pytest: query=%s %s, key=%s %s, "
            "layout=%s/%s, sparse_mode=%s",
            testcase_name,
            tuple(query.shape),
            query.dtype,
            tuple(key.shape),
            key.dtype,
            layout_query,
            layout_kv,
            sparse_mode,
        )


def expect_error(kwargs):
    value = kwargs.get("expect_error", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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
    if expect_error(kwargs):
        return None
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
