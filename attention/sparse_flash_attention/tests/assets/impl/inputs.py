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

import logging

import torch

from . import golden as golden_module


class SparseFlashAttentionInputAdapter:
    """Delegate input construction to pytest while keeping adapter state local."""

    def golden_module(self):
        return golden_module

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
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        attention_mode,
        return_softmax_lse,
        rope_head_dim,
    ):
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

    def generate(
        self,
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        *,
        block_table=None,
        actual_seq_lengths_query=None,
        actual_seq_lengths_kv=None,
        query_rope=None,
        key_rope=None,
        sinks=None,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=0,
        return_softmax_lse=False,
        testcase_name=None,
        input_ranges=None,
        **kwargs,
    ):
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
        outputs, generated, _ = pytest_golden.compute_cpu(
            generated, params, return_bench=False
        )
        if outputs is None:
            raise RuntimeError("SparseFlashAttention pytest compute_cpu failed")

        self.copy_tensor(query, generated.get("query"), "query")
        self.copy_tensor(key, generated.get("key_cache", generated.get("key")), "key")
        self.copy_tensor(
            value, generated.get("value_cache", generated.get("value")), "value"
        )
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
        self.copy_tensor(query_rope, generated.get("query_rope"), "query_rope")
        self.copy_tensor(
            key_rope,
            generated.get("key_rope_cache", generated.get("key_rope")),
            "key_rope",
        )
        if sinks is not None:
            self.copy_tensor(sinks, generated.get("sinks"), "sinks")

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
                "block_table",
                "actual_seq_lengths_query",
                "actual_seq_lengths_kv",
                "query_rope",
                "key_rope",
                "sinks",
                "sparse_block_size",
                "layout_query",
                "layout_kv",
                "sparse_mode",
                "attention_mode",
                "return_softmax_lse",
            )
        }
        for name, item in reference_inputs.items():
            values = kwargs.get(name + "_values")
            if values is None:
                continue
            if not torch.is_tensor(item):
                raise ValueError(f"{name}_values requires a declared tensor")
            source = torch.as_tensor(values, dtype=item.dtype)
            if source.shape != item.shape:
                raise ValueError(f"{name}_values shape mismatch")
            locals_tensor = locals()[name]
            locals_tensor.copy_(source.to(locals_tensor.device))
            item.copy_(source)
        reference_inputs.update(
            {
                name: kwargs[name]
                for name in ("pre_tokens", "next_tokens")
                if name in kwargs
            }
        )
        golden_module.CASE_DATA.put(
            testcase_name, {"reference_inputs": reference_inputs}
        )
        logging.info(
            "[%s] SFA CPU inputs ready from pytest: query=%s %s, key=%s %s, "
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


def generate_sfa_inputs(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    attention_mode=0,
    return_softmax_lse=False,
    testcase_name=None,
    input_ranges=None,
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
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sinks=sinks,
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


def generate_sfa_inputs_aclnn_v2(
    query,
    key,
    value,
    sparseIndices,
    blockTable=None,
    actualSeqLengthsQuery=None,
    actualSeqLengthsKv=None,
    queryRope=None,
    keyRope=None,
    sinks=None,
    scaleValue=None,
    sparseBlockSizeOptional=None,
    layoutQueryOptional=None,
    layoutKvOptional=None,
    sparseMode=None,
    preTokens=None,
    nextTokens=None,
    attentionMode=None,
    returnSoftmaxLse=False,
    attentionOut=None,
    softmaxMax=None,
    softmaxSum=None,  # 输出
    testcase_name=None,
    input_ranges=None,
    **kwargs,
):
    # 10 个输入 tensor → 9 个 others → 3 个输出 tensor。
    if expect_error(kwargs):
        return None
    return INPUT_ADAPTER.generate(
        query,
        key,
        value,
        sparseIndices,
        scaleValue,
        block_table=blockTable,
        actual_seq_lengths_query=actualSeqLengthsQuery,
        actual_seq_lengths_kv=actualSeqLengthsKv,
        query_rope=queryRope,
        key_rope=keyRope,
        sinks=sinks,
        sparse_block_size=sparseBlockSizeOptional,
        layout_query=layoutQueryOptional,
        layout_kv=layoutKvOptional,
        sparse_mode=sparseMode,
        pre_tokens=preTokens,
        next_tokens=nextTokens,
        attention_mode=attentionMode,
        return_softmax_lse=returnSoftmaxLse,
        testcase_name=testcase_name,
        input_ranges=input_ranges,
        **kwargs,
    )


def generate_sfa_inputs_aclnn(
    query,
    key,
    value,
    sparseIndices,
    blockTable=None,
    actualSeqLengthsQuery=None,
    actualSeqLengthsKv=None,
    queryRope=None,
    keyRope=None,
    scaleValue=None,
    sparseBlockSizeOptional=None,
    layoutQueryOptional=None,
    layoutKvOptional=None,
    sparseMode=None,
    preTokens=None,
    nextTokens=None,
    attentionMode=None,
    returnSoftmaxLse=False,
    attentionOut=None,
    softmaxMax=None,
    softmaxSum=None,
    testcase_name=None,
    input_ranges=None,
    **kwargs,
):
    # V1（aclnnSparseFlashAttention）无 sinks：9 个输入 tensor → 9 个 others → 3 个输出 tensor。
    if expect_error(kwargs):
        return None
    return INPUT_ADAPTER.generate(
        query,
        key,
        value,
        sparseIndices,
        scaleValue,
        block_table=blockTable,
        actual_seq_lengths_query=actualSeqLengthsQuery,
        actual_seq_lengths_kv=actualSeqLengthsKv,
        query_rope=queryRope,
        key_rope=keyRope,
        sinks=None,
        sparse_block_size=sparseBlockSizeOptional,
        layout_query=layoutQueryOptional,
        layout_kv=layoutKvOptional,
        sparse_mode=sparseMode,
        pre_tokens=preTokens,
        next_tokens=nextTokens,
        attention_mode=attentionMode,
        return_softmax_lse=returnSoftmaxLse,
        testcase_name=testcase_name,
        input_ranges=input_ranges,
        **kwargs,
    )


INPUT_ADAPTER = SparseFlashAttentionInputAdapter()


__input__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "generate_sfa_inputs",
        "torch.ops.cann_ops_transformer.sparse_flash_attention": (
            "generate_sfa_inputs"
        ),
    },
    "aclnn": {
        "aclnnSparseFlashAttention": "generate_sfa_inputs_aclnn",
        "aclnnSparseFlashAttentionV2": "generate_sfa_inputs_aclnn_v2",
    },
}
