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

import pytest
import torch
import torchair
from torchair.configs.compiler_config import CompilerConfig

try:
    import torch_npu
except ImportError as import_error:  # pragma: no cover - depends on NPU environment
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR = import_error
else:
    TORCH_NPU_IMPORT_ERROR = None

import custom_ops
import combined_kv_cache


QBSA_METADATA_OUTPUT_SIZE = 2048


def _to_npu(tensor):
    if tensor is None:
        return None
    return tensor.npu()


def _to_cpu(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _check_runtime_available():
    if torch_npu is None:
        pytest.skip(f"torch_npu is not available: {TORCH_NPU_IMPORT_ERROR}")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU device is not available (aclInit failed or no /dev/davinci*).")
    if not custom_ops.has_quant_block_sparse_attn_op():
        detail = "; ".join(custom_ops.load_errors()[:3])
        suffix = f" Load errors: {detail}" if detail else ""
        pytest.skip(f"torch.ops.custom.npu_quant_block_sparse_attn is not registered.{suffix}")
    if not custom_ops.has_quant_block_sparse_attn_metadata_op():
        detail = "; ".join(custom_ops.load_errors()[:3])
        suffix = f" Load errors: {detail}" if detail else ""
        pytest.skip(f"torch.ops.custom.npu_quant_block_sparse_attn_metadata is not registered.{suffix}")


def _prepare_kv_inputs(inputs):
    if "kv_cache_storage" not in inputs:
        return None, _to_npu(inputs["key"]), _to_npu(inputs["value"]), _to_npu(inputs["k_descale"])

    kv_cache_storage = _to_npu(inputs["kv_cache_storage"])
    combined_kv_cache.assert_combined_kv_views(kv_cache_storage, inputs["kv_cache_meta"])
    key, value, k_descale = combined_kv_cache.make_combined_kv_views(
        kv_cache_storage, inputs["kv_cache_meta"])
    return kv_cache_storage, key, value, k_descale


def _normalize_npu_output(output, return_softmax_lse):
    if isinstance(output, dict):
        attention_out = output.get("attention_out")
        softmax_lse = output.get("softmax_lse")
    elif isinstance(output, (tuple, list)):
        attention_out = output[0]
        softmax_lse = output[1] if len(output) > 1 else None
    else:
        attention_out = output
        softmax_lse = None

    attention_out_cpu = _to_cpu(attention_out)
    nonzero_count = int(torch.count_nonzero(attention_out_cpu).item()) if attention_out_cpu is not None else 0
    total_count = int(attention_out_cpu.numel()) if attention_out_cpu is not None else 0
    result = {"attention_out": attention_out_cpu}
    if return_softmax_lse and softmax_lse is not None:
        result["softmax_lse"] = _to_cpu(softmax_lse)
    return result


def _golden_output(test_data):
    inputs = test_data["input"]
    if "golden" in test_data:
        golden = {"attention_out": test_data["golden"]["attention_out"]}
        if inputs["return_softmax_lse"]:
            golden["softmax_lse"] = test_data["golden"]["softmax_lse"]
        return golden

    golden = {"attention_out": test_data["cpu_output"]}
    if inputs["return_softmax_lse"]:
        golden["softmax_lse"] = test_data["cpu_softmax_lse"]
    return golden


def _metadata_needs_generation(metadata):
    if metadata is None:
        return True
    if metadata.numel() != QBSA_METADATA_OUTPUT_SIZE:
        return True
    if metadata.device.type == "cpu" and int(torch.count_nonzero(metadata).item()) == 0:
        return True
    return False


def _call_npu_quant_block_sparse_attn_metadata(test_data):
    inputs = test_data["input"]
    metadata_input = test_data.get("metadata_input", {})
    sparse_seq_len = _to_npu(inputs["sparse_seq_len"])
    metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        sparse_seq_len,
        metadata_input["num_heads_q"],
        metadata_input["num_heads_kv"],
        metadata_input["head_dim"],
        cu_seqlens_q=_to_npu(inputs["cu_seqlens_q"]),
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=_to_npu(inputs["seqused_kv"]),
        batch_size=metadata_input.get("batch_size", int(inputs["sparse_seq_len"].shape[0])),
        sparse_block_size_q=inputs["sparse_q_block_size"],
        sparse_block_size_k=inputs["sparse_kv_block_size"],
        quant_mode=inputs["quant_mode"],
        mask_mode=inputs["mask_mode"],
        layout_q=metadata_input.get("layout_q", inputs["layout_q"]),
        layout_kv=metadata_input.get("layout_kv", inputs["layout_kv"]),
        layout_sparse_indices=metadata_input.get("layout_sparse_indices", inputs["layout_sparse_indices"]),
    )
    torch_npu.npu.synchronize()
    return metadata


def _prepare_metadata(test_data):
    metadata = test_data["input"].get("metadata")
    if not _metadata_needs_generation(metadata):
        return _to_npu(metadata)
    return _call_npu_quant_block_sparse_attn_metadata(test_data)


def _call_npu_quant_block_sparse_attn(test_data, device_id):
    _check_runtime_available()
    torch_npu.npu.set_device(device_id)
    inputs = test_data["input"]
    kv_cache_storage, key, value, k_descale = _prepare_kv_inputs(inputs)
    metadata = _prepare_metadata(test_data)
    output = torch.ops.custom.npu_quant_block_sparse_attn(
        _to_npu(inputs["query"]),
        key,
        value,
        _to_npu(inputs["q_descale"]),
        k_descale,
        _to_npu(inputs["v_descale"]),
        _to_npu(inputs["p_scale"]),
        _to_npu(inputs["sparse_indices"]),
        _to_npu(inputs["sparse_seq_len"]),
        _to_npu(inputs.get("atten_mask")),
        inputs["softmax_scale"],
        inputs["sparse_q_block_size"],
        inputs["sparse_kv_block_size"],
        cu_seqlens_q=_to_npu(inputs["cu_seqlens_q"]),
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=_to_npu(inputs["seqused_kv"]),
        block_table=_to_npu(inputs["block_table"]),
        metadata=metadata,
        layout_kv=inputs["layout_kv"],
        layout_q=inputs["layout_q"],
        layout_sparse_indices=inputs["layout_sparse_indices"],
        layout_out=inputs["layout_out"],
        quant_mode=inputs["quant_mode"],
        mask_mode=inputs["mask_mode"],
        return_softmax_lse=inputs["return_softmax_lse"],
    )
    torch_npu.npu.synchronize()
    del kv_cache_storage
    return _normalize_npu_output(output, inputs["return_softmax_lse"])


class _QuantBlockSparseAttnGraph(torch.nn.Module):
    def __init__(self, inputs, kv_cache_meta=None):
        super().__init__()
        self.inputs = inputs
        self.kv_cache_meta = kv_cache_meta

    def _rebuild_combined_kv_views(self, kv_cache_storage):
        meta = self.kv_cache_meta
        fp8_storage = kv_cache_storage.view(torch.float8_e4m3fn)
        key = torch.as_strided(
            fp8_storage,
            (meta["block_num"], meta["n2"], meta["block_size"], meta["d_size"]),
            (meta["pa_block_stride"], meta["block_size"] * meta["d_size"], meta["d_size"], 1),
            meta["key_offset_bytes"])
        value = torch.as_strided(
            fp8_storage,
            (meta["block_num"], meta["n2"], meta["block_size"], meta["dv_size"]),
            (meta["pa_block_stride"], meta["block_size"] * meta["dv_size"], meta["dv_size"], 1),
            meta["value_offset_bytes"])
        fp32_storage = kv_cache_storage.view(torch.float32)
        k_descale = torch.as_strided(
            fp32_storage,
            (meta["block_num"], meta["n2"], meta["block_size"], 1),
            (meta["pa_block_stride"] // 4, meta["block_size"], 1, 1),
            meta["k_scale_offset_bytes"] // 4)
        return key, value, k_descale

    def forward(self, query, kv_cache_storage, q_descale, v_descale, p_scale, sparse_indices,
                sparse_seq_len, atten_mask, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                block_table, metadata):
        key, value, k_descale = self._rebuild_combined_kv_views(kv_cache_storage)
        return torch.ops.custom.npu_quant_block_sparse_attn(
            query, key, value, q_descale, k_descale, v_descale, p_scale, sparse_indices, sparse_seq_len,
            atten_mask, self.inputs["softmax_scale"], self.inputs["sparse_q_block_size"],
            self.inputs["sparse_kv_block_size"],
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=None, seqused_q=None,
            seqused_kv=seqused_kv, block_table=block_table, metadata=metadata,
            layout_kv=self.inputs["layout_kv"], layout_q=self.inputs["layout_q"],
            layout_sparse_indices=self.inputs["layout_sparse_indices"],
   	        layout_out=self.inputs["layout_out"], quant_mode=self.inputs["quant_mode"],
            mask_mode=self.inputs["mask_mode"], return_softmax_lse=self.inputs["return_softmax_lse"])


def _mark_static_tensors(*tensors):
    for tensor in tensors:
        if tensor is not None:
            torch._dynamo.mark_static(tensor)


def test_quant_block_sparse_attn_process_graph(test_data, device_id=0):
    _check_runtime_available()
    torch_npu.npu.set_device(device_id)
    inputs = test_data["input"]
    kv_cache_storage, key, value, k_descale = _prepare_kv_inputs(inputs)
    if kv_cache_storage is None:
        pytest.skip("graph mode currently requires combined KV cache storage")
    combined_kv_cache.assert_combined_kv_views(kv_cache_storage, inputs["kv_cache_meta"])
    metadata = _prepare_metadata(test_data)
    query = _to_npu(inputs["query"])
    kv_cache_storage = _to_npu(inputs["kv_cache_storage"])
    q_descale = _to_npu(inputs["q_descale"])
    v_descale = _to_npu(inputs["v_descale"])
    p_scale = _to_npu(inputs["p_scale"])
    sparse_indices = _to_npu(inputs["sparse_indices"])
    sparse_seq_len = _to_npu(inputs["sparse_seq_len"])
    atten_mask = _to_npu(inputs.get("atten_mask"))
    cu_seqlens_q = _to_npu(inputs["cu_seqlens_q"])
    cu_seqlens_kv = _to_npu(inputs["cu_seqlens_kv"])
    seqused_q = _to_npu(inputs["seqused_q"])
    seqused_kv = _to_npu(inputs["seqused_kv"])
    block_table = _to_npu(inputs["block_table"])

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    _mark_static_tensors(
        query, kv_cache_storage, q_descale, v_descale, p_scale, sparse_indices,
        sparse_seq_len, atten_mask, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        block_table, metadata,
    )
    torch._dynamo.reset()
    model = torch.compile(
        _QuantBlockSparseAttnGraph(inputs, inputs["kv_cache_meta"]).npu(),
        fullgraph=True, backend=npu_backend, dynamic=False)
    output = model(
        query, kv_cache_storage, q_descale, v_descale, p_scale, sparse_indices,
        sparse_seq_len, atten_mask, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        block_table, metadata)
    torch_npu.npu.synchronize()
    del kv_cache_storage
    return _normalize_npu_output(output, inputs["return_softmax_lse"]), _golden_output(test_data)


def test_quant_block_sparse_attn_process_ci(test_data, device_id=0):
    npu_result = _call_npu_quant_block_sparse_attn(test_data, device_id)
    return npu_result, _golden_output(test_data)
