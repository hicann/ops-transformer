# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


MLA_METADATA_OP_NAME = "flash_mla_with_kvcache_metadata"


def _get_core_nums():
    """从硬件获取 AIC/AIV 核数（与 aclnn host 侧 GetCurrentPlatformInfo 同源）。"""
    props = torch.npu.get_device_properties()
    return props.cube_core_num, props.vector_core_num


def _calculate_batch_size(cache_seqlens: torch.Tensor):
    """batch size 由必传的 cache_seqlens 长度推导（每 batch 一项，即 size(0)）。"""
    return cache_seqlens.size(0)


def _calculate_metadata_size(batch_size, aic_core_num, aiv_core_num):
    """计算 metadata tensor 的对齐后大小。

    MLA 硬约束 kv head num == 1（B2/D16 上界公式，禁止按 num_heads_kv>1 放大）：
    最坏容量 = ((aic + aiv) * batch_size * 1 + 1) * 16 个 int32（每核 16 word，
    外加 1 个 header），按 4096 个 INT32 元素对齐（与 flash_attn.py 一致）。
    核数从硬件获取，见 _get_core_nums。
    """
    metadata_size = ((aic_core_num + aiv_core_num) * batch_size * 1 + 1) * 16
    return ((metadata_size + 4095) // 4096) * 4096


def _check_metadata_capacity(metadata, batch_size, aic_core_num, aiv_core_num):
    """显式容量校验（B2/D16）：metadata 张量必须不小于 kv==1 上界容量。"""
    required = _calculate_metadata_size(batch_size, aic_core_num, aiv_core_num)
    torch._check(
        metadata.numel() >= required,
        lambda: f"The metadata tensor is too small: {metadata.numel()} elements, required at least "
        f"{required} elements for batch_size={batch_size} (capacity = ((aic {aic_core_num} + aiv {aiv_core_num})"
        f" * batch_size * 1 + 1) * 16, aligned to 4096 int32 elements; kv head num is 1 for MLA).",
    )


class FlashMlaWithKvcacheOpBuilder(OpBuilder):
    def __init__(self):
        # Builder name == torch 公开接口名（照 flash_attn.py 惯例）；仅用作 JIT 缓存键与
        # @impl 注册名，sources() 指向同名 cpp（csrc/attention/flash_mla_with_kvcache.cpp）。
        super(FlashMlaWithKvcacheOpBuilder, self).__init__(
            "flash_mla_with_kvcache", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/flash_mla_with_kvcache.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature (m0070 interface: no v, no q_rope/k_rope)."""
        return [
            "flash_mla_with_kvcache_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "int? max_seqlen_q=None, int? max_seqlen_kv=None, "
            "int head_dim_qk=576, int head_dim_v=512, "
            'int? mask_mode=None, str? layout_q="BSND") -> Tensor',
            "flash_mla_with_kvcache(Tensor q, Tensor k_cache, "
            "Tensor? block_table=None, Tensor? cache_seqlens=None, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "Tensor? attn_mask=None, Tensor? metadata=None, "
            "int head_dim_v=512, float softmax_scale=1.0, int mask_mode=0, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="BSND", str layout_kv="PA_BBND", str? layout_out="BSND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + MLA_METADATA_OP_NAME)
        def flash_mla_with_kvcache_metadata_meta(
            cache_seqlens: torch.Tensor,
            num_heads_q: int,
            num_heads_kv: int,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None,
            head_dim_qk: int = 576,
            head_dim_v: int = 512,
            mask_mode: Optional[int] = None,
            layout_q: Optional[str] = None,
        ):
            b_size = _calculate_batch_size(cache_seqlens)
            aic_core_num, aiv_core_num = _get_core_nums()
            metadata_size = _calculate_metadata_size(b_size, aic_core_num, aiv_core_num)
            return torch.empty((metadata_size,), dtype=torch.int32, device="meta")

        @torch.library.register_fake("cann_ops_transformer::flash_mla_with_kvcache")
        def flash_mla_meta(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            block_table: Optional[torch.Tensor] = None,
            cache_seqlens: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            head_dim_v: Optional[int] = 512,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[int] = 0,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "PA_BBND",
            layout_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
        ):
            layout_q = "BSND" if layout_q is None else layout_q
            layout_out = "BSND" if layout_out is None else layout_out
            # MLA: attn_out 最后一维 = head_dim_v（nope/value 宽度 512）；q/k_cache 最后维为
            # 576（nope 512 + rope 64），rope 已合并进输入，故输出 D 不能取 q 最后维（对照 flash_attn.py
            # 由 v 推导 D 的逻辑，MLA 无独立 v 输入，改用 head_dim_v）。
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                d_size = head_dim_v
                softmax_out_size = (n_size, t_size)
            elif layout_q == "BSND":
                b_size = q.size(0)
                s_size = q.size(1)
                n_size = q.size(2)
                d_size = head_dim_v
                softmax_out_size = (b_size, n_size, s_size)
            else:
                b_size = q.size(0)
                n_size = q.size(1)
                s_size = q.size(2)
                d_size = head_dim_v
                softmax_out_size = (b_size, n_size, s_size)

            # 输出默认 BSND；显式布局必须与 query 一致，或为 TND→NTD。
            if layout_out is not None:
                torch._check(
                    layout_out == layout_q
                    or (layout_q == "TND" and layout_out == "NTD"),
                    lambda: f"layout_out must equal layout_q ({layout_q}), but got {layout_out}",
                )
            if layout_q == "TND":
                if layout_out == "NTD":
                    attention_out_size = (n_size, t_size, d_size)
                else:
                    attention_out_size = (t_size, n_size, d_size)
            elif layout_q == "BSND":
                attention_out_size = (b_size, s_size, n_size, d_size)
            else:
                attention_out_size = (b_size, n_size, s_size, d_size)

            return (
                torch.empty(attention_out_size, dtype=q.dtype, device="meta"),
                torch.empty(softmax_out_size, dtype=torch.float, device="meta"),
            )


# Instantiate the builder
flash_mla_with_kvcache_op_builder = FlashMlaWithKvcacheOpBuilder()
flash_mla_with_kvcache_op_builder._ensure_initialized()  # 照 flash_attn.py:167：加载时延迟初始化（define schema）


@impl(get_as_library(), MLA_METADATA_OP_NAME, "PrivateUse1")
def flash_mla_with_kvcache_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_qk: int = 576,
    head_dim_v: int = 512,
    mask_mode: Optional[int] = 0,
    layout_q: Optional[str] = "BSND",
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    b_size = _calculate_batch_size(cache_seqlens)
    aic_core_num, aiv_core_num = _get_core_nums()

    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = 0 if mask_mode is None else mask_mode
    layout_q = "BSND" if layout_q is None else layout_q

    op_module = flash_mla_with_kvcache_op_builder.load()
    metadata_size = _calculate_metadata_size(b_size, aic_core_num, aiv_core_num)
    output = torch.empty((metadata_size,), dtype=torch.int32, device="npu")

    return op_module.flash_mla_with_kvcache_metadata(
        cu_seqlens_q,
        cache_seqlens,
        seqused_q,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
        aic_core_num,
        aiv_core_num,
        output,
    )


@impl(get_as_library(), flash_mla_with_kvcache_op_builder.name, "PrivateUse1")
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    head_dim_v: Optional[int] = 512,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[int] = 0,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    Dispatcher implementation for NPU (DeepSeek MLA flash attention).
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    aic_core_num, aiv_core_num = _get_core_nums()
    if metadata is not None:
        # 显式容量校验（B2/D16）：主算子侧对用户传入的 metadata 做防呆
        batch_size = None
        if seqused_q is not None:
            batch_size = seqused_q.size(0)
        elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
            batch_size = cu_seqlens_q.size(0) - 1
        if batch_size is not None:
            _check_metadata_capacity(metadata, batch_size, aic_core_num, aiv_core_num)

    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "PA_BBND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out
    softmax_scale = 1.0 if softmax_scale is None else softmax_scale

    op_module = flash_mla_with_kvcache_op_builder.load()
    return op_module.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        seqused_q,
        attn_mask,
        metadata,
        head_dim_v,
        softmax_scale,
        mask_mode,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        return_softmax_lse,
        aic_core_num,
        aiv_core_num,
    )


flash_mla_with_kvcache = torch.ops.cann_ops_transformer.flash_mla_with_kvcache
flash_mla_with_kvcache_metadata = (
    torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata
)
