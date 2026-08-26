# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class UndGenQkvRmsNormRopeCacheOpBuilder(OpBuilder):
    def __init__(self):
        super(UndGenQkvRmsNormRopeCacheOpBuilder, self).__init__(
            "und_gen_qkv_rms_norm_rope_cache", category="posembedding"
        )

    def sources(self):
        return ["csrc/posembedding/und_gen_qkv_rms_norm_rope_cache.cpp"]

    def schema(self) -> List[str]:
        return [
            # k_cache/v_cache 由算子原地写入，必须带 (a!)/(b!) 别名标注，
            # 否则 functionalization / torch.compile 会把这个原地写当成无副作用而丢掉 cache 更新
            "und_gen_qkv_rms_norm_rope_cache(Tensor und_qkv, Tensor und_weights_q, Tensor und_weights_k, "
            "Tensor cos_sin_cache, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor slot_mapping, Tensor positions, "
            # 默认值必须写在 schema 里：顶层 cann_ops_transformer.<op> 拿到的是 torch.ops
            # 句柄而非本文件的 wrapper，wrapper 的 def 默认值在那条路径上不生效
            "Tensor? gen_qkv=None, Tensor? gen_weights_q=None, Tensor? gen_weights_k=None, "
            "Tensor? cat_indices=None, *, int num_heads_q=8, int num_heads_k=1, "
            "int num_heads_v=1, float norm_eps=1e-6, int[] mrope_section=[]) -> Tensor"
        ]

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def und_gen_qkv_rms_norm_rope_cache_meta(
            und_qkv: torch.Tensor,
            und_weights_q: torch.Tensor,
            und_weights_k: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            positions: torch.Tensor,
            gen_qkv: Optional[torch.Tensor] = None,
            gen_weights_q: Optional[torch.Tensor] = None,
            gen_weights_k: Optional[torch.Tensor] = None,
            cat_indices: Optional[torch.Tensor] = None,
            *,
            num_heads_q: int = 8,
            num_heads_k: int = 1,
            num_heads_v: int = 1,
            norm_eps: float = 1e-6,
            mrope_section: Tuple[int, ...] = (),
        ) -> torch.Tensor:
            head_dim = und_qkv.shape[2]
            total = und_qkv.shape[0] + (0 if gen_qkv is None else gen_qkv.shape[0])
            return torch.empty(
                [total, num_heads_q, head_dim], dtype=und_qkv.dtype, device="meta"
            )


und_gen_qkv_rms_norm_rope_cache_op_builder = UndGenQkvRmsNormRopeCacheOpBuilder()
und_gen_qkv_rms_norm_rope_cache_op_builder._ensure_initialized()


@impl(get_as_library(), "und_gen_qkv_rms_norm_rope_cache", "PrivateUse1")
def und_gen_qkv_rms_norm_rope_cache(
    und_qkv: torch.Tensor,
    und_weights_q: torch.Tensor,
    und_weights_k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    gen_qkv: Optional[torch.Tensor] = None,
    gen_weights_q: Optional[torch.Tensor] = None,
    gen_weights_k: Optional[torch.Tensor] = None,
    cat_indices: Optional[torch.Tensor] = None,
    *,
    num_heads_q: int = 8,
    num_heads_k: int = 1,
    num_heads_v: int = 1,
    norm_eps: float = 1e-6,
    mrope_section: Tuple[int, ...] = (),
) -> torch.Tensor:
    """und/gen 两段 QKV 融合 RMSNorm + MRoPE，并把 K/V 写入分页 KV Cache，封装 aclnnUndGenQkvRmsNormRopeCache。

    按 cat_indices 间接寻址把理解段（undecoded）与生成段（generated）拼成一条输出序列，
    逐 token 沿头维度拆出 Q/K/V，对 Q 和 K 逐头做 RMSNorm 与 MRoPE，V 不参与；Q 作为返回值
    输出，K 和 V 按 slot_mapping 写入分页 KV Cache。RMSNorm 权重按源 token 落在理解段还是
    生成段在两套权重间选择。计算全程 float32，返回值与 KV Cache 均为 bfloat16。

    记 T = und_len + gen_len 为输出 token 总数，N = Hq + Hk + Hv，D 为头维度（固定 128），
    Bn / Bs 为 KV Cache 的页数与页内行数。

    Args:
        und_qkv (Tensor): 理解段 Q/K/V 融合输入，头维度按 [Hq,Hk,Hv] 排布，
            shape [und_len,N,D]，dtype bfloat16。
        und_weights_q (Tensor): 理解段 Q 分支的 RMSNorm 权重，shape [D]，dtype bfloat16。
        und_weights_k (Tensor): 理解段 K 分支的 RMSNorm 权重，shape [D]，dtype bfloat16。
        cos_sin_cache (Tensor): 位置编码表，前 D/2 列为 cos、后 D/2 列为 sin，
            shape [max_pos,D]，dtype float32。
        k_cache (Tensor): K Cache，**原地更新**，必须内存连续，shape [Bn,Bs,Hk,D]，dtype bfloat16。
        v_cache (Tensor): V Cache，**原地更新**，必须内存连续，shape [Bn,Bs,Hv,D]，dtype bfloat16。
        slot_mapping (Tensor): 每个输出 token 写入 cache 的 slot 索引，
            slot = block_idx * Bs + row_idx，shape [T]，dtype int64。
        positions (Tensor): MRoPE 的 T/H/W 三轴位置索引，逐行对应一个轴，shape [3,T]，dtype int64。
        gen_qkv (Tensor, optional): 生成段 Q/K/V 融合输入，N 和 D 维须与 und_qkv 一致，
            shape [gen_len,N,D]，dtype bfloat16。默认 None，但当前版本必须传入且 gen_len 为正，
            纯 prefill 暂不支持。
        gen_weights_q (Tensor, optional): 生成段 Q 分支 RMSNorm 权重，shape [D]。默认 None，
            当前版本必须传入，且需与 gen_weights_k 成对。
        gen_weights_k (Tensor, optional): 生成段 K 分支 RMSNorm 权重，shape [D]。默认 None，
            当前版本必须传入，且需与 gen_weights_q 成对。
        cat_indices (Tensor, optional): 输出 token 到源 token 的映射 out_t -> src_t，取值小于
            und_len 时取理解段、否则取生成段，shape [T]，dtype int64。默认 None，
            当前版本必须传入，单序列恒等映射暂不支持。
        num_heads_q (int): Q 头数 Hq，默认 8。
        num_heads_k (int): K 头数 Hk，默认 1。
        num_heads_v (int): V 头数 Hv，默认 1。
        norm_eps (float): RMSNorm 防除零参数，必须为正数，默认 1e-6。
        mrope_section (Tuple[int, ...]): MRoPE 的 T/H/W section 参数 [t,h,w]，长度为 0 或 3，
            默认 () 即退化为标准 RoPE（三轴同源）。

    Returns:
        Tensor: q，理解段与生成段拼接后的 Q 输出，shape [T,Hq,D]，dtype bfloat16。
            k_cache / v_cache 为原地更新，不在返回值中，调用后直接读入参 Tensor 即可。

    Examples:
        >>> q = cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
        ...     und_qkv, w_q, w_k, cos_sin_cache, k_cache, v_cache, slot_mapping, positions,
        ...     gen_qkv, gen_w_q, gen_w_k, cat_indices,
        ...     num_heads_q=8, num_heads_k=1, num_heads_v=1,
        ...     norm_eps=1e-6, mrope_section=[16, 16, 16])
    """
    op_module = und_gen_qkv_rms_norm_rope_cache_op_builder.load()
    return op_module.und_gen_qkv_rms_norm_rope_cache(
        und_qkv,
        und_weights_q,
        und_weights_k,
        cos_sin_cache,
        k_cache,
        v_cache,
        slot_mapping,
        positions,
        gen_qkv,
        gen_weights_q,
        gen_weights_k,
        cat_indices,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        norm_eps,
        list(mrope_section),
    )
