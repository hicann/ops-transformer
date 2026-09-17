# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""flash_mla_with_kvcache golden: 参照 flash_attn_golden 架构的独立实现。

算子契约 (flash_mla.txt + 安装包 cann_ops_transformer schema):
- D == head_dim_qk == 576(MLA 硬约束, nope 512 + rope 64 = QK 完整宽度);
  DV == head_dim_v == 512(V 投影宽度, kv 复用只取 K 的 nope 部分)。
- layout_kv 支持三种 PA 缓存布局: PA_NZ 5D (total_blocks, 1, DSUB=36,
  block_size, 16) / PA_BBND 4D (NB, BS, kv, D) / PA_BNBD 4D (NB, kv, BS, D)
  (nope 前 32 块 + rope 后 4 块)。
- block_table (B, max_blocks) int32, 值 = k_cache 中的块索引;
  cache_seqlens (B,) int32, 逐 batch cache 长度(接口参数名)。
- mask_mode: 0 = 无掩码全可见; 3 = RIGHT_DOWN 右对齐 causal(decode 时
  q 位于序列末尾, 可见该序列全部 cache key; prefill 多 token 行内 causal)。
- 输出: attn_out dtype 同 q(宽 = head_dim_v); softmax_lse float32。形状:
  BNSD -> (B,N,S,DV) / (B,N,S); BSND -> (B,S,N,DV) / (B,N,S);
  TND  -> (T,N,DV) / (N,T)。return_softmax_lse=False 时 lse 为 zeros(1)。

实现架构与 ``golden/flash_attn_golden.py`` 一致(QK 宽 D=576、V 宽 DV=512,
flash golden 的 v_dim 独立机制天然覆盖): forward_golden(17 参对齐算子)
-> _compute_per_batch(逐 batch 防 OOM) -> _extract(PA_NZ 块还原)
-> _compute_tiled_single(块式 online softmax) / _compute_full_single(全 bmm);
mask_mode=3 用右对齐 causal(q_pos = q_idx + (skv - sq))。完全自包含,
不依赖 FusedInferAttentionScoreGolden / FlashAttnGolden。
"""

import math
from typing import Optional, Tuple

import torch

from .attention_math import attention_single, gather_pages

_NZ_BLK_ELEM = 16  # PA_NZ blk_elem (fp16/bf16)


class FlashMlaWithKvcacheGolden:
    """flash_mla_with_kvcache 参考实现 (参照 FlashAttnGolden 架构)。

    不需要 __init__ 配置 —— 所有几何参数在 forward_golden 内
    从输入张量/算子参数推导:
      num_heads_q  <- TND/BNSD dim1, BSND dim2
      head_dim_qk  <- q.shape[-1] (QK 全宽, 硬约束 576)
      head_dim_v   <- 算子参数(硬约束 512)
      rope_dim     <- head_dim_qk - head_dim_v (= 64)
      block_size   <- k_cache.shape[3] (PA_NZ 5D 的块长)
      softmax_scale<- 算子参数, 缺省 1/sqrt(head_dim_qk)
    输入参数与算子 flash_mla_with_kvcache 17 参对齐(额外保留 framework
    内部参数 device_mode: 'cpu' 时辅助张量搬 CPU 计算)。
    """

    TILE_SIZE = 2048
    EPSILON = 1e-20
    TILED_CROSSOVER = 3 * 1024 * 1024  # S1*S2 > this -> tiled saves memory

    # ==========================================================================
    # Geometry extraction (forward_golden 开头统一提取/校验)
    # ==========================================================================

    @staticmethod
    def _derive_geometry(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        layout_q: str,
        layout_kv: str,
        head_dim_v: int,
    ) -> dict:
        """计算前统一从输入张量提取/校验全部几何参数(布局相关集中于此)。

        各布局维序: TND (T,N,D), BNSD (B,N,S,D), BSND (B,S,N,D);
        KV(均 PA 缓存): PA_NZ 5D (NB, kv, DSUB, BS, BLK=16),
                        PA_BBND 4D (NB, BS, kv, D),
                        PA_BNBD 4D (NB, kv, BS, D)。
        head_dim_v 由调用方传入(算子 17 参之一, MLA 硬约束 512, V 宽 = nope);
        head_dim_qk 由 q 宽度校验(576 = nope 512 + rope 64)。
        Returns dict: num_heads_q / head_dim_qk / head_dim_v / rope_dim /
                      block_size / batch / seq_decl
        - batch: BNSD/BSND = q.shape[0]; TND = None(由 cu_seqlens_q 决定)
        - seq_decl: 声明序列长, BNSD = q.shape[2], BSND = q.shape[1];
                    TND = None(由 cu_seqlens_q 决定)
        """
        if layout_q not in ("BNSD", "BSND", "TND"):
            raise ValueError(f"unsupported layout_q: {layout_q}")
        if layout_kv not in ("PA_NZ", "PA_BBND", "PA_BNBD"):
            raise ValueError(
                f"unsupported layout_kv: {layout_kv} (仅 PA_NZ/PA_BBND/PA_BNBD)"
            )

        head_dim_qk = q.shape[-1]
        if head_dim_qk != 576:
            raise ValueError(
                f"q last dim (head_dim_qk) 必须为 576(MLA 硬约束, nope 512 + "
                f"rope 64), got {head_dim_qk}"
            )
        # MLA 硬约束: V 宽 = nope 宽, 必须为 512
        if head_dim_v != 512:
            raise ValueError(f"head_dim_v 必须为 512(MLA 硬约束), got {head_dim_v}")

        # KV 布局维度校验与块长/宽提取
        if layout_kv == "PA_NZ":
            if k_cache.dim() != 5:
                raise ValueError(
                    f"layout_kv=PA_NZ 要求 k_cache 5 维 (NB, kv, DSUB, BS, "
                    f"16), got dim={k_cache.dim()}"
                )
            if k_cache.shape[1] != 1:
                raise ValueError(
                    f"k_cache dim1 (kv heads) must be 1 (MLA 硬约束), "
                    f"got {k_cache.shape[1]}"
                )
            if k_cache.shape[-1] != _NZ_BLK_ELEM:
                raise ValueError(
                    f"k_cache last dim must be NZ blk_elem={_NZ_BLK_ELEM}, "
                    f"got {k_cache.shape[-1]}"
                )
            block_size = k_cache.shape[3]  # BS
            k_dim = k_cache.shape[2] * k_cache.shape[4]  # DSUB*BLK = 576
            if k_dim != head_dim_qk:
                raise ValueError(f"k_cache 宽度 {k_dim} != head_dim_qk({head_dim_qk})")
        elif layout_kv == "PA_BBND":
            if k_cache.dim() != 4:
                raise ValueError(
                    f"layout_kv=PA_BBND 要求 k_cache 4 维 (NB, BS, kv, D), "
                    f"got dim={k_cache.dim()}"
                )
            if k_cache.shape[2] != 1:
                raise ValueError(
                    f"k_cache dim2 (kv heads) must be 1 (MLA 硬约束), "
                    f"got {k_cache.shape[2]}"
                )
            block_size = k_cache.shape[1]  # BS
            k_dim = k_cache.shape[-1]  # D
            if k_dim != head_dim_qk:
                raise ValueError(f"k_cache 宽度 {k_dim} != head_dim_qk({head_dim_qk})")
        else:  # PA_BNBD
            if k_cache.dim() != 4:
                raise ValueError(
                    f"layout_kv=PA_BNBD 要求 k_cache 4 维 (NB, kv, BS, D), "
                    f"got dim={k_cache.dim()}"
                )
            if k_cache.shape[1] != 1:
                raise ValueError(
                    f"k_cache dim1 (kv heads) must be 1 (MLA 硬约束), "
                    f"got {k_cache.shape[1]}"
                )
            block_size = k_cache.shape[2]  # BS
            k_dim = k_cache.shape[-1]  # D
            if k_dim != head_dim_qk:
                raise ValueError(f"k_cache 宽度 {k_dim} != head_dim_qk({head_dim_qk})")

        # N 位置随 layout: TND/BNSD 在 dim1, BSND (B,S,N,D) 在 dim2
        if layout_q == "BSND":
            num_heads_q, seq_decl, batch = q.shape[2], q.shape[1], q.shape[0]
        elif layout_q == "BNSD":
            num_heads_q, seq_decl, batch = q.shape[1], q.shape[2], q.shape[0]
        else:  # TND
            num_heads_q, seq_decl, batch = q.shape[1], None, None

        return {
            "num_heads_q": num_heads_q,
            "head_dim_qk": head_dim_qk,
            "head_dim_v": head_dim_v,
            "rope_dim": head_dim_qk - head_dim_v,  # = 64
            "block_size": block_size,
            "batch": batch,
            "seq_decl": seq_decl,
            "layout_kv": layout_kv,
        }

    # ==========================================================================
    # Public entry point (参数与算子 17 参对齐)
    # ==========================================================================

    @torch.no_grad()
    def forward_golden(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        head_dim_v: int = 512,
        softmax_scale: Optional[float] = None,
        mask_mode: int = 0,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "TND",
        layout_kv: str = "PA_NZ",
        layout_out: Optional[str] = None,
        return_softmax_lse: bool = False,
        device_mode: str = "npu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 flash_mla_with_kvcache 参考实现。

        q: BNSD (B,N,S,576) / BSND (B,S,N,576) / TND (T,N,576);
        k_cache: PA_NZ 5D (total_blocks, 1, DSUB=36, block_size, 16)。
        cu_seqlens_q: TND 必传, 累计含前导 0; seqused_q: 每 batch 实际 Q 长。
        返回 (attn_out, softmax_lse); attn_out dtype 同 q, lse float32。
        """
        if device_mode not in ("cpu", "npu"):
            raise ValueError(f"device_mode must be 'cpu' or 'npu', got {device_mode}")
        layout_out = layout_q if layout_out is None else layout_out
        if layout_kv not in ("PA_NZ", "PA_BBND", "PA_BNBD"):
            raise ValueError(
                f"unsupported layout_kv: {layout_kv} (仅 PA_NZ/PA_BBND/PA_BNBD)"
            )
        if mask_mode not in (0, 3):
            raise ValueError(f"mask_mode 仅支持 0/3, got {mask_mode}")
        if layout_q == "TND" and cu_seqlens_q is None:
            raise ValueError("layout_q=TND 要求 cu_seqlens_q")
        # mask_mode=3 (RIGHT_DOWN): golden 按右对齐位置自构造, 忽略 attn_mask
        if mask_mode == 3:
            attn_mask = None

        # ---- 几何参数统一提取(head_dim_v 为算子参数, TND/BNSD/BSND 的 N/S/B
        # 与三 PA KV 布局的维度语义均在 _derive_geometry 内集中处理) ----
        geo = self._derive_geometry(q, k_cache, layout_q, layout_kv, head_dim_v)
        self._geo = geo  # 供 _compute_per_batch 使用
        self._softmax_scale = 1.0 / math.sqrt(geo["head_dim_qk"])

        if device_mode == "npu":
            for name, t in [("q", q), ("k_cache", k_cache)]:
                if t.device.type != "npu":
                    raise ValueError(
                        f"device_mode='npu' requires {name} to be on NPU, "
                        f"got {t.device}"
                    )

        orig_dtype = q.dtype
        scale = softmax_scale if softmax_scale is not None else self._softmax_scale

        # 1. cpu 模式: 辅助张量搬 CPU(q/k 由调用方自备, 计算留在 q.device)
        if device_mode == "cpu":
            block_table = block_table.cpu()
            cache_seqlens = cache_seqlens.cpu()
            if cu_seqlens_q is not None:
                cu_seqlens_q = cu_seqlens_q.cpu()
            if seqused_q is not None:
                seqused_q = seqused_q.cpu()

        cu_q_list = cu_seqlens_q.tolist() if cu_seqlens_q is not None else None
        cache_lens = [int(x) for x in cache_seqlens.tolist()]

        # 2. 推导每 batch 序列长: seqused_q 优先, cu diff 次之, 否则声明 S
        if seqused_q is not None:
            q_lens = seqused_q.tolist()
        elif cu_q_list is not None:
            q_lens = [
                cu_q_list[b + 1] - cu_q_list[b] for b in range(len(cu_q_list) - 1)
            ]
        else:
            q_lens = None
        kv_lens = cache_lens  # PA cache 长度 = cache_seqlens(接口参数)

        batch = len(cu_q_list) - 1 if layout_q == "TND" else geo["batch"]
        if len(cache_lens) != batch or any(x < 0 for x in cache_lens):
            raise ValueError(f"cache_seqlens must contain {batch} nonnegative lengths")
        if q_lens is not None and len(q_lens) != batch:
            raise ValueError(f"seqused_q must contain {batch} lengths")
        if layout_q == "TND" and (
            cu_q_list[0] != 0
            or cu_q_list[-1] != q.shape[0]
            or any(a > b for a, b in zip(cu_q_list, cu_q_list[1:]))
        ):
            raise ValueError(
                "cu_seqlens_q must monotonically cover the TND tensor from 0"
            )

        # 值域校验: seqused_q 不允许超过物理/声明序列长(禁止静默截断或越界错算;
        # 与 loader/backend 的长度校验互补, 违反即清晰报错)
        if q_lens is not None:
            for b_idx, ql in enumerate(q_lens):
                if layout_q == "TND":
                    phys_len = cu_q_list[b_idx + 1] - cu_q_list[b_idx]
                else:
                    phys_len = geo["seq_decl"]
                if not (0 <= ql <= phys_len):
                    raise ValueError(
                        f"seqused_q[{b_idx}]={ql} 超出该 batch 物理/声明序列长 "
                        f"{phys_len}(layout_q={layout_q}); 序列实际长度不得越过张量长度"
                    )

        # Resolve auto compute_mode (同 flash golden 口径)
        if q_lens is not None:
            max_sq = max(q_lens, default=0)
            max_product = max((sq * skv for sq, skv in zip(q_lens, kv_lens)), default=0)
        else:
            max_sq = geo["seq_decl"]
            max_skv = max(kv_lens, default=0)
            max_product = max_sq * max_skv
        if max_product * geo["num_heads_q"] > self.TILED_CROSSOVER:
            compute_mode = "tiled"
        else:
            compute_mode = "full"
        self.last_compute_mode = compute_mode

        # 3. 逐 batch 计算(防 OOM)
        out_native, lse_f32 = self._compute_per_batch(
            q,
            k_cache,
            q_lens,
            kv_lens,
            layout_q,
            cu_q_list,
            block_table,
            mask_mode,
            attn_mask,
            scale,
            compute_mode,
        )

        # 4. 输出布局转换 + dtype
        if layout_q == "TND":
            attn_out = out_native.to(orig_dtype)  # (T, N, DV)
        else:
            out_bnsd = out_native.to(orig_dtype)  # (B, N, S, DV)
            if layout_q == "BNSD":
                attn_out = out_bnsd
            else:  # BSND
                attn_out = out_bnsd.permute(0, 2, 1, 3).contiguous()

        # 4b. 目标输出布局(与 fia golden 同口径; golden 内部完成, 外部不得变换)
        if layout_out != layout_q:
            if layout_q == "TND" and layout_out == "NTD":
                attn_out = attn_out.permute(1, 0, 2).contiguous()
            elif layout_q == "BNSD" and layout_out == "NBSD":
                attn_out = attn_out.permute(1, 0, 2, 3).contiguous()
            elif layout_q == "BSND" and layout_out == "NBSD":
                attn_out = attn_out.permute(2, 0, 1, 3).contiguous()
            elif layout_q == "BNSD" and layout_out == "BSND":
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
            elif layout_q == "BSND" and layout_out == "BNSD":
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
            else:
                raise ValueError(
                    f"Unsupported golden output conversion {layout_q} -> {layout_out}"
                )

        # 5. softmax_lse (算子契约: TND -> (N, T), 非 TND -> (B, N, S))
        if not return_softmax_lse:
            softmax_lse = torch.zeros(1, dtype=torch.float32, device=attn_out.device)
        elif layout_q == "TND":
            softmax_lse = lse_f32.t().contiguous()  # (T, N) -> (N, T)
        else:
            softmax_lse = lse_f32

        return attn_out, softmax_lse

    # ==========================================================================
    # Per-batch compute
    # ==========================================================================

    def _compute_per_batch(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        q_lens: Optional[list],
        kv_lens: list,
        layout_q: str,
        cu_q: Optional[list],
        block_table: torch.Tensor,
        mask_mode: int,
        attn_mask: Optional[torch.Tensor],
        scale: float,
        compute_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """逐 batch 计算, 每 batch q/k 保持输入 dtype 转 BNSD 后进入单头 compute。

        kv 恒为 PA_NZ 单头(MLA): k 还原为 (1, 1, skv, D), V = K 的 nope 部分
        (宽 head_dim_v), QK 内积用 D=576 全宽(= nope+rope 之和)。
        """
        N_q = self._geo["num_heads_q"]
        dv = self._geo["head_dim_v"]
        seq_decl = self._geo["seq_decl"]  # BNSD/BSND 声明序列长(TND=None)
        group = N_q  # num_heads_kv == 1, 单 KV 头广播到所有 Q 头

        if layout_q == "TND":
            B = len(cu_q) - 1
        else:
            B = self._geo["batch"]

        if q_lens is None:
            if layout_q == "TND":
                q_lens = [cu_q[b + 1] - cu_q[b] for b in range(B)]
            else:  # BNSD/BSND: 声明长度
                q_lens = [seq_decl] * B

        is_tnd_out = layout_q == "TND"
        if is_tnd_out:
            total_q = q.shape[0]
            out_tnd = torch.zeros(total_q, N_q, dv, dtype=q.dtype, device=q.device)
            lse_tnd = torch.zeros(total_q, N_q, dtype=torch.float32, device=q.device)
            out = lse = None
        else:
            out = torch.zeros(B, N_q, seq_decl, dv, dtype=q.dtype, device=q.device)
            lse = torch.zeros(B, N_q, seq_decl, dtype=torch.float32, device=q.device)
            out_tnd = lse_tnd = None

        for b in range(B):
            sq = q_lens[b]
            skv = kv_lens[b]
            if is_tnd_out:
                q_start = cu_q[b]
            # seqused 截断行(物理 > 计算)的 LSE: 算子输出 +inf。
            # 必须先于空批/空 KV 分支执行, 保证 skv<=0 或 sq<=0 时截断行
            # 同样 +inf(否则被 early-continue 短路成 0)。
            if is_tnd_out:
                phys = cu_q[b + 1] - cu_q[b]
                if phys > sq:
                    lse_tnd[q_start + sq : q_start + phys, :] = float("inf")
            elif sq < seq_decl:
                lse[b, :, sq:] = float("inf")
            if sq <= 0 or skv <= 0:
                # 空 batch / 空 KV: 算子 LSE 输出 +inf
                if is_tnd_out:
                    lse_tnd[q_start : q_start + sq, :] = float("inf")
                else:
                    lse[b, :, :] = float("inf")
                continue

            q_b, k_b = self._extract_batch_bnsd(
                q,
                k_cache,
                b,
                sq,
                skv,
                layout_q,
                self._geo["layout_kv"],
                cu_q,
                block_table,
            )
            # V = K 的 nope 部分 (kv 复用, rope 不参与 V 投影)
            v_b = k_b[..., :dv]

            # 单 KV 头 -> 所有 Q 头广播; masked 用右对齐 causal 自构造
            mask_b = (
                None
                if (compute_mode == "tiled" and mask_mode == 3)
                else self._build_batch_mask(
                    sq,
                    skv,
                    mask_mode,
                    attn_mask,
                    q.device,
                    b,
                )
            )

            delta = skv - sq
            k_bh = k_b[0, 0, :skv, :]  # (Skv, D=576)
            v_bh = v_b[0, 0, :skv, :]  # (Skv, DV=512)
            # Fold heads into query rows. Each KV tile is widened once for
            # the whole row tile instead of once per head (critical for long KV).
            per_head_mask = (
                attn_mask is not None
                and attn_mask.ndim == 4
                and attn_mask.shape[1] != 1
            )
            head_step = 1 if per_head_mask else group
            for first in range(0, group, head_step):
                last = min(first + head_step, group)
                q_heads = q_b[0, first:last, :sq, :]
                selected_mask = mask_b
                if per_head_mask:
                    selected_mask = self._build_batch_mask(
                        sq, skv, mask_mode, attn_mask, q.device, b, head_index=first
                    )
                flat_q = q_heads.reshape(-1, q_heads.shape[-1])
                if compute_mode == "tiled":

                    def mask_fn(qs, qe, ks, ke):
                        positions = torch.arange(qs, qe, device=q.device) % sq
                        if mask_mode == 3:
                            return self._build_mask_mode_mask(
                                positions, torch.arange(ks, ke, device=q.device), delta
                            )
                        if selected_mask is None:
                            return None
                        return selected_mask[:, ks:ke].index_select(0, positions)

                    def kv_bounds(qs, qe):
                        if mask_mode != 3:
                            return 0, skv
                        # A row tile can straddle heads; each head restarts Q.
                        last_q = sq - 1 if qs // sq != (qe - 1) // sq else (qe - 1) % sq
                        return 0, last_q + delta + 1

                    o, l = attention_single(
                        flat_q,
                        k_bh,
                        v_bh,
                        scale,
                        self.TILE_SIZE,
                        mask_fn,
                        kv_bounds=kv_bounds,
                    )
                else:
                    if selected_mask is not None and last - first > 1:
                        selected_mask = selected_mask.repeat(last - first, 1)
                    o, l = self._compute_full_single(
                        flat_q, k_bh, v_bh, selected_mask, scale
                    )
                o = o.reshape(last - first, sq, dv)
                l = l.reshape(last - first, sq)
                if is_tnd_out:
                    out_tnd[q_start : q_start + sq, first:last, :] = o.permute(1, 0, 2)
                    lse_tnd[q_start : q_start + sq, first:last] = l.T
                else:
                    out[b, first:last, :sq, :] = o
                    lse[b, first:last, :sq] = l

            del q_b, k_b, v_b

        if is_tnd_out:
            return out_tnd, lse_tnd
        return out, lse

    def _extract_batch_bnsd(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        b: int,
        sq: int,
        skv: int,
        layout_q: str,
        layout_kv: str,
        cu_q: Optional[list],
        block_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取单 batch q/k 为 (1, N, S, D) in input dtype (PA 块还原)。

        k 还原为 QK 全宽 D(=head_dim_qk); 布局:
        PA_NZ 5D block (DSUB, BS, BLK) -> (BS, D);
        PA_BBND block (BS, kv, D) / PA_BNBD block (kv, BS, D)。
        """
        # Q
        if layout_q == "BNSD":
            q_b = q[b : b + 1, :, :sq, :]
        elif layout_q == "BSND":
            q_b = q[b : b + 1, :sq, :, :].permute(0, 2, 1, 3)
        else:  # TND
            s, e = cu_q[b], cu_q[b + 1]
            q_b = q[s:e].unsqueeze(0).permute(0, 2, 1, 3)

        k_b = gather_pages(k_cache, block_table, b, skv, layout_kv)
        return q_b, k_b

    # ==========================================================================
    # Tiled (block-wise online softmax)
    # ==========================================================================

    def _compute_tiled_single(
        self,
        q: torch.Tensor,  # (Sq, D) input dtype
        k: torch.Tensor,  # (Skv, D) input dtype
        v: torch.Tensor,  # (Skv, DV) input dtype
        mask: Optional[torch.Tensor],  # (Sq, Skv) bool, True=mask out
        scale: float,
        mask_mode: int = 0,
        delta: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Online softmax with input-dtype Cube operands and FP32 accumulators."""

        def mask_fn(qs, qe, ks, ke):
            if mask_mode == 3:
                q_idx = torch.arange(qs, qe, device=q.device)
                k_idx = torch.arange(ks, ke, device=q.device)
                return self._build_mask_mode_mask(q_idx, k_idx, delta)
            return None if mask is None else mask[qs:qe, ks:ke]

        return attention_single(
            q,
            k,
            v,
            scale,
            self.TILE_SIZE,
            mask_fn,
            kv_bounds=(lambda qs, qe: (0, qe + delta)) if mask_mode == 3 else None,
        )

    # ==========================================================================
    # Full (non-tiled) golden
    # ==========================================================================

    def _compute_full_single(
        self,
        q: torch.Tensor,  # (Sq, D) input dtype
        k: torch.Tensor,  # (Skv, D) input dtype
        v: torch.Tensor,  # (Skv, DV) input dtype
        mask: Optional[torch.Tensor],  # (Sq, Skv) bool
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-tile mixed-precision reference; exp weights round before PV."""
        return attention_single(
            q,
            k,
            v,
            scale,
            max(q.shape[0], k.shape[0], 1),
            lambda qs, qe, ks, ke: mask,
            None,
        )

    # ==========================================================================
    # Mask generation (右对齐 causal, 同 flash golden)
    # ==========================================================================

    @staticmethod
    def _build_mask_mode_mask(
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        delta: int,
    ) -> Optional[torch.Tensor]:
        """mask_mode=3 (RIGHT_DOWN): q 序列右对齐到 KV 序列末尾。

        Q[i] 位于 KV 位置 i + (Skv - Sq); 屏蔽 k_pos > q_pos。
        decode(Sq=1) 时 q 位于末尾 => 全部 cache key 可见。
        """
        q_pos = q_idx.unsqueeze(1) + delta
        k_pos = k_idx.unsqueeze(0)
        return k_pos > q_pos

    def _build_batch_mask(
        self,
        sq: int,
        skv: int,
        mask_mode: int,
        attn_mask: Optional[torch.Tensor],
        device: torch.device,
        batch_index: int,
        head_index: int = 0,
    ) -> Optional[torch.Tensor]:
        """Build (sq, skv) bool mask for a single batch. True = mask out."""
        if mask_mode == 3:
            q_idx = torch.arange(sq, device=device)
            k_idx = torch.arange(skv, device=device)
            return self._build_mask_mode_mask(q_idx, k_idx, skv - sq)
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask[
                    0 if attn_mask.shape[0] == 1 else batch_index,
                    0 if attn_mask.shape[1] == 1 else head_index,
                ]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[0 if attn_mask.shape[0] == 1 else batch_index]
            return attn_mask[:sq, :skv].to(device=device, dtype=torch.bool)
        return None
