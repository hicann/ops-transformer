#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
PoolKeyIndexer 统一标杆实现(CPU 金标 + NPU 组合标杆, 按输入设备自动选择)。

统一入口 ``pool_key_indexer_reference(...)``:
  - 输入 tensor 在 CPU  -> CPU 金标实现(逐 batch 精确计算, 全布局能力, 速度慢,
    作为精度基准);
  - 输入 tensor 在 NPU  -> NPU 组合标杆实现(纯 torch 标准算子向量化, 无 Python
    逐行循环, 作为自定义算子的性能对比基线)。

两种实现接口能力一致(支持 BSND/TND query, BSND/TND/PA_BBND key, 不等长 batch、
0 池/0 长度 batch、PA 0 轴非连续 key、BSND 可选 actual_seq 等), 计算语义一致:

  1. 池级分数计算(逐 batch, float32):
        dot[i, n, j] = dot(query[i, n, :], pool_key[j, :]) / sqrt(headDim)  # 缩放点积
        score[i, j]  = sum_n weights[i, n] * relu(dot[i, n, j])             # 多头加权聚合
     分数通道全程 fp32(q/k/weights 上转 fp32 计算), 与算子 cube 侧 fp16 输入
     fp32 累加 + vector 侧 fp32 加权求和的精度管线对齐; 若用 fp16 分数会引入
     ~1e-3 舍入, 近并列分数翻转 TopK 选择, 导致与算子输出不可比。
  2. 因果可见池数(mask_mode=3, rightDownCausal 右对齐因果):
        L          = 池数 * pool_size + pool_tail_k[b]            # 原始 token 总数
        valid[i]   = clamp(floor((L - S1 + i + 1) / pool_size), 0, 池数)
     即 query 第 i 行只能看到前 valid[i] 个池; mask_mode=0 时所有池可见。
  3. 池级 TopK: 取 k = topk // pool_size 个池, 分数降序选取
     (标杆在同分时按下标升序取; NPU 并列分数取舍可能不同, 由集合比对吸收)。
  4. 索引展开: 选中池 p 展开为 token 索引
        p * pool_size, p * pool_size + 1, ..., p * pool_size + pool_size - 1
  5. 尾块追加(输出 topk 位置起, 容量 pool_size - 1):
        mask_mode=0: visible = pool_tail_k
        mask_mode=3: visible = max(0, min(pool_tail_k, gpos - (L - pool_tail_k) + 1))
                      （按尾池真实起点 L-pool_tail_k 因果截断, 与 kernel 修复后
                       公式一致; 不可用选池预算 topk 代替尾池起点）
                      其中 gpos = L - S1 + i (query 行的全局 token 位置)
        第 t 个可见尾 token 的索引为 L - pool_tail_k + t
  6. 输出:
        indices: (B, S1, topk+pool_size-1) 或 (T1, topk+pool_size-1), int32, 无效 -1
        values:  (B, S1, topk//pool_size)   或 (T1, topk//pool_size),  float32, 无效 -inf

NPU 实现说明:
  - 不等长 batch 通过"归一化到等长稠密布局 + 有效性掩码"向量化完成:
    query 按批 gather 填充到 (B, S1max), pool_key 归一化到 (B, Pmax)
    (TND gather / PA 经 block_table gather), 无效行/列以 -inf 分数掩蔽,
    输出按有效性反填充(unpad);
  - TopK 并列分数取舍与 CPU 标杆可能不同, 由集合比对策略吸收(同既有套件)。

可直接运行本文件做自验(需 NPU):  python pool_key_indexer_reference.py
  1. CPU 实现自验(密集/PA 布局输出 shape/取值范围/交叉一致性);
  2. NPU 实现 vs CPU 实现一致性自验(等长 + 不等长 + 0 池 + PA 非连续 +
     BSND actual_seq 等, indices 集合比对 + values 参考比对)。
"""

# ---------------------------------------------------------------------------
# ST 用例说明（aclnnPoolKeyIndexer 泛化精度 ST 用例（取 ATK 泛化资产前 100 条））
#
# 本文件为配套执行器，与同目录 ``atk_aclnnPoolKeyIndexer.json``（70 条用例）联合使用：
#
#   atk task -n <node.yaml> -c <atk_aclnnPoolKeyIndexer.json> -p <executor_aclnnPoolKeyIndexer.py> -s 0 -e 70 -mt 1
#
# 用例截取自转测泛化精度资产前 100 条（shape 从小到大递增）中
# 按实测耗时剔除慢例后的 70 条（910B 实测总时长 251s），覆盖各 layout 与 dtype 组合；节点配置需含 pyaclnn accuracy 节点与 npu accuracy 标杆节点。
# --------------------------------------------------------------------------

import math

import torch

__all__ = ["pool_key_indexer_reference"]

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _as_int_list(x):
    """int tensor / list / 标量 -> list[int]"""
    if x is None:
        return None
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _prefix_to_counts(prefix):
    """前缀和序列 -> 每 batch 数量列表"""
    p = _as_int_list(prefix)
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]


def _topk_desc(scores, k):
    """降序 TopK; NPU topk 不可用时回退到 sort。

    NPU 上 torch.topk 插件内部走 NonZero/AICPU 路径, 在亿级 score 上触发
    aicpu timeout(E30008, 看门狗 ~10min), 且超时为进程级内错无法 try/except
    捕获, 会级联饿死同 worker 的后续用例(实测 k=1024/84M 与 k=256/446M 均
    触发) —— NPU 上一律走 sort(AICore 向量路径, 行为稳定)。

    并列取舍: 与 _reference_cpu 金标约定一致, 同分按下标升序选取
    (金标用 argsort(..., stable=True) 显式实现)。CPU 侧用 stable sort
    保证该约定 —— oversized 用例的 cpu_benchmark 真值由本函数所在的
    _reference_npu_blocked 分块路径产生(替代 _reference_cpu), 并列取舍
    必须与被替代的金标一致, 否则 relu 产生的精确 0 分并列区会使 TopK
    边界行集合比对大面积翻转(实测 3 用例 23%~59% 行误判失败)。
    NPU 侧 torch_npu sort 不保证 stable 参数可用, 保持非稳定(仅影响
    参考性 bm-vs-truth 指标, 不影响 op-vs-truth 精度门禁)。
    """
    if getattr(scores, "is_npu", False):
        sorted_vals, sorted_idx = torch.sort(scores, dim=-1, descending=True)
        return sorted_vals[..., :k], sorted_idx[..., :k]
    # CPU: stable sort —— 同分按下标升序(金标约定), topk 并列顺序未定义不用
    sorted_vals, sorted_idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    return sorted_vals[..., :k], sorted_idx[..., :k]


def _validate_args(query, pool_key, layout_q, layout_k, topk, pool_size, mask_mode):
    """公共参数校验(两种实现共用)。"""
    if layout_q not in ("BSND", "TND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_k not in ("BSND", "TND", "PA_BBND"):
        raise ValueError(f"unsupported layout_k: {layout_k}")
    if mask_mode not in (0, 3):
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if topk % pool_size != 0:
        raise ValueError(f"topk({topk}) must be divisible by pool_size({pool_size})")
    if pool_key.shape[-2] != 1:
        raise ValueError(f"pool_key N2 must be 1, got {pool_key.shape[-2]}")
    if query.shape[-1] != pool_key.shape[-1]:
        raise ValueError(
            f"headDim mismatch: query {query.shape[-1]} vs pool_key {pool_key.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# CPU 金标实现(逐 batch 精确计算)
# ---------------------------------------------------------------------------


def _split_query(query, weights, layout_q, actual_seq_q):
    """拆分 query/weights 为逐 batch 的 (Sb, N1, D) / (Sb, N1) float32 列表。

    返回 (q_list, w_list, counts, batch, s1_dim):
      - counts: 每 batch 有效 query token 数
      - s1_dim: BSND 输出的 S1 维(query.shape[1], 含无效行)
    """
    if layout_q == "BSND":
        batch, s1_dim, n1, d = query.shape
        if actual_seq_q is not None:
            counts = _as_int_list(actual_seq_q)
        else:
            counts = [s1_dim] * batch
        q_list = [query[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        w_list = [weights[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        return q_list, w_list, counts, batch, s1_dim
    if layout_q == "TND":
        t1, n1, d = query.shape
        if actual_seq_q is None:
            raise ValueError("layout_q='TND' requires actual_seq_q (prefix sums)")
        counts = _prefix_to_counts(actual_seq_q)
        q_list, w_list, start = [], [], 0
        for c in counts:
            q_list.append(query[start : start + c].to(torch.float32))
            w_list.append(weights[start : start + c].to(torch.float32))
            start += c
        return q_list, w_list, counts, len(counts), t1
    raise ValueError(f"unsupported layout_q: {layout_q}")


def _split_key(pool_key, layout_k, actual_seq_k, block_table):
    """拆分 pool_key 为逐 batch 的 (Pb, D) float32 列表。

    - BSND:    pool_key (B, S2, N2, D), actual_seq_k 可选(每 batch 池数, 非前缀和);
    - TND:     pool_key (T2, N2, D),    actual_seq_k 必传(池数前缀和);
    - PA_BBND: pool_key (blockNum, blockSize, N2, D), 块内每行一个池,
               需 block_table(B, maxBlocks) 与 actual_seq_k(每 batch 池数, 非前缀和);
               逻辑池 p 属于物理块 block_table[b][p // blockSize] 的第 p % blockSize 行,
               空槽(-1)跳过; 支持 0 轴非连续(strided view)。
    """
    if layout_k == "BSND":
        batch, s2, n2, d = pool_key.shape
        if actual_seq_k is not None:
            counts = _as_int_list(actual_seq_k)
        else:
            counts = [s2] * batch
        return [pool_key[b, :c, 0, :].to(torch.float32) for b, c in enumerate(counts)]
    if layout_k == "TND":
        t2, n2, d = pool_key.shape
        if actual_seq_k is None:
            raise ValueError("layout_k='TND' requires actual_seq_k (prefix sums)")
        counts = _prefix_to_counts(actual_seq_k)
        keys, start = [], 0
        for c in counts:
            keys.append(pool_key[start : start + c, 0, :].to(torch.float32))
            start += c
        return keys
    if layout_k == "PA_BBND":
        if block_table is None:
            raise ValueError("layout_k='PA_BBND' requires block_table")
        if actual_seq_k is None:
            raise ValueError(
                "layout_k='PA_BBND' requires actual_seq_k (pools per batch)"
            )
        block_num, block_size, n2, d = pool_key.shape
        bt = block_table.detach().cpu()
        batch, max_blocks = bt.shape
        counts = _as_int_list(actual_seq_k)
        keys = []
        for b in range(batch):
            need, rows, pos = counts[b], [], 0
            for t in range(max_blocks):
                if pos >= need:
                    break
                blk = int(bt[b, t])
                if blk < 0:
                    continue
                take = min(block_size, need - pos)
                rows.append(pool_key[blk, :take, 0, :].to(torch.float32))
                pos += take
            if rows:
                keys.append(torch.cat(rows, dim=0))
            else:
                keys.append(pool_key.new_zeros((0, d)).to(torch.float32))
        return keys
    raise ValueError(f"unsupported layout_k: {layout_k}")


def _e8m0_decode(t):
    """E8M0 raw(uint8) -> float32 scale: 2^(raw-127), raw==255 -> NaN。

    与 QLIv2 golden 的 e8m0_raw_to_float 一致(0xFF 约定为 NaN, 不拦截)。
    """
    raw = t.detach().cpu().to(torch.uint8)
    scale = torch.pow(2.0, raw.to(torch.float32) - 127.0)
    return torch.where(raw == 255, torch.full_like(scale, float("nan")), scale)


def _split_q_descale_generic(ds, layout_q, q_counts):
    """通用 q_descale 拆分(任意尾维, 已解码 float) -> 逐 batch (Sb, N1, ...) 列表。"""
    if layout_q == "BSND":
        return [ds[b, :c] for b, c in enumerate(q_counts)]
    out, start = [], 0
    for c in q_counts:
        out.append(ds[start : start + c])
        start += c
    return out


def _split_k_descale_generic(
    ds, layout_k, actual_seq_k, k_counts, block_table=None, block_size=None
):
    """通用 k_descale 拆分(任意尾维, 已解码 float) -> 逐 batch (Pb, ...) 列表。

    支持与 pool_key 相同的三布局(BSND 切批 / TND 前缀切 / PA 经 block_table)。
    """
    if layout_k == "BSND":
        return [ds[b, :c] for b, c in enumerate(k_counts)]
    if layout_k == "TND":
        out, start = [], 0
        for c in k_counts:
            out.append(ds[start : start + c])
            start += c
        return out
    # PA_BBND: (blockNum, blockSize, N2, ...) 经 block_table 重排逻辑池序
    if block_table is None:
        raise ValueError("PA k_descale requires block_table")
    bt = block_table.detach().cpu()
    batch, max_blocks = bt.shape
    out = []
    for b in range(batch):
        need, rows, pos = k_counts[b], [], 0
        for t in range(max_blocks):
            if pos >= need:
                break
            blk = int(bt[b, t])
            if blk < 0:
                continue
            take = min(block_size, need - pos)
            rows.append(ds[blk, :take, 0])
            pos += take
        out.append(
            torch.cat(rows, dim=0) if rows else ds.new_zeros((0,) + ds.shape[3:])
        )
    return out


def _split_q_descale(q_descale, layout_q, actual_seq_q, q_counts):
    """q_descale(mode=0, per-token-head) -> 逐 batch (Sb, N1) float32 列表。"""
    if q_descale is None:
        return None
    ds = q_descale.detach().cpu().to(torch.float32)
    if layout_q == "BSND":
        batch, _, n1 = ds.shape
        return [ds[b, :c] for b, c in enumerate(q_counts)]
    # TND: (T1, N1) 按前缀和切
    out, start = [], 0
    for c in q_counts:
        out.append(ds[start : start + c])
        start += c
    return out


def _split_k_descale(
    k_descale, layout_k, actual_seq_k, k_counts, block_table=None, block_size=None
):
    """k_descale(mode=0) -> 逐 batch (Pb,) float32 列表。

    BSND (B,S2,N2) -> [b, :c, 0]; TND (T2,N2) 前缀切;
    PA (blockNum,blockSize,N2) 与 pool_key 同构: 逻辑池 p 的 scale 在
    物理块 block_table[b][p//bs] 第 p%bs 行。
    """
    if k_descale is None:
        return None
    ds = k_descale.detach().cpu().to(torch.float32)
    if layout_k == "BSND":
        return [ds[b, :c, 0] for b, c in enumerate(k_counts)]
    if layout_k == "TND":
        out, start = [], 0
        for c in k_counts:
            out.append(ds[start : start + c, 0])
            start += c
        return out
    # PA_BBND
    if block_table is None:
        raise ValueError("PA k_descale requires block_table")
    bt = block_table.detach().cpu()
    batch, max_blocks = bt.shape
    out = []
    for b in range(batch):
        need, vals, pos = k_counts[b], [], 0
        for t in range(max_blocks):
            if pos >= need:
                break
            blk = int(bt[b, t])
            if blk < 0:
                continue
            take = min(block_size, need - pos)
            vals.append(ds[blk, :take, 0])
            pos += take
        out.append(torch.cat(vals, dim=0) if vals else ds.new_zeros((0,)))
    return out


def _compute_batch_loop(
    q_b,
    w_b,
    k_b,
    tail_k,
    topk,
    pool_size,
    mask_mode,
    q_scale_b=None,
    k_scale_b=None,
    mx_q_scale_b=None,
    mx_k_scale_b=None,
    int8_semantics=False,
    sparse_indices_mode=0,
):
    """单个 batch 的池级 TopK 选择与索引展开(逐行 Python 循环, 等价性对账基准)。

    入参:
      q_b: (S, N1, D) float32;  w_b: (S, N1) float32;  k_b: (P, D) float32;
      tail_k: 尾部有效 token 数; topk/pool_size/mask_mode: 算子属性;
      q_scale_b/k_scale_b: mode=0 的 per-token-head 反量化系数(float32):
        q_scale_b (S, N1) / None; k_scale_b (P,) / None。
      mx_q_scale_b/mx_k_scale_b: mode=1 的 mxFP8 scale(float32, 已解码):
        mx_q_scale_b (S, N1, D/32) / None; mx_k_scale_b (P, D/32) / None。
        按 per-32-D-group 反量化后计算(参考 QLIv2 cal_atten_per_batch_mx)。

    返回:
      token_idx: (S, topk + pool_size - 1) int32, 无效 -1
      pool_val:  (S, topk // pool_size) float32,   无效 -inf
    """
    s_len, n1, d = q_b.shape
    p_len = k_b.shape[0]
    sparse_count = topk // pool_size
    out_len = topk + pool_size - 1
    l_orig = p_len * pool_size + tail_k  # 原始 token 总数

    token_idx = torch.full((s_len, out_len), -1, dtype=torch.int32)
    pool_val = torch.full((s_len, sparse_count), _NEG_INF, dtype=torch.float32)
    if s_len == 0:
        return token_idx, pool_val
    # p_len==0(纯尾块批次, 0 池 + pool_tail_k>0)不再短路: 池选择自然为空,
    # 步骤 5 尾块追加照常执行(l_orig = tail_k, seqTailStart = 0)

    # 步骤 1: 池级分数
    #   mode=0: score[i,j] = (Σ_n relu(Σ_d q·k) · w[i,n] · dq) · dk / sqrt(d)
    #           (q_scale 乘入 weight、k_scale 乘入聚合和, 与 kernel 数值口径一致;
    #            scale 恒正可从 D 维点积提出, 数学等价于先反量化)
    #   mode=1: q/k 先按 per-32-D-group scale 反量化, 再点积(参考 QLIv2 MX golden)
    #   非量化: score[i,j] = Σ_n w[i,n] · relu(q[i,n]·k[j]/sqrt(d))
    scale = 1.0 / math.sqrt(d)
    qk_scale_phys = 1.0  # int8: values 输出补乘的物理缩放(排序键域不含, 见 int8 分支)
    if mx_q_scale_b is not None or mx_k_scale_b is not None:
        # mxFP8: 展开 group scale 到 D 维后反量化
        # 0 元素 batch(0 池/0 行): reshape(x, -1) 对 0 元素含糊会抛 RuntimeError,
        # 跳过缩放(空张量无元素可缩放, 语义不变)
        if mx_q_scale_b is not None and mx_q_scale_b.numel() > 0:
            mxq = (
                mx_q_scale_b.reshape(mx_q_scale_b.shape[0], -1)
                if mx_q_scale_b.dim() > 2
                else mx_q_scale_b
            )  # (S, N1*D/32) -> (S, N1, D/32)
            mxq = mxq.reshape(mxq.shape[0], -1, d // 32)
            q_b = q_b * mxq.repeat_interleave(32, dim=-1)
        if mx_k_scale_b is not None and mx_k_scale_b.numel() > 0:
            mxk = mx_k_scale_b.reshape(mx_k_scale_b.shape[0], -1)  # (P, D/32)
            k_b = k_b * mxk.repeat_interleave(32, dim=-1)
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        # [键域对齐 QLIV2] mxFP8 排序域 = bf16(QLIV2 golden cal_atten_per_batch_mx:
        # to_be_sort_ele.to(torch.bfloat16); kernel 侧 uint16 bf16 sortable key 同域)
        scores = torch.einsum("sn,snp->sp", w_b, dot.clamp_min(0.0)) * scale
        scores = scores.to(torch.bfloat16).to(torch.float32)
    elif int8_semantics:
        # quant_mode=2 七步逐位口径(对齐 master 27b8f5eab arch22 键域契约, 与 QLIv2 一致):
        # INT32 精确点积 -> fp32×2^-10 -> FP16 RNE(带 FP16_MAX clip) -> ReLU
        # -> wq FP16 域预乘 -> FP32 聚合 -> ×dk -> bf16 RNE round-trip。
        # 排序键域不含 ×qkScale(1/√d×2^10)——物理缩放仅补乘进 values 输出
        # (kernel: Muls(qkScale,1.0f) 摘除 + SortAll 前 bf16 round-trip + Extract 后补乘)。
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        t = dot * 2.0**-10
        sh = t.clamp(-65504.0, 65504.0).to(torch.float16).to(torch.float32)
        sh = sh.clamp_min(0.0)
        wq = (w_b.to(torch.float16) * q_scale_b.to(torch.float16)).to(torch.float32)
        scores = torch.einsum("sn,snp->sp", wq, sh)
        if k_scale_b is not None:
            scores = scores * k_scale_b.reshape(1, p_len)
        # [键域对齐 QLIV2·终版] int8 排序域 = fp32 直接稳定排序, 不做 bf16 round-trip
        # (QLIV2 golden cal_atten_per_batch_int8: reduce_sum.clone() 无 bf16;
        #  kernel 侧 arch35/arch22 int8 同为 fp32 域 —— 逐位对齐, 无需桶仲裁)
        qk_scale_phys = scale * 1024.0
    else:
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        dot = dot.clamp_min(0.0)
        if q_scale_b is not None:
            # q_scale 乘入 weight(per-token-head, 与 kernel 一致)
            w_eff = w_b * q_scale_b  # (S, N1)
        else:
            w_eff = w_b
        scores = torch.einsum("sn,snp->sp", w_eff, dot)  # (S, P)
        if k_scale_b is not None:
            # k_scale 乘入聚合和(每池 1 个标量; squeeze 吸收 N2=1 维)
            scores = scores * k_scale_b.reshape(1, p_len)
        scores = scores * scale
        if q_scale_b is not None:
            # [键域对齐 QLIV2] per-token FP8(mode0) 排序域 = bf16(QLIV2 golden
            # cal_atten_per_batch_fp8: to_be_sort_ele.to(torch.bfloat16)); 非量化不量化键域
            scores = scores.to(torch.bfloat16).to(torch.float32)

    arange_pool = torch.arange(pool_size, dtype=torch.int64)
    for i in range(s_len):
        # 步骤 2: 因果可见池数
        if mask_mode == 3:
            valid = (l_orig - s_len + i + 1) // pool_size  # floor 除法
            valid = max(0, min(valid, p_len))
        else:
            valid = p_len
        select = min(valid, sparse_count)
        if select > 0:
            # 步骤 3: 池级 TopK(分数降序, 同分按下标升序)
            order = torch.argsort(scores[i, :valid], descending=True, stable=True)
            picked = order[:select]
            # 步骤 4: 展开为 token 索引(依次填入, 总长 select*pool_size <= topk)
            tokens = picked.to(torch.int64).unsqueeze(1) * pool_size + arange_pool
            flat = tokens.reshape(-1)
            token_idx[i, : flat.numel()] = flat.to(torch.int32)
            pool_val[i, :select] = scores[i, picked] * qk_scale_phys

        # 步骤 5: 尾块追加(V1 legacy: 输出 topk 位置起, 容量 pool_size-1;
        # V2 compact: 紧跟有效展开 select*pool_size 起, -1 全部位于行尾)
        if pool_size > 1 and tail_k > 0:
            if mask_mode == 0:
                visible = tail_k
            else:
                gpos = l_orig - s_len + i
                visible = max(0, min(tail_k, gpos - (l_orig - tail_k) + 1))
            if visible > 0:
                tail_start = topk if sparse_indices_mode == 0 else select * pool_size
                tail_tokens = torch.arange(
                    l_orig - tail_k, l_orig - tail_k + visible, dtype=torch.int32
                )
                token_idx[i, tail_start : tail_start + visible] = tail_tokens

    return token_idx, pool_val


def _compute_batch_vec(
    q_b,
    w_b,
    k_b,
    tail_k,
    topk,
    pool_size,
    mask_mode,
    q_scale_b=None,
    k_scale_b=None,
    mx_q_scale_b=None,
    mx_k_scale_b=None,
    int8_semantics=False,
    sparse_indices_mode=0,
):
    """_compute_batch 的向量化等价实现(步骤 2~5 批处理, 消除逐行 Python 循环)。

    与逐行版逐元素等价:
      - 步骤 2~3: 将 [valid_i, P) 列分数置 -inf 后对整行做 stable 降序 argsort,
        前 select_i 名即「前缀 TopK」—— stable 排序保持同分下标升序, 与
        argsort(scores[i, :valid], stable) 的相对序一致;
      - 步骤 4: (picked × pool_size + arange) 行铺平, 列掩码裁到 sel_i*ps;
      - 步骤 5: 尾块基址 l_orig-tail_k 对所有行相同, 可见数 vis_i 向量化后
        按 tail 槽位逐列 scatter。
    规模收益: S1 可达 10^5 行(非量化套 211 例 >=100k), 逐行 argsort 为
    CPU 金标主要耗时; 向量化后整批一次 argsort。
    """
    s_len, n1, d = q_b.shape
    p_len = k_b.shape[0]
    sparse_count = topk // pool_size
    out_len = topk + pool_size - 1
    l_orig = p_len * pool_size + tail_k

    token_idx = torch.full((s_len, out_len), -1, dtype=torch.int32)
    pool_val = torch.full((s_len, sparse_count), _NEG_INF, dtype=torch.float32)
    if s_len == 0:
        return token_idx, pool_val
    # p_len==0(纯尾块批次, 0 池 + pool_tail_k>0)不再短路: 池选择自然为空,
    # 步骤 5 尾块追加照常执行(l_orig = tail_k, seqTailStart = 0)

    scale = 1.0 / math.sqrt(d)
    qk_scale_phys = 1.0  # int8: values 输出补乘的物理缩放(排序键域不含, 见 int8 分支)
    if mx_q_scale_b is not None or mx_k_scale_b is not None:
        # mxFP8: 展开 group scale 到 D 维后反量化(0 元素 batch 跳过, 见逐行版注释)
        if mx_q_scale_b is not None and mx_q_scale_b.numel() > 0:
            mxq = (
                mx_q_scale_b.reshape(mx_q_scale_b.shape[0], -1)
                if mx_q_scale_b.dim() > 2
                else mx_q_scale_b
            )
            mxq = mxq.reshape(mxq.shape[0], -1, d // 32)
            q_b = q_b * mxq.repeat_interleave(32, dim=-1)
        if mx_k_scale_b is not None and mx_k_scale_b.numel() > 0:
            mxk = mx_k_scale_b.reshape(mx_k_scale_b.shape[0], -1)
            k_b = k_b * mxk.repeat_interleave(32, dim=-1)
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        # [键域对齐 QLIV2] mxFP8 排序域 = bf16(QLIV2 golden cal_atten_per_batch_mx:
        # to_be_sort_ele.to(torch.bfloat16); kernel 侧 uint16 bf16 sortable key 同域)
        scores = torch.einsum("sn,snp->sp", w_b, dot.clamp_min(0.0)) * scale
        scores = scores.to(torch.bfloat16).to(torch.float32)
    elif int8_semantics:
        # quant_mode=2 七步逐位口径(与 _compute_batch_loop int8 分支一致, 对齐
        # master 27b8f5eab arch22 键域契约): 排序键域不含 ×qkScale, bf16 RNE
        # round-trip; ×qkScale 物理缩放仅补乘进 values 输出(qk_scale_phys)。
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        t = dot * 2.0**-10
        sh = t.clamp(-65504.0, 65504.0).to(torch.float16).to(torch.float32)
        sh = sh.clamp_min(0.0)
        wq = (w_b.to(torch.float16) * q_scale_b.to(torch.float16)).to(torch.float32)
        scores = torch.einsum("sn,snp->sp", wq, sh)
        if k_scale_b is not None:
            scores = scores * k_scale_b.reshape(1, p_len)
        # [键域对齐 QLIV2·终版] int8 排序域 = fp32 直接稳定排序, 不做 bf16 round-trip
        # (QLIV2 golden cal_atten_per_batch_int8: reduce_sum.clone() 无 bf16;
        #  kernel 侧 arch35/arch22 int8 同为 fp32 域 —— 逐位对齐, 无需桶仲裁)
        qk_scale_phys = scale * 1024.0
    else:
        dot = torch.einsum("snd,pd->snp", q_b, k_b)
        dot = dot.clamp_min(0.0)
        if q_scale_b is not None:
            w_eff = w_b * q_scale_b
        else:
            w_eff = w_b
        scores = torch.einsum("sn,snp->sp", w_eff, dot)
        if k_scale_b is not None:
            scores = scores * k_scale_b.reshape(1, p_len)
        scores = scores * scale
        if q_scale_b is not None:
            # [键域对齐 QLIV2] per-token FP8(mode0) 排序域 = bf16; 非量化不量化键域
            scores = scores.to(torch.bfloat16).to(torch.float32)

    # 步骤 2: 每行可见池数(S,)
    if mask_mode == 3:
        vis_pool = (l_orig - s_len + torch.arange(s_len) + 1) // pool_size
        vis_pool = torch.clamp(vis_pool, 0, p_len)
    else:
        vis_pool = torch.full((s_len,), p_len, dtype=torch.int64)
    sel_vec = torch.minimum(vis_pool, torch.tensor(sparse_count, dtype=torch.int64))

    # 步骤 3: 前缀 TopK —— 无效列置 -inf 后整行 stable 降序排序
    col_ar = torch.arange(p_len)
    prefix_mask = col_ar.unsqueeze(0) < vis_pool.unsqueeze(1)
    masked_scores = torch.where(prefix_mask, scores, torch.full_like(scores, _NEG_INF))
    order = torch.argsort(masked_scores, dim=1, descending=True, stable=True)

    # 步骤 4: 取前 sparse_count 名(池数 < sparse 时按 K=min(sparse,P) 裁剪,
    # 尾部补默认值), 展开 + 行长裁剪 —— 与逐行版 select=min(valid, sparse) 一致
    k_sel = min(sparse_count, p_len)
    top_picks = order[:, :k_sel]  # (S, k_sel)
    pick_valid = torch.arange(k_sel).unsqueeze(0) < sel_vec.unsqueeze(1)
    picked_scores = torch.gather(masked_scores, 1, top_picks)
    picked_scores = (
        picked_scores * qk_scale_phys
    )  # int8: values 补乘物理缩放(-inf 不变)
    pool_val_rows = torch.where(
        pick_valid, picked_scores, torch.full_like(picked_scores, _NEG_INF)
    )
    if k_sel < sparse_count:
        pool_val_rows = torch.cat(
            [
                pool_val_rows,
                torch.full((s_len, sparse_count - k_sel), _NEG_INF),
            ],
            dim=1,
        )
    pool_val = pool_val_rows
    tokens = top_picks.unsqueeze(-1) * pool_size + torch.arange(
        pool_size, dtype=torch.int64
    ).reshape(1, 1, -1)  # (S, k_sel, ps)
    flat_tokens = tokens.reshape(s_len, k_sel * pool_size).to(torch.int32)
    flat_valid = torch.arange(k_sel * pool_size).unsqueeze(0) < (
        sel_vec * pool_size
    ).unsqueeze(1)
    if k_sel * pool_size < sparse_count * pool_size:
        flat_tokens = torch.cat(
            [
                flat_tokens,
                torch.full(
                    (s_len, (sparse_count - k_sel) * pool_size), -1, dtype=torch.int32
                ),
            ],
            dim=1,
        )
        flat_valid = torch.cat(
            [
                flat_valid,
                torch.zeros(
                    (s_len, (sparse_count - k_sel) * pool_size), dtype=torch.bool
                ),
            ],
            dim=1,
        )
    token_idx[:, : sparse_count * pool_size] = torch.where(
        flat_valid, flat_tokens, torch.full_like(flat_tokens, -1)
    )

    # 步骤 5: 尾块追加(基址各行相同, 可见数向量化, 按 tail 槽位逐列 scatter)
    # V1 legacy: 固定偏移 topk 起; V2 compact: 紧跟各行有效展开(expand_total 起)
    if pool_size > 1 and tail_k > 0:
        if mask_mode == 0:
            vis_tail = torch.full((s_len,), tail_k, dtype=torch.int64)
        else:
            gpos = l_orig - s_len + torch.arange(s_len, dtype=torch.int64)
            vis_tail = torch.clamp(gpos - (l_orig - tail_k) + 1, 0, tail_k)
        tail_base = l_orig - tail_k
        if sparse_indices_mode == 0:
            for j in range(pool_size - 1):
                rows = vis_tail > j
                if bool(rows.any()):
                    token_idx[rows, topk + j] = int(tail_base + j)
        else:
            # 每行 expand_total = 有效展开 token 数(flat_valid 前缀计数)
            expand_total = flat_valid.sum(dim=1).to(torch.int64)
            for j in range(pool_size - 1):
                rows = vis_tail > j
                if bool(rows.any()):
                    token_idx[rows, expand_total[rows] + j] = int(tail_base + j)

    return token_idx, pool_val


def _compute_batch(*args, **kwargs):
    """金标单 batch 入口: 默认走向量化实现, ATK_PKI_GOLDEN_LOOP=1 回退逐行版
    (A/B 等价性对账与排查用, 两版语义逐元素一致, 见 pki_golden_vec_equivalence.py)。"""
    import os as _os

    if _os.environ.get("ATK_PKI_GOLDEN_LOOP", "") == "1":
        return _compute_batch_loop(*args, **kwargs)
    return _compute_batch_vec(*args, **kwargs)


def _reference_cpu(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    return_value=False,
    q_descale=None,
    k_descale=None,
    quant_mode=-1,
    sparse_indices_mode=0,
):
    """CPU 金标实现(逐 batch Python 循环, 精确语义, 全布局能力)。"""
    # ---- 布局归一化: 逐 batch 拆分 ----
    q_list, w_list, q_counts, batch, s1_dim = _split_query(
        query, weights, layout_q, actual_seq_q
    )
    k_list = _split_key(pool_key, layout_k, actual_seq_k, block_table)
    if len(k_list) != batch:
        raise ValueError(f"batch mismatch: query {batch} vs pool_key {len(k_list)}")
    tail_list = _as_int_list(pool_tail_k) if pool_tail_k is not None else [0] * batch
    if len(tail_list) != batch:
        raise ValueError(f"pool_tail_k length {len(tail_list)} != batch {batch}")

    # ---- 量化 descale 拆分 ----
    # mode=0: q_descale (…,N1) float, dim==N1+1; k_descale (…,N2) float
    # mode=1: q_descale (…,N1,D/64,2) e8m0; k_descale (…,N2,D/64,2) e8m0
    # dispatch 依据: mode=1 的 descale 是 e8m0(uint8/float8_e8m0fnu)或解码后
    # 尾维为 D/32 的 float —— 由 run_golden 统一解码后传入(此时 shape 尾维
    # 为 D/32, dim 比 mode=0 多一维), 按尾维长度判别(D==128 时 D/32=4)
    q_scale_list = None
    k_scale_list = None
    mx_q_scale_list = None
    mx_k_scale_list = None
    is_mx = quant_mode == 1
    if q_descale is not None:
        if not is_mx:
            q_scale_list = _split_q_descale(q_descale, layout_q, actual_seq_q, q_counts)
        else:
            # mxFP8: 已解码 float, 尾维 [D/32](run_golden 展平 [D/64,2] 后)
            mx_q_scale_list = _split_q_descale_generic(q_descale, layout_q, q_counts)
    if k_descale is not None:
        k_counts = [k.shape[0] for k in k_list]
        block_size = pool_key.shape[1] if layout_k == "PA_BBND" else None
        if not is_mx:
            k_scale_list = _split_k_descale(
                k_descale, layout_k, actual_seq_k, k_counts, block_table, block_size
            )
        else:
            mx_k_scale_list = _split_k_descale_generic(
                k_descale, layout_k, actual_seq_k, k_counts, block_table, block_size
            )

    # ---- 逐 batch 计算 ----
    idx_rows, val_rows = [], []
    for b in range(batch):
        token_idx, pool_val = _compute_batch(
            q_list[b],
            w_list[b],
            k_list[b],
            tail_list[b],
            topk,
            pool_size,
            mask_mode,
            q_scale_list[b] if q_scale_list is not None else None,
            k_scale_list[b] if k_scale_list is not None else None,
            mx_q_scale_list[b] if mx_q_scale_list is not None else None,
            mx_k_scale_list[b] if mx_k_scale_list is not None else None,
            int8_semantics=(quant_mode == 2),
            sparse_indices_mode=sparse_indices_mode,
        )
        idx_rows.append(token_idx)
        val_rows.append(pool_val)

    # ---- 组装输出 ----
    out_len = topk + pool_size - 1
    sparse_count = topk // pool_size
    if layout_q == "BSND":
        indices = torch.full((batch, s1_dim, out_len), -1, dtype=torch.int32)
        values = torch.full(
            (batch, s1_dim, sparse_count), _NEG_INF, dtype=torch.float32
        )
        for b in range(batch):
            c = q_counts[b]
            indices[b, :c] = idx_rows[b]
            values[b, :c] = val_rows[b]
    else:  # TND: 所有行有效, 按序拼接
        indices = (
            torch.cat(idx_rows, dim=0)
            if idx_rows
            else torch.zeros((0, out_len), dtype=torch.int32)
        )
        values = (
            torch.cat(val_rows, dim=0)
            if val_rows
            else torch.zeros((0, sparse_count), dtype=torch.float32)
        )

    if not return_value:
        values = torch.empty(0, dtype=torch.float32)
    return indices, values


# ---------------------------------------------------------------------------
# NPU 组合标杆实现(向量化, 支持不等长 batch)
# ---------------------------------------------------------------------------


def _gather_rows(src, index_2d, fp8_via_fp16=False):
    """按 (B, L) 索引矩阵 gather 首维: src (T, ...) -> (B, L, ...)。

    fp8_via_fp16(性能路径专属, 由 _reference_npu(perf_mode) 传入): torch_npu
    Index 核不支持 fp8 dtype(aclnnIndex 161002, 仅在 NPU 上发生), 以 fp16
    中转(精确无损, 行 gather 语义不变)。精度路径(量化标杆恒 route CPU/
    blocked, fp8 在 CPU 上 Index 核原生支持)恒为 False, 保持原行为。
    """
    if fp8_via_fp16 and src.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        src = src.to(torch.float16)
    flat = src[index_2d.reshape(-1)]
    return flat.reshape(index_2d.shape[0], index_2d.shape[1], *src.shape[1:])


def _normalize_query_npu(
    query, weights, actual_seq_q, layout_q, device, fp8_via_fp16=False
):
    """query/weights 归一化为等长稠密布局。

    返回 (q, w, q_counts_t, row_valid):
      - q: (B, S1m, N1, D); w: (B, S1m, N1) —— TND 时按批 gather 填充,
        BSND 时原样(仅记录行有效性);
      - q_counts_t: (B,) 每 batch 有效 query 行数(int64, device);
      - row_valid: (B, S1m) 有效行掩码(填充行 False)。
    """
    if layout_q == "BSND":
        b, s1m, n1, d = query.shape
        if actual_seq_q is not None:
            counts = _as_int_list(actual_seq_q)  # 非前缀和
            if len(counts) != b:
                raise ValueError(f"actual_seq_q length {len(counts)} != batch {b}")
        else:
            counts = [s1m] * b
        q_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
        row_valid = torch.arange(s1m, device=device)[None, :] < q_counts_t[:, None]
        return query, weights, q_counts_t, row_valid

    # TND: 前缀和 -> 逐批 gather 填充到 (B, S1m)
    counts = _prefix_to_counts(actual_seq_q)
    b = len(counts)
    t1, n1, d = query.shape
    if t1 != sum(counts):
        raise ValueError(f"actual_seq_q 总和({sum(counts)}) != T1({t1})")
    q_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
    s1m = max(counts) if counts else 0
    if s1m == 0:  # 全空 query: 上层直接短路返回空输出
        return (
            query.new_zeros((b, 0, n1, d)),
            weights.new_zeros((b, 0, n1)),
            q_counts_t,
            torch.zeros((b, 0), dtype=torch.bool, device=device),
        )
    row = torch.arange(s1m, device=device)
    starts = q_counts_t.cumsum(0) - q_counts_t
    gather_idx = starts[:, None] + row[None, :]  # (B, S1m)
    row_valid = row[None, :] < q_counts_t[:, None]
    safe_idx = gather_idx.masked_fill(~row_valid, 0).clamp(0, t1 - 1)
    q = _gather_rows(query, safe_idx, fp8_via_fp16)
    w = _gather_rows(weights, safe_idx, fp8_via_fp16)
    return q, w, q_counts_t, row_valid


def _normalize_key_npu(
    pool_key, actual_seq_k, block_table, layout_k, device, fp8_via_fp16=False
):
    """pool_key 归一化为等长稠密布局 (B, Pm, D)。

    返回 (k, k_counts_t, pool_valid):
      - k: (B, Pm, D) —— TND 按批 gather 填充, PA 经 block_table gather,
        BSND 原样(仅记录列有效性);
      - k_counts_t: (B,) 每 batch 有效池数(int64, device);
      - pool_valid: (B, Pm) 有效池掩码。
    """
    if layout_k == "BSND":
        b, s2, n2, d = pool_key.shape
        if actual_seq_k is not None:
            counts = _as_int_list(actual_seq_k)  # 非前缀和
            if len(counts) != b:
                raise ValueError(f"actual_seq_k length {len(counts)} != batch {b}")
        else:
            counts = [s2] * b
        k_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
        pool_valid = torch.arange(s2, device=device)[None, :] < k_counts_t[:, None]
        return pool_key[:, :, 0, :], k_counts_t, pool_valid

    if layout_k == "TND":
        counts = _prefix_to_counts(actual_seq_k)
        b = len(counts)
        t2, n2, d = pool_key.shape
        if t2 != sum(counts):
            raise ValueError(f"actual_seq_k 总和({sum(counts)}) != T2({t2})")
        k_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
        pm = max(counts) if counts else 0
        if pm == 0:  # 全部 batch 0 池
            k = pool_key.new_zeros((b, 0, d))
            return k, k_counts_t, torch.zeros((b, 0), dtype=torch.bool, device=device)
        col = torch.arange(pm, device=device)
        starts = k_counts_t.cumsum(0) - k_counts_t
        gather_idx = starts[:, None] + col[None, :]  # (B, Pm)
        pool_valid = col[None, :] < k_counts_t[:, None]
        safe_idx = gather_idx.masked_fill(~pool_valid, 0).clamp(0, t2 - 1)
        k = _gather_rows(pool_key[:, 0, :], safe_idx, fp8_via_fp16)  # (B, Pm, D)
        return k, k_counts_t, pool_valid

    # PA_BBND: 经 block_table gather 归一化(块内每行 = 1 个池, LIV2 语义)
    counts = _as_int_list(actual_seq_k)  # 非前缀和(每 batch 池数)
    b = len(counts)
    block_num, block_size, n2, d = pool_key.shape
    k_counts_t = torch.tensor(counts, dtype=torch.int64, device=device)
    pm = max(counts) if counts else 0
    if pm == 0 or block_num == 0:  # 全部 batch 0 池
        k = pool_key.new_zeros((b, 0, d))
        return k, k_counts_t, torch.zeros((b, 0), dtype=torch.bool, device=device)
    if block_table is None:
        raise ValueError("layout_k='PA_BBND' requires block_table")
    p_idx = torch.arange(pm, device=device)
    blk_col = (p_idx // block_size).clamp(0, block_table.shape[1] - 1)  # (Pm,)
    # index_select 避免 torch_npu(2.9.0.post2) advanced indexing bug:
    # x[:, idx] 在 NPU 上输出周期性错乱(result[b][j] = x_flat[(b+j)%n]),
    # CPU 正确; index_select/gather 实测正确
    blk = block_table.index_select(1, blk_col).long()  # (B, Pm), 空槽 -1
    pool_valid = p_idx[None, :] < k_counts_t[:, None]
    blk_safe = blk.clamp(min=0)  # 无效槽位指向物理块 0, 数据由掩码丢弃
    rows = blk_safe * block_size + (p_idx % block_size)[None, :]  # (B, Pm)
    # 0 轴非连续 key: reshape 在视图不可用时自动拷贝, 数据语义不变
    flat = pool_key[:, :, 0, :].reshape(block_num * block_size, d)
    if fp8_via_fp16 and flat.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    ):
        # 同 _gather_rows(fp8_via_fp16): 仅性能路径, Index 核不支持 fp8,
        # fp16 中转精确无损; 精度路径 fp8 恒在 CPU, 不走中转
        flat = flat.to(torch.float16)
    k = flat[rows.reshape(-1)].reshape(b, pm, d)
    return k, k_counts_t, pool_valid


def _reference_npu(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    return_value=False,
    sparse_indices_mode=0,
    q_descale=None,
    k_descale=None,
    quant_mode=-1,
    perf_mode=False,
):
    """NPU 组合标杆实现(torch 标准算子向量化, 输入/输出不跨设备)。"""
    device = query.device
    sparse = topk // pool_size
    out_len = topk + pool_size - 1

    # ---- query 归一化: (B, S1m, N1, D) + 行有效性 ----
    q, w, q_counts_t, row_valid = _normalize_query_npu(
        query,
        weights,
        actual_seq_q,
        layout_q,
        device,
        fp8_via_fp16=perf_mode,
    )
    b, s1m, n1, d = q.shape
    if layout_q == "TND" and s1m == 0:  # 全空 query: 直接返回空输出
        indices = torch.zeros((0, out_len), dtype=torch.int32, device=device)
        if return_value:
            values = torch.full(
                (0, sparse), _NEG_INF, dtype=torch.float32, device=device
            )
        else:
            values = torch.empty(0, dtype=torch.float32, device=device)
        return indices, values

    # ---- key 归一化: (B, Pm, D) + 池有效性 ----
    k, k_counts_t, pool_valid = _normalize_key_npu(
        pool_key,
        actual_seq_k,
        block_table,
        layout_k,
        device,
        fp8_via_fp16=perf_mode,
    )
    if k_counts_t.shape[0] != b:
        raise ValueError(f"batch mismatch: query {b} vs pool_key {k_counts_t.shape[0]}")
    pm = k.shape[1]
    tail = pool_tail_k.detach().to(device=device, dtype=torch.int64).reshape(b)
    lens = k_counts_t * pool_size + tail  # 每 batch 原始 token 总数

    # ---- 步骤 1: 池级分数(fp32 通道), 无效池列置 -inf ----
    if pm > 0:
        if (
            perf_mode
            and quant_mode in (0, 1)
            and (q_descale is not None or k_descale is not None)
        ):
            # ---- 性能路径专属分支(performance_device; perf_mode=True 即
            # with_output=False): fp8 标杆 NPU 稠密口径 —— 精度任务(accuracy,
            # with_output=True)的量化标杆恒 route CPU(route_cpu), 本分支对
            # 精度验证不可达; 打分口径与 CPU 金标 _compute_batch 逐模式对齐
            # (mode0 标量 descale / mode1 mxFP8 group 反量化), 等价性验证见
            # review/perf_fp8_fix_validate.py(NPU 标杆 vs CPU 金标 8/8)。
            # descale 归一化:
            #   qds: (B,S1m,N1) [mode0] 或 (B,S1m,N1,D/32) [mode1 mxFP8]
            #   kds: (B,Pm)     [mode0] 或 (B,Pm,D/32)     [mode1 mxFP8]
            #   TND 行 gather / PA 经 block_table gather / BSND 原生; gather 用
            #   安全索引, 无效行/列由 masked_fill 掩码丢弃(填充值不参与排序)
            qds = None
            if q_descale is not None:
                qds = q_descale.to(device=device, dtype=torch.float32)
                if layout_q == "TND" and qds.dim() >= 2:
                    # TND (T1,N1[,G]) -> 按 batch gather/pad 到 (B,S1m,N1[,G])
                    t1 = qds.shape[0]
                    if t1 == s1m and b == 1:
                        qds = qds.unsqueeze(0)  # 均匀 batch=1
                    elif t1 == b * s1m:
                        qds = qds.reshape(b, s1m, *qds.shape[1:])  # 均匀多 batch
                    else:
                        # 不等长 TND: 按 q_counts gather
                        starts_q = q_counts_t.cumsum(0) - q_counts_t
                        col_q = torch.arange(s1m, device=device)
                        gidx_q = (starts_q[:, None] + col_q[None, :]).clamp(0, t1 - 1)
                        mask_q = col_q[None, :] < q_counts_t[:, None]
                        safe_q = gidx_q.masked_fill(~mask_q, 0)
                        qds = torch.stack([qds[safe_q[bi]] for bi in range(b)])
                elif qds.dim() == 2:
                    qds = qds.unsqueeze(0)  # fallback
                if qds.shape[0] == 1 and b > 1:
                    qds = qds.expand(b, *qds.shape[1:])  # broadcast batch
                if qds.dim() == 3 and qds.shape[-1] == 1 and w.shape[-1] > 1:
                    qds = qds.expand(*w.shape)  # N2=1 广播到 N1
            kds = None
            if k_descale is not None:
                kds = k_descale.to(device=device, dtype=torch.float32)
                if layout_k == "TND" and kds.dim() >= 2:
                    # TND (T2,N2[,G]) -> 按 batch gather/pad 到 (B,Pm[,G])
                    t2 = kds.shape[0]
                    if t2 == pm and b == 1:
                        kds = kds.reshape(1, pm, *kds.shape[2:])
                    elif t2 == b * pm:
                        kds = kds.reshape(b, pm, *kds.shape[2:])
                    else:
                        starts_k = k_counts_t.cumsum(0) - k_counts_t
                        col_k = torch.arange(pm, device=device)
                        gidx_k = (starts_k[:, None] + col_k[None, :]).clamp(0, t2 - 1)
                        mask_k = col_k[None, :] < k_counts_t[:, None]
                        safe_k = gidx_k.masked_fill(~mask_k, 0)
                        kds_flat = kds[:, 0] if kds.shape[1] == 1 else kds.mean(dim=1)
                        kds = torch.stack([kds_flat[safe_k[bi]] for bi in range(b)])
                elif layout_k == "PA_BBND" and kds.dim() >= 3:
                    # PA (blockNum, blockSize, N2[,G]) -> 经 block_table gather
                    bn, bs = kds.shape[0], kds.shape[1]
                    kds_flat = kds[:, :, 0].reshape(bn * bs, *kds.shape[3:])
                    # 重算 rows(与 _normalize_key_npu PA 分支同逻辑)
                    p_idx2 = torch.arange(pm, device=device)
                    blk_col2 = (p_idx2 // bs).clamp(0, block_table.shape[1] - 1)
                    blk2 = block_table.index_select(1, blk_col2).long()
                    blk_safe2 = blk2.clamp(min=0)
                    rows2 = blk_safe2 * bs + (p_idx2 % bs)[None, :]
                    kds = torch.stack([kds_flat[rows2[bi]] for bi in range(b)])
                elif kds.dim() in (3, 4):
                    # BSND (B,S2,N2[,G]): N2=1 时取 [:, :, 0] -> (B,Pm[,G])
                    kds = kds[:, :, 0] if kds.shape[2] == 1 else kds.mean(dim=2)
                if kds.dim() == 1:
                    kds = kds.reshape(1, -1)
                if kds.shape[0] == 1 and b > 1:
                    kds = kds.expand(b, *kds.shape[1:])
            if quant_mode == 1:
                # mxFP8(与 CPU 金标 mx 分支同口径): q/k 按 per-32-D group scale
                # 整体反量化 -> 点积 -> relu -> w 聚合 -> ×1/√d -> bf16 RNE
                q32 = q.float()
                if qds is not None:
                    if qds.dim() == 3:
                        qds = qds.unsqueeze(-1)  # (B,S1m,N1) -> (B,S1m,N1,1)
                    q32 = q32 * qds.repeat_interleave(32, dim=-1)
                k32 = k.float()
                if kds is not None:
                    if kds.dim() == 2:
                        kds = kds.unsqueeze(-1)  # (B,Pm) -> (B,Pm,1)
                    k32 = k32 * kds.repeat_interleave(32, dim=-1)
                dot = torch.matmul(
                    q32, k32.transpose(-1, -2).unsqueeze(1)
                )  # (B, S1m, N1, Pm)
                scores = torch.matmul(
                    w.float().unsqueeze(2), dot.clamp_min(0.0)
                ).squeeze(2)
                scores = (scores * (d**-0.5)).to(torch.bfloat16).to(torch.float32)
            else:
                # mode 0(fp8 标量 descale; 与 CPU 金标同口径): relu 点积 ->
                # w×qscale 聚合 -> ×kscale -> ×1/√d -> bf16 RNE round-trip
                dot = torch.matmul(
                    q.float(), k.transpose(-1, -2).unsqueeze(1).float()
                ).clamp_min(0.0)  # (B, S1m, N1, Pm)
                w_eff = w.float() * qds if qds is not None else w.float()
                scores = torch.matmul(w_eff.unsqueeze(2), dot).squeeze(2)
                if kds is not None:
                    scores = scores * kds.unsqueeze(1)  # (B,1,Pm)
                if qds is not None:
                    scores = (scores * (d**-0.5)).to(torch.bfloat16).to(torch.float32)
                else:
                    scores = scores * (d**-0.5)
        elif q_descale is not None:
            # 量化路径(int8): dot/1024 -> relu -> fp16 round-trip -> wq 预乘
            # 加权 -> ×k_descale; 排序域不含 ×(1/√d)(对齐算子 fp32 排序)
            dot_raw = torch.matmul(
                q.float(), k.transpose(-1, -2).unsqueeze(1).float()
            )  # (B, S1m, N1, Pm)
            t = (dot_raw * 0.0009765625).clamp_min(0.0)  # /1024 + relu
            t = t.to(torch.float16).to(torch.float32)  # fp16 round-trip
            # q_descale 归一化到 (B, S1m, N1) 匹配 w
            qds = q_descale.to(device=device, dtype=torch.float32)
            if layout_q == "TND" and qds.dim() == 2:
                # TND (T1,N1) -> 按 batch gather/pad 到 (B,S1m,N1)
                t1, nn1 = qds.shape
                if t1 == s1m and b == 1:
                    qds = qds.unsqueeze(0)  # 均匀 batch=1: (1,S1m,N1)
                elif t1 == b * s1m:
                    qds = qds.reshape(b, s1m, nn1)  # 均匀多 batch
                else:
                    # 不等长 TND: 按 q_counts gather
                    starts_q = q_counts_t.cumsum(0) - q_counts_t
                    col_q = torch.arange(s1m, device=device)
                    gidx_q = (starts_q[:, None] + col_q[None, :]).clamp(0, t1 - 1)
                    mask_q = col_q[None, :] < q_counts_t[:, None]
                    safe_q = gidx_q.masked_fill(~mask_q, 0)
                    qds_flat = qds  # (T1, N1)
                    qds = torch.stack(
                        [qds_flat[safe_q[bi]] for bi in range(b)]
                    )  # (B, S1m, N1)
            elif qds.dim() == 2:
                qds = qds.unsqueeze(0)  # fallback
            if qds.shape[0] == 1 and b > 1:
                qds = qds.expand(b, *qds.shape[1:])  # broadcast batch
            if qds.shape[-1] == 1 and w.shape[-1] > 1:
                qds = qds.expand(*w.shape)  # N2=1 广播到 N1
            wq = (w.to(torch.float16) * qds.to(torch.float16)).to(torch.float32)
            scores = torch.matmul(wq.unsqueeze(2).float(), t).squeeze(2)
            # k_descale 归一化到 (B, Pm) 后广播为 (B, 1, Pm)
            if k_descale is not None:
                kds = k_descale.to(device=device, dtype=torch.float32)
                if layout_k == "TND" and kds.dim() == 2:
                    # TND (T2,N2) -> 按 batch gather/pad 到 (B,Pm)
                    t2, _ = kds.shape
                    if t2 == pm and b == 1:
                        kds = kds.reshape(1, pm)
                    elif t2 == b * pm:
                        kds = kds.reshape(b, pm)
                    else:
                        starts_k = k_counts_t.cumsum(0) - k_counts_t
                        col_k = torch.arange(pm, device=device)
                        gidx_k = (starts_k[:, None] + col_k[None, :]).clamp(0, t2 - 1)
                        mask_k = col_k[None, :] < k_counts_t[:, None]
                        safe_k = gidx_k.masked_fill(~mask_k, 0)
                        kds_flat = kds[:, 0] if kds.shape[1] == 1 else kds.mean(dim=1)
                        kds = torch.stack(
                            [kds_flat[safe_k[bi]] for bi in range(b)]
                        )  # (B, Pm)
                elif layout_k == "PA_BBND" and kds.dim() == 3:
                    # PA (blockNum, blockSize, N2) -> 经 block_table gather 到 (B,Pm)
                    bn, bs, _ = kds.shape
                    kds_flat = kds[:, :, 0].reshape(bn * bs)  # (blockNum*blockSize,)
                    # 复用 _normalize_key_npu 已算好的 rows
                    # rows: (B, Pm) -> kds: (B, Pm)
                    if "rows" in dir():
                        kds = torch.stack([kds_flat[rows[bi]] for bi in range(b)])
                    else:
                        # 重算 rows(与 _normalize_key_npu PA 分支同逻辑)
                        block_num, block_size = bn, bs
                        p_idx2 = torch.arange(pm, device=device)
                        blk_col2 = (p_idx2 // block_size).clamp(
                            0, block_table.shape[1] - 1
                        )
                        blk2 = block_table.index_select(1, blk_col2).long()
                        blk_safe2 = blk2.clamp(min=0)
                        rows2 = blk_safe2 * block_size + (p_idx2 % block_size)[None, :]
                        kds = torch.stack(
                            [kds_flat[rows2[bi]] for bi in range(b)]
                        )  # (B, Pm)
                elif kds.dim() == 3:
                    # BSND (B,S2,N2): N2=1 时取 [:, :, 0] -> (B, Pm)
                    kds = kds[:, :, 0] if kds.shape[2] == 1 else kds.mean(dim=2)
                if kds.dim() == 1:
                    kds = kds.reshape(1, -1)
                if kds.shape[0] == 1 and b > 1:
                    kds = kds.expand(b, pm)
                scores = scores * kds.unsqueeze(1)  # (B,1,Pm)
        else:
            dot = torch.relu(
                torch.matmul(q.float(), k.transpose(-1, -2).unsqueeze(1).float())
            )  # (B, S1m, N1, Pm)
            scores = torch.matmul(w.unsqueeze(2).float(), dot).squeeze(2) * (
                d**-0.5
            )  # (B, S1m, Pm)
        scores = scores.masked_fill(~pool_valid.unsqueeze(1), _NEG_INF)

        # ---- 步骤 2: 因果可见池数(右对齐因果; 上界由池有效性掩码覆盖) ----
        if mask_mode == 3:
            row = torch.arange(s1m, device=device)
            valid = torch.div(
                lens[:, None] - q_counts_t[:, None] + row[None, :] + 1,
                pool_size,
                rounding_mode="floor",
            ).clamp(min=0)  # (B, S1m)
            col = torch.arange(pm, device=device)
            scores = scores.masked_fill(
                col[None, None, :] >= valid[:, :, None], _NEG_INF
            )

        # ---- 步骤 3: 池级 TopK(分数降序) ----
        k_sel = min(sparse, pm)
        vals, pool_idx = _topk_desc(scores, k_sel)  # (B, S1m, k_sel)
    else:  # 全部 batch 0 池: 无池可选
        k_sel = 0
        vals = torch.full((b, s1m, 0), _NEG_INF, dtype=torch.float32, device=device)
        pool_idx = torch.zeros((b, s1m, 0), dtype=torch.int64, device=device)

    # ---- 步骤 4: 选中池展开为 token 索引, 无效位置 -1 ----
    if pool_size > 1:
        off = torch.arange(pool_size, device=device)
        expanded = (pool_idx.unsqueeze(-1) * pool_size + off).reshape(
            b, s1m, k_sel * pool_size
        )
        fin = (
            torch.isfinite(vals)
            .unsqueeze(-1)
            .expand(b, s1m, k_sel, pool_size)
            .reshape(b, s1m, k_sel * pool_size)
        )
    else:
        expanded = pool_idx
        fin = torch.isfinite(vals)
    token_idx = torch.where(fin, expanded, -1)  # (B, S1m, k_sel*ps)
    if token_idx.shape[-1] < out_len:
        token_idx = torch.nn.functional.pad(
            token_idx, (0, out_len - token_idx.shape[-1]), value=-1
        )

    # ---- 步骤 5: 尾块追加(输出 topk 位置起, 容量 pool_size - 1) ----
    if pool_size > 1:
        t = torch.arange(pool_size - 1, device=device)
        if mask_mode == 3:
            row = torch.arange(s1m, device=device)
            gpos = lens[:, None] - q_counts_t[:, None] + row[None, :]
            # 尾池真实起点 lens-tail 的因果截断(修复后 kernel 口径)
            vis = torch.clamp(
                torch.minimum(tail[:, None], gpos - (lens - tail)[:, None] + 1),
                min=0,
            )
        else:
            vis = tail[:, None].expand(b, s1m)
        base = (lens - tail)[:, None, None] + t[None, None, :]
        tail_blk = torch.where(t[None, None, :] < vis[:, :, None], base, -1)
        # 0 池 batch 的尾块即全部有效 token(lens = tail, seqTailStart = 0),
        # 与 _reference_cpu 纯尾块语义一致, 不清零
        if sparse_indices_mode == 1:
            # V2 compact: 尾 token 紧跟各行有效展开(expand_total 起), -1 全在行尾
            expand_total = fin.sum(dim=-1)  # (b, s1m) 有效展开 token 数(前缀)
            col = torch.arange(out_len, device=device)
            in_tail = (col >= expand_total[..., None]) & (
                col < (expand_total + vis)[..., None]
            )
            # k_counts_t/lens-tail 为 (B,) 1-D, 须升维 (B,1,1) 广播至 (B,S1m,outLen)
            in_tail = in_tail & (k_counts_t != 0)[:, None, None]
            tail_val = (lens - tail)[:, None, None] + (col - expand_total[..., None])
            token_idx = torch.where(in_tail, tail_val.to(token_idx.dtype), token_idx)
        else:
            token_idx = torch.cat(
                [token_idx[..., :topk], tail_blk.to(token_idx.dtype)], dim=-1
            )

    # ---- 步骤 6: 组装输出(dtype: int32 / float32) ----
    if layout_q == "BSND" and actual_seq_q is not None:
        # BSND + actual_seq_q: 有效长度外的行填 -1 / -inf
        token_idx = token_idx.masked_fill(~row_valid.unsqueeze(-1), -1)
        if return_value:
            vals = vals.masked_fill(~row_valid.unsqueeze(-1), _NEG_INF)

    if return_value:
        values = vals.to(torch.float32)
        if k_sel < sparse:
            values = torch.nn.functional.pad(
                values, (0, sparse - k_sel), value=_NEG_INF
            )
    else:
        values = torch.empty(0, dtype=torch.float32, device=device)

    if layout_q == "TND":
        # 反填充: 按行有效性取回全局行序(批内行序 = 前缀和顺序)
        token_idx = token_idx[row_valid]
        if return_value:
            values = values[row_valid]
    return token_idx.to(torch.int32), values


def _reference_npu_blocked(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    return_value=False,
    q_descale=None,
    k_descale=None,
    quant_mode=-1,
    sparse_indices_mode=0,
):
    """NPU 组合标杆的分块 streaming 实现(与 _reference_npu 数值语义一致)。

    _reference_npu 一次性物化 (B, S1m, Pm) 稠密 scores —— 极端 shape
    (ps=1 + 池数 11~17 万)下单矩阵 50~500GB, NPU HBM 放不下; 中大 shape
    虽能放下, 但 worker 长跑后 caching allocator 碎片化, 数 GB 级分配
    逐步劣化直至 aicpu 看门狗超时(进程级不可捕获, 级联饿死后续用例)。

    本实现峰值内存 = S1m*(N1 逐头中间)*blockW + S1m*k_sel*2:
      - 逐 batch 处理, pool 维按 blockW 分块;
      - N1 头维拆为逐头循环(w 可负, 不能提前乘入 q —— relu 作用在
        dot 之后), 每头仅物化 (S1m, blockW) fp32 中间;
      - 块内 topk + 与累积缓冲 cat-topk 归并(增量 streaming topk);
      - TND 反填充用显式行号 index_select(int 索引走 gather), 避免
        布尔索引触发 torch_npu AdvancedIndex/NonZero 的 AICPU 超时。

    量化 descale 与 _compute_batch 数值口径一致:
      - mode=0: q_scale 乘入 weight(per-token-head)、k_scale 乘入
        聚合和(每池标量), scale 恒正可从点积提出, 数学等价于先反量化;
      - mode=1(mxFP8): q/k 先按 per-32-D-group scale 反量化再点积。
      scale 拆分复用稠密版 _split_*_descale 函数(逻辑池序与
      _normalize_key_npu 的 block_table gather 序一致), 拆出结果只
      覆盖有效行/池, 以 1 填充到 s1m/pm 对齐 densify 形状(填充行由
      掩码/丢弃逻辑处理, 数值无影响)。

    blockW 下限 2*k_sel(保证归并信息不丢失), 上限 16384(控制 sort 规模)。
    """
    device = query.device
    sparse = topk // pool_size
    out_len = topk + pool_size - 1

    q, w, q_counts_t, row_valid = _normalize_query_npu(
        query, weights, actual_seq_q, layout_q, device
    )
    b, s1m, n1, d = q.shape
    if layout_q == "TND" and s1m == 0:  # 全空 query: 同 _reference_npu 短路
        indices = torch.zeros((0, out_len), dtype=torch.int32, device=device)
        if return_value:
            values = torch.full(
                (0, sparse), _NEG_INF, dtype=torch.float32, device=device
            )
        else:
            values = torch.empty(0, dtype=torch.float32, device=device)
        return indices, values

    k, k_counts_t, pool_valid = _normalize_key_npu(
        pool_key, actual_seq_k, block_table, layout_k, device
    )
    pm = k.shape[1]
    tail = pool_tail_k.detach().to(device=device, dtype=torch.int64).reshape(b)
    lens = k_counts_t * pool_size + tail
    scale = float(d) ** -0.5
    k_sel = min(sparse, pm)

    block_w = min(16384, max(2 * k_sel if k_sel > 0 else 4096, 4096))
    col_ar = torch.arange(pm, device=device)

    # 量化 descale 拆分(复用稠密版 _split_*_descale: 逻辑池序与
    # _normalize_key_npu 的 block_table gather 序一致; q_counts/k_counts
    # 均为逐 batch 有效数, 与稠密版口径相同)
    q_scale_list = k_scale_list = mx_q_scale_list = mx_k_scale_list = None
    if q_descale is not None or k_descale is not None:
        q_counts_l = [int(v) for v in q_counts_t.tolist()]
        k_counts_l = [int(v) for v in k_counts_t.tolist()]
        ds_block_size = pool_key.shape[1] if layout_k == "PA_BBND" else None
        is_mx = quant_mode == 1
        if q_descale is not None:
            if not is_mx:
                q_scale_list = _split_q_descale(
                    q_descale, layout_q, actual_seq_q, q_counts_l
                )
            else:
                mx_q_scale_list = _split_q_descale_generic(
                    q_descale, layout_q, q_counts_l
                )
        if k_descale is not None:
            if not is_mx:
                k_scale_list = _split_k_descale(
                    k_descale,
                    layout_k,
                    actual_seq_k,
                    k_counts_l,
                    block_table,
                    ds_block_size,
                )
            else:
                mx_k_scale_list = _split_k_descale_generic(
                    k_descale,
                    layout_k,
                    actual_seq_k,
                    k_counts_l,
                    block_table,
                    ds_block_size,
                )
        q_scale_list = (
            [s.to(device) for s in q_scale_list] if q_scale_list is not None else None
        )
        k_scale_list = (
            [s.to(device) for s in k_scale_list] if k_scale_list is not None else None
        )
        mx_q_scale_list = (
            [s.to(device) for s in mx_q_scale_list]
            if mx_q_scale_list is not None
            else None
        )
        mx_k_scale_list = (
            [s.to(device) for s in mx_k_scale_list]
            if mx_k_scale_list is not None
            else None
        )

    idx_list, val_list = [], []
    for bi in range(b):
        cnt = int(q_counts_t[bi])
        n_pools = int(k_counts_t[bi])
        if s1m == 0 or pm == 0 or k_sel == 0:
            # 无任何池的形态(整场 pm==0 / k_sel==0): 全 -1 短路
            idx_list.append(
                torch.full((s1m, out_len), -1, dtype=torch.int64, device=device)
            )
            val_list.append(
                torch.full((s1m, sparse), _NEG_INF, dtype=torch.float32, device=device)
            )
            continue

        q_b = q[bi].float()  # (S1m, N1, D)
        w_b = w[bi].float()  # (S1m, N1)
        k_b = k[bi].float()  # (Pm, D)
        pv = pool_valid[bi]  # (Pm,)
        # int8(mode=2) 七步逐位契约(与 _compute_batch int8 分支同口径, 对齐
        # master 27b8f5eab arch22 键域契约): 排序键域不含 ×qkScale,
        # wq 在 FP16 域预乘, 块内分数 bf16 RNE round-trip, values 补乘物理缩放
        is_int8 = quant_mode == 2
        wq_b = None
        # 量化 q 侧: mode=0 scale 乘入 weight; mode=1 整体反量化 q。
        # scale 只覆盖有效行, 以 1 填充到 s1m(填充行由后续掩码/丢弃处理)
        if q_scale_list is not None:
            qs = q_scale_list[bi]  # (Sb, N1)
            if qs.shape[0] < s1m:
                qs = torch.cat(
                    [
                        qs,
                        torch.ones(
                            (s1m - qs.shape[0], n1), dtype=qs.dtype, device=device
                        ),
                    ],
                    dim=0,
                )
            if is_int8:
                # wq FP16 域预乘(与稠密 int8 分支一致)
                wq_b = (w[bi].to(torch.float16) * qs.to(torch.float16)).to(
                    torch.float32
                )
            w_b = w_b * qs
        if mx_q_scale_list is not None:
            mq = mx_q_scale_list[bi]  # (Sb, N1, D/32) 或 (Sb, D/32)
            if mq.dim() == 2:
                mq = mq.unsqueeze(1)
            if mq.shape[0] < s1m:
                mq = torch.cat(
                    [
                        mq,
                        torch.ones(
                            (s1m - mq.shape[0],) + tuple(mq.shape[1:]),
                            dtype=mq.dtype,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            q_b = q_b * mq.repeat_interleave(32, dim=-1)
        # 量化 k 侧 scale: 同样以 1 填充到 pm, 块内切片应用
        ks = None
        if k_scale_list is not None:
            ks = k_scale_list[bi]  # (Pb,)
            if ks.shape[0] < pm:
                ks = torch.cat(
                    [ks, torch.ones(pm - ks.shape[0], dtype=ks.dtype, device=device)]
                )
        mk = None
        if mx_k_scale_list is not None:
            _mk = mx_k_scale_list[bi]
            # 0 池 batch: 0 元素 reshape(x, -1) 含糊抛错, 空张量直接给 (0, D/32) 形状
            mk = (
                _mk.reshape(_mk.shape[0], d // 32)
                if _mk.numel() > 0
                else _mk.reshape(0, d // 32)
            )  # (Pb, D/32)
            if mk.shape[0] < pm:
                mk = torch.cat(
                    [
                        mk,
                        torch.ones(
                            (pm - mk.shape[0], d // 32), dtype=mk.dtype, device=device
                        ),
                    ]
                )
        if mask_mode == 3:
            row_ar = torch.arange(s1m, device=device)
            valid_row = torch.div(
                int(lens[bi]) - cnt + row_ar + 1, pool_size, rounding_mode="floor"
            ).clamp(min=0)  # (S1m,)
        else:
            valid_row = None

        cum_v = torch.full((s1m, k_sel), _NEG_INF, dtype=torch.float32, device=device)
        cum_i = torch.full((s1m, k_sel), -1, dtype=torch.int64, device=device)
        # 单块(pm <= block_w)时 cat-argsort 归并是纯冗余: 累积缓冲全 -inf,
        # 白排一遍 2*k_sel 宽(case 633 类 8192x16384 argsort 即来自此处,
        # 属设备上最大的单 sort 调用), 直接采用本块 topk 结果
        single_block = pm <= block_w
        for lo in range(0, pm, block_w):
            hi = min(lo + block_w, pm)
            bw = hi - lo
            kb = k_b[lo:hi]  # (bw, D)
            if mk is not None:
                # mode=1: k 按 per-32-D-group scale 反量化(块内切片,
                # 与 _compute_batch 的 repeat_interleave 口径一致)
                kb = kb * mk[lo:hi].repeat_interleave(32, dim=-1)
            s_blk = torch.zeros((s1m, bw), dtype=torch.float32, device=device)
            for n in range(n1):
                if is_int8:
                    # int8 七步: dot×2^-10 -> FP16 RNE(±65504 clip) -> ReLU
                    # -> wq FP16 域预乘加权聚合(排序键域, 不含 ×qkScale)
                    dot_n = q_b[:, n] @ kb.transpose(0, 1)
                    t_n = (
                        (dot_n * 2.0**-10)
                        .clamp(-65504.0, 65504.0)
                        .to(torch.float16)
                        .to(torch.float32)
                        .clamp_min(0.0)
                    )
                    s_blk += wq_b[:, n : n + 1] * t_n
                else:
                    dot_n = torch.relu(q_b[:, n] @ kb.transpose(0, 1))  # (S1m, bw)
                    s_blk += w_b[:, n : n + 1] * dot_n
            if is_int8:
                # [键域对齐 QLIV2] int8 排序域 = fp32 直接稳定排序(无 bf16)
                if ks is not None:
                    s_blk *= ks[lo:hi]
            else:
                s_blk *= scale
                if ks is not None:
                    # mode=0: k_scale 乘入聚合和(每池标量, 块内切片)
                    s_blk *= ks[lo:hi]
                if not is_int8:
                    # [键域对齐 QLIV2] FP8 家族(mode0/1)排序域 = bf16(与稠密金标
                    # _compute_batch/_compute_batch_vec 及 kernel uint16 键同域)
                    s_blk = s_blk.to(torch.bfloat16).to(torch.float32)
            # 布尔 AdvancedIndex setitem 在 torch_npu 上走 NonZero/动态索引
            # 路径, 大 shape 下有 AICPU 超时/vector core 异常前科(见
            # _topk_desc 注释), 统一改 masked_fill_(与 _reference_npu 稠密
            # 路径同款安全算子)
            s_blk.masked_fill_(~pv[None, lo:hi], _NEG_INF)
            if valid_row is not None:
                s_blk.masked_fill_(col_ar[None, lo:hi] >= valid_row[:, None], _NEG_INF)

            kk = min(k_sel, bw)
            v_blk, i_blk = _topk_desc(s_blk, kk)  # (S1m, kk)
            i_blk = i_blk + lo
            if kk < k_sel:
                pad_v = torch.full(
                    (s1m, k_sel - kk), _NEG_INF, dtype=torch.float32, device=device
                )
                pad_i = torch.full(
                    (s1m, k_sel - kk), -1, dtype=torch.int64, device=device
                )
                v_blk = torch.cat([v_blk, pad_v], dim=-1)
                i_blk = torch.cat([i_blk, pad_i], dim=-1)
            if single_block:
                cum_v, cum_i = v_blk, i_blk
                continue
            # 增量归并: cat 累积缓冲 + 本块 topk, 再取 k_sel。
            # CPU 侧 stable 归并(同分按下标升序, 金标约定): cum 来自更早的
            # 块(池下标更小)且内部已按约定有序, v_blk 同块内也已稳定有序,
            # stable sort 保持 [cum, v_blk] 的原始相对顺序 => 并列分数全局
            # 按池下标升序, 与 _reference_cpu/_topk_desc(CPU stable) 一致;
            # NPU 侧 torch_npu sort 不保证 stable 可用, 保持非稳定(仅 bm
            # 参考指标路径, 不影响 op-vs-truth 精度门禁)
            cat_v = torch.cat([cum_v, v_blk], dim=-1)
            cat_i = torch.cat([cum_i, i_blk], dim=-1)
            if cat_v.device.type == "cpu":
                order = torch.argsort(cat_v, dim=-1, descending=True, stable=True)[
                    ..., :k_sel
                ]
            else:
                order = torch.argsort(cat_v, dim=-1, descending=True)[..., :k_sel]
            cum_v = cat_v.gather(-1, order)
            cum_i = cat_i.gather(-1, order)

        # ---- 展开为 token 索引(同 _reference_npu 语义) ----
        if pool_size > 1:
            off = torch.arange(pool_size, device=device)
            expanded = (cum_i.unsqueeze(-1) * pool_size + off).reshape(
                s1m, k_sel * pool_size
            )
            fin = (
                torch.isfinite(cum_v)
                .unsqueeze(-1)
                .expand(s1m, k_sel, pool_size)
                .reshape(s1m, k_sel * pool_size)
            )
        else:
            expanded = cum_i
            fin = torch.isfinite(cum_v)
        token_b = torch.where(fin, expanded, -1)  # (S1m, k_sel*ps)
        if token_b.shape[-1] < out_len:
            token_b = torch.nn.functional.pad(
                token_b, (0, out_len - token_b.shape[-1]), value=-1
            )
        elif token_b.shape[-1] > out_len:
            token_b = token_b[..., :out_len]

        # ---- 尾块追加 ----
        if pool_size > 1:
            t = torch.arange(pool_size - 1, device=device)
            if mask_mode == 3:
                row_ar = torch.arange(s1m, device=device)
                gpos = int(lens[bi]) - cnt + row_ar
                # 尾池真实起点 lens-tail 的因果截断(修复后 kernel 口径)
                vis = torch.clamp(
                    torch.minimum(
                        torch.full_like(gpos, int(tail[bi])),
                        gpos - (int(lens[bi]) - int(tail[bi])) + 1,
                    ),
                    min=0,
                )
            else:
                vis = torch.full(
                    (s1m,), int(tail[bi]), dtype=torch.int64, device=device
                )
            base = (int(lens[bi]) - int(tail[bi])) + t
            tail_blk = torch.where(t[None, :] < vis[:, None], base[None, :], -1)
            if sparse_indices_mode == 1:
                # V2 compact: 尾 token 紧跟各行有效展开, -1 全在行尾
                expand_total = fin.sum(dim=-1)  # (S1m,) 有效展开 token 数(前缀)
                col = torch.arange(out_len, device=device)
                in_tail = (col >= expand_total[:, None]) & (
                    col < (expand_total + vis)[:, None]
                )
                tail_val = (int(lens[bi]) - int(tail[bi])) + (
                    col - expand_total[:, None]
                )
                token_b = torch.where(in_tail, tail_val.to(token_b.dtype), token_b)
            else:
                token_b = torch.cat(
                    [token_b[..., :topk], tail_blk.to(token_b.dtype)], dim=-1
                )

        vals_b = cum_v
        if is_int8:
            # int8: values 输出补乘物理缩放 ×qkScale = scale×2^10(-inf 不变)
            vals_b = vals_b * (scale * 1024.0)
        if k_sel < sparse:
            vals_b = torch.nn.functional.pad(
                vals_b, (0, sparse - k_sel), value=_NEG_INF
            )
        idx_list.append(token_b)
        val_list.append(vals_b)

    token_idx = torch.stack(idx_list, dim=0)  # (B, S1m, out_len)
    values = torch.stack(val_list, dim=0)  # (B, S1m, sparse)

    if layout_q == "BSND" and actual_seq_q is not None:
        token_idx = token_idx.masked_fill(~row_valid.unsqueeze(-1), -1)
        if return_value:
            values = values.masked_fill(~row_valid.unsqueeze(-1), _NEG_INF)

    if layout_q == "TND":
        # 反填充: 显式有效行号 index_select(int 索引走 gather,
        # 避免布尔索引触发 AdvancedIndex/NonZero 的 AICPU 超时)。
        # batch b 局部行 j -> 展平索引 b*S1m + j(有效 j < counts[b])
        rows_per_batch = q_counts_t.tolist()
        sel_parts = [
            torch.arange(bi * s1m, bi * s1m + int(c), device=device)
            for bi, c in enumerate(rows_per_batch)
        ]
        sel = (
            torch.cat(sel_parts)
            if sel_parts
            else torch.zeros(0, dtype=torch.int64, device=device)
        )
        flat_idx = token_idx.reshape(b * s1m, out_len)
        token_idx = flat_idx.index_select(0, sel)
        if return_value:
            flat_val = values.reshape(b * s1m, sparse)
            values = flat_val.index_select(0, sel)

    if return_value:
        values = values.to(torch.float32)
    else:
        values = torch.empty(0, dtype=torch.float32, device=device)
    return token_idx.to(torch.int32), values


# ---------------------------------------------------------------------------
# 对外统一入口
# ---------------------------------------------------------------------------


def pool_key_indexer_reference(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    return_value=False,
    q_descale=None,
    k_descale=None,
    quant_mode=-1,
    sparse_indices_mode=0,
    perf_mode=False,
):
    """PoolKeyIndexer 统一标杆(按输入设备自动选择 CPU/NPU 实现)。

    入参说明:
      query: Query 张量, dtype float16/bfloat16/float32(内部统一转 float32 计算)。
        - layout_q="BSND": shape (B, S1, N1, D);
        - layout_q="TND":  shape (T1, N1, D), T1 为所有 batch 的 query token 总和。
      pool_key: 池级 Key 张量(每行一个 pool 的 key), dtype 与 query 一致:
        - layout_k="BSND":    shape (B, S2, N2, D), S2 为每 batch 池数;
        - layout_k="TND":     shape (T2, N2, D), T2 为所有 batch 池数总和;
        - layout_k="PA_BBND": shape (blockNum, blockSize, N2, D), 块内每行一个池,
          需同时传入 block_table 与 actual_seq_k; 支持 0 轴非连续(strided view)。
      weights: 多头权重, shape (B, S1, N1)(BSND) 或 (T1, N1)(TND)。
      pool_tail_k: 每 batch 尾部不完整 pool 的有效 token 数, shape (B,), int64,
        取值范围 [0, pool_size-1]。
      actual_seq_q: 可选, 每 batch query 有效 token 数:
        - layout_q="TND":  必传, 前缀和形式(长度 B, 末项等于 T1);
        - layout_q="BSND": 可传, 每 batch 有效数(非前缀和); None 表示全部有效。
      actual_seq_k: 可选, 每 batch 有效池数:
        - layout_k="TND":     必传, 前缀和形式(末项等于 T2);
        - layout_k="PA_BBND": 必传, 每 batch 池数(非前缀和);
        - layout_k="BSND":    可传, 每 batch 池数(非前缀和); None 表示全部有效。
      block_table: layout_k="PA_BBND" 时必传, shape (B, maxBlockNumPerSeq), int32,
        元素为物理块编号, 无效槽位 -1。
      layout_q: query 布局, "BSND" 或 "TND"。
      layout_k: pool_key 布局, "BSND"、"TND" 或 "PA_BBND"。
      topk: 展开后保留的 token 数, 需满足 topk % pool_size == 0。
      pool_size: 每个 pool 包含的 token 数, 取值 [1, 128]。
      mask_mode: 0 = defaultMask(无掩码); 3 = rightDownCausal(右对齐因果)。
      return_value: True 时输出 values; False 时 values 为空 tensor。

    返回值:
      (indices, values):
        indices: int32; BSND 时 (B, S1, topk+pool_size-1), TND 时
          (T1, topk+pool_size-1); 无效位置填 -1。
        values: float32; return_value=True 时 BSND 为 (B, S1, topk//pool_size),
          TND 为 (T1, topk//pool_size), 无效位置填 -inf; return_value=False 时
          为空 tensor (0,)。

    设备选择:
      - 输入 tensor 在 CPU -> CPU 金标实现(精度基准);
      - 输入 tensor 在 NPU -> NPU 组合标杆实现(性能基线);
      - 输出与输入同设备。两种实现接口能力与计算语义一致(见模块 docstring)。
    """
    _validate_args(query, pool_key, layout_q, layout_k, topk, pool_size, mask_mode)
    if query.device.type == "cpu":
        return _reference_cpu(
            query,
            pool_key,
            weights,
            pool_tail_k,
            actual_seq_q=actual_seq_q,
            actual_seq_k=actual_seq_k,
            block_table=block_table,
            layout_q=layout_q,
            layout_k=layout_k,
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            return_value=return_value,
            q_descale=q_descale,
            k_descale=k_descale,
            quant_mode=quant_mode,
            sparse_indices_mode=sparse_indices_mode,
        )
    return _reference_npu(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
        sparse_indices_mode=sparse_indices_mode,
        perf_mode=perf_mode,
        q_descale=q_descale,
        k_descale=k_descale,
        quant_mode=quant_mode,
    )


"""
PoolKeyIndexer ATK 自定义执行方式 (BaseApi) + 三节点共享输入规整工具。

注册名: "pool_key_indexer" (对应 YAML api_type)

执行拓扑(双标杆精度标准 pki_double_benchmark):
  - operator  (pyaclnn): aclnn 接口调用自定义算子, 原始精度输入;
  - npu_golden(npu)     : NPU 组合标杆(与 pytest 的 _reference_npu 一致), 原始精度输入;
  - 真值      (cpu_benchmark, ATK benchmark 任务自动触发): CPU 金标 + 升精度输入
    (fp16/bf16 -> fp32), 与 pytest --ref-device cpu 的金标一致。

本 Api 按输入所在设备自动路由(与 pytest 的 pool_key_indexer_reference 一致):
  - 输入在 CPU(真值任务, 升精度后)  -> CPU 金标实现(_reference_cpu);
  - 输入在 NPU(npu_golden 节点)     -> NPU 组合标杆实现(_reference_npu);
  - 稠密化超限(oversized)          -> 一律搬 CPU 分块标杆(_reference_npu_blocked),
    NPU 分块路径已废弃(大 sort/topk 有 ERR 507035 打挂设备前科, 见
    _NPU_GOLDEN_MAX_* 阈值处注释)。

输入规整(prepare_pki_inputs)为全节点共用(算子适配器/标杆/真值):
ATK 随机生成的 actualSeq / blockTable 需要确定性规整为合法形态,
规整只依赖输入数据本身(种子固定), 保证三方消费到完全一致的输入。
"""

import math

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# pool_key_indexer_reference / _e8m0_decode / _reference_npu(_blocked)
# 已合入本文件上半部分, 直接引用

LAYOUT_MAP = {0: "BSND", 1: "TND", 2: "PA_BBND"}

# 输入索引 (与 YAML inputs 顺序一致)
IDX_QUERY = 0
IDX_POOL_KEY = 1
IDX_WEIGHTS = 2
IDX_POOL_TAIL_K = 3
IDX_ACT_SEQ_Q = 4
IDX_ACT_SEQ_K = 5
IDX_BLOCK_TABLE = 6
IDX_Q_DESCALE = 7
IDX_K_DESCALE = 8
IDX_LAYOUT_Q = 9
IDX_LAYOUT_K = 10
IDX_TOPK = 11
IDX_POOL_SIZE = 12
IDX_MASK_MODE = 13
IDX_QUANT_MODE = 14
IDX_RETURN_VALUE = 15
IDX_TND_UNEQUAL = 16
IDX_ZERO_Q_BATCH = 17
IDX_ZERO_POOL_BATCH = 18
IDX_KEY_NON_CONTIG = 19
# sparse_indices_mode(索引输出布局模式, V2 接口专有): 1=紧凑布局(compact); 缺省/0=V1 legacy。
# 仅 pool_key_indexer_v2compact.yaml 声明该输入(位置 20), 旧 YAML 无此字段。
IDX_SPARSE_INDICES_MODE = 20


def _parse_layout(raw):
    if isinstance(raw, (int, float)):
        return LAYOUT_MAP.get(int(raw), "BSND")
    return str(raw)


def _parse_bool(raw, default=False):
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    return default


def _tensor_or_none(t):
    """空 tensor(shape[0]) -> None(可选输入未启用)。"""
    if t is None:
        return None
    if hasattr(t, "shape") and t.numel() == 0:
        return None
    return t


def _normalize_prefix_seq(seq, total):
    """ATK 随机生成的 TND actualSeq 规整为合法前缀和(末项 = total)。

    只依赖 len(seq) 与 total, 确定性输出, 三节点一致。
    """
    b = len(seq)
    if b <= 1:
        return torch.tensor([total], dtype=torch.int64)
    lens = [total // b] * b
    lens[-1] += total - sum(lens)
    return torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)


def _distribute(total, b):
    """total 均分到 b 份, 余数逐份 +1 分散(不堆在末 batch)。"""
    base = total // b
    rem = total - base * b
    return [base + 1 if i < rem else base for i in range(b)]


def _split_batch_lens(b, total, mode="uniform", max_ratio=2.0):
    """将 total 确定性拆分为 b 份逐 batch 长度(与随机 tensor 值解耦)。

    mode:
      - "uniform": 均分(默认)
      - "unequal": 头 batch 占大头, 其余均分(不等长 batch)
      - "zero_head": 首 batch 为 0, 其余按 unequal 拆(0 长度/0 池 batch)
    b<2 或余量不足时退化为 uniform。

    max_ratio: 单 batch 长度相对均匀份额的上限。必须设上限:
    NPU 组合标杆按 (B, S1m, Pm) 稠密化, 若头 batch 按 total 的固定比例
    (如 60%)取值, 大 batch 用例单 batch 峰值放大 ~b/2 倍, 稠密矩阵
    随 batch 维再乘 B, 内存 O(b^2) 爆炸(实测 51 batch 时 dot 矩阵
    ~77GB 导致 NPU OOM)。上限 2x 保证峰值 <= 均匀基线的 ~4x。
    (zero_head 的峰值上限按有效份额 total/(b-1) 计, 略超 2x*total/b
    属预期。)
    """
    if b <= 0 or total < 0:
        return []
    if b == 1:
        return [total]
    if mode == "zero_head":
        if total < b - 1:  # 其余 batch 各至少 1 个
            mode = "uniform"
        else:
            return [0] + _split_batch_lens(b - 1, total, "unequal", max_ratio)
    if mode == "unequal":
        eff = b - 1
        head = max(1, int(total * 0.6))
        head = min(head, int(max_ratio * total) // b)
        head = min(head, total - eff)
        head = max(head, 1)
        return [head] + _distribute(total - head, eff)
    return _distribute(total, b)


def make_pa_key_non_contig(key):
    """PA key 0 轴非连续视图: 物理块嵌入 2 倍缓冲偶数行(与 pytest
    make_pa_key_non_contig 语义一致), stride(0) = 2 * blockSize * N2 * D。

    必须在目标设备上调用(非稠密视图跨设备 .to() 会被拍平为连续,
    丢失 stride 信息)。"""
    bn, bs, n2, d = key.shape
    big = torch.zeros(2 * bn, bs, n2, d, dtype=key.dtype, device=key.device)
    big[0::2] = key
    return big[::2]


def _construct_block_table(batch_size, act_seq_k, block_num, block_size):
    """PA_BBND 合法 block_table 构造(与 pytest pack_pa_key 语义一致)。

    每 batch 池独立连续填块, 物理块随机排布一次(种子固定),
    bt[b][i] 指向的物理块真实存放 batch b 第 i 组池; 0 池 batch 不占块。
    """
    max_block_num_per_seq = 0
    block_num_per_batch = []
    for b in range(batch_size):
        cur_k = (
            int(act_seq_k[b].item())
            if torch.is_tensor(act_seq_k)
            else int(act_seq_k[b])
        )
        cur_blocks = math.ceil(cur_k / block_size) if cur_k > 0 else 0
        block_num_per_batch.append(cur_blocks)
        max_block_num_per_seq = max(max_block_num_per_seq, cur_blocks)

    rng = np.random.RandomState(block_num * 31 + batch_size * 7 + max_block_num_per_seq)
    block_id_list = rng.permutation(block_num).astype(np.int32)
    bt = np.full((batch_size, max_block_num_per_seq), -1, dtype=np.int32)
    cur_id = 0
    for b in range(batch_size):
        for i in range(block_num_per_batch[b]):
            if cur_id < block_num:
                bt[b][i] = block_id_list[cur_id]
                cur_id += 1
    return torch.from_numpy(bt).to(torch.int32)


def prepare_pki_inputs(input_data: InputDataset, build_non_contig=True):
    """解析并规整 ATK 输入, 算子/标杆/真值三方一致消费。

    build_non_contig: 是否在 key 所在设备构造 PA 非连续视图。
    pyaclnn 适配器传 False(输入在 npu:0, 需搬到目标设备后再构造,
    避免跨设备 .to() 拍平视图), 由适配器自行构造并读取 keyStride0。

    返回 dict:
      query/pool_key/weights/pool_tail_k/actual_seq_q/actual_seq_k/block_table
      layout_q/layout_k/topk/pool_size/mask_mode/quant_mode/return_value
      tnd_unequal/zero_q_batch/zero_pool_batch/key_non_contig
      q_descale/k_descale(算子侧: mode=0 fp32 scale / mode=1 uint8 E8M0 raw)
      q_descale_scales/k_descale_scales(标杆侧: mode=1 已解码 [.., D/32] float;
      mode=0 与原 tensor 相同; 非量化恒 None)
    """
    args = list(input_data.args) if input_data.args else []
    kw = dict(input_data.kwargs) if input_data.kwargs else {}

    def pick(idx, name):
        if idx < len(args) and args[idx] is not None:
            return args[idx]
        return kw.get(name)

    def pick_attr(idx, name, default):
        value = pick(idx, name)
        return default if value is None else value

    query = pick(IDX_QUERY, "query")
    key = pick(IDX_POOL_KEY, "poolKey")
    weights = pick(IDX_WEIGHTS, "weights")
    pool_tail_k = _tensor_or_none(pick(IDX_POOL_TAIL_K, "poolTailK"))
    act_seq_q = _tensor_or_none(pick(IDX_ACT_SEQ_Q, "actualSeqQ"))
    act_seq_k = _tensor_or_none(pick(IDX_ACT_SEQ_K, "actualSeqK"))
    block_table = _tensor_or_none(pick(IDX_BLOCK_TABLE, "blockTable"))

    # 属性双路读取: 生成态在 args, 落盘加载态在 kwargs(按名)
    layout_q = _parse_layout(pick_attr(IDX_LAYOUT_Q, "layout_q", "BSND"))
    layout_k = _parse_layout(pick_attr(IDX_LAYOUT_K, "layout_k", "BSND"))
    topk = int(pick_attr(IDX_TOPK, "topk", 2048))
    pool_size = int(pick_attr(IDX_POOL_SIZE, "pool_size", 1))
    mask_mode = int(pick_attr(IDX_MASK_MODE, "mask_mode", 3))
    quant_mode = int(pick_attr(IDX_QUANT_MODE, "quant_mode", -1))
    return_value = _parse_bool(pick_attr(IDX_RETURN_VALUE, "return_value", False))
    # 场景标志(旧 JSON 无字段时回退 False)
    tnd_unequal = _parse_bool(pick_attr(IDX_TND_UNEQUAL, "tnd_unequal", False))
    zero_q_batch = _parse_bool(pick_attr(IDX_ZERO_Q_BATCH, "zero_q_batch", False))
    zero_pool_batch = _parse_bool(
        pick_attr(IDX_ZERO_POOL_BATCH, "zero_pool_batch", False)
    )
    key_non_contig = _parse_bool(pick_attr(IDX_KEY_NON_CONTIG, "key_non_contig", False))
    # sparse_indices_mode(索引输出布局模式): V2 yaml 声明该属性; 旧 YAML/旧 JSON 无字段回退 0(legacy)
    sparse_indices_mode = int(
        pick_attr(IDX_SPARSE_INDICES_MODE, "sparse_indices_mode", 0)
    )

    # TND actualSeq: 按场景标志确定性拆分(不等长/0长度 batch), 前缀和
    if layout_q == "TND" and act_seq_q is not None:
        mode = (
            "zero_head" if zero_q_batch else ("unequal" if tnd_unequal else "uniform")
        )
        lens = _split_batch_lens(len(act_seq_q), query.shape[0], mode)
        act_seq_q = torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)
    if layout_k == "TND" and act_seq_k is not None:
        mode = (
            "zero_head"
            if zero_pool_batch
            else ("unequal" if tnd_unequal else "uniform")
        )
        lens = _split_batch_lens(len(act_seq_k), key.shape[0], mode)
        act_seq_k = torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)

    # PA_BBND: actualSeqK 为每 batch 池数(非前缀和), 构造合法 block_table
    if layout_k == "PA_BBND":
        b_pa = len(act_seq_k) if act_seq_k is not None else query.shape[0]
        if zero_pool_batch or tnd_unequal:
            # 不等长/0池: 按 blockNum 拆分物理块(总量精确匹配 shape),
            # 每 batch 池数 = 块数 * blockSize
            mode = "zero_head" if zero_pool_batch else "unequal"
            blocks = _split_batch_lens(b_pa, key.shape[0], mode)
            act_seq_k = torch.tensor(
                [bl * key.shape[1] for bl in blocks], dtype=torch.int64
            )
        elif act_seq_k is None:
            act_seq_k = torch.full((b_pa,), key.shape[1], dtype=torch.int64)
        block_table = _construct_block_table(
            b_pa, act_seq_k, key.shape[0], key.shape[1]
        )

    # PA key 0 轴非连续: 在 key 所在设备构造视图
    # (pyaclnn 适配器传 build_non_contig=False, 由其在目标设备自行构造)
    if (
        build_non_contig
        and key_non_contig
        and layout_k == "PA_BBND"
        and key.is_contiguous()
    ):
        key = make_pa_key_non_contig(key)

    if pool_tail_k is None:
        b = query.shape[0] if layout_q == "BSND" else len(act_seq_q)
        pool_tail_k = torch.zeros((b,), dtype=torch.int64)

    # ---- 量化 descale 规整(quant_mode=0/1; 非量化恒为 None, 行为不变) ----
    # q_descale/k_descale:
    #   - prep["q_descale"]/["k_descale"]: 算子侧原始形态
    #       mode=0: FLOAT scale(真值任务升精度后可能是 fp64, 统一回 fp32, 精确);
    #       mode=1: uint8 E8M0 raw(适配层视图为 float8_e8m0fnu 下传);
    #   - prep["q_descale_scales"]/["k_descale_scales"]: 标杆侧解码形态
    #       mode=0: 同原 tensor; mode=1: 2^(raw-127) 解码 + 尾维 [D/64,2]
    #       展平为 [D/32](与 pytest run_golden 的 e8m0 预处理一致)
    q_descale = _tensor_or_none(pick(IDX_Q_DESCALE, "qDescale"))
    k_descale = _tensor_or_none(pick(IDX_K_DESCALE, "kDescale"))
    if quant_mode not in (0, 1, 2):
        q_descale = None
        k_descale = None
    q_descale_scales = None
    k_descale_scales = None
    if q_descale is not None or k_descale is not None:
        if quant_mode in (0, 2):
            if q_descale is not None:
                q_descale_scales = q_descale.to(torch.float32)
                if quant_mode == 0:
                    # mode=0: descale 原生 FLOAT, 算子侧统一 fp32
                    q_descale = q_descale_scales
                # mode=2(int8): 算子侧保持 fp16 原始 tensor(host tiling 要求
                # FLOAT16), 标杆侧消费 fp32 解码形态(q_descale_scales)
            if k_descale is not None:
                k_descale_scales = k_descale.to(torch.float32)
                if quant_mode == 0:
                    k_descale = k_descale_scales
        else:  # mxFP8: raw -> float scale, 尾维 [D/64, 2] 展平为 D/32
            if q_descale is not None:
                q_descale_scales = _e8m0_decode(q_descale).reshape(
                    *q_descale.shape[:-2], -1
                )
            if k_descale is not None:
                k_descale_scales = _e8m0_decode(k_descale).reshape(
                    *k_descale.shape[:-2], -1
                )

    # 规整产生的 tensor 归一到 query 所在设备(CPU 真值 / NPU 标杆与算子),
    # 保证 reference/算子侧输入设备一致
    if hasattr(query, "device"):
        dev = query.device
        act_seq_q = act_seq_q.to(dev) if act_seq_q is not None else None
        act_seq_k = act_seq_k.to(dev) if act_seq_k is not None else None
        block_table = block_table.to(dev) if block_table is not None else None
        pool_tail_k = pool_tail_k.to(dev)
        q_descale = q_descale.to(dev) if q_descale is not None else None
        k_descale = k_descale.to(dev) if k_descale is not None else None
        q_descale_scales = (
            q_descale_scales.to(dev) if q_descale_scales is not None else None
        )
        k_descale_scales = (
            k_descale_scales.to(dev) if k_descale_scales is not None else None
        )

    return dict(
        query=query,
        pool_key=key,
        weights=weights,
        pool_tail_k=pool_tail_k,
        actual_seq_q=act_seq_q,
        actual_seq_k=act_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
        tnd_unequal=tnd_unequal,
        zero_q_batch=zero_q_batch,
        zero_pool_batch=zero_pool_batch,
        key_non_contig=key_non_contig,
        sparse_indices_mode=sparse_indices_mode,
        q_descale=q_descale,
        k_descale=k_descale,
        q_descale_scales=q_descale_scales,
        k_descale_scales=k_descale_scales,
    )


def pki_output_stubs(prep):
    """is_get_output_task 场景(输出规格推导): 返回正确 shape/dtype 的空 tensor。

    异常用例占位: pool_size/topk 可能为非法值(0/负数), 占位规格钳 1 防除零
    (占位 shape 不参与异常用例判定, 仅需可分配)。"""
    query = prep["query"]
    layout_q = prep["layout_q"]
    topk = max(1, int(prep["topk"]))
    pool_size = max(1, int(prep["pool_size"]))
    out_len = topk + pool_size - 1
    sparse = topk // pool_size
    if layout_q == "BSND":
        idx_shape = [query.shape[0], query.shape[1], out_len]
        val_shape = [query.shape[0], query.shape[1], sparse]
    else:
        idx_shape = [query.shape[0], out_len]
        val_shape = [query.shape[0], sparse]
    indices = torch.empty(idx_shape, dtype=torch.int32)
    if prep["return_value"]:
        values = torch.empty(val_shape, dtype=torch.float32)
    else:
        values = torch.empty(0, dtype=torch.float32)
    return indices, values


def _estimate_npu_golden_load(prep):
    """估算 NPU 组合标杆稠密化后的峰值元素数。

    _reference_npu 按 (B, S1m, Pm) 稠密化: scores 一份 + sort 工作区 ~2x +
    展开输出 (B, S1m, topk+ps-1) 多份 —— 极端 shape(zero_q/unequal 使 S1m
    翻倍、大 topk)下峰值数十 GB, NPU 64GB HBM 放不下: 实测 OOM 或挂死
    (AdvancedIndex/NonZero aicpu timeout ~10min 且进程级不可捕获, 会级联
    饿死 worker 内后续用例)。返回 (scores元素数, 展开输出元素数) 供上限判断。
    """
    q = prep["query"]
    ps = prep["pool_size"]
    topk = prep["topk"]
    if prep["layout_q"] == "BSND":
        b, s1m = int(q.shape[0]), int(q.shape[1])
    else:
        aq = prep["actual_seq_q"]
        if aq is not None and len(aq) > 0:
            lens = [int(aq[0])] + [int(aq[i] - aq[i - 1]) for i in range(1, len(aq))]
            b, s1m = len(lens), (max(lens) if lens else 0)
        else:
            b, s1m = 1, int(q.shape[0])
    ak = prep["actual_seq_k"]
    key = prep["pool_key"]
    if prep["layout_k"] == "BSND":
        pm = int(key.shape[1])
    elif ak is not None and len(ak) > 0:
        if prep["layout_k"] == "TND":
            lensk = [int(ak[0])] + [int(ak[i] - ak[i - 1]) for i in range(1, len(ak))]
            pm = max(lensk) if lensk else 0
        else:  # PA: 每 batch 池数(非前缀和)
            pm = int(max(int(v) for v in ak))
    else:
        pm = int(key.shape[0])
    return b * s1m * pm, b * s1m * (topk + ps - 1)


# NPU 组合标杆稠密化元素上限(超过走分块 streaming 路径 _reference_npu_blocked,
# 数值语义一致): 原稠密实现一次性物化 (B,S1m,Pm) scores, 极端 shape(ps=1 +
# 池数 11~17 万)下单矩阵 50~500GB 放不下; 中大 shape 放得下但 worker 长跑后
# caching allocator 碎片化, 数 GB 级分配劣化直至 aicpu 看门狗超时(进程级
# 不可捕获, 级联饿死后续用例)。blocked 实现峰值仅 S1m*blockW + S1m*k_sel*2,
# 实测极端 case(ps=1/11~17 万池)11~39s 完成(稠密版 CPU 降级路径 30~90 分钟
# 仍超时)。CPU 向量化仅作 blocked 亦 OOM 的兜底。
_NPU_GOLDEN_MAX_SCORE_ELEMS = 50_000_000
_NPU_GOLDEN_MAX_TOKEN_ELEMS = 250_000_000
# oversized 一律搬 CPU 跑分块标杆(数值语义一致), 不再保留 NPU 分块路径:
# NPU 长跑 worker 上大 sort/大张量算子存在非确定 vector core 异常(ERR 507035)
# 前科 —— 2026-09-10 全量跑中 case 633(67M scores, 8192x16384 argsort)的
# npu_golden 打崩 device 2, 级联秒败后续 124 例(算子侧无责, 该段算子任务
# 从未执行); 2026-09-12 case 1186(57309x16384 topk/argsort, scores 3.28e9)
# 在 device 2/3 双双确定性复现并打挂设备, 同批 1187~1199 段被殃及秒败。
# 且 1186~1399 巨型带用例单块 sort 规模 8e8~1.7e10, 全部不低于 1186 的
# 确定性崩溃点, 在 NPU 上本就无通过可能, CPU 是其唯一可执行路径。
# golden 仅为参考节点(bm-vs-truth 为 reference-only 指标, 比较器从落盘
# 文件读取且统一 .cpu()), 搬 CPU 后 256 核主机秒级~分钟级完成; 极端
# s1m≈1M 用例小时级、峰值内存 O(s1m*(blockW+3*k_sel)) 达数百 GB, 全量跑
# 时须避免多例巨型金标并发挤兑主机内存。注: 不通过调小 block_w 省内存
# —— 归并宽度 2*k_sel 不可缩减, block_w 变小只会增加归并轮数(更慢),
# 对占大头的归并缓冲(s1m*2*k_sel*12B)无收益。


def _is_pki_abnormal_case(task_result):
    """case 配置了 expected_error_msg => 异常值用例(ATK 走错误信息匹配判定)。

    异常用例下: 标杆/真值节点不执行参考计算(非法输入会让参考实现崩溃,
    而精度比对对异常用例不生效, 由错误信息匹配一票定案), 仅返回占位输出;
    pyaclnn 适配器按原始非法输入直传(本套件异常用例均基于 BSND/BSND 声明,
    prepare_pki_inputs 对该布局不做任何重建, 非法参数原样到达 host 校验)。
    """
    try:
        case_cfg = getattr(task_result, "case_config", None)
        return bool(getattr(case_cfg, "expected_error_msg", None))
    except Exception:
        return False


@register("pool_key_indexer")
class PKIApi(BaseApi):
    """PoolKeyIndexer 统一标杆执行器(CPU 金标 / NPU 组合标杆, 按输入设备路由)。"""

    def __init__(self, task_result):
        super().__init__(task_result)
        self.run_modes = []

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        prep = prepare_pki_inputs(input_data)

        if getattr(self.task_result, "is_get_output_task", False):
            return pki_output_stubs(prep)
        if _is_pki_abnormal_case(self.task_result):
            return pki_output_stubs(prep)

        # 量化场景: NPU 组合标杆暂不支持 descale(pool_key_indexer_reference
        # 约定), 统一搬 CPU 走金标 —— 输入已在 CPU 时无额外搬运。
        # oversized 的量化用例同样走 CPU 分块标杆(descale 已在
        # _reference_npu_blocked 原生支持, 见下方 oversized 路由)
        # quant_mode=2(int8) 与 mode=0 同构: 正 descale 可提出 ReLU,
        # int8 点积在 fp32 中精确, 金标按 mode=0 的 scale 折算路径计算
        quant = prep["quant_mode"] in (0, 1, 2)
        # 环境开关 ATK_PKI_GOLDEN_DEVICE=cpu: 强制金标走 CPU。共享卡高载下
        # NPU 金标的大 sort/topk 撞 507034 vector core timeout(批量误杀);
        # 两套 yaml 均 skip_values_compare=true, 裁决仅依赖 indices vs CPU
        # 真值, NPU 金标仅作 reference-only 标定 —— 搬 CPU 对判定无损。
        import os as _os

        force_cpu_golden = _os.environ.get("ATK_PKI_GOLDEN_DEVICE", "").lower() == "cpu"
        # 性能测试(with_output=False)不搬 CPU: 量化标杆在 NPU 上执行才有 Device
        # 性能可言; 精度比较(with_output=True)维持 CPU 路由(NPU 标杆暂不支持
        # descale, 大 sort/topk 有 vector 核超时前科)
        route_cpu = (quant or force_cpu_golden) and with_output
        if route_cpu and prep["query"].device.type != "cpu":
            prep = {
                k: (v.cpu() if hasattr(v, "cpu") and torch.is_tensor(v) else v)
                for k, v in prep.items()
            }

        # 统一标杆(与 pytest 一致):
        #   输入在 CPU(真值任务升精度后) -> CPU 金标; 输入在 NPU -> NPU 组合标杆
        #   极端 shape(稠密化超限) -> 一律搬 CPU 走分块 streaming 实现(数值
        #     语义一致, 峰值内存 S1m*blockW + S1m*k_sel*2):
        #     CPU 分块向量化数分钟内完成(原 _reference_cpu 逐行 Python 循环
        #     在 11 万行规格下 30~90 分钟超时); NPU 分块路径已废弃 —— 其大
        #     sort/topk 存在 ERR 507035 vector 核异常前科, 会打挂设备(见
        #     _NPU_GOLDEN_MAX_* 阈值处注释)。
        scores_elems, token_elems = _estimate_npu_golden_load(prep)
        oversized = (
            scores_elems > _NPU_GOLDEN_MAX_SCORE_ELEMS
            or token_elems > _NPU_GOLDEN_MAX_TOKEN_ELEMS
        )
        # 性能测试(with_output=False): oversized 也走 NPU 标杆, 不搬 CPU
        # —— Device 性能如实记录(稠密矩阵超 NPU 内存时自然失败, 属真实限制)
        if oversized and not with_output:
            oversized = False
        if oversized:
            if prep["query"].device.type == "npu":
                # oversized 一律搬 CPU 跑分块标杆(含量化场景 —— descale
                # 已在 _reference_npu_blocked 原生支持, 数值口径与稠密版
                # _compute_batch 一致), 设备风险彻底清零(见
                # _NPU_GOLDEN_MAX_* 阈值处注释: NPU 分块的大 sort/topk 有
                # ERR 507035 vector 核异常前科, case 633 打崩 device 2 级联
                # 秒败 124 例, case 1186 在 device 2/3 确定性复现打挂设备,
                # 算子侧无责 —— pyaclnn 任务从未执行)。empty_cache 释放
                # 搬运后的 NPU 缓存块, 避免碎片化殃及同 case 后续的
                # pyaclnn 算子任务。
                prep = {
                    k: (v.cpu() if torch.is_tensor(v) else v) for k, v in prep.items()
                }
                torch.npu.empty_cache()
            # common 必须在设备路由(可能整体搬 CPU)之后构建: 若提前构建,
            # block_table/actual_seq_* 留在 NPU 与已搬 CPU 的 query/pool_key
            # 混用, PA_BBND 场景 index_select 直接跨设备报错(全量跑实测
            # 20 例 PA+oversized 用例失败, 均为本 bug)
            common = dict(
                actual_seq_q=prep["actual_seq_q"],
                actual_seq_k=prep["actual_seq_k"],
                block_table=prep["block_table"],
                layout_q=prep["layout_q"],
                layout_k=prep["layout_k"],
                topk=prep["topk"],
                pool_size=prep["pool_size"],
                mask_mode=prep["mask_mode"],
                return_value=prep["return_value"],
            )
            indices, values = _reference_npu_blocked(
                prep["query"],
                prep["pool_key"],
                prep["weights"],
                prep["pool_tail_k"],
                **common,
                q_descale=prep["q_descale_scales"] if quant else None,
                k_descale=prep["k_descale_scales"] if quant else None,
                quant_mode=prep["quant_mode"] if quant else -1,
                sparse_indices_mode=prep.get("sparse_indices_mode", 0),
            )
            return indices, values
        # perf_mode 仅性能任务(performance_device, with_output=False)为 True:
        # 精度任务(accuracy, with_output=True)恒 False —— NPU 标杆走原口径,
        # fp8 分支/索引中转对精度验证不可达(量化精度标杆恒 route CPU)
        indices, values = pool_key_indexer_reference(
            prep["query"],
            prep["pool_key"],
            prep["weights"],
            prep["pool_tail_k"],
            actual_seq_q=prep["actual_seq_q"],
            actual_seq_k=prep["actual_seq_k"],
            block_table=prep["block_table"],
            layout_q=prep["layout_q"],
            layout_k=prep["layout_k"],
            topk=prep["topk"],
            pool_size=prep["pool_size"],
            mask_mode=prep["mask_mode"],
            return_value=prep["return_value"],
            q_descale=prep["q_descale_scales"] if quant else None,
            k_descale=prep["k_descale_scales"] if quant else None,
            quant_mode=prep["quant_mode"] if quant else -1,
            sparse_indices_mode=prep.get("sparse_indices_mode", 0),
            perf_mode=not with_output,
        )
        return indices, values


"""
PoolKeyIndexer ATK PyAclnn 适配器 (AclnnBaseApi)。

注册名: "aclnn_pool_key_indexer" (对应 YAML aclnn_api_type)

职责(主要完成 aclnn 接口的测试):
  1. init_by_input_data: 构造 aclnn C 接口入参, 调用 aclnnPoolKeyIndexer;
     - poolTailK / actualSeqQ / actualSeqK 为 ValueDepend 输入,
       传 aclIntArray*(Host 侧 int64 值), 未启用时传空指针;
     - blockTable / qDescale / kDescale 为可选 aclTensor*, 未启用传空指针;
     - 属性顺序与 op_def 声明一致: topk, poolSize, layoutQuery, layoutKey,
       maskMode, quantMode, returnValue, keyStride0, kDescaleStride0;
     - 输出 sparseIndicesOut(INT32) / sparseValuesOut(FLOAT, return_value=False
       时为空 tensor), 规格由标杆节点运行结果(output_info_list)提供。
  2. 算子输出保持原始结果(不做任何归一化/重排后处理), 由精度比较插件
     (pool_key_indexer_compare.py) 按 pytest 口径比对。

aclnn 接口签名(与 op_host/pool_key_indexer_def.cpp 声明顺序一致):
  aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
      const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights,
      const aclIntArray* poolTailK,
      const aclIntArray* actualSeqLengthsQueryOptional,
      const aclIntArray* actualSeqLengthsKeyOptional,
      const aclTensor* blockTableOptional,
      const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional,
      int64_t topk, int64_t poolSize,
      char* layoutQueryOptional, char* layoutKeyOptional,
      int64_t maskMode, int64_t quantMode, bool returnValue,
      int64_t keyStride0, int64_t kDescaleStride0,
      const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut,
      uint64_t* workspaceSize, aclOpExecutor** executor)
"""

import ctypes

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr, AclIntArray

# pool_key_indexer_api 模块已合入本文件(PKIApi 注册 + prepare_pki_inputs/
# make_pa_key_non_contig 工具), 直接引用, 无需 import

# E8M0 dtype 映射补充: atk acl_wrapper 的 TORCH_TO_ACLTYPE 未收录
# torch.float8_e8m0fnu(仅 e4m3/e5m2), 量化 mxFP8(quant_mode=1) descale 需要
# ACL_FLOAT8_E8M0 —— 增量补充, 不影响既有映射与场景。
# 兼容性守卫: 部分 atk 版本(如 26.4.30)的 AclDataType 未收录 ACL_FLOAT8_E8M0,
# 此时跳过补充(该版本不支持 mxFP8 descale, 非量化场景不受影响),
# 避免模块级 AttributeError 导致整个适配器注册失败。
from atk.tasks.backends.lib_interface import acl_wrapper as _acl_wrapper

_ACL_FLOAT8_E8M0 = getattr(
    getattr(_acl_wrapper, "AclDataType", None), "ACL_FLOAT8_E8M0", None
)
if (
    _ACL_FLOAT8_E8M0 is not None
    and hasattr(_acl_wrapper, "TORCH_TO_ACLTYPE")
    and "torch.float8_e8m0fnu" not in _acl_wrapper.TORCH_TO_ACLTYPE
):
    _acl_wrapper.TORCH_TO_ACLTYPE["torch.float8_e8m0fnu"] = _ACL_FLOAT8_E8M0

# ValueDepend 可选输入的空指针类型(与 aclIntArray* 签名校验匹配)
IntAryPtr = ctypes.POINTER(AclIntArray)

_TORCH_OUTPUT_DTYPES = (
    "torch.int32",
    "torch.float32",
    "int32",
    "fp32",
    "float32",
)


def _parse_torch_dtype(raw, default):
    mapping = {
        "torch.int32": torch.int32,
        "torch.float32": torch.float32,
        "int32": torch.int32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(str(raw), default)


def _recover_device(device_id):
    """aicore 异常后复位并重新初始化 NPU 设备(进程内后续用例不受污染)。"""
    import acl
    import os
    import time

    try:
        torch.npu.empty_cache()
    except Exception:
        pass

    try:
        acl.rt.reset_device(device_id)
        time.sleep(1)
        acl.rt.set_device(device_id)
    except Exception:
        pass

    try:
        torch.npu.set_device(device_id)
        torch.npu.empty_cache()
        _ = torch.zeros(1, device=f"npu:{device_id}")
        torch.npu.synchronize()
    except Exception:
        os._exit(1)


@register("aclnn_pool_key_indexer")
class PKIAclnnApi(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super().__init__(task_result, backend)
        self.run_modes = []

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize("
            "const aclTensor* query, "
            "const aclTensor* poolKey, "
            "const aclTensor* weights, "
            "const aclIntArray* poolTailK, "
            "const aclIntArray* actualSeqLengthsQueryOptional, "
            "const aclIntArray* actualSeqLengthsKeyOptional, "
            "const aclTensor* blockTableOptional, "
            "const aclTensor* qDescaleOptional, "
            "const aclTensor* kDescaleOptional, "
            "int64_t topk, "
            "int64_t poolSize, "
            "char* layoutQueryOptional, "
            "char* layoutKeyOptional, "
            "int64_t maskMode, "
            "int64_t quantMode, "
            "bool returnValue, "
            "int64_t keyStride0, "
            "int64_t kDescaleStride0, "
            "const aclTensor* sparseIndicesOut, "
            "const aclTensor* sparseValuesOut, "
            "uint64_t* workspaceSize, "
            "aclOpExecutor** executor)"
        )

    def _get_device_id(self):
        device_id = 0
        if hasattr(self.backend, "device_id") and self.backend.device_id is not None:
            device_id = self.backend.device_id
        elif getattr(self.task_result, "device_id", None) is not None:
            device_id = self.task_result.device_id
        return int(device_id)

    def _init_abnormal_raw(self, input_data: InputDataset, device_id: int, dev: str):
        """异常用例 raw 旁路: 非法输入原样下发(跳过一切伴随参数补全/重建)。

        prepare_pki_inputs 会"修复"部分非法性(非量化场景丢弃 descale、
        TND/PA 重建 act_seq/block_table), 使目标异常被测试侧消解; 异常用例
        必须保证算子 host 校验按原始声明拒绝。属性仍按声明解析(非法属性
        取值本身就是要下发的非法性)。
        """
        args = list(input_data.args) if input_data.args else []
        kw = dict(input_data.kwargs) if input_data.kwargs else {}

        def pick(idx, name):
            if idx < len(args) and args[idx] is not None:
                return args[idx]
            return kw.get(name)

        def pick_attr(idx, name, default):
            value = pick(idx, name)
            return default if value is None else value

        def _to_dev(tensor):
            if tensor is None or not hasattr(tensor, "to"):
                return tensor
            if tensor.device.type == "npu" and tensor.device.index == device_id:
                return tensor
            return tensor.to(dev)

        def _tensor(tensor):
            result = self.backend.convert_input_data(_to_dev(tensor), name="tensor")
            if isinstance(result, list):
                result = result[0]
            return result

        def _int_array(tensor):
            if tensor is None:
                return IntAryPtr()
            values = [
                ctypes.c_int64(int(v))
                for v in tensor.detach().cpu().reshape(-1).tolist()
            ]
            return self.backend.convert_input_data(values)[0]

        def _tensor_or_null(tensor):
            # 必选输入空 tensor 保留空形态(触发 host "must not be empty" 校验),
            # 仅真正的 None(声明 [0] 的可选输入)映射空指针
            if tensor is None:
                return TensorPtr()
            return _tensor(tensor)

        # 必选输入: 原样下发(空 tensor 保留空形态, 触发"must not be empty"校验)
        query = pick(IDX_QUERY, "query")
        key = pick(IDX_POOL_KEY, "poolKey")
        weights = pick(IDX_WEIGHTS, "weights")
        pool_tail_k = pick(IDX_POOL_TAIL_K, "poolTailK")

        # [复核2026-11-12] mxFP8(quant_mode=1) descale 存储约定与正常路径一致:
        # uint8 E8M0 raw 在适配层视图为 float8_e8m0fnu 下传(prepare_pki_inputs
        # 同款), 否则 raw 旁路以 ACL_UINT8(4) 到达 tiling 的
        # "q_descale dtype must be FLOAT8_E8M0" 前置检查, 使 k_descale dtype/
        # 尾维等目标检查不可达(fp8 abnormal 6 例误报根因之一)。
        raw_quant_mode = int(pick_attr(IDX_QUANT_MODE, "quant_mode", -1))

        def _e8m0_storage_view(tensor):
            if (
                raw_quant_mode == 1
                and tensor is not None
                and getattr(tensor, "dtype", None) == torch.uint8
            ):
                return tensor.view(torch.float8_e8m0fnu)
            return tensor

        input_args = [
            _tensor_or_null(query),
            _tensor_or_null(key),
            _tensor_or_null(weights),
            _int_array(
                pool_tail_k
                if (
                    pool_tail_k is not None
                    and getattr(pool_tail_k, "numel", lambda: 1)() > 0
                )
                else None
            ),
            _int_array(_tensor_or_none(pick(IDX_ACT_SEQ_Q, "actualSeqQ"))),
            _int_array(_tensor_or_none(pick(IDX_ACT_SEQ_K, "actualSeqK"))),
            _tensor_or_null(_tensor_or_none(pick(IDX_BLOCK_TABLE, "blockTable"))),
            _tensor_or_null(
                _e8m0_storage_view(_tensor_or_none(pick(IDX_Q_DESCALE, "qDescale")))
            ),
            _tensor_or_null(
                _e8m0_storage_view(_tensor_or_none(pick(IDX_K_DESCALE, "kDescale")))
            ),
        ]

        # 9 个 attr 入参(与 op_def 声明顺序一致), 原样下发(含非法取值)
        input_args.append(ctypes.c_int64(int(pick_attr(IDX_TOPK, "topk", 2048))))
        input_args.append(
            ctypes.c_int64(int(pick_attr(IDX_POOL_SIZE, "pool_size", 16)))
        )
        layout_q = str(pick_attr(IDX_LAYOUT_Q, "layout_q", "BSND"))
        layout_k = str(pick_attr(IDX_LAYOUT_K, "layout_k", "BSND"))
        input_args.append(ctypes.c_char_p(layout_q.encode("utf-8")))
        input_args.append(ctypes.c_char_p(layout_k.encode("utf-8")))
        input_args.append(ctypes.c_int64(int(pick_attr(IDX_MASK_MODE, "mask_mode", 3))))
        input_args.append(
            ctypes.c_int64(int(pick_attr(IDX_QUANT_MODE, "quant_mode", -1)))
        )
        input_args.append(
            ctypes.c_bool(
                _parse_bool(pick_attr(IDX_RETURN_VALUE, "return_value", False))
            )
        )
        input_args.append(ctypes.c_int64(-1))  # keyStride0
        input_args.append(ctypes.c_int64(-1))  # kDescaleStride0

        # 输出 tensor: 规格来自标杆节点 output_info_list(异常用例为占位规格)
        self._saved_output_structs = []
        output_packages = []
        for index in range(len(self.task_result.output_info_list)):
            out = self.task_result.output_info_list[index]
            out_shape = (
                list(out.shape)
                if hasattr(out, "shape")
                else list(out.get("shape", [1]))
            )
            if index == 0:
                torch_dtype = torch.int32
            elif index == 1:
                torch_dtype = torch.float32
            else:
                out_dtype = (
                    out.dtype
                    if hasattr(out, "dtype")
                    else out.get("dtype", "torch.float32")
                )
                torch_dtype = _parse_torch_dtype(out_dtype, torch.float32)
            pytensor = torch.empty(out_shape, dtype=torch_dtype, device=dev)
            acl_struct = self.backend.torch_tensor_to_acl(pytensor)
            self._saved_output_structs.append(acl_struct)
            output_packages.append(acl_struct)

        input_args.extend(output_packages)
        return input_args, output_packages

    def init_by_input_data(self, input_data: InputDataset):
        self._saved_device_id = self._get_device_id()
        device_id = self._saved_device_id
        dev = f"npu:{device_id}"

        try:
            torch.npu.set_device(device_id)
            torch.npu.synchronize()
        except RuntimeError:
            _recover_device(device_id)

        # 异常用例 raw 旁路: 非法输入原样下发, 保证 host 校验按原始声明拒绝
        if _is_pki_abnormal_case(self.task_result):
            return self._init_abnormal_raw(input_data, device_id, dev)

        # 三节点一致的输入规整(TND 前缀和 / PA block_table / 场景标志);
        # 非连续 key 由本适配器在目标设备构造(输入在 npu:0, 跨设备搬运会
        # 拍平视图, prepare_pki_inputs 传 build_non_contig=False)
        prep = prepare_pki_inputs(input_data, build_non_contig=False)

        def _to_dev(tensor):
            if tensor is None or not hasattr(tensor, "to"):
                return tensor
            if tensor.device.type == "npu" and tensor.device.index == device_id:
                return tensor
            return tensor.to(dev)

        # PA key 0 轴非连续: 搬到目标设备后构造视图, keyStride0 取实际 stride
        key = _to_dev(prep["pool_key"])
        if (
            prep["key_non_contig"]
            and prep["layout_k"] == "PA_BBND"
            and key.is_contiguous()
        ):
            key = make_pa_key_non_contig(key)
        key_stride0 = -1
        if not key.is_contiguous():
            key_stride0 = int(key.stride(0))  # 元素单位, 与 torch_extension csrc 一致

        def _tensor(tensor):
            # aclTensor 输入(已搬到目标设备, 避免 create_acl_tensor 静默跨设备)
            result = self.backend.convert_input_data(_to_dev(tensor), name="tensor")
            if isinstance(result, list):
                result = result[0]
            return result

        def _int_array(tensor):
            # ValueDepend 输入: aclnn 期望 aclIntArray*(Host 侧 int64 值)
            if tensor is None:
                return IntAryPtr()  # 空 aclIntArray 指针(可选输入)
            values = [
                ctypes.c_int64(int(v))
                for v in tensor.detach().cpu().reshape(-1).tolist()
            ]
            return self.backend.convert_input_data(values)[0]

        input_args = []
        # 9 个输入: query/poolKey/weights/blockTable 为 aclTensor,
        # poolTailK/actualSeqQ/actualSeqK 为 aclIntArray(Host 侧取值)
        input_args.append(_tensor(prep["query"]))
        input_args.append(_tensor(key))
        input_args.append(_tensor(prep["weights"]))

        input_args.append(_int_array(prep["pool_tail_k"]))
        input_args.append(_int_array(prep["actual_seq_q"]))
        input_args.append(_int_array(prep["actual_seq_k"]))

        if prep["block_table"] is not None:
            input_args.append(_tensor(prep["block_table"]))
        else:
            input_args.append(TensorPtr())

        # qDescale / kDescale: 量化场景传实际 tensor, 非量化为空指针
        #   mode=0: FLOAT scale 直传;
        #   mode=1: uint8 raw 视图为 float8_e8m0fnu(E8M0, 1B 元素, 视图不改布局)
        if prep["q_descale"] is not None:
            qd = _to_dev(prep["q_descale"])
            if prep["quant_mode"] == 1:
                qd = qd.view(torch.float8_e8m0fnu)
            input_args.append(_tensor(qd))
        else:
            input_args.append(TensorPtr())  # qDescale
        if prep["k_descale"] is not None:
            kd = _to_dev(prep["k_descale"])
            if prep["quant_mode"] == 1:
                kd = kd.view(torch.float8_e8m0fnu)
            input_args.append(_tensor(kd))
        else:
            input_args.append(TensorPtr())  # kDescale

        # 9 个 attr 入参(与 op_def 声明顺序一致):
        # topk, poolSize, layoutQuery, layoutKey, maskMode, quantMode,
        # returnValue, keyStride0, kDescaleStride0
        # keyStride0: 非连续 key 传实际 stride(0)(元素单位); 连续传 -1
        # (tiling 回退 shape 推导); kDescaleStride0 非量化场景恒 -1
        input_args.append(ctypes.c_int64(prep["topk"]))
        input_args.append(ctypes.c_int64(prep["pool_size"]))
        input_args.append(ctypes.c_char_p(prep["layout_q"].encode("utf-8")))
        input_args.append(ctypes.c_char_p(prep["layout_k"].encode("utf-8")))
        input_args.append(ctypes.c_int64(prep["mask_mode"]))
        input_args.append(ctypes.c_int64(prep["quant_mode"]))
        input_args.append(ctypes.c_bool(prep["return_value"]))
        input_args.append(ctypes.c_int64(key_stride0))
        input_args.append(ctypes.c_int64(-1))  # kDescaleStride0

        # 输出 tensor: indices INT32, values FLOAT(规格来自标杆节点 output_info_list)
        self._saved_output_structs = []
        output_packages = []
        for index in range(len(self.task_result.output_info_list)):
            out = self.task_result.output_info_list[index]
            out_shape = (
                list(out.shape)
                if hasattr(out, "shape")
                else list(out.get("shape", [1]))
            )
            if index == 0:
                torch_dtype = torch.int32  # sparseIndicesOut
            elif index == 1:
                torch_dtype = torch.float32  # sparseValuesOut (固定 FLOAT)
            else:
                out_dtype = (
                    out.dtype
                    if hasattr(out, "dtype")
                    else out.get("dtype", "torch.float32")
                )
                torch_dtype = _parse_torch_dtype(out_dtype, torch.float32)

            pytensor = torch.empty(out_shape, dtype=torch_dtype, device=dev)
            acl_struct = self.backend.torch_tensor_to_acl(pytensor)
            self._saved_output_structs.append(acl_struct)
            output_packages.append(acl_struct)

        input_args.extend(output_packages)
        return input_args, output_packages

    def __call__(self):
        device_id = getattr(self, "_saved_device_id", 0)
        try:
            torch.npu.set_device(device_id)
            super().__call__()
            torch.npu.synchronize()
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "507015" in msg
                or "EZ9999" in msg
                or "EE9999" in msg
                or "aicore" in msg.lower()
            ):
                _recover_device(device_id)
            raise

    def after_call(self, output_packages):
        # 直接返回 NPU 侧输出 tensor(算子原始结果, 不做归一化),
        # 后续 save_output_data 阶段统一 .cpu() 落盘
        outputs = []
        for output_pack in output_packages:
            if hasattr(output_pack, "pytensor"):
                outputs.append(output_pack.pytensor)
        return outputs


@register("aclnn_pool_key_indexer_v2")
class PKIAclnnApiV2(PKIAclnnApi):
    """PoolKeyIndexer V2(aclnnPoolKeyIndexerV2, sparse_indices_mode 属性)PyAclnn 适配器。

    问题2(前缀消费合同)锚定用例配套: V2 接口签名 = V1 + ``int64_t sparseIndicesMode``
    (位于 kDescaleStride0 之后、输出参数之前), 布局语义见
    pool_key_indexer_v2compact.yaml。复用 V1 适配器的全部入参构造, 仅在
    输出参数前插入 sparse_indices_mode 属性。
    """

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnPoolKeyIndexerV2GetWorkspaceSize("
            "const aclTensor* query, "
            "const aclTensor* poolKey, "
            "const aclTensor* weights, "
            "const aclIntArray* poolTailK, "
            "const aclIntArray* actualSeqLengthsQueryOptional, "
            "const aclIntArray* actualSeqLengthsKeyOptional, "
            "const aclTensor* blockTableOptional, "
            "const aclTensor* qDescaleOptional, "
            "const aclTensor* kDescaleOptional, "
            "int64_t topk, "
            "int64_t poolSize, "
            "char* layoutQueryOptional, "
            "char* layoutKeyOptional, "
            "int64_t maskMode, "
            "int64_t quantMode, "
            "bool returnValue, "
            "int64_t keyStride0, "
            "int64_t kDescaleStride0, "
            "int64_t sparseIndicesMode, "
            "const aclTensor* sparseIndicesOut, "
            "const aclTensor* sparseValuesOut, "
            "uint64_t* workspaceSize, "
            "aclOpExecutor** executor)"
        )

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # sparse_indices_mode 属性插在输出参数之前(与 V2 C 签名一致);
        # 取值来自用例声明(锚定 yaml 恒为 1=compact)
        prep = prepare_pki_inputs(input_data, build_non_contig=False)
        sparse_indices_mode = int(prep.get("sparse_indices_mode", 1))
        input_args.insert(
            len(input_args) - len(output_packages), ctypes.c_int64(sparse_indices_mode)
        )
        return input_args, output_packages


"""
PoolKeyIndexer ATK 双标杆精度比较插件 (pki_double_benchmark)。

注册名: "pki_double_benchmark" (对应 YAML standard.acc, 名字包含 "benchmark"
以触发 ATK 框架的真值任务: 标杆升精度(fp16/bf16->fp32)在 CPU 上重跑,
结果落盘到 {output}/cpu_benchmark/{save_name}/{id}/)。

三方数据:
  - operator  : pyaclnn 算子输出(pyaclnn_operator/), 原始精度输入;
  - 标杆      : NPU 组合标杆输出(npu_npu_golden/), 原始精度输入;
  - 真值      : CPU 金标升精度输出(cpu_benchmark/), ATK benchmark 任务生成。

比对策略(与 pytest tests/pytest/result_compare.py 口径一致):
  1. indices(第一输出 sparseIndices, INT32) —— 集合对比法(唯一索引门禁):
     算子 vs 真值, 逐行提取有效索引(>=0, 无效位 -1)升序排序后做集合相等
     比较, 要求 100% 行通过(允许 TopK 输出顺序不同, 选中索引集合不得有差异);
     同时校验算子无效位取值必须精确为 -1。
  2. values(第二输出 sparseValues, FLOAT32) —— 按索引排序后双标杆对比:
     每行提取 (pool 索引, value) 对并按索引排序(TopK 顺序差异被吸收),
     算子与标杆分别与真值配对计算相对误差:
       Actual    = 算子 vs 真值   -> max_re / avg_re / rmse
       Benchmark = 标杆 vs 真值   -> max_re / avg_re / rmse
     门禁(双标杆: 标杆和算子都跟真值对比, 以标杆固有误差标定容差):
       Actual有效值 / Benchmark有效值 <= 对应 ratio;
       误差低于 abs_floor 视为 0(双方均达到数值噪声 floor 时通过);
       标杆有效误差为 0 而算子有效误差 > 0 时判定失败。

YAML 配置(standard.acc.pki_double_benchmark):
  max_re_ratio: 2        # max_re  的 Actual/Benchmark 比例上限
  avg_re_ratio: 1.2      # avg_re  的 Actual/Benchmark 比例上限
  rmse_ratio: 1.2        # rmse    的 Actual/Benchmark 比例上限
  row_pass_ratio: 1.0    # indices 集合比对行通过率阈值(1.0 = 严格 100%)
  abs_floor: 1.0e-5      # 误差噪声 floor(低于该值视为 0)
  denom_floor: 1.0e-3    # 相对误差分母下限(小值域用绝对误差口径)
  skip_values_compare: false  # values 精度比对开关: true=跳过 values 比对
                              # (rv=True 用例 values 直接判定通过, 仅保留
                              #  indices 门禁); 默认 false=正常比对
"""

import math
import os

import torch

from atk.common.log import Logger
from atk.common.utils import get_output_path_by_case, torch_load_safe
from atk.configs.nodes_config import BENCHMARK_PATH_PATTERN
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare

logging = Logger().get_logger()


def _to_3d(tensor):
    """(B, S, L) / (T, L) -> (B, S, L), 并转 CPU。"""
    t = tensor.cpu()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def _case_input_value(case_config, name, default=None):
    """从用例配置读取输入/属性取值(range_values[0])。"""
    for inp in case_config.inputs:
        if getattr(inp, "name", None) == name:
            rv = getattr(inp, "range_values", None)
            if rv:
                return rv[0]
            return default
    return default


def _indices_layout_compare(
    op_indices, truth_indices, topk, pool_size, sparse_indices_mode
):
    """indices 布局合同比对(问题2 前缀消费合同的确定性门禁)。

    集合比对(_indices_set_compare)对行内位置不敏感 —— tail 落在固定偏移
    topk(V1 legacy)还是紧跟有效展开(V2 compact)、中间是否有 -1 空洞,
    在集合口径下恒等。本函数以真值(标杆按 sparse_indices_mode 产出合同布局)为基准,
    做两级位置敏感断言, 且对 TopK 并列翻转免疫(翻转不改变行内有效位数):

      1. -1 掩码逐位一致: 有效位的"位置"必须与真值完全相同 ——
         V2 紧凑布局回归(尾 token 落错位/前缀中空/行尾非-1)与
         V1 legacy 布局回归(tail 不在 [topk, ...) 段)都会使掩码错位;
      2. 尾 token 值精确一致: 尾 token 是确定性追加(非 TopK 选择,
         无并列翻转歧义), 在真值尾位上取值必须逐元素相等。
         尾位从真值自证推导: V1 = 有效位且列 >= topk 的段;
             V2 = 行尾最后一个连续有效段(首个 -1 之前的有效后缀)。

    返回 (ok, error_info)。
    """
    op = _to_3d(op_indices)
    truth = _to_3d(truth_indices)
    if op.shape != truth.shape:
        return (
            False,
            f"layout compare shape mismatch: {tuple(op.shape)} vs {tuple(truth.shape)}",
        )

    op_neg = op == -1
    tr_neg = truth == -1
    mask_diff = op_neg != tr_neg
    if bool(mask_diff.any()):
        b_idx, s_idx, c_idx = mask_diff.nonzero(as_tuple=True)
        n = int(mask_diff.sum())
        show = [
            f"({int(b)}, {int(s)}, {int(c)}): op={int(op[b, s, c])} truth={int(truth[b, s, c])}"
            for b, s, c in list(zip(b_idx, s_idx, c_idx))[:4]
        ]
        return (
            False,
            f"layout contract violated: -1 mask differs from truth at {n} slots "
            f"(sparse_indices_mode={sparse_indices_mode}); first diffs: {show}",
        )

    # 尾 token 位置(从真值掩码几何无关推导)与取值精确比对
    width = truth.shape[-1]
    col = torch.arange(width, device=truth.device)[None, None, :]
    if sparse_indices_mode == 0:
        # V1 legacy: 尾 token 恒位于 [topk, topk+pool_size-1) 段
        tail_mask = (~tr_neg) & (col >= topk)
    else:
        # V2 compact: 有效区为单一前缀 [0, P), P = expandTotal + vtk, 其中
        # expandTotal ≡ 0 (mod ps) 且 0 <= vtk < ps  =>  vtk = P mod ps,
        # 尾段 = [P - P mod ps, P)(全 -1 行 P=0 -> 空尾)
        first_gap = torch.where(
            tr_neg.any(dim=-1),
            tr_neg.float().argmax(dim=-1),
            torch.full_like(tr_neg[..., 0], width, dtype=torch.long),
        )
        prefix_len = first_gap  # 有效前缀长度
        vtk = prefix_len % pool_size
        expand_total = prefix_len - vtk
        tail_mask = (col >= expand_total[..., None]) & (col < prefix_len[..., None])
    tail_diff = tail_mask & (op != truth)
    if bool(tail_diff.any()):
        b_idx, s_idx, c_idx = tail_diff.nonzero(as_tuple=True)
        n = int(tail_diff.sum())
        show = [
            f"({int(b)}, {int(s)}, {int(c)}): op={int(op[b, s, c])} truth={int(truth[b, s, c])}"
            for b, s, c in list(zip(b_idx, s_idx, c_idx))[:4]
        ]
        return (
            False,
            f"layout contract violated: tail token mismatch at {n} slots "
            f"(sparse_indices_mode={sparse_indices_mode}); first diffs: {show}",
        )
    return (True, "")


def _row_pairs(idx_row, val_row, pool_size, sparse):
    """从 indices/values 单行提取 (pool_idx, value) 对, 按索引排序。

    values 第 k 个有效值对应 indices 前 topk 列中第 k 个被选中的池
    (其 token 展开块首元素 = indices[k * pool_size], pool = token // pool_size)。
    """
    pairs = []
    for k in range(sparse):
        tok = int(idx_row[k * pool_size])
        val = float(val_row[k])
        if tok >= 0 and math.isfinite(val):
            pairs.append((tok // pool_size, val))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _pair_rel_errors(pairs_a, pairs_truth, denom_floor):
    """按 pool 索引交集配对, 计算相对误差列表(小值域用绝对误差口径)。"""
    truth_map = dict(pairs_truth)
    errors = []
    for pool_idx, value in pairs_a:
        truth_value = truth_map.get(pool_idx)
        if truth_value is None:
            continue
        errors.append(abs(value - truth_value) / max(abs(truth_value), denom_floor))
    return errors


def _metrics_of(errors):
    """(max_re, avg_re, rmse) 聚合。"""
    if not errors:
        return 0.0, 0.0, 0.0
    tensor = torch.tensor(errors, dtype=torch.float64)
    max_re = tensor.max().item()
    avg_re = tensor.mean().item()
    rmse = tensor.pow(2).mean().sqrt().item()
    return max_re, avg_re, rmse


def _indices_set_compare(op_indices, truth_indices, row_pass_ratio, tie_ctx=None):
    """indices 集合比对(算子 vs 真值), 与 pytest compare_indices 一致,
    可选并列翻转容忍(tie_ctx)。

    并列翻转容忍(为什么需要): TopK 边界处两池分数差可低至 1e-7,
    远小于算子 cube 侧与 CPU 真值的固有数值偏差(~1e-4), 排名相邻的
    近并列分数取舍会随计算路径翻转 —— 属数值噪声而非选池错误。

    仲裁策略(两级):
      1) 分数仲裁(优先, rv=True 有 values 时): 对差异行提取
         truth-only 池 A(取真值分数) 与 op-only 池 B(取算子分数),
         按分数排序配对, 全部 |sA - sB| <= tie_rtol*max(|sA|,denom_floor)
         则该行判为并列翻转(严格可信);
      2) 结构仲裁(rv=False 无 values 时): 差集等量且 <= max_flip_pools
         判为翻转(启发式), 受 flip_row_ratio 总量比例约束兜底
         (大量翻转仍失败, 防选池错误蒙混)。
      差集不等量/分数差过大/结构翻转超比例 -> 该行失败。

    tie_ctx: None(严格集合比对) 或 dict(
        op_values, truth_values, pool_size, sparse,
        tie_rtol, denom_floor, max_flip_pools, flip_row_ratio)

    返回 (ok, error_info, diag): ok=是否通过; diag=诊断明细。
    """
    op = _to_3d(op_indices)
    truth = _to_3d(truth_indices)
    if op.shape != truth.shape:
        return (
            False,
            (
                f"indices shape mismatch: operator {tuple(op.shape)} vs truth {tuple(truth.shape)}"
            ),
            "",
        )

    # 无效位取值校验: 接口约定无效位置必须精确为 -1
    neg = op[op < 0]
    if neg.numel() > 0 and not bool((neg == -1).all()):
        bad = neg[neg != -1][:8].tolist()
        return False, f"invalid slots must be exactly -1, got {bad}", ""

    # ---- 快速路径: 整批向量化预筛相等行 ----
    # 逐行 Python 循环(排序+set 构造)在大 rows×topk 下每行毫秒级, 10 万行
    # ×topk 8192 总耗时可到 10+ 分钟, 撞 celery compare 队列软时限
    # (SoftTimeLimitExceeded, 与 atk -to 挂钩; 默认 60s 必超)。行内升序排序
    # 后逐元素全等 <=> 两行多重集合相等(有效 token 唯一, 与集合比较等价;
    # 无效位已校验全为 -1, 参与排序不引入差异)。相等行直接判 pass, 仅
    # 差异行走下方逐行差集+仲裁慢路径(差异行通常为 0 或并列翻转少量行)。
    op_sorted, _ = op.sort(dim=-1)
    tr_sorted, _ = truth.sort(dim=-1)
    eq_rows = (op_sorted == tr_sorted).all(dim=-1)  # (B, S1m)

    # values 预处理(分数仲裁用): 空输出(rv=False)时仅剩结构仲裁
    tie_ctx = tie_ctx or {}
    op_vals = tie_ctx.get("op_values")
    truth_vals = tie_ctx.get("truth_values")
    has_values = (
        op_vals is not None
        and truth_vals is not None
        and getattr(op_vals, "numel", lambda: 0)() > 0
        and getattr(truth_vals, "numel", lambda: 0)() > 0
    )
    if has_values:
        op_vals = _to_3d(op_vals).to(torch.float32)
        truth_vals = _to_3d(truth_vals).to(torch.float32)
        if op_vals.shape[:2] != op.shape[:2] or truth_vals.shape[:2] != truth.shape[:2]:
            has_values = False

    pool_size = int(tie_ctx.get("pool_size", 1))
    sparse = int(tie_ctx.get("sparse", 1))
    tie_rtol = float(tie_ctx.get("tie_rtol", 0.0))
    denom_floor = float(tie_ctx.get("denom_floor", 1e-3))
    max_flip_pools = int(tie_ctx.get("max_flip_pools", 2))
    flip_row_ratio = float(tie_ctx.get("flip_row_ratio", 0.0))

    def _row_pool_val_map(idx_row, val_row):
        return dict(_row_pairs(idx_row, val_row, pool_size, sparse))

    total_rows = op.shape[0] * op.shape[1]
    pass_rows = 0
    tie_rows = 0  # 分数仲裁通过的翻转行(严格可信)
    struct_rows = 0  # 结构仲裁通过的翻转行(启发式, 受比例约束)
    first_diffs = []
    eq_mask = eq_rows.reshape(-1).tolist()  # 展平行序与下方双循环一致
    row_iter = 0
    for b in range(op.shape[0]):
        for s in range(op.shape[1]):
            if eq_mask[row_iter]:
                pass_rows += 1
                row_iter += 1
                continue
            row_iter += 1
            g_row = truth[b, s]
            n_row = op[b, s]
            g_valid = g_row[g_row >= 0].sort().values
            n_valid = n_row[n_row >= 0].sort().values
            if g_valid.numel() == n_valid.numel() and torch.equal(g_valid, n_valid):
                pass_rows += 1
                continue
            # ---- 差异行: 翻转仲裁 ----
            # 差集换算到 pool 粒度(ps>1 时一个池的翻转 = ps 个连续 token)
            g_set = set(int(v) // pool_size for v in g_valid.tolist())
            n_set = set(int(v) // pool_size for v in n_valid.tolist())
            only_g = sorted(g_set - n_set)  # truth-only pools A
            only_n = sorted(n_set - g_set)  # op-only pools B
            # 仲裁容量: 有分数仲裁(has_values)时不限翻转池数 —— 逐对校验分数差,
            # 全部并列才容忍, 数学上与大翻转数不矛盾; 随机数据下 relu 产生大量
            # 精确 0 分, topk 边界落入 0 分并列区时单行翻转可达数十池(测例实测
            # 中位 17), 按 max_flip_pools=2 封顶会把合法并列翻转误判为失败。
            # 无分数(rv=False)仅剩结构仲裁, 维持小容量启发式(防选池错误蒙混)。
            if has_values and tie_rtol > 0:
                flip_cap = max(len(only_g), len(only_n))
            else:
                flip_cap = max_flip_pools
            tolerated = False
            if len(only_g) == len(only_n) and 0 < len(only_g) <= flip_cap:
                if has_values and tie_rtol > 0:
                    # 分数仲裁: A 取真值分数, B 取算子分数, 排序配对比分数差
                    t_map = _row_pool_val_map(g_row, truth_vals[b, s])
                    o_map = _row_pool_val_map(n_row, op_vals[b, s])
                    scores_a = sorted(t_map.get(p, float("-inf")) for p in only_g)
                    scores_b = sorted(o_map.get(p, float("-inf")) for p in only_n)
                    if all(
                        math.isfinite(a) and math.isfinite(bb)
                        for a, bb in zip(scores_a, scores_b)
                    ):
                        tolerated = all(
                            abs(a - bb) <= tie_rtol * max(abs(a), denom_floor)
                            for a, bb in zip(scores_a, scores_b)
                        )
                elif not has_values and max_flip_pools > 0:
                    # 结构仲裁: 等量小差集视为翻转(无分数可验证)
                    tolerated = True
            if tolerated:
                if has_values:
                    tie_rows += 1
                else:
                    struct_rows += 1
            elif len(first_diffs) < 3:
                first_diffs.append(
                    f"row({b},{s}): golden-only{only_g[:6]} npu-only{only_n[:6]}"
                )
    fail_rows = total_rows - pass_rows - tie_rows - struct_rows
    ratio = (pass_rows + tie_rows + struct_rows) / total_rows if total_rows > 0 else 0.0
    struct_ratio = struct_rows / total_rows if total_rows > 0 else 0.0
    ok = ratio >= row_pass_ratio and struct_ratio <= flip_row_ratio
    info = (
        f"indices set compare: {pass_rows}/{total_rows} matched"
        f" (+{tie_rows} tie-flip, +{struct_rows} struct-flip)"
        f" ({ratio:.2%})"
    )
    if not ok:
        if struct_ratio > flip_row_ratio:
            info += (
                f"; struct-flip ratio {struct_ratio:.2%} > {flip_row_ratio:.2%}"
                f" (mass flips without value arbitration)"
            )
        info += "; first diffs: " + "; ".join(first_diffs)
    return (
        ok,
        info,
        f"rows={total_rows}, matched={pass_rows}, tie={tie_rows}, struct={struct_rows}",
    )


def _values_double_benchmark(
    op_indices,
    op_values,
    bm_indices,
    bm_values,
    truth_indices,
    truth_values,
    pool_size,
    topk,
    denom_floor,
    abs_floor,
    ratio_thresholds,
):
    """values 按索引排序后的双标杆对比。

    返回 (ok, error_info)。
    """
    sparse = topk // pool_size
    op_idx = _to_3d(op_indices)
    bm_idx = _to_3d(bm_indices)
    truth_idx = _to_3d(truth_indices)

    op_vals = _to_3d(op_values).to(torch.float32)
    bm_vals = _to_3d(bm_values).to(torch.float32)
    truth_vals = _to_3d(truth_values).to(torch.float32)

    if op_vals.shape != truth_vals.shape:
        return False, (
            f"values shape mismatch: operator {tuple(op_vals.shape)} vs truth {tuple(truth_vals.shape)}"
        )

    actual_errors = []
    benchmark_errors = []
    for b in range(truth_idx.shape[0]):
        for s in range(truth_idx.shape[1]):
            truth_pairs = _row_pairs(
                truth_idx[b, s], truth_vals[b, s], pool_size, sparse
            )
            op_pairs = _row_pairs(op_idx[b, s], op_vals[b, s], pool_size, sparse)
            bm_pairs = _row_pairs(bm_idx[b, s], bm_vals[b, s], pool_size, sparse)
            actual_errors.extend(_pair_rel_errors(op_pairs, truth_pairs, denom_floor))
            benchmark_errors.extend(
                _pair_rel_errors(bm_pairs, truth_pairs, denom_floor)
            )

    a_max, a_avg, a_rmse = _metrics_of(actual_errors)
    b_max, b_avg, b_rmse = _metrics_of(benchmark_errors)

    def _effective(value):
        return value if value > abs_floor else 0.0

    checks = []
    ok = True
    for name, a_val, b_val, ratio in (
        ("max_re", a_max, b_max, ratio_thresholds["max_re_ratio"]),
        ("avg_re", a_avg, b_avg, ratio_thresholds["avg_re_ratio"]),
        ("rmse", a_rmse, b_rmse, ratio_thresholds["rmse_ratio"]),
    ):
        eff_a = _effective(a_val)
        eff_b = _effective(b_val)
        if eff_a == 0.0:
            checks.append(f"{name}: actual={a_val:.3e} <= floor(pass)")
        elif eff_b == 0.0:
            ok = False
            checks.append(
                f"{name}: actual={a_val:.3e} > floor but benchmark={b_val:.3e} exact(FAIL)"
            )
        else:
            real_ratio = eff_a / eff_b
            passed = real_ratio <= ratio
            ok = ok and passed
            checks.append(
                f"{name}: actual/benchmark={real_ratio:.3f}"
                f"(a={a_val:.3e}, b={b_val:.3e}, thd={ratio})"
                f"{'pass' if passed else 'FAIL'}"
            )

    info = (
        f"values double benchmark (pairs: actual={len(actual_errors)}, "
        f"benchmark={len(benchmark_errors)}): " + "; ".join(checks)
    )
    return ok, info


@ACCURACY_REGISTRY.register("pki_double_benchmark")
class PKIDoubleBenchmarkCompare(BaseAccuracyCompare):
    """PoolKeyIndexer 双标杆精度比较(indices 集合比对 + values 双标杆)。"""

    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)
        self.max_re_ratio = float(kwargs.get("max_re_ratio", 2.0))
        self.avg_re_ratio = float(kwargs.get("avg_re_ratio", 1.2))
        self.rmse_ratio = float(kwargs.get("rmse_ratio", 1.2))
        self.row_pass_ratio = float(kwargs.get("row_pass_ratio", 1.0))
        self.abs_floor = float(kwargs.get("abs_floor", 1e-5))
        self.denom_floor = float(kwargs.get("denom_floor", 1e-3))
        # 索引并列翻转容忍参数(见 _indices_set_compare docstring)
        self.tie_rtol = float(kwargs.get("tie_rtol", 1e-2))
        self.max_flip_pools = int(kwargs.get("max_flip_pools", 2))
        self.flip_row_ratio = float(kwargs.get("flip_row_ratio", 1e-2))
        # values 精度比对开关: true=跳过 values 比对(仅保留 indices 门禁),
        # 默认 false=正常比对(旧用例 JSON 无该字段时同样回退到 False)
        self.skip_values_compare = bool(kwargs.get("skip_values_compare", False))

    def _load_output(self, output_dir, filename):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return None
        return torch_load_safe(path)

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        """抽象方法实体化(本比较器整体接管 precision_compare_for_case,
        该方法不会被调用, 仅满足 BaseAccuracyCompare 实例化约束)。"""
        return AccuracyConfig(filename=data_file, result=True)

    def precision_compare_for_case(
        self, main_node, remote_node, is_npu=False, device_id=0
    ):
        """三方(算子/标杆/真值)对比, 产出 output_0/output_1 两条精度结果。"""
        operator_dir = get_output_path_by_case(
            main_node.get_output_path(), main_node.get_backend_name(), self.case_config
        )
        benchmark_dir = get_output_path_by_case(
            remote_node.get_output_path(),
            remote_node.get_backend_name(),
            self.case_config,
        )
        benchmark_device = getattr(
            getattr(self.executor, "task_result", None), "benchmark_device", "cpu"
        )
        truth_dir = get_output_path_by_case(
            main_node.get_output_path(),
            BENCHMARK_PATH_PATTERN.format(benchmark_device),
            self.case_config,
        )

        op_indices = self._load_output(operator_dir, "output_0.pt")
        op_values = self._load_output(operator_dir, "output_1.pt")
        bm_indices = self._load_output(benchmark_dir, "output_0.pt")
        bm_values = self._load_output(benchmark_dir, "output_1.pt")
        truth_indices = self._load_output(truth_dir, "output_0.pt")
        truth_values = self._load_output(truth_dir, "output_1.pt")

        if op_indices is None or op_values is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=f"operator output missing in {operator_dir}",
                )
            ]
        if truth_indices is None or truth_values is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=(
                        f"truth(cpu_benchmark) output missing in {truth_dir}; "
                        f"check standard name contains 'benchmark' and bm task ran"
                    ),
                )
            ]
        if bm_indices is None or bm_values is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=f"benchmark node output missing in {benchmark_dir}",
                )
            ]

        # ---- 1. indices 集合比对(算子 vs 真值, 唯一索引门禁, 并列翻转容忍) ----
        topk = int(_case_input_value(self.case_config, "topk", 2048))
        pool_size = int(_case_input_value(self.case_config, "pool_size", 1))
        tie_ctx = dict(
            op_values=op_values,
            truth_values=truth_values,
            pool_size=pool_size,
            sparse=max(1, topk // pool_size),
            tie_rtol=self.tie_rtol,
            denom_floor=self.denom_floor,
            max_flip_pools=self.max_flip_pools,
            flip_row_ratio=self.flip_row_ratio,
        )
        idx_ok, idx_info, _ = _indices_set_compare(
            op_indices, truth_indices, self.row_pass_ratio, tie_ctx=tie_ctx
        )

        # 布局合同比对(问题2 前缀消费合同门禁): -1 掩码逐位一致 + 尾 token
        # 值精确一致, 对 TopK 并列翻转免疫; V1/V2 布局版本取自用例声明
        sparse_indices_mode = int(
            _case_input_value(self.case_config, "sparse_indices_mode", 0)
        )
        if idx_ok:
            lay_ok, lay_info = _indices_layout_compare(
                op_indices, truth_indices, topk, pool_size, sparse_indices_mode
            )
            if not lay_ok:
                idx_ok = False
                idx_info = f"{idx_info}; {lay_info}"

        # 标杆 vs 真值集合匹配率(仅诊断, 不判定)
        bm_ok, bm_diag, _ = _indices_set_compare(
            bm_indices, truth_indices, self.row_pass_ratio
        )

        idx_result = AccuracyConfig(
            filename="output_0.pt",
            result=idx_ok,
            error_info=idx_info + f"; [ref] bm-vs-truth: {bm_diag} (reference only)",
        )

        # ---- 2. values 按索引排序后双标杆对比 ----

        if self.skip_values_compare:
            val_result = AccuracyConfig(
                filename="output_1.pt",
                result=True,
                error_info="values compare disabled by config (skip_values_compare=true)",
            )
        elif truth_values.numel() == 0 and op_values.numel() == 0:
            val_result = AccuracyConfig(
                filename="output_1.pt",
                result=True,
                error_info="values empty (return_value=False), skip compare",
            )
        else:
            val_ok, val_info = _values_double_benchmark(
                op_indices,
                op_values,
                bm_indices,
                bm_values,
                truth_indices,
                truth_values,
                pool_size,
                topk,
                self.denom_floor,
                self.abs_floor,
                {
                    "max_re_ratio": self.max_re_ratio,
                    "avg_re_ratio": self.avg_re_ratio,
                    "rmse_ratio": self.rmse_ratio,
                },
            )
            val_result = AccuracyConfig(
                filename="output_1.pt",
                result=val_ok,
                error_info=val_info,
            )

        logging.info(
            f"[case {self.case_config.id}] pki_double_benchmark: "
            f"indices={'PASS' if idx_ok else 'FAIL'} | {idx_result.error_info}"
        )
        logging.info(
            f"[case {self.case_config.id}] pki_double_benchmark: "
            f"values={'PASS' if val_result.result else 'FAIL'} | {val_result.error_info}"
        )
        return [idx_result, val_result]


# ============================================================================
# torch 接口执行器(torch_pool_key_indexer)
#
# 与 aclnn 直调套件互补的 torch dispatcher 链路覆盖: npu 主节点调 torch 公共入口
# cann_ops_transformer.pool_key_indexer(V1: sparse_indices_mode=0 /
# V2: sparse_indices_mode=1, 由用例声明驱动), 内部 JIT 桥接并路由
# aclnnPoolKeyIndexer / aclnnPoolKeyIndexerV2; cpu 真值与标杆路径复用
# PKIApi 统一标杆。精度判定沿用 pki_double_benchmark(indices vs CPU 真值门禁,
# skip_values_compare=true 下 values 比对关闭)。
#
# 主节点/标杆节点区分: torch 路径仅在主(算子)节点生效。以 task_result 上的
# 节点名属性判定(torch 套件 node.yaml: 算子节点名 torch_api / 标杆节点名 npu);
# 属性探测失败时回退 device 判据(双 npu 节点均走 torch 路径, indices 门禁
# 仍以 CPU 真值为准, 判定有效性不受影响, 首次调用时打点日志留痕)。
# ============================================================================

_TORCH_PKI_ENTRY = None
_TORCH_PKI_ROLE_LOGGED = False


def _torch_pki_entry():
    """懒加载 torch 公共入口: 优先 vendor 子包(最新代码), 回退整包。"""
    global _TORCH_PKI_ENTRY
    if _TORCH_PKI_ENTRY is None:
        entry = None
        try:
            import cann_ops_transformer_gjw_pki as _sub

            entry = _sub.pool_key_indexer
            logging.info(
                "[torch_pki] entry = cann_ops_transformer_gjw_pki.pool_key_indexer"
            )
        except Exception:
            pass
        if entry is None:
            import cann_ops_transformer as _cot

            entry = _cot.pool_key_indexer
            logging.info("[torch_pki] entry = cann_ops_transformer.pool_key_indexer")
        _TORCH_PKI_ENTRY = entry
    return _TORCH_PKI_ENTRY


def _torch_pki_node_name(task_result):
    """探测 benchmark/标杆角色判定用的节点名属性。"""
    for attr in ("node_name", "node", "name", "node_id"):
        value = getattr(task_result, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _torch_descale(t, quant):
    """mxFP8(mode=1) torch 链路 descale 视图: uint8 raw -> float8_e8m0fnu
    (torch 公共入口校验 descale dtype 必须为 float8_e8m0fnu, 与 aclnn 适配器的
    view 口径一致); 其余模式原样返回。"""
    if quant == 1 and t is not None and hasattr(t, "view") and t.dtype == torch.uint8:
        return t.view(torch.float8_e8m0fnu)
    return t


@register("torch_pool_key_indexer")
class PKITorchApi(PKIApi):
    """PoolKeyIndexer torch 接口执行器(npu=torch 公共入口 / 其他=统一标杆)。"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        global _TORCH_PKI_ROLE_LOGGED
        device = getattr(self, "device", None)
        node_name = _torch_pki_node_name(self.task_result)
        if not _TORCH_PKI_ROLE_LOGGED:
            _TORCH_PKI_ROLE_LOGGED = True
            logging.info(
                f"[torch_pki] role probe: device={device} node_name={node_name!r} "
                f"task_result_attrs={[a for a in dir(self.task_result) if not a.startswith('_')]}"
            )
        # 主节点(npu + 非标杆节点名)走 torch 接口; 标杆/真值路径回落统一标杆
        is_benchmark_node = node_name in ("npu", "benchmark", "cpu")
        if device == "npu" and not is_benchmark_node:
            prep = prepare_pki_inputs(input_data)
            if getattr(self.task_result, "is_get_output_task", False):
                return pki_output_stubs(prep)
            if _is_pki_abnormal_case(self.task_result):
                return pki_output_stubs(prep)
            entry = _torch_pki_entry()
            quant = prep["quant_mode"]
            logging.info(
                f"[torch_pki] exec via torch interface: layout_q={prep['layout_q']} "
                f"layout_k={prep['layout_k']} topk={prep['topk']} ps={prep['pool_size']} "
                f"quant={quant} sparse_mode={prep.get('sparse_indices_mode', 0)}"
            )
            indices, values = entry(
                prep["query"],
                prep["pool_key"],
                prep["weights"],
                prep["pool_tail_k"],
                actual_seq_q=prep["actual_seq_q"],
                actual_seq_k=prep["actual_seq_k"],
                block_table=prep["block_table"],
                q_descale=_torch_descale(prep["q_descale"], quant),
                k_descale=_torch_descale(prep["k_descale"], quant),
                layout_q=prep["layout_q"],
                layout_k=prep["layout_k"],
                topk=prep["topk"],
                pool_size=prep["pool_size"],
                mask_mode=prep["mask_mode"],
                quant_mode=quant,
                return_value=prep["return_value"],
                sparse_indices_mode=prep.get("sparse_indices_mode", 0),
            )
            return indices, values
        return super().__call__(input_data, with_output)
