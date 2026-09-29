# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CPU reference for ORI_SPARSE, ORI_CMP_SPARSE and CMP_SPARSE.

Packed KV decodes to shared K/V in logical [nope448, rope64] order.
Each query gathers the specified optional TopK prefix (all K columns if omitted)
(zero-length sides are skipped; empty queries return zero and LSE=sinks). Attention uses
online softmax staging (sinks seed the running max/sum; per-tile running
max/sum with BF16 probability staging and FP32 accumulation).
"""

import math

import test_constants as C

FP8_E4M3_MAX = 448.0  # fp8 e4m3fn 最大可表示幅值
FP4_E2M1_MAX = 6.0  # fp4 e2m1 最大可表示幅值（OCP MX 无 inf/NaN）
# e2m1 全部 16 个码点值（按码点序 S-E-E-M；全部在 bf16 精确可表示）
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _torch():
    """惰性 import torch（golden 用 torch 的 fp8/bf16 dtype；测试侧 importorskip）。"""
    import torch

    return torch


# =============================================================================
# 逐段 golden helper（各自独立可调用，支撑分段对拍）
# =============================================================================


def e2m1_encode(n_f32):
    """f32 → e2m1 码点（uint8）。就近舍入；平局取**幅值较小**者
    （`argmin(|n - E2M1_VALUES|)` 首个最小下标，= 上游 `mx_quant_fp4_tool` 约定）。"""
    torch = _torch()
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    # 分块仅限制16倍距离表的临时内存；argmin及平局选择与整块计算完全一致。
    # golden 一律在 CPU 上计算：输入先搬回 CPU，避免在 NPU 上调用小算子。
    flat = n_f32.cpu().reshape(-1)
    codes = torch.empty(flat.shape, dtype=torch.uint8)
    for start in range(0, flat.numel(), 1 << 20):
        chunk = flat[start : start + (1 << 20)]
        dist = (chunk.unsqueeze(-1) - table).abs()
        codes[start : start + chunk.numel()] = dist.argmin(dim=-1).to(torch.uint8)
    return codes.reshape(n_f32.shape)


def e2m1_decode(enc):
    """e2m1 码点 → f32 值（查表）。"""
    torch = _torch()
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    return table[enc.long()].float()


def pack_fp4_to_uint8(enc):
    """e2m1 码点 [..., 2m] → 打包字节 [..., m]：**偶元素在低 nibble、奇元素在高位**
    （= 上游 `mx_quant_fp4_tool.pack_fp4_to_uint8`：`packed = (enc[2k+1]<<4) | enc[2k]`）。"""
    torch = _torch()
    enc = enc.to(torch.uint8)
    return (enc[..., 1::2] << 4) | enc[..., 0::2]


def unpack_uint8_to_fp4(packed):
    """打包字节 → e2m1 码点（`pack_fp4_to_uint8` 的逆）：低 nibble = 偶元素。"""
    torch = _torch()
    packed = packed.cpu()  # golden 只在 CPU 上计算
    out = torch.zeros(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = (packed >> 4) & 0x0F
    return out


def quantize_group(vec_f32, side):
    """按侧分组量化整行 512 元素（物理序 nope[448]‖rope[64]）。

    side="ori" → group32 FP8 E4M3：返回 (fp8 张量 [S,512], scale bf16 [S,16])。
    side="cmp" → group16 FP4 E2M1（x2 打包）：返回 (packed uint8 [S,256], scale bf16 [S,32])。
    scale = amax/fmax 转 bf16；量化用 bf16 回读的 scale（与 kernel 数值流一致）。
    """
    torch = _torch()
    assert side in ("ori", "cmp")
    rows = vec_f32.shape[0]
    g, ng = (
        (C.GROUP_SIZE_ORI, C.NUM_GROUPS_ORI)
        if side == "ori"
        else (C.GROUP_SIZE_CMP, C.NUM_GROUPS_CMP)
    )
    grp = vec_f32.reshape(rows, ng, g)
    amax = grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [rows,ng,1]
    fmax = FP8_E4M3_MAX if side == "ori" else FP4_E2M1_MAX
    scale_bf16 = (amax / fmax).to(torch.bfloat16)
    scale_used = scale_bf16.float().clamp(min=1e-30)
    n = (grp / scale_used).clamp(-fmax, fmax)
    if side == "ori":
        q = n.to(torch.float8_e4m3fn)  # torch CPU 支持
        return q.reshape(rows, ng * g), scale_bf16.reshape(rows, ng)
    enc = e2m1_encode(n)  # [rows,ng,g] uint8 码点
    packed = pack_fp4_to_uint8(enc)  # [rows,ng,g/2]
    return packed.reshape(rows, ng * g // 2), scale_bf16.reshape(rows, ng)


def dequant_group(data, scale, side):
    """ori group32 / cmp group16 反量化 → bf16 [rows,512]（物理序 nope‖rope）。

    kernel 侧数值流：ori 走 fp8→f32→bf16（cast 精确/RN）→ vmul bf16；cmp 走 LUT 查表
    （e2m1 值在 bf16 精确）→ vmul bf16。两者等价于 bf16(码值) × bf16(scale) → RN bf16。"""
    torch = _torch()
    assert side in ("ori", "cmp")
    rows = scale.shape[0]
    g, ng = (
        (C.GROUP_SIZE_ORI, C.NUM_GROUPS_ORI)
        if side == "ori"
        else (C.GROUP_SIZE_CMP, C.NUM_GROUPS_CMP)
    )
    if side == "ori":
        k_bf16 = data.float().to(torch.bfloat16).reshape(rows, ng, g)
    else:
        codes = unpack_uint8_to_fp4(data.reshape(rows, ng * g // 2))
        k_bf16 = e2m1_decode(codes).to(torch.bfloat16).reshape(rows, ng, g)
    deq = (k_bf16 * scale.float().reshape(rows, ng, 1)).to(torch.bfloat16)
    return deq.reshape(rows, ng * g)


def kv_dequant(kv):
    """直接反量化为计算序 [nope448 ‖ rope64]，物理与逻辑顺序一致。"""
    return dequant_group(kv["data"], kv["scale"], kv["side"])


def _topk_lengths(total_queries, width, topk_length):
    """Normalize [T1]/[T1,1] prefixes; omission means full index width."""
    torch = _torch()
    if topk_length is None:
        return torch.full((total_queries,), width, dtype=torch.int32)
    lengths = torch.as_tensor(topk_length)
    if lengths.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("topk_length must contain integers")
    if lengths.ndim != 1 and not (lengths.ndim == 2 and lengths.shape[1] == 1):
        raise ValueError("topk_length must have shape [T1] or [T1,1]")
    lengths = lengths.reshape(-1)
    if lengths.numel() != total_queries:
        raise ValueError(f"topk_length must contain {total_queries} query lengths")
    if (lengths < 0).any() or (lengths > width).any():
        raise ValueError("topk_length must be between zero and index table width")
    return lengths.to(torch.int32)


def mqsmla_cpu_benchmark(ref, scenario=None, quant_mode=None, return_softmax_lse=False):
    """CPU flash attention: gather one KV tile, QK, online softmax, then PV.

    Follows the reference golden's RUN_MODE=1 recurrence. Sinks seed max/sum;
    probabilities round to BF16 before PV, while accumulation stays FP32.
    Only a [N1, TILE_N] score tile is materialized, never full-query scores.
    """
    torch = _torch()
    scenario = ref["scenario"] if scenario is None else scenario
    _validate_scenario(scenario)
    quant_mode = ref["quant_mode"] if quant_mode is None else quant_mode
    if int(quant_mode) != int(C.QuantMode.CONTIGUOUS):
        raise ValueError("only quant_mode=1 is supported")
    sides = scenario_sides(scenario)
    decoded = {side: kv_dequant(ref[side]) for side in sides}
    lengths = {
        side: _topk_lengths(
            ref["T1"],
            ref[f"{side}_sparse_indices"].shape[1],
            ref[f"{side}_topk_length"],
        )
        for side in sides
    }
    output = torch.empty_like(ref["q"])
    lse = torch.empty((1, ref["T1"], ref["n1"]), dtype=torch.float32)
    for row in range(ref["T1"]):
        if ref["T1"] >= 8192 and row % 8192 == 0:
            print(f"[mqsmla CPU golden] query {row}/{ref['T1']}", flush=True)
        batch = int(ref["query_batch"][row])
        q = ref["q"][row].float()
        row_max = ref["sinks"].clone()
        row_sum = torch.ones(ref["n1"], dtype=torch.float32)
        acc = torch.zeros((ref["n1"], C.D), dtype=torch.float32)
        for side in sides:
            width = int(lengths[side][row])
            indices = ref[f"{side}_sparse_indices"][row, :width].long()
            pool_length = int(ref[f"{side}_seqlens"][batch])
            if ((indices < 0) | (indices >= pool_length)).any():
                raise ValueError("active sparse index exceeds KV pool length")
            offset = ref[f"{side}_offsets"][batch]
            for start in range(0, width, C.TILE_N):
                kv = decoded[side][offset + indices[start : start + C.TILE_N]].float()
                scores = (q @ kv.T) * ref["softmax_scale"]
                # The device's masked reduction includes zero-valued lanes.
                tile_max = scores.amax(dim=-1).clamp_min(0.0)
                new_max = torch.maximum(row_max, tile_max)
                rescale = torch.exp(row_max - new_max)
                probabilities = torch.exp(scores - new_max[:, None])
                row_sum = rescale * row_sum + probabilities.sum(dim=-1)
                acc = (
                    acc * rescale[:, None]
                    + probabilities.to(torch.bfloat16).float() @ kv
                )
                row_max = new_max
        output[row] = (acc / row_sum[:, None]).to(torch.bfloat16)
        lse[0, row] = row_max + torch.log(row_sum)
    return output, lse if return_softmax_lse else torch.empty((0,), dtype=torch.float32)


# =============================================================================
# 输入构造
# =============================================================================


def _validate_scenario(scenario):
    if scenario not in (
        C.Scenario.ORI_SPARSE,
        C.Scenario.ORI_CMP_SPARSE,
        C.Scenario.CMP_SPARSE,
    ):
        raise ValueError("only ORI_SPARSE, ORI_CMP_SPARSE and CMP_SPARSE are supported")


def scenario_sides(scenario):
    """Active KV sides per scenario: ori-only, ori+cmp, or cmp-only."""
    _validate_scenario(scenario)
    if scenario == C.Scenario.ORI_SPARSE:
        return ("ori",)
    if scenario == C.Scenario.CMP_SPARSE:
        return ("cmp",)
    return ("ori", "cmp")


def query_lengths(batch_size, seqlen_q, cu_seqlens_q=None, seqused_q=None):
    """Resolve unpadded TND lengths; explicit query metadata overrides S1."""
    torch = _torch()

    def vector(value, name, size):
        tensor = torch.as_tensor(value, device="cpu")
        if tensor.ndim != 1 or tensor.numel() != size:
            raise ValueError(
                f"{name} must be a one-dimensional vector of {size} integers"
            )
        if tensor.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(f"{name} must contain integers")
        tensor = tensor.to(torch.int64)
        if (tensor < 0).any() or (tensor > torch.iinfo(torch.int32).max).any():
            raise ValueError(f"{name} must contain non-negative int32 values")
        return tensor

    used = None if seqused_q is None else vector(seqused_q, "seqused_q", batch_size)
    if cu_seqlens_q is not None:
        cumulative = vector(cu_seqlens_q, "cu_seqlens_q", batch_size + 1)
        if cumulative[0] != 0:
            raise ValueError("cu_seqlens_q must start at 0")
        lengths = cumulative.diff()
        if used is not None and not torch.equal(lengths, used):
            raise ValueError(
                "seqused_q must equal adjacent differences of cu_seqlens_q (no query padding)"
            )
    elif used is not None:
        lengths = used
    else:
        value = torch.as_tensor(seqlen_q, device="cpu")
        if value.ndim == 0:
            value = value.repeat(batch_size)
        lengths = vector(value, "S1", batch_size)
    if (lengths <= 0).any():
        raise ValueError("query lengths must be positive for every batch")
    if lengths.sum() > torch.iinfo(torch.int32).max:
        raise ValueError("total query length must fit in int32")
    cumulative = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))).to(
        torch.int32
    )
    return (
        lengths.to(torch.int32),
        cumulative,
        None if used is None else used.to(torch.int32),
    )


def make_inputs(
    scenario=C.Scenario.ORI_SPARSE,
    quant_mode=C.QuantMode.CONTIGUOUS,
    layout_q="TND",
    layout_kv="PA_BBND",
    batch_size=1,
    seqlen_q=64,
    seqlen_kv=128,
    n1=64,
    cmp_seqlen=128,
    cmp_topk=16,
    seed=0,
    dist="norm",
    ori_topk=0,
    ori_topk_length=None,
    cmp_topk_length=None,
    ori_kv_topk_mode="fullK",
    cmp_kv_topk_mode="fullK",
    cu_seqlens_q=None,
    seqused_q=None,
):
    """Generate all batches together in TND order, including sparse tables.

    Sequence lengths accept a scalar or B lengths. KV pools contain the sum
    of batch lengths; sparse indices remain batch-local logical token indices.
    Random indices sample with replacement, so generation needs only [T1,K]
    storage. Explicit lengths override the mode; otherwise fullK uses K
    and random generates a positive prefix length per query.
    CMP_SPARSE builds only the cmp pool (S2C/K2); seqlen_kv/ori_topk are
    unused there.
    """
    torch = _torch()
    _validate_scenario(scenario)
    if int(quant_mode) != int(C.QuantMode.CONTIGUOUS):
        raise ValueError("only quant_mode=1 is supported")
    if layout_q != "TND" or layout_kv != "PA_BBND":
        raise ValueError("only layout_q=TND and layout_kv=PA_BBND are supported")

    def batch_lengths(value):
        lengths = torch.as_tensor(value, dtype=torch.int32).reshape(-1)
        if lengths.numel() == 1:
            lengths = lengths.repeat(batch_size)
        if lengths.numel() != batch_size or (lengths <= 0).any():
            raise ValueError("sequence lengths must be positive and contain B entries")
        return lengths

    q_lengths, cu_seqlens_q, seqused_q = query_lengths(
        batch_size, seqlen_q, cu_seqlens_q, seqused_q
    )
    sides = scenario_sides(scenario)
    query_batch = torch.repeat_interleave(torch.arange(batch_size), q_lengths.long())
    T1, D = int(q_lengths.sum()), C.D
    g = torch.Generator().manual_seed(seed)

    if dist != "norm":
        raise ValueError(f"dist supports only 'norm', got {dist!r}")

    def _rand(*shape):
        """使用固定seed的正态分布生成输入。"""
        return torch.randn(*shape, generator=g)

    # Generate directly into the final BF16 storage in bounded query chunks.
    # The largest white-box cases have 262144 queries; a full-size FP32 randn
    # temporary alongside q and its golden exceeds host memory during batch_save.
    q = torch.empty((T1, n1, D), dtype=torch.bfloat16)
    for q_start in range(0, T1, 8192):
        q_stop = min(q_start + 8192, T1)
        q_f32 = _rand(q_stop - q_start, n1, D)
        q_f32.mul_(0.5)
        q[q_start:q_stop] = q_f32
    del q_f32
    sinks = _rand(n1).float()
    ref = {
        "scenario": scenario,
        "quant_mode": quant_mode,
        "B": batch_size,
        "T1": T1,
        "D": D,
        "n1": n1,
        "q": q,
        "sinks": sinks,
        "softmax_scale": 1.0 / math.sqrt(D),
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "query_batch": query_batch,
    }

    def make_side(side, pool_lengths, width, topk_length, variant):
        if topk_length is None and variant not in ("fullK", "random"):
            raise ValueError("topk mode must be fullK or random")
        offsets = pool_lengths.cumsum(0) - pool_lengths
        vec = _rand(int(pool_lengths.sum()), D)
        vec.mul_(0.5)
        # 保持既有seed对应的逻辑KV值：历史随机流先生成rope，再生成nope。
        # 量化前整理为公共nope/rope顺序，scale也按此顺序生成。
        vec = torch.cat((vec[:, C.D_ROPE :], vec[:, : C.D_ROPE]), dim=-1)
        data, scale = quantize_group(vec, side)
        ref[side] = {"side": side, "data": data, "scale": scale}
        ref[f"{side}_seqlens"] = pool_lengths
        ref[f"{side}_offsets"] = offsets
        pool_per_query = pool_lengths[query_batch]
        sparse_rng = torch.Generator().manual_seed(
            seed * 1000003 + (17 if side == "ori" else 91)
        )
        if topk_length is not None:
            lengths = _topk_lengths(T1, width, topk_length)
        elif variant == "random":
            capacity = pool_per_query.clamp_max(width)
            if (capacity == 0).any():
                raise ValueError("random topk requires a positive index width")
            lengths = (torch.rand(T1, generator=sparse_rng) * capacity).to(
                torch.int32
            ) + 1
        else:
            lengths = torch.full((T1,), width, dtype=torch.int32)
        indices = (
            torch.rand((T1, width), generator=sparse_rng) * pool_per_query[:, None]
        ).to(torch.int32)
        indices.masked_fill_(torch.arange(width)[None, :] >= lengths[:, None], -1)
        ref[f"{side}_sparse_indices"] = indices
        ref[f"{side}_topk_length"] = lengths

    if "ori" in sides:
        make_side(
            "ori", batch_lengths(seqlen_kv), ori_topk, ori_topk_length, ori_kv_topk_mode
        )
    if "cmp" in sides:
        make_side(
            "cmp",
            batch_lengths(cmp_seqlen),
            cmp_topk,
            cmp_topk_length,
            cmp_kv_topk_mode,
        )
    inputs = {
        "q": q,
        "sinks": sinks,
        "quant_mode": int(quant_mode),
        "softmax_scale": ref["softmax_scale"],
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
    }
    return inputs, ref


# =============================================================================
# v0 PA_BBND 物理布局打包（新量化 v2；契约见 mixed_quant_sparse_flash_mla.md）
# =============================================================================
#
# PA 把 KV 存成物理块，用 blockTable 做逻辑→物理映射。取 token s2Idx（参考 GetKeyOffset isPa :306-318）：
#   blkIdx=s2Idx/blockSize; off=s2Idx%blockSize; phys=blockTable[blkIdx]; keyOffset=phys*paBlockStride + off*dSizeVInput。
# 本 golden 把「物理块 stride」化简为「物理行 = phys*blockSize + off」（paBlockStride=blockSize*rowBytes），
# 于是 PA 寻址 = W6 行索引 + 一次 blockTable 标量查表。逻辑 K̃（dequant 值）与布局无关。
#
# ★ 新量化 v2（quant_mode=1 唯一编码，两池行宽**分家**）：
#   ori_kv (FP8 E4M3)    544 B/token = nope[448]+rope[64] 全 fp8 @0..512 ‖ scale 16×bf16 @512
#   cmp_kv (FP4 E2M1 x2) 320 B/token = nope[448]+rope[64] 全 fp4x2 @0..256 ‖ scale 32×bf16 @256
#   物理字节序 = nope[448] ‖ rope[64]；scale i 覆盖物理序元素 [g*i, g*(i+1))，无 pad 字段。
#   ori 的 g=32，cmp 的 g=16。


def _kv_row_bytes(kv):
    """逻辑 KV 一段 → 物理**逻辑行**（uint8），行宽由 `kv["side"]` 携带：

    ori → ``{"row": [S,544]}``（512 B fp8 数据 ‖ 16×bf16 scale）。
    cmp → ``{"row": [S,320]}``（256 B fp4x2 打包数据 ‖ 32×bf16 scale）。"""
    torch = _torch()
    side = kv["side"]
    data_b = kv["data"].contiguous().view(torch.uint8)  # ori [S,512] / cmp [S,256]
    scale_b = kv["scale"].contiguous().view(torch.uint8)  # ori [S,32] / cmp [S,64]
    S, data_len = data_b.shape
    scale_bytes = C.KV_SCALE_BYTES_ORI if side == "ori" else C.KV_SCALE_BYTES_CMP
    row = torch.zeros(S, data_len + scale_bytes, dtype=torch.uint8)
    row[:, :data_len] = data_b
    row[:, data_len:] = scale_b
    assert row.shape[1] == (C.KV_ROW_BYTES_ORI if side == "ori" else C.KV_ROW_BYTES_CMP)
    return {"row": row}


def pack_pa_layout(
    ref, quant_mode=None, block_size=64, spare_blocks=2, seed=17, cmp_block_size=None
):
    """Pack all batches directly into one shuffled physical pool per side.

    Block tables have shape [B, max_blocks]. Unused physical rows contain
    random bytes, and each batch owns distinct blocks even at partial tails.
    """
    torch = _torch()
    quant_mode = ref["quant_mode"] if quant_mode is None else quant_mode
    assert int(quant_mode) == int(C.QuantMode.CONTIGUOUS)
    g = torch.Generator().manual_seed(seed)
    out = {}
    sides = scenario_sides(ref["scenario"])
    for side in sides:
        bs = (
            block_size
            if side == "ori" or cmp_block_size is None
            else int(cmp_block_size)
        )
        lengths = ref[f"{side}_seqlens"].long()
        blocks = (lengths + bs - 1) // bs
        block_offsets = blocks.cumsum(0) - blocks
        total_blocks = int(blocks.sum()) + spare_blocks
        permutation = torch.randperm(total_blocks, generator=g)
        columns = torch.arange(int(blocks.max()))[None, :]
        valid = columns < blocks[:, None]
        table = torch.zeros((ref["B"], int(blocks.max())), dtype=torch.int32)
        table[valid] = permutation[(block_offsets[:, None] + columns)[valid]].to(
            torch.int32
        )
        logical = _kv_row_bytes(ref[side])["row"]
        physical = torch.randint(
            0,
            256,
            (total_blocks * bs, logical.shape[1]),
            generator=g,
            dtype=torch.uint8,
        )
        batch = torch.repeat_interleave(torch.arange(ref["B"]), lengths)
        token = torch.arange(int(lengths.sum())) - ref[f"{side}_offsets"][batch]
        physical_row = table[batch, token // bs].long() * bs + token % bs
        physical[physical_row] = logical
        out[side] = {
            "phys": {"row": physical},
            "block_table": table,
            "block_size": bs,
            "num_phys_blocks": total_blocks,
            "max_block_num_per_batch": table.shape[1],
        }
    return out
