# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import math
import torch_npu
from torchair.configs.compiler_config import CompilerConfig
import torchair as tng
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)
# ==============================================================================
# 配置区
# ==============================================================================
# GRAPH_PATH: 0=单算子, 5=动态图, 7=aclgraph
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 16
N_kv = 2
D = 128

ACTUAL_SEQ_Q = [32 * 1024]
ACTUAL_SEQ_KV = [32 * 1024]

# Layout 选择
INPUT_LAYOUT = "NTD_TND"
OUTPUT_LAYOUT = "TND"
Q_SCALE_LAYOUT = "NT"

# PA KV Cache Layout
KV_CACHE_LAYOUT = "BnNBsD"

# Data Range
Q_DATA_RANGE = (-400, 400)
K_DATA_RANGE = (-400, 400)
V_DATA_RANGE = (-400, 400)

ENABLE_PA = True
ENABLE_LSE = True
BLOCK_SIZE = 128
SPARSE_MODE = 3
SCALE_VALUE = None
IS_CONTIGUOUS = True

# Seed
SEED_Q = 54
SEED_K = 3
SEED_V = 20
SEED_BLOCK_TABLE = 1234
FP8_DTYPE = torch.float8_e4m3fn
OUTPUT_DETYPE = torch.bfloat16
P_SCALE = 1.0
EPSILON = 1e-20

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256

SAVE_PT = False
SAVE_PT_DIR = ""

# 重复运行次数，每次重新生成输入数据
RUN_TIMES = 5


# ==============================================================================
# 数据生成函数
# ==============================================================================
def get_fp8_per_token_head_quant_scale(tensor):
    """
    用于生成 query/key quant scale
    per-token-head quant scale: shape (B, N, S, 1)
    """
    tensor = tensor.contiguous()
    B, N, S, _ = tensor.shape
    fp8_e4m3_max = 448.0
    row_max = torch.abs(tensor).max(dim=3, keepdim=True).values
    row_max = torch.max(row_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / row_max
    return scale.view(B, N, S, 1).float().contiguous()


def get_fp8_per_head_quant_scale(tensor):
    """
    用于生成 value quant scale
    per-head quant scale: shape (1, N, 1, 1)
    """
    tensor = tensor.contiguous()
    fp8_e4m3_max = 448.0
    head_max = torch.abs(tensor).amax(dim=(0, 2, 3), keepdim=True)
    head_max = torch.max(head_max, torch.tensor(1e-8, device=tensor.device))
    scale = fp8_e4m3_max / head_max
    return scale.float().contiguous()


def quant_fp16_to_fp8(tensor, scale):
    """将 fp16 数据量化为 fp8_e4m3"""
    tensor = tensor.contiguous()
    scale = scale.contiguous()
    result = tensor.float() * scale
    result = torch.clamp(result, -448.0, 448.0)
    return result.to(FP8_DTYPE).contiguous()


def create_block_table(actual_seq_kv, block_size, seed=SEED_BLOCK_TABLE, run_idx=0):
    """创建 block table"""
    block_num_per_batch = [
        math.ceil(int(seq_len) / block_size) for seq_len in actual_seq_kv
    ]
    total_blocks = sum(block_num_per_batch)
    max_blocks = max(block_num_per_batch)
    block_idx_list = np.random.default_rng(seed + run_idx).permutation(
        np.arange(total_blocks, dtype=np.int32)
    )
    block_table = np.full((len(actual_seq_kv), max_blocks), -1, dtype=np.int32)
    idx = 0

    for b_index, block_num in enumerate(block_num_per_batch):
        block_table[b_index, :block_num] = block_idx_list[idx : idx + block_num]
        idx += block_num
    return block_table


def bnsd_to_k_cache(k_fp8_bnsd, k_scale_fp32_bnsd, seq_lens, block_size, block_table):
    """BNSD to PA K cache, with k scale (fp32) stored in the 4 extra rows"""
    k_fp8_bnsd = k_fp8_bnsd.contiguous()
    k_scale_fp32_bnsd = k_scale_fp32_bnsd.contiguous()
    B_dim, N_dim, S_dim, D_dim = k_fp8_bnsd.shape
    scale_rows = 4
    block_num_per_seq = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_seq)

    cache = torch.zeros(
        (total_blocks, N_dim, block_size + scale_rows, D_dim),
        dtype=torch.uint8,
        device=k_fp8_bnsd.device,
    ).contiguous()

    for b in range(B_dim):
        bid_table = block_table[b]
        for blk_idx in range(block_num_per_seq[b]):
            blockid = int(bid_table[blk_idx])
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_lens[b])
            valid = end_s - start_s
            if valid <= 0:
                continue
            k_data = k_fp8_bnsd[b, :, start_s:end_s, :].contiguous()
            cache[blockid, :, :valid, :] = k_data.view(torch.uint8)
            scales_all = k_scale_fp32_bnsd[
                b, :, start_s:end_s, 0
            ].contiguous()  # (N, valid)
            scale_buf = torch.zeros(
                N_dim, scale_rows, D_dim // 4, dtype=torch.float32, device=cache.device
            )
            flat_scale = scale_buf.reshape(N_dim, -1)  # (N, 128)
            if valid <= flat_scale.shape[1]:
                flat_scale[:, :valid] = scales_all
            cache[blockid, :, block_size : block_size + scale_rows, :] = scale_buf.view(
                torch.uint8
            ).reshape(N_dim, scale_rows, D_dim)

    return (
        cache.view(FP8_DTYPE)
        .reshape(total_blocks, N_dim, block_size + scale_rows, D_dim)
        .contiguous()
    )


def bnsd_to_v_cache(tensor_bnsd, seq_lens, block_size, block_table):
    """BNSD to V cache - V cache 使用 FP8 类型"""
    tensor_bnsd = tensor_bnsd.contiguous()
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)

    # V cache 使用 FP8 类型
    out_cache = torch.zeros(
        (total_blocks, heads, block_size + 4, dim), dtype=FP8_DTYPE, device=device
    ).contiguous()

    for b in range(batch):
        for blk_idx in range(block_num_per_batch[b]):
            block_id = int(block_table[b, blk_idx].item())
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[block_id, :, :valid_len, :] = tensor_bnsd[
                b, :, block_offset : block_offset + valid_len, :
            ].contiguous()

    return out_cache.contiguous()


def generate_data(run_idx=0):
    """生成 BNSD FP16 Q/K/V 并做 FP8 量化

    run_idx: 第几次运行，用于偏移随机种子，保证每次调用产生不同的输入数据
    """
    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV) if max(ACTUAL_SEQ_KV) > 0 else 1
    logger.info(f"[INFO] [run {run_idx}] max_sq={max_sq}, max_skv={max_skv}")

    # 使用随机数据
    np.random.seed(SEED_Q + run_idx)
    q_amp_hi = max(abs(Q_DATA_RANGE[0]), abs(Q_DATA_RANGE[1]))
    q_amp_lo = q_amp_hi * 0.01
    q_token_amps = np.power(
        10.0,
        np.random.uniform(
            np.log10(q_amp_lo), np.log10(q_amp_hi), size=(B, N_q, max_sq, 1)
        ),
    ).astype(np.float32)  # (B, N_q, max_sq, 1)
    q_base = np.random.uniform(low=-1.0, high=1.0, size=(B, N_q, max_sq, D)).astype(
        np.float32
    )
    q_data = (q_base * q_token_amps).astype(np.float16)
    q_fp16 = torch.from_numpy(q_data)

    np.random.seed(SEED_K + run_idx)
    k_amp_hi = max(abs(K_DATA_RANGE[0]), abs(K_DATA_RANGE[1]))
    k_amp_lo = 1.0
    k_token_amps = np.power(
        10.0,
        np.random.uniform(
            np.log10(k_amp_lo), np.log10(k_amp_hi), size=(B, N_kv, max_skv, 1)
        ),
    ).astype(np.float32)  # (B, N_kv, max_skv, 1)
    k_base = np.random.uniform(low=-1.0, high=1.0, size=(B, N_kv, max_skv, D)).astype(
        np.float32
    )
    k_data = (k_base * k_token_amps).astype(np.float16)
    k_fp16 = torch.from_numpy(k_data)

    np.random.seed(SEED_V + run_idx)
    v_head_amps = np.power(
        10.0, np.random.uniform(0.0, np.log10(V_DATA_RANGE[1]), size=(B, N_kv, 1, 1))
    ).astype(np.float32)  # (B, N_kv, 1, 1) —— 幅度范围 [1, V_DATA_RANGE[1]]
    v_data_base = np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(B, N_kv, max_skv, D),
    ).astype(np.float32)
    v_data = (v_data_base * v_head_amps).astype(np.float16)
    v_fp16 = torch.from_numpy(v_data)

    q_fp16 = q_fp16.cpu().contiguous()
    k_fp16 = k_fp16.cpu().contiguous()
    v_fp16 = v_fp16.cpu().contiguous()

    # 计算量化scale
    quant_scale_q = get_fp8_per_token_head_quant_scale(q_fp16)
    quant_scale_k = get_fp8_per_token_head_quant_scale(k_fp16)
    quant_scale_v = get_fp8_per_head_quant_scale(v_fp16)

    # 反量化scale
    dequant_scale_q = (1.0 / quant_scale_q).contiguous()
    dequant_scale_k = (1.0 / quant_scale_k).contiguous()
    dequant_scale_v = (1.0 / quant_scale_v).contiguous()

    # 量化到fp8
    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k)
    v_fp8 = quant_fp16_to_fp8(v_fp16, quant_scale_v)

    if max(ACTUAL_SEQ_KV) == 0:
        real_skv = max(ACTUAL_SEQ_KV)
        k_fp8 = k_fp8[:, :, :real_skv, :].contiguous()
        v_fp8 = v_fp8[:, :, :real_skv, :].contiguous()

    logger.info(
        f"[INFO] [run {run_idx}] q_fp8 shape: {q_fp8.shape}, dtype: {q_fp8.dtype}"
    )
    logger.info(
        f"[INFO] [run {run_idx}] k_fp8 shape: {k_fp8.shape}, dtype: {k_fp8.dtype}"
    )
    logger.info(
        f"[INFO] [run {run_idx}] v_fp8 shape: {v_fp8.shape}, dtype: {v_fp8.dtype}"
    )

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32).cpu().contiguous()

    return (
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
    )


# ==============================================================================
# Layout 转换
# ==============================================================================
def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    """BNSD → 各种 layout"""
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    tensor = tensor.cpu().contiguous()
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)

    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return (
            tensor[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N * D)
            .contiguous()
        )
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t : t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    elif layout == "NTD_TND":
        T = sum(seq_lens)
        result = torch.zeros((N, T, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[n, t : t + act_s, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_scale_to_layout(tensor, seq_lens, scale_type):
    """Scale to layout"""
    tensor = tensor.cpu().contiguous()
    if scale_type == "deq_q":
        B, N, _, _ = tensor.shape
        T = sum(seq_lens)
        if Q_SCALE_LAYOUT == "NT":
            result = torch.zeros((N, T), dtype=torch.float32)
            t = 0
            for b in range(B):
                act_s = seq_lens[b]
                for n in range(N):
                    result[n, t : t + act_s] = tensor[b, n, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif Q_SCALE_LAYOUT == "TN":
            result = torch.zeros((T, N), dtype=torch.float32)
            t = 0
            for b in range(B):
                act_s = seq_lens[b]
                for n in range(N):
                    result[t : t + act_s, n] = tensor[b, n, :act_s, 0]
                t += act_s
            return result.contiguous()
        elif Q_SCALE_LAYOUT == "BNSD":
            result = torch.zeros((B, N, max(seq_lens), 1), dtype=torch.float32)
            for b in range(B):
                act_s = seq_lens[b]
                for n in range(N):
                    result[b, n, :act_s, 0] = tensor[b, n, :act_s, 0]
            return result.contiguous()
        else:
            return tensor.float().contiguous()
    elif scale_type == "deq_v":
        return tensor.reshape(tensor.shape[1]).float().contiguous()
    return tensor.squeeze(-1).contiguous()


def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result


# ==============================================================================
# NPU 调用
# GRAPH_PATH: 0=单算子, 5=动态图, 7=aclgraph
# ==============================================================================
def get_npu_fa_kwargs():
    return dict(
        query_quant_mode=3,
        key_quant_mode=3,
        value_quant_mode=2,
        query_dtype=FP8_DTYPE,
        key_dtype=FP8_DTYPE,
        value_dtype=FP8_DTYPE,
        dequant_scale_query_dtype=torch.float32,
        dequant_scale_key_dtype=torch.float32,
        dequant_scale_value_dtype=torch.float32,
        return_softmax_lse=ENABLE_LSE,
    )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        mask,
        actual_seq_q,
        actual_seq_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout,
        block_size,
        out_dtype,
    ):
        atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
            q,
            k,
            v,
            atten_mask=mask,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            dequant_scale_query=dequant_scale_q,
            dequant_scale_key=dequant_scale_k,
            dequant_scale_value=dequant_scale_v,
            block_table=block_table,
            block_size=block_size,
            num_query_heads=q_n,
            num_key_value_heads=kv_n,
            softmax_scale=softmax_scale,
            input_layout=layout,
            sparse_mode=SPARSE_MODE,
            quant_scale_p=p_scale,
            out_dtype=out_dtype,
            **get_npu_fa_kwargs(),
        )
        return atten_out, lse_out


def call_npu_fa_op(
    q,
    k,
    v,
    mask,
    actual_seq_q,
    actual_seq_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    block_size,
    out_dtype,
):
    torch.npu.synchronize()
    atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        atten_mask=mask,
        actual_seq_qlen=actual_seq_q,
        actual_seq_kvlen=actual_seq_kv,
        dequant_scale_query=dequant_scale_q,
        dequant_scale_key=dequant_scale_k,
        dequant_scale_value=dequant_scale_v,
        block_table=block_table,
        block_size=block_size,
        num_query_heads=q_n,
        num_key_value_heads=kv_n,
        softmax_scale=softmax_scale,
        input_layout=layout,
        sparse_mode=SPARSE_MODE,
        quant_scale_p=p_scale,
        out_dtype=out_dtype,
        **get_npu_fa_kwargs(),
    )
    torch.npu.synchronize()
    return atten_out, lse_out


def fia_gqa_torch_npu(
    q,
    k,
    v,
    mask,
    actual_seq_q,
    actual_seq_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    block_size,
    out_dtype,
):
    if GRAPH_PATH == 0:
        logger.info("[INFO] GRAPH_PATH == 0, single operator mode...")
        return call_npu_fa_op(
            q,
            k,
            v,
            mask,
            actual_seq_q,
            actual_seq_kv,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout,
            block_size,
            out_dtype,
        )

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    config = CompilerConfig()
    with torch.no_grad():
        torch.npu.synchronize()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        fa_args = (
            q,
            k,
            v,
            mask,
            actual_seq_q,
            actual_seq_kv,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout,
            block_size,
            out_dtype,
        )

        if GRAPH_PATH == 5:
            logger.info("[INFO] GRAPH_PATH == 5, dynamic graph...")
            torch._dynamo.reset()
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            atten_out, lse_out = npu_mode(*fa_args)
        elif GRAPH_PATH == 7:
            logger.info("[INFO] GRAPH_PATH == 7, aclgraph...")
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            config.mode = "reduce-overhead"
            npu_mode = torch.compile(
                npu_mode, fullgraph=True, backend=npu_backend, dynamic=True
            )
            for t in (
                q,
                k,
                v,
                mask,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                block_table,
            ):
                if t is not None:
                    torch._dynamo.mark_static(t)
            atten_out, lse_out = npu_mode(*fa_args)
        else:
            raise ValueError(
                f"Unsupported GRAPH_PATH: {GRAPH_PATH}, only support 0/5/7"
            )

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out


def fa_run_npu(
    q,
    k,
    v,
    mask,
    actual_seq_q,
    actual_seq_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    block_size,
    q_n,
    kv_n,
    softmax_scale,
    layout,
    out_dtype,
):
    """将数据转移到NPU上并调用NPU算子"""
    device_id = DEVICE_ID
    torch_npu.npu.set_device(int(device_id))

    # 确保所有输入都在NPU上且有正确的数据类型
    q = q.npu()
    k = k.npu()
    v = v.npu()

    # 取出deq_k，数据在NPU上，且是float32类型
    k_pa_f32 = (
        k.view(torch.uint8)  # (Bn, N, Bs+4, D)   → uint8，仍是 view，stride 不变
        .view(
            k.shape[0], k.shape[1], -1
        )  # (Bn, N, 16896)     → 合并最后两维，仍是 view
        .view(torch.float32)  # (Bn, N, 4224)      → float32，仍是 view
    )
    dequant_scale_k = k_pa_f32[:, :, -BLOCK_SIZE:]  # (Bn, N, 128) float32

    # dequant scales 必须是 float32 类型
    dequant_scale_q = dequant_scale_q.float().npu()
    dequant_scale_v = dequant_scale_v.float().npu()
    p_scale = p_scale.float().npu()

    if not IS_CONTIGUOUS and ENABLE_PA:
        fake_kscale_tensor = torch.ones_like(dequant_scale_k)
        double_kscale = torch.stack([dequant_scale_k, fake_kscale_tensor], dim=2)
        double_kscale = double_kscale.npu()
        dequant_scale_k = double_kscale[:, :, 0]  # 覆写为非连续

    # block_table 必须是 int32 类型
    block_table = block_table.int().npu() if ENABLE_PA else None

    # mask 如果有的话，转换为 bool 类型
    if mask is not None:
        mask = mask.bool().npu()

    # 将kv从cache中取切片
    k = k[:, :, :128, :]
    v = v[:, :, :128, :]

    # 打印调试信息
    logger.info(f"[INFO] q dtype: {q.dtype}, shape: {q.shape}")
    logger.info(f"[INFO] k dtype: {k.dtype}, shape: {k.shape}")
    logger.info(f"[INFO] v dtype: {v.dtype}, shape: {v.shape}")
    logger.info(
        f"[INFO] deq_q dtype: {dequant_scale_q.dtype}, shape: {dequant_scale_q.shape}"
    )
    logger.info(
        f"[INFO] deq_k dtype: {dequant_scale_k.dtype}, shape: {dequant_scale_k.shape}"
    )
    logger.info(
        f"[INFO] deq_v dtype: {dequant_scale_v.dtype}, shape: {dequant_scale_v.shape}"
    )
    logger.info(f"[INFO] NPU input layout: {layout}, sparse_mode: {SPARSE_MODE}")
    logger.info(
        f"[INFO] key is_contiguous: {k.is_contiguous()}, value is_contiguous: {v.is_contiguous()}"
    )
    logger.info(f"[INFO] key stride: {k.stride()}, value stride: {v.stride()}")
    logger.info(f"[INFO] deq_k is_contiguous: {dequant_scale_k.is_contiguous()}")
    logger.info(f"[INFO] deq_k stride: {dequant_scale_k.stride()}")
    logger.info("[INFO] --- tensor devices ---")
    logger.info(f"  q.device={q.device}, k.device={k.device}, v.device={v.device}")
    logger.info(
        f"  deq_q.device={dequant_scale_q.device}, deq_k.device={dequant_scale_k.device}, deq_v.device={dequant_scale_v.device}"
    )
    logger.info(
        f"  p_scale.device={p_scale.device}, block_table.device={block_table.device if block_table is not None else None}"
    )
    logger.info(f"  mask.device={mask.device if mask is not None else None}")

    # 从这里就开始进入NPU调用流程，不再处理数据
    atten_out, lse_out = fia_gqa_torch_npu(
        q,
        k,
        v,
        mask,
        actual_seq_q,
        actual_seq_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout,
        block_size,
        out_dtype,
    )

    if GRAPH_PATH == 0:
        atten_out = atten_out.cpu()
        lse_out = lse_out.cpu()
    return atten_out, lse_out


def npu_fp8_full_quant(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    run_idx=0,
):
    """Main NPU quant function - prepares data and calls NPU

    run_idx: 第几次运行，用于偏移 block_table 种子，保证每次输入不同
    """
    softmax_scale = 1.0 / math.sqrt(D)
    T = sum(actual_seq_q)
    out_dtype = OUTPUT_DETYPE

    accum_seq_q = (
        make_accum_seq(actual_seq_q)
        if INPUT_LAYOUT in ("NTD_TND", "TND")
        else actual_seq_q
    )

    npu_input_layout = INPUT_LAYOUT

    # 确保 q 是 FP8 类型
    q_npu = convert_q_bnsd_to_layout(q_fp8, actual_seq_q, npu_input_layout)

    # dequant scales 使用 float32
    deq_q_npu = convert_scale_to_layout(dequant_scale_q, ACTUAL_SEQ_Q, "deq_q")
    deq_v_npu = convert_scale_to_layout(dequant_scale_v, ACTUAL_SEQ_KV, "deq_v")

    if SPARSE_MODE == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
    else:
        mask = None

    if ENABLE_PA:
        # 在CPU上准备block table和cache
        block_table = create_block_table(ACTUAL_SEQ_KV, BLOCK_SIZE, run_idx=run_idx)
        block_table_tensor = torch.as_tensor(block_table, dtype=torch.int32)
        # 生成k和v的cache
        k_pa = bnsd_to_k_cache(
            k_fp8, dequant_scale_k, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table
        )
        v_pa = bnsd_to_v_cache(v_fp8, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table)

        # 重要！！！此处取出deq_k其实没用，因为传入NPU时会使用.npu(), 导致stride变成连续的！！！
        # 在这里取出deq_k只是为了保存为pt文件（其实也用不到deq_k的pt文件）
        # 将deq_k从k_pa中提取出来，确保deq_k和k_pa共享内存
        k_pa_f32 = (
            k_pa.view(torch.uint8)  # (Bn, N, Bs+4, D)   → uint8，仍是 view，stride 不变
            .view(
                k_pa.shape[0], k_pa.shape[1], -1
            )  # (Bn, N, 16896)     → 合并最后两维，仍是 view
            .view(torch.float32)  # (Bn, N, 4224)      → float32，仍是 view
        )
        deq_k_npu = k_pa_f32[:, :, -BLOCK_SIZE:]  # (Bn, N, 128) float32

        # 构造kvcache非连续
        if not IS_CONTIGUOUS:
            kv_cache = torch.stack([k_pa, v_pa], dim=2)
            kv_cache = kv_cache.npu()
            k_pa = kv_cache[:, :, 0]
            v_pa = kv_cache[:, :, 1]

        if SAVE_PT:
            import os

            os.makedirs(SAVE_PT_DIR, exist_ok=True)
            torch.save(q_npu, os.path.join(SAVE_PT_DIR, "q_fp8.pt"))
            torch.save(k_pa, os.path.join(SAVE_PT_DIR, "k_fp8.pt"))
            torch.save(v_pa, os.path.join(SAVE_PT_DIR, "v_fp8.pt"))
            torch.save(deq_q_npu, os.path.join(SAVE_PT_DIR, "deq_q.pt"))
            torch.save(deq_k_npu, os.path.join(SAVE_PT_DIR, "deq_k.pt"))
            torch.save(deq_v_npu, os.path.join(SAVE_PT_DIR, "deq_v.pt"))
            torch.save(block_table_tensor, os.path.join(SAVE_PT_DIR, "block_table.pt"))
            torch.save(accum_seq_q, os.path.join(SAVE_PT_DIR, "seq_q.pt"))
            torch.save(actual_seq_kv, os.path.join(SAVE_PT_DIR, "seq_kv.pt"))
            torch.save(mask, os.path.join(SAVE_PT_DIR, "mask.pt"))

        output = fa_run_npu(
            q_npu,
            k_pa,
            v_pa,
            mask,
            accum_seq_q,
            actual_seq_kv,
            deq_q_npu,
            deq_k_npu,
            deq_v_npu,
            p_scale,
            block_table_tensor,
            BLOCK_SIZE,
            N_q,
            N_kv,
            softmax_scale,
            npu_input_layout,
            out_dtype,
        )
    else:
        raise NotImplementedError("当前仅支持 PA 模式")

    atten_out = output[0]
    T_actual = sum(actual_seq_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    return output


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    logger.info(
        "FIA Full Quant GQA NPU ({} runs, refresh input each run)".format(RUN_TIMES)
    )
    print("=" * 60)
    logger.info(
        f"[INFO] Scene: {'PA' if ENABLE_PA else 'noPA'}, INPUT_LAYOUT: {INPUT_LAYOUT}, OUTPUT_LAYOUT: {OUTPUT_LAYOUT}"
    )
    logger.info(f"[INFO] B={B}, N_q={N_q}, N_kv={N_kv}, D={D}")
    logger.info(f"[INFO] ACTUAL_SEQ_Q={ACTUAL_SEQ_Q}")
    logger.info(f"[INFO] ACTUAL_SEQ_KV={ACTUAL_SEQ_KV}")

    for run_idx in range(RUN_TIMES):
        print("\n" + "=" * 60)
        logger.info(f"[Run {run_idx + 1}/{RUN_TIMES}] start")
        print("=" * 60)

        logger.info(
            f"\n[Run {run_idx + 1} Step 1] data generation (seed offset run_idx={run_idx})"
        )
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
        ) = generate_data(run_idx)

        logger.info(f"\n[Run {run_idx + 1} Step 2] NPU: run NPU computation")
        npu_out = npu_fp8_full_quant(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            ACTUAL_SEQ_Q,
            ACTUAL_SEQ_KV,
            run_idx=run_idx,
        )
        logger.info(
            f"[INFO] [run {run_idx}] npu_out[0] shape: {npu_out[0].shape}, dtype: {npu_out[0].dtype}"
        )
        if ENABLE_LSE:
            logger.info(
                f"[INFO] [run {run_idx}] npu_out[1] shape: {npu_out[1].shape}, dtype: {npu_out[1].dtype}"
            )
        logger.info("#" * 60)

    print("\n" + "=" * 60)
    logger.info(f"[INFO] All {RUN_TIMES}  runs completed")
    print("=" * 60)
