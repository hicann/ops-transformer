import torch
import numpy as np
import copy
from ml_dtypes import bfloat16 as np_bfloat16
from ml_dtypes import float8_e4m3fn as np_float8_e4m3fn

__golden__ = {
    "kernel": {"stem_oam_prep_paged_kv": "stem_oam_prep_paged_kv_golden"},
    "aclnn": {"aclnnStemOamPrepPagedKv": "aclnn_stem_oam_prep_paged_kv_golden"}
}

def _numpy_to_torch(t, dtype=torch.float32):
    if t is None:
        return None
    if isinstance(t, np.ndarray) and t.dtype == np_bfloat16:
        return torch.form_numpy(t.view(np.int16).copy()).view(torch.bfloat16).to(dtype)
    if isinstance(t, np.ndarray) and t.dtype == np_float8_e4m3fn:
        return torch.form_numpy(t.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    if hasattr(t, "numpy"):
        return t.to(dtype)
    return torch.from_numpy(t).to(dtype)

def stem_oam_prep_paged_kv(
    kcache_fp8,
    vcache_fp8,
    kscale_fp32,
    vscale,
    kv_indices,
    kv_seq_lens,
    kv_block_size,
    lambda_mag=0.3,
    kv_layout=0,
    stem_block_size=128,
    stem_stride=16
):
    """Pure PyTorch golden for stem_oam_prep_paged_kv.

    Supports both Layout A (interleaved) and Layout B (contiguous).

    K Processing:
      de-page -> Cast FP32 -> x kScaleCache -> group sum (anti-diag flip) -> Cast BF16
    V Processing:
      de-page -> Cast FP32 -> x vScale -> L2 Norm -> Max -> Log
      -> Global Normalize (mu/sigma over ALL k_down_len) -> ReLU -> x lambda -> Block Average

    Args:
        kcache_fp8:     [total_blocks, kvBlockSize, H_kv, 128] (Layout A)
                        [total_blocks, H_kv, kvBlockSize, 128] (Layout B), FP8_E4M3FN
        vcache_fp8:     same shape/layout as kcache_fp8, FP8_E4M3FN
        kscale_fp32:    [total_blocks, kvBlockSize, H_kv, 1] (Layout A)
                        [total_blocks, H_kv, kvBlockSize, 1] (Layout B), FP32
        vscale:         [H_kv], FP32
        kv_indices:     [batch, max_kv_blocks], INT32
        kv_seq_lens:    [batch], INT32
        kv_block_size:  64 or 128
        lambda_mag:     float, default 0.3
        kv_layout:      0=Layout A, 1=Layout B
        stem_block_size: must be multiple of 32, <=256, default 128
        stem_stride:    must be multiple of 16, <=64, S<=B, default 16

    Returns:
        kflat:  [batch, H_kv, max_Kb, stem_stride * 128] BF16
        v_bias: [batch, H_kv, max_Kb] FP32
    """
    device = kcache_fp8.device
    num_batch = kv_seq_lens.shape[0]

    if kv_layout == 0:
        num_head_kv = kcache_fp8.shape[2]
    else:
        num_head_kv = kcache_fp8.shape[1]

    dim_qk = kcache_fp8.shape[3]
    dim_v = vcache_fp8.shape[3]
    R = stem_block_size // stem_stride
    epsilon = 1e-6

    k_padded_lens = (
        (kv_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    ).to(torch.int32)
    max_k_padded = k_padded_lens.max().item()
    max_Kb = max_k_padded // stem_block_size

    kflat_out = torch.zeros(
        num_batch, num_head_kv, max_Kb, stem_stride * dim_qk,
        dtype=torch.bfloat16, device=device,
    )
    v_bias_out = torch.zeros(
        num_batch, num_head_kv, max_Kb,
        dtype=torch.float32, device=device,
    )

    kcache_f32 = kcache_fp8.to(torch.float32)
    vcache_f32 = vcache_fp8.to(torch.float32)

    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        k_padded = k_padded_lens[b].item()
        num_stem_blocks = k_padded // stem_block_size
        k_down_len = k_padded // stem_stride

        num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size
        block_ids = kv_indices[b, :num_kv_blocks]

        # De-page K and V into dense FP32 (zero-padded beyond kv_len)
        if kv_layout == 0:
            k_paged = kcache_f32[block_ids].reshape(-1, num_head_kv, dim_qk)
            v_paged = vcache_f32[block_ids].reshape(-1, num_head_kv, dim_v)
            kScale_paged = kscale_fp32[block_ids].reshape(-1, num_head_kv, 1)
        else:
            k_paged = kcache_f32[block_ids].permute(0, 2, 1, 3).reshape(-1, num_head_kv, dim_qk)
            v_paged = vcache_f32[block_ids].permute(0, 2, 1, 3).reshape(-1, num_head_kv, dim_v)
            kScale_paged = kscale_fp32[block_ids].permute(0, 2, 1, 3).reshape(-1, num_head_kv, 1)

        k_dense = torch.zeros(k_padded, num_head_kv, dim_qk, dtype=torch.float32, device=device)
        v_dense = torch.zeros(k_padded, num_head_kv, dim_v, dtype=torch.float32, device=device)
        kScale_dense = torch.zeros(k_padded, num_head_kv, 1, dtype=torch.float32, device=device)

        actual_rows = min(kv_len, num_kv_blocks * kv_block_size)
        k_dense[:actual_rows] = k_paged.reshape(-1, num_head_kv, dim_qk)[:actual_rows]
        v_dense[:actual_rows] = v_paged.reshape(-1, num_head_kv, dim_v)[:actual_rows]
        kScale_dense[:actual_rows] = kScale_paged.reshape(-1, num_head_kv, 1)[:actual_rows]

        k_scaled = (
            k_dense[:num_stem_blocks * stem_block_size]
            * kScale_dense[:num_stem_blocks * stem_block_size]
        )

        for h in range(num_head_kv):
            k_h = k_scaled[:, h, :]
            k_blocks = k_h.reshape(num_stem_blocks, R, stem_stride, dim_qk)
            k_group_sum = k_blocks.sum(dim=1)
            k_group_rev = k_group_sum.flip(1)
            kflat_out[b, h, :num_stem_blocks] = k_group_rev.reshape(
                num_stem_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

            vscale_h = vscale[h].item()
            v_h = v_dense[:num_stem_blocks * stem_block_size, h, :] * vscale_h
            v_rows = v_h.reshape(k_down_len, stem_stride, dim_v)
            norms = torch.sqrt((v_rows ** 2).sum(dim=-1))
            row_ids = torch.arange(
                num_stem_blocks * stem_block_size, device=device,
            ).reshape(k_down_len, stem_stride)
            norms = torch.where(row_ids < kv_len, norms, torch.zeros_like(norms))
            v_norm_down = norms.max(dim=-1).values

            if k_down_len > 0:
                log_vals = torch.log(v_norm_down + epsilon)
                v_mean = log_vals.mean()
                if k_down_len > 1:
                    v_std = log_vals.std(unbiased=True)
                else:
                    v_std = torch.tensor(0.0, device=device)
                inv_std = 1.0 / (v_std + epsilon)
                normalized = (log_vals - v_mean) * inv_std
                v_final = lambda_mag * torch.relu(normalized)
                v_final_blocks = v_final[:num_stem_blocks * R].reshape(num_stem_blocks, R)
                v_bias_out[b, h, :num_stem_blocks] = v_final_blocks.mean(dim=1)

    return kflat_out, v_bias_out

def aclnn_stem_oam_prep_paged_kv_golden(
    kCache,
    vCache,
    kvIndices,
    kvSeqLens,
    kScaleCache,
    vScale,
    lambdaMag,
    kvLayout,
    stemBlockSize,
    stemStride,
    kFlat,
    vBias,
    **kwargs
):
    kvSeqLens = torch.tensor(kvSeqLens, dtype=torch.int32)
    kvBlockSize = kCache.shape[1]
    cacheLayout = 0
    if kvLayout == "BNBD":
        cacheLayout = 1
        kvBlockSize = kCache.shape[2]
    
    kflat_out, v_bias_out = stem_oam_prep_paged_kv(kCache, vCache, kScaleCache, vScale, kvIndices, kvSeqLens, 
                                                kvBlockSize, lambdaMag, cacheLayout, stemBlockSize, stemStride)

    return (kflat_out, v_bias_out)

def stem_oam_prep_paged_kv_golden(
    kCache,
    vCache,
    kvIndices,
    kvSeqLens,
    kScaleCache,
    vScale,
    lambdaMag,
    kvLayout,
    stemBlockSize,
    stemStride,
    **kwargs
):
    kCache = _numpy_to_torch(kCache, torch.float8_e4m3fn)
    vCache = _numpy_to_torch(vCache, torch.float8_e4m3fn)
    kvIndices = _numpy_to_torch(kvIndices, torch.int32)
    kvSeqLens = _numpy_to_torch(kvSeqLens, torch.int32)
    kScaleCache = _numpy_to_torch(kScaleCache)
    vScale = _numpy_to_torch(vScale)
    kvBlockSize = kCache.shape[1]
    cacheLayout = 0
    if kvLayout == "BNBD":
        cacheLayout = 1
        kvBlockSize = kCache.shape[2]
    
    kflat_out, v_bias_out = stem_oam_prep_paged_kv(kCache, vCache, kScaleCache, vScale, kvIndices, kvSeqLens, 
                                                kvBlockSize, lambdaMag, stemBlockSize, stemStride, cacheLayout)
    kflat_out = kflat_out.to(torch.float32).numpy()
    v_bias_out = v_bias_out.numpy()

    return (kflat_out, v_bias_out)
