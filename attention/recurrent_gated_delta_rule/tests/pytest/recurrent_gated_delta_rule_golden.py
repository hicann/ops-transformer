# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: E712
import torch
import torch_npu
import numpy as np
import logging
import datetime
import os
import sys
import argparse
import gc

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch.npu.config.allow_internal_format = True

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# SKIP_GOLDEN=1 时跳过 CPU golden 计算与精度对比，仅执行 NPU 算子（用于加快随机泛化、配合 mssanitizer 检测）
SKIP_GOLDEN = os.environ.get("SKIP_GOLDEN", "0") == "1"


def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level="INFO"):
    logger.info(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


# fuzz 中precision_method == 1的精度对比方式
def check_result(expect, result, data_type, pct_thd=0.005):
    expect_flat = expect.reshape(-1)
    result_flat = result.reshape(-1)
    total = result_flat.numel()

    if total == 0 and expect_flat.numel() == 0:
        print_log(
            'The npu_output is [], and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return 100.0, "Pass"
    start = 0
    end = total - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if total != expect_flat.numel():
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (total, expect_flat.numel())
        )
        return 0.0, result
    if data_type == "bfloat16":
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.0078125
        atol = 0.0001
        max_error_idx = 10000000
    else:
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.005
        atol = 0.000025
        max_error_idx = 10000000

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    eps = 10e-10
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    chunk_size = 1 << 22

    overflows_count = 0
    err_count = 0
    inf_samples = []
    nan_samples = []
    err_idx_parts = []
    err_diff_parts = []

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        r_chunk = result_flat[chunk_start:chunk_end].cpu().to(torch.float32).numpy()
        c_chunk = expect_flat[chunk_start:chunk_end].cpu().to(torch.float32).numpy()

        inf_mask = np.isinf(c_chunk)
        nan_mask = np.isnan(c_chunk)
        overflows_count += int(inf_mask.sum()) + int(nan_mask.sum())
        if len(inf_samples) < 10:
            for li in np.where(inf_mask)[0]:
                if len(inf_samples) >= 10:
                    break
                inf_samples.append(c_chunk[li])
        if len(nan_samples) < 10:
            for li in np.where(nan_mask)[0]:
                if len(nan_samples) >= 10:
                    break
                nan_samples.append(c_chunk[li])

        diff_chunk = np.isclose(r_chunk, c_chunk, rtol=rtol, atol=atol, equal_nan=True)
        local_err = np.where(~diff_chunk)[0]
        err_count += local_err.size
        if local_err.size > 0:
            global_err = local_err + chunk_start
            er = r_chunk[local_err]
            ec = c_chunk[local_err]
            ed = np.abs(ec - er) / (
                np.maximum(np.maximum(np.abs(er), np.abs(ec)), b2) + eps + eps
            )
            err_idx_parts.append(global_err)
            err_diff_parts.append(ed)

    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                np.array(inf_samples)[0:10] if inf_samples else np.array([]),
                np.array(nan_samples)[0:10] if nan_samples else np.array([]),
            )
        )

    err_idx = (
        np.concatenate(err_idx_parts) if err_idx_parts else np.array([], dtype=np.int64)
    )
    err_diff = (
        np.concatenate(err_diff_parts)
        if err_diff_parts
        else np.array([], dtype=np.float64)
    )

    fulfill_percent = float(split_count - err_count) / float(split_count) * 100.0

    display_threshold = 1 << 20
    if total <= display_threshold:
        disp_real = result_flat.cpu().to(torch.float32).numpy()
        disp_compe = expect_flat.cpu().to(torch.float32).numpy()
        display_output_np_isclose(disp_real, disp_compe, start, end)
    else:
        print_log(
            "---------------------------------------------------------------------------------------"
        )
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
        print_log(
            "---------------------------------------------------------------------------------------"
        )
        n_sample = 10
        r_head = result_flat[:n_sample].cpu().to(torch.float32).numpy()
        c_head = expect_flat[:n_sample].cpu().to(torch.float32).numpy()
        r_tail = result_flat[total - n_sample :].cpu().to(torch.float32).numpy()
        c_tail = expect_flat[total - n_sample :].cpu().to(torch.float32).numpy()
        for i in range(n_sample):
            diff_abs = abs(np.float64(c_head[i]) - np.float64(r_head[i]))
            diff_rate = cal_relative_diff_np_isclose(r_head[i], c_head[i])
            print_log(
                "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                % (i + 1, c_head[i], r_head[i], diff_abs, diff_rate)
            )
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(n_sample):
            diff_abs = abs(np.float64(c_tail[i]) - np.float64(r_tail[i]))
            diff_rate = cal_relative_diff_np_isclose(r_tail[i], c_tail[i])
            print_log(
                "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                % (
                    total - n_sample + i + 1,
                    c_tail[i],
                    r_tail[i],
                    diff_abs,
                    diff_rate,
                )
            )

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        err_limit = min(len(err_idx), max_error_idx)
        if err_limit > 0:
            err_indices = err_idx[:err_limit]
            err_tensor = torch.as_tensor(err_indices, dtype=torch.long)
            err_r = (
                result_flat[err_tensor.to(result_flat.device)]
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            err_c = expect_flat[err_tensor].cpu().to(torch.float32).numpy()
            display_error_output(
                err_r, err_c, np.arange(err_limit), err_diff[:err_limit]
            )
    return fulfill_percent, result


def cpu_recurrent_gated_delta_rule(
    q,
    k,
    v,
    state,
    beta,
    scale_value,
    act_seq_len,
    ssm_state_indices,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    T, n_heads_qk, Dk = q.shape
    T, n_heads_v, Dv = v.shape
    B = act_seq_len.shape[0]
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    initial_state = state.to(torch.float32).clone()
    beta = beta.to(torch.float32)
    output = torch.empty_like(v).to(torch.float32)
    g = (
        torch.ones(T, n_heads_v).to(torch.float32)
        if g is None
        else g.to(torch.float32).exp()
    )
    gk = (
        torch.ones(T, n_heads_v, Dk).to(torch.float32)
        if gk is None
        else gk.to(torch.float32).exp()
    )

    q = q * scale_value
    seq_start = 0
    for i in range(B):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[
                ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]
            ]
        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + act_seq_len[i]):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_qk)]  # [Dk]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_qk)]  # [Dk]
                v_i = v[slot_id][head_id]  # [Dv]
                beta_i = beta[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                S = S * alpha_i
                alphak_i = gk[slot_id][head_id]  # [Dk]
                S = S * alphak_i.unsqueeze(-2)

                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i  # [Dv]
                S_ = y[:, None] * k_i[None, :]  # [Dv, Dk]
                S = S + S_  # [Dv, Dk]
                initial_state[ssm_state_indices[slot_id]][head_id] = S

                output[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)  # [Dv]
        seq_start += act_seq_len[i]
    output_golden = output
    state_golden = initial_state

    logger.info(f"output_golden.shape: {output_golden.shape}")
    logger.info(f"state_golden.shape: {state_golden.shape}")
    output_golden = torch.tensor(output_golden).to(q.dtype)
    return output_golden, state_golden


def rand_range(shape, data_range=[-10, 10], dtype=torch.bfloat16, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(
        shape, dtype=dtype, device=device
    )


def adjust_range(datarange):
    left, right = datarange
    if right < 0:
        return [left, right]
    if left > 0:
        return [-right, -left]
    return [left, 0]


# 最近一条用例的精度达标率（PctRlt），供 conftest.py 落 CSV；SKIP_GOLDEN 时保持 None
LAST_OUT_PCT = None
LAST_STATE_PCT = None


def run_recurrent_gated_delta_rule_eager(
    B,
    mtp,
    nk,
    nv,
    dk,
    dv,
    actual_seq_lengths=None,
    ssm_state_indices=None,
    has_gamma="False",
    has_gamma_k="False",
    has_num_accepted_tokens="False",
    scale_value=None,
    num_accepted_tokens=None,
    block_num=None,
    data_type=torch.bfloat16,
    state_data_type=None,
    query_datarange=[-10, 10],
    key_datarange=[-10, 10],
    value_datarange=[-10, 10],
    gamma_datarange=[0, 1],
    gamma_k_datarange=[0, 1],
    beta_datarange=[0, 1],
    state_datarange=[-10, 10],
    state_non_contiguous=False,
):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== set input params finish ========================
    if state_data_type is None:
        state_data_type = data_type
    block_num = B * mtp if block_num is None else block_num
    if scale_value is None:
        scale_value = dk**-0.5
    if actual_seq_lengths is None:
        actual_seq_lengths = [mtp] * B
    if has_num_accepted_tokens == True and num_accepted_tokens is None:
        num_accepted_tokens = (
            torch.tensor([torch.randint(0, h, (1,)) for h in actual_seq_lengths]) + 1
        )
    T = int(sum(actual_seq_lengths))
    actual_seq_lengths = torch.tensor(actual_seq_lengths)
    if ssm_state_indices is None:
        ssm_state_indices = torch.arange(T, dtype=torch.int32)
    # ======================== set input params finish ========================
    # ======================== check input params start ========================
    if len(actual_seq_lengths) != B:
        logger.error(
            f"Error: the len of seqused is {len(actual_seq_lengths)}, it should be B({B})"
        )
        return
    if has_num_accepted_tokens == True and len(num_accepted_tokens) != B:
        logger.error(
            f"Error: the len of num_accepted_tokens is {len(num_accepted_tokens)}, it should be B({B})"
        )
        return
    for i in range(B):
        act_seq = actual_seq_lengths[i]
        if act_seq <= 0 or act_seq > mtp:
            logger.error(
                f"Error: actual_seq_lengths[{i}] is {act_seq}, it should be > 0 and <= mtp({mtp})"
            )
            return
        if has_num_accepted_tokens == True:
            accepted_token = num_accepted_tokens[i]
            if accepted_token < 1 or accepted_token > act_seq:
                logger.error(
                    f"Error: num_accepted_tokens[{i}] is {accepted_token}, it should >= 1 and <= actual_seq_lengths[{i}]({act_seq})"
                )
                return
    if len(ssm_state_indices) != T:
        logger.error(
            f"Error: the len of ssm_state_indices is {len(ssm_state_indices)}, it should be T({T})"
        )
        return
    for i in range(T):
        idx = ssm_state_indices[i]
        if idx < 0 or idx > block_num:
            logger.error(
                f"Error: ssm_state_indices[{i}] is {idx}, it should >= 0 and < block_num({block_num})"
            )
            return
    # ======================== check input params finish ========================
    # ======================== gen input data start =============================
    # SKIP_GOLDEN 模式（random_npu/memcheck）无 CPU golden 消费，直接在 NPU 上生成大张量，
    # 省 host 峰值内存与 H2D 搬运；带 golden 模式必须保留 host 生成。
    gen_device = "npu:%s" % DEVICE_ID if SKIP_GOLDEN else None
    query = rand_range((T, nk, dk), query_datarange, data_type, device=gen_device)
    key = rand_range((T, nk, dk), key_datarange, data_type, device=gen_device)
    value = rand_range((T, nv, dv), value_datarange, data_type, device=gen_device)
    g = (
        rand_range(
            (T, nv),
            adjust_range(gamma_datarange),
            dtype=torch.float32,
            device=gen_device,
        )
        if has_gamma == True
        else None
    )
    gk = (
        rand_range(
            (T, nv, dk),
            adjust_range(gamma_k_datarange),
            dtype=torch.float32,
            device=gen_device,
        )
        if has_gamma_k == True
        else None
    )
    beta = rand_range((T, nv), beta_datarange, data_type, device=gen_device)
    num_accepted_tokens = (
        torch.tensor(num_accepted_tokens, dtype=torch.int32)
        if has_num_accepted_tokens == True
        else None
    )
    state = rand_range(
        (block_num, nv, dv, dk), state_datarange, state_data_type, device=gen_device
    )
    act_seq_len = torch.tensor(actual_seq_lengths, dtype=torch.int32)
    ssm_state_indices = torch.tensor(ssm_state_indices, dtype=torch.int32)

    # 对于query和key数据范围大于[-1, 1]的情况加入归一化处理，避免数据过大导致计算结果全为inf或者nan的情况
    if query_datarange[0] < -1 or query_datarange[1] > 1:
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
    if key_datarange[0] < -1 or key_datarange[1] > 1:
        key = torch.nn.functional.normalize(key, p=2, dim=-1)

    # ======================== gen input data finish =============================

    # ======================== execute cpu start =================================
    if SKIP_GOLDEN:
        logger.info("SKIP_GOLDEN=1, skip cpu golden execute")
    else:
        cpu_out, cpu_state_ouput = cpu_recurrent_gated_delta_rule(
            query,
            key,
            value,
            state,
            beta,
            scale_value,
            act_seq_len,
            ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=gk,
        )
    # ======================== execute cpu finish ================================

    # ======================== execute npu start =================================
    query = query.to("npu:%s" % DEVICE_ID)
    key = key.to("npu:%s" % DEVICE_ID)
    value = value.to("npu:%s" % DEVICE_ID)
    state = state.to("npu:%s" % DEVICE_ID)
    if state_non_contiguous:
        padded_state = torch.zeros(
            block_num, nv, dv + 1, dk, dtype=state.dtype, device=state.device
        )
        padded_state[:, :, :dv, :] = state
        state = padded_state[:, :, :dv, :]
        logger.info(
            f"state non-contiguous: shape={state.shape}, strides={state.stride()}, "
            f"is_contiguous={state.is_contiguous()}"
        )
    beta = beta.to("npu:%s" % DEVICE_ID)
    act_seq_len = act_seq_len.to("npu:%s" % DEVICE_ID)
    ssm_state_indices = ssm_state_indices.to("npu:%s" % DEVICE_ID)
    num_accepted_tokens = (
        num_accepted_tokens.to("npu:%s" % DEVICE_ID)
        if has_num_accepted_tokens == True
        else None
    )
    g = g.to("npu:%s" % DEVICE_ID) if has_gamma == True else None
    gk = gk.to("npu:%s" % DEVICE_ID) if has_gamma_k == True else None

    # ======================== execute npu finish ================================
    # start run custom ops
    if state_non_contiguous:
        init_padded = padded_state.clone()
        init_state = init_padded[:, :, :dv, :]
    else:
        init_state = state.clone()
    npu_out = torch_npu.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        init_state,
        beta=beta,
        scale=scale_value,
        actual_seq_lengths=act_seq_len,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        g=g,
        gk=gk,
    )
    npu_state_out = init_state
    logger.info(f"query: shape {query.shape}, dtype: {query.dtype}")
    logger.info(f"key: shape {key.shape}, dtype: {key.dtype}")
    logger.info(f"value: shape {value.shape}, dtype: {value.dtype}")
    logger.info(f"state: shape {state.shape}, dtype: {state.dtype}")
    logger.info(f"beta: shape {beta.shape}, dtype: {beta.dtype}")
    logger.info(
        f"act_seq_len: shape {act_seq_len.shape[0]}, dtype: {act_seq_len.dtype}"
    )
    logger.info(
        f"ssm_state_indices: shape {ssm_state_indices.shape[0]}, dtype: {ssm_state_indices.dtype}"
    )

    # 结果精度对比
    out_data_type = str(npu_out.dtype)
    state_data_type_str = str(npu_state_out.dtype)

    del query, key, value, state, beta, act_seq_len, ssm_state_indices
    del num_accepted_tokens, g, gk, init_state
    if state_non_contiguous:
        del padded_state, init_padded
    torch.npu.empty_cache()

    if SKIP_GOLDEN:
        logger.info("SKIP_GOLDEN=1, skip precision check, npu execute finished")
        del npu_out, npu_state_out
        gc.collect()
        torch.npu.empty_cache()
        return

    logger.info(
        "--------------------------------------------------------------check result-------------------------------------------------------------"
    )
    out_pct, out_result = check_result(cpu_out, npu_out, out_data_type)
    logger.info(
        "--------------------------------------------------------------check state output-------------------------------------------------------------"
    )
    state_pct, state_result = check_result(
        cpu_state_ouput,
        npu_state_out,
        state_data_type_str,
    )

    global LAST_OUT_PCT, LAST_STATE_PCT
    LAST_OUT_PCT = out_pct
    LAST_STATE_PCT = state_pct

    del cpu_out, cpu_state_ouput, npu_out, npu_state_out
    gc.collect()
    torch.npu.empty_cache()

    assert out_result == "Pass", (
        f"output precision check failed: pass_rate={out_pct:.4f}%"
    )
    assert state_result == "Pass", (
        f"state precision check failed: pass_rate={state_pct:.4f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", required=True, type=int, help="batch size")
    parser.add_argument(
        "--mtp", required=True, type=int, help="max sequence length, for every batch"
    )
    parser.add_argument("--nk", required=True, type=int, help="0 < nk <= 256")
    parser.add_argument(
        "--nv",
        required=True,
        type=int,
        help="nv should be a multiple of nk and 0 < nv <= 256",
    )
    parser.add_argument("--dk", required=True, type=int, help="0 < dk <= 256")
    parser.add_argument("--dv", required=True, type=int, help="0 < dv <= 256")
    parser.add_argument(
        "--actual_seq_lengths",
        type=int,
        nargs="*",
        help="sequence of every batch should not greated than mtp, len is B",
    )
    parser.add_argument(
        "--ssm_state_indices",
        type=int,
        nargs="*",
        help="map index from input sequence to state matrix, len is B",
    )
    parser.add_argument(
        "--has_gamma", type=str, default="False", help="whether use gamma"
    )
    parser.add_argument(
        "--has_gamma_k", type=str, default="False", help="whether use gamma k"
    )
    parser.add_argument(
        "--has_num_accepted_tokens",
        type=str,
        default="False",
        help="whether use num_accepted_tokens",
    )
    parser.add_argument(
        "--scale_value", type=float, default=None, help="query scaling factor"
    )
    parser.add_argument(
        "--block_num",
        type=int,
        default=None,
        help="block_num should not be less than the sum of actual_seq_lengths",
    )
    parser.add_argument("--data_type", type=str, default="bfloat16", help="bfloat16")
    parser.add_argument("--query_datarange", type=list, default=[-10, 10])
    parser.add_argument("--key_datarange", type=list, default=[-10, 10])
    parser.add_argument("--value_datarange", type=list, default=[-10, 10])
    parser.add_argument("--gamma_datarange", type=list, default=[0, 1])
    parser.add_argument("--gamma_k_datarange", type=list, default=[0, 1])
    parser.add_argument("--beta_datarange", type=list, default=[0, 1])
    parser.add_argument("--state_datarange", type=list, default=[-10, 10])
    args = parser.parse_args()

    if (
        args.data_type == "float16"
        or args.data_type == "FP16"
        or args.data_type == "fp16"
    ):
        data_type = torch.float16
    elif (
        args.data_type == "bfloat16"
        or args.data_type == "BF16"
        or args.data_type == "bf16"
    ):
        data_type = torch.bfloat16
    else:
        raise ValueError("Error: data_type only support bfloat16 and float16")
        sys.exit(1)

    run_recurrent_gated_delta_rule_eager(
        args.B,
        args.mtp,
        args.nk,
        args.nv,
        args.dk,
        args.dv,
        args.actual_seq_lengths,
        args.ssm_state_indices,
        args.has_gamma,
        args.has_gamma_k,
        args.has_num_accepted_tokens,
        args.scale_value,
        args.block_num,
        data_type,
        args.query_datarange,
        args.key_datarange,
        args.value_datarange,
        args.gamma_datarange,
        args.gamma_k_datarange,
        args.beta_datarange,
        args.state_datarange,
    )
