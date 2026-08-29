# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import random
import torch
import torch_npu
from cann_ops_transformer.ops import quant_compressor
import numpy as np
import logging
import datetime
import os
import sys
import argparse
from hif8_utils import hif8_to_fp32_lut, sample_hif8_value_uniform

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch.npu.config.allow_internal_format = True

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


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
    print(
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
    real_data = result.cpu().numpy().ravel()
    data_compe = expect.cpu().numpy().ravel()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return 100.0, "Pass"
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.size, data_compe.size)
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

    if np.array_equal(real_data, data_compe):
        fulfill_percent = 100.0
        pct_thd_pct = (1 - pct_thd) * 100.0
        display_output_np_isclose(real_data, data_compe, start, end)
        print_log(
            "---------------------------------------------------------------------------------------"
        )
        print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
        print_log(
            "---------------------------------------------------------------------------------------"
        )
        print_log(
            "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
            % (rtol, atol, pct_thd_pct, fulfill_percent, "Pass")
        )
        return fulfill_percent, "Pass"

    overflows_count = int(np.count_nonzero(np.isinf(data_compe))) + int(
        np.count_nonzero(np.isnan(data_compe))
    )
    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                data_compe[np.isinf(data_compe)][0:10],
                data_compe[np.isnan(data_compe)][0:10],
            )
        )

    if str(real_data.dtype) == "bfloat16":
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    elif str(real_data.dtype) == "float8_e4m3fn":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    elif str(real_data.dtype) == "float8_e5m2":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        nan_pos_inf = np.isposinf(real_data)
        real_data[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(real_data)
        real_data[nan_neg_inf] = -57344

        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        nan_pos_inf = np.isposinf(data_compe)
        data_compe[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(data_compe)
        data_compe[nan_neg_inf] = -57344

        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    err_idx = np.where(~diff_result)[0]

    b2 = float((1.0 / (1 << 14)) / diff_thd)
    eps = 10e-10
    if err_idx.size > 0:
        if str(data_compe.dtype) == "bool":
            data_compe = data_compe.astype(np.int8)
            real_data = real_data.astype(np.int8)
        err_real = real_data[err_idx]
        err_expect = data_compe[err_idx]
        err_abs = np.abs(err_expect - err_real)
        err_max = np.maximum(np.abs(err_real), np.abs(err_expect))
        err_b = np.maximum(err_max, b2) + eps
        err_diff = err_abs / err_b
    else:
        err_diff = np.array([], dtype=np.float64)

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    display_output_np_isclose(real_data, data_compe, start, end)
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
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return fulfill_percent, result


def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
    if seqused is not None:
        return seqused[batch_idx]
    else:
        if cu_seqlens is not None:
            return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            return S


def softmax_columns(z):
    """
    对输入矩阵 z 按列计算 Softmax。
    参数:
    z -- 一个二维NumPy数组（矩阵），形状为 (行, 列)
    返回:
    一个与 z 形状相同的数组，其中每一列都是一个概率分布（该列元素之和为1）。
    """
    # 仅支持float32计算
    # 1. 数值稳定技巧：减去每列的最大值，防止指数计算溢出
    z_max = np.max(z, axis=0, keepdims=True)  # 计算每列最大值，保持二维维度
    z_stable = z - z_max

    # 2. 计算指数
    exp_z = np.exp(z_stable)

    # 3.对齐NPU exp 硬件行为：FP32 denorm flush-to-zero
    exp_z[exp_z < np.finfo(np.float32).tiny] = 0.0

    # 4. 按列求和（axis=0），并进行归一化
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# state.shape is (block_num, block_size, coff * head_dim), is numpy type
# new_state.shape is (s, coff * head_dim), data writed to state
# block_table.shape is (B, (s_max + block_size - 1) // block_size)
def write_state_page_cache(
    state,
    update_position,
    sc_new_state,
    b_idx,
    start_seq_idx,
    end_seq_idx,
    block_table,
    cache_mode=quant_compressor.CacheMode.LINEAR_BUFFER,
):
    block_size = state.shape[1]  # 应该从state，不是从block_table
    seq_cnt = end_seq_idx - start_seq_idx
    finish_cnt = 0
    if cache_mode == quant_compressor.CacheMode.RING_BUFFER:
        block_id = block_table[b_idx]
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        if cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER:
            block_id = block_table[b_idx][cur_seq_id // block_size]
        block_start_seq_id = cur_seq_id % block_size
        can_write_seq_cnt = block_size - block_start_seq_id
        if can_write_seq_cnt > seq_cnt - finish_cnt:
            can_write_seq_cnt = seq_cnt - finish_cnt
        # block_id为0表示block无效,不写数据
        if (
            cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER and block_id != 0
        ) or cache_mode == quant_compressor.CacheMode.RING_BUFFER:
            state[
                block_id : (block_id + 1),
                block_start_seq_id : (block_start_seq_id + can_write_seq_cnt),
                :,
            ] = sc_new_state[finish_cnt : (finish_cnt + can_write_seq_cnt), :]
            update_position[
                block_id : (block_id + 1),
                block_start_seq_id : (block_start_seq_id + can_write_seq_cnt),
                :,
            ] = True
        finish_cnt = finish_cnt + can_write_seq_cnt


# state.shape is (block_num, block_size, coff * head_dim)
# block_table.shape is (B, (s_max + block_size - 1) // block_size)
# when coff_id = 0, read pre data; when coff_id = 1, read cur data
# out.shape is (end_seq_idx - start_seq_idx, head_dim)
def read_state_page_cache(
    state,
    b_idx,
    start_seq_idx,
    end_seq_idx,
    block_table,
    d_start,
    d_end,
    cache_mode=quant_compressor.CacheMode.LINEAR_BUFFER,
):
    result = np.zeros(
        shape=(end_seq_idx - start_seq_idx, d_end - d_start), dtype=np.float32
    )
    block_size = state.shape[1]  # 应该从state，不是从block_table
    seq_cnt = end_seq_idx - start_seq_idx
    if cache_mode == quant_compressor.CacheMode.RING_BUFFER:
        block_id = block_table[b_idx]
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        if cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER:
            block_id = block_table[b_idx][cur_seq_id // block_size]
        block_start_seq_id = cur_seq_id % block_size
        can_read_seq_cnt = block_size - block_start_seq_id
        if can_read_seq_cnt > seq_cnt - finish_cnt:
            can_read_seq_cnt = seq_cnt - finish_cnt
        # 如果block_id为0, 输出错误日志, 但是依然读数据
        if cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER and block_id == 0:
            print(
                f"Error Info: [read_state_page_cache] block_id of block_table is 0, b_idx={b_idx} cur_seq_id={cur_seq_id} block_size={block_size}"
            )
        result[finish_cnt : (finish_cnt + can_read_seq_cnt), :] = state[
            block_id : (block_id + 1),
            block_start_seq_id : (block_start_seq_id + can_read_seq_cnt),
            d_start:d_end,
        ]
        finish_cnt = finish_cnt + can_read_seq_cnt
    return result


def cpu_compressor(
    x,
    wkv,
    wgate,
    x_descale,
    wkv_descale,
    wgate_descale,
    kv_state,
    score_state,
    update_kv_position,
    update_score_position,
    ape,
    block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    cmp_ratio=4,
    coff=1,
    cache_mode=quant_compressor.CacheMode.LINEAR_BUFFER,
    quant_mode=quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL,
):
    x = x.numpy()
    wkv = wkv.numpy()
    wgate = wgate.numpy()
    x_descale = x_descale.to(torch.float32).numpy()
    wkv_descale = wkv_descale.to(torch.float32).numpy()
    wgate_descale = wgate_descale.to(torch.float32).numpy()
    kv_state = kv_state.numpy()
    score_state = score_state.numpy()
    ape = ape.numpy()
    matmul_dtype = np.float32
    # 算子实现是先将两个descale相乘
    x_fp32 = hif8_to_fp32_lut(x)
    new_kv_state = np.matmul(x_fp32, hif8_to_fp32_lut(wkv.T), dtype=matmul_dtype)
    new_score_state = np.matmul(x_fp32, hif8_to_fp32_lut(wgate.T), dtype=matmul_dtype)

    if (
        quant_mode
        == quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL
    ):
        new_kv_state = new_kv_state * (x_descale * wkv_descale)
        new_score_state = new_score_state * (x_descale * wgate_descale)
    else:
        raise ValueError(f"Error: unsupport quantmode {quant_mode}")

    B = len(start_pos)
    head_dim = wkv.shape[0] // coff
    bs_combine_flag = False
    if cu_seqlens is not None:
        bs_combine_flag = True

    # 将B和S合轴, 便于后面取索引
    if not bs_combine_flag:
        S = x.shape[1]
        new_kv_state = new_kv_state.reshape(B * S, new_kv_state.shape[-1])
        new_score_state = new_score_state.reshape(B * S, new_score_state.shape[-1])
        cmp_kv = np.zeros(
            shape=(B, (S + cmp_ratio - 1) // cmp_ratio, head_dim), dtype=matmul_dtype
        )
    else:
        cmp_kv = np.zeros(
            shape=(min(x.shape[0], x.shape[0] // cmp_ratio + B), head_dim),
            dtype=matmul_dtype,
        )

    cmp_kv_mask = np.zeros_like(cmp_kv, dtype=bool)
    out_cu_seqlen = [0] * (B + 1)
    out_seqused = [0] * B

    out_sum_sc_cnt = 0
    for b_idx in range(B):
        batch_out_sc_id = 0
        batch_start_pos = start_pos[b_idx]
        # get_seq_used_by_batch
        if seqused is not None:
            batch_seq_used = seqused[b_idx]
        else:
            if not bs_combine_flag:
                batch_seq_used = x.shape[1]  # x的shape为3维, BSh
            else:
                batch_seq_used = (
                    cu_seqlens[b_idx + 1] - cu_seqlens[b_idx]
                )  # x的shape为2维, Th
        # 小于compress_seq_id的seq_id为需要压缩的token
        compress_seq_id = (batch_start_pos + batch_seq_used) // cmp_ratio * cmp_ratio

        batch_seq_idx = 0
        while batch_seq_idx < batch_seq_used:
            # 获取要处理的S方向起始和结束点, 已加start_pos
            start_seq_idx = batch_start_pos + batch_seq_idx
            end_seq_idx = start_seq_idx // cmp_ratio * cmp_ratio + cmp_ratio
            if end_seq_idx > (batch_start_pos + batch_seq_used):
                end_seq_idx = batch_start_pos + batch_seq_used

            # calc offset T of one sc, for get data from new state
            base_offset = cu_seqlens[b_idx] if bs_combine_flag else b_idx * x.shape[1]
            start_offset = base_offset + (start_seq_idx - batch_start_pos)
            end_offset = base_offset + (end_seq_idx - batch_start_pos)

            # add ape
            start_seq_id_in_sc = start_seq_idx % cmp_ratio
            end_seq_idx_in_sc = start_seq_id_in_sc + (end_seq_idx - start_seq_idx)
            new_score_state[start_offset:end_offset, :] = np.add(
                new_score_state[start_offset:end_offset, :],
                ape[start_seq_id_in_sc:end_seq_idx_in_sc, :],
            )
            # 1.判断块是否需要存储到state
            # 2.判断块是否需要压缩
            if cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER:
                save_flag = True
            else:
                save_flag = (
                    True
                    if start_seq_idx >= (compress_seq_id - (coff - 1) * cmp_ratio)
                    else False
                )
            compress_flag = True if start_seq_idx < compress_seq_id else False

            if save_flag:
                tmp_kv_state = new_kv_state[start_offset:end_offset, :]
                tmp_score_state = new_score_state[start_offset:end_offset, :]
                write_state_page_cache(
                    kv_state,
                    update_kv_position,
                    tmp_kv_state,
                    b_idx,
                    start_seq_idx,
                    end_seq_idx,
                    block_table,
                    cache_mode=cache_mode,
                )
                write_state_page_cache(
                    score_state,
                    update_score_position,
                    tmp_score_state,
                    b_idx,
                    start_seq_idx,
                    end_seq_idx,
                    block_table,
                    cache_mode=cache_mode,
                )

            if compress_flag:
                # init value and shape of one sc state
                sc_kv_state = np.zeros(
                    shape=(coff, cmp_ratio, head_dim), dtype=matmul_dtype
                )
                sc_score_state = np.full(
                    shape=(coff, cmp_ratio, head_dim),
                    fill_value=-float("inf"),
                    dtype=matmul_dtype,
                )

                # fill cur data
                coff_id = coff - 1
                d_start = coff_id * head_dim
                d_end = (coff_id + 1) * head_dim
                cnt_from_state = 0
                if batch_start_pos == start_seq_idx:
                    # 第一块
                    cnt_from_state = batch_start_pos % cmp_ratio
                    if cnt_from_state > 0:
                        copy_start_seq_id = batch_start_pos - cnt_from_state
                        copy_end_seq_id = batch_start_pos
                        sc_kv_state[coff_id, 0:cnt_from_state, :] = (
                            read_state_page_cache(
                                kv_state,
                                b_idx,
                                copy_start_seq_id,
                                copy_end_seq_id,
                                block_table,
                                d_start,
                                d_end,
                                cache_mode=cache_mode,
                            )
                        )
                        sc_score_state[coff_id, 0:cnt_from_state, :] = (
                            read_state_page_cache(
                                score_state,
                                b_idx,
                                copy_start_seq_id,
                                copy_end_seq_id,
                                block_table,
                                d_start,
                                d_end,
                                cache_mode=cache_mode,
                            )
                        )
                sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = new_kv_state[
                    start_offset:end_offset, d_start:d_end
                ]
                sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = new_score_state[
                    start_offset:end_offset, d_start:d_end
                ]

                # fill pre data
                if coff == 2:
                    coff_id = 0
                    d_start = coff_id * head_dim
                    d_end = (coff_id + 1) * head_dim
                    cnt_from_state = 0
                    if batch_start_pos == start_seq_idx:
                        # 第一块
                        # when batch_start_pos < cmp_ratio, no exist pre data, no need copy;
                        # others, all pre data need copy from state
                        cnt_from_state = cmp_ratio
                        if batch_start_pos >= cmp_ratio:
                            copy_start_seq_id = (
                                batch_start_pos
                                - batch_start_pos % cmp_ratio
                                - cmp_ratio
                            )
                            copy_end_seq_id = copy_start_seq_id + cnt_from_state
                            sc_kv_state[coff_id, 0:cnt_from_state, :] = (
                                read_state_page_cache(
                                    kv_state,
                                    b_idx,
                                    copy_start_seq_id,
                                    copy_end_seq_id,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                )
                            )
                            sc_score_state[coff_id, 0:cnt_from_state, :] = (
                                read_state_page_cache(
                                    score_state,
                                    b_idx,
                                    copy_start_seq_id,
                                    copy_end_seq_id,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                )
                            )
                    elif start_seq_idx - cmp_ratio < batch_start_pos:
                        # 第二块, pre数据部分在state中
                        cnt_from_state = batch_start_pos % cmp_ratio
                        if cnt_from_state > 0:  # 不可能为0
                            copy_start_seq_id = (
                                batch_start_pos - batch_start_pos % cmp_ratio
                            )
                            copy_end_seq_id = batch_start_pos
                            sc_kv_state[coff_id, 0:cnt_from_state, :] = (
                                read_state_page_cache(
                                    kv_state,
                                    b_idx,
                                    copy_start_seq_id,
                                    copy_end_seq_id,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                )
                            )
                            sc_score_state[coff_id, 0:cnt_from_state, :] = (
                                read_state_page_cache(
                                    score_state,
                                    b_idx,
                                    copy_start_seq_id,
                                    copy_end_seq_id,
                                    block_table,
                                    d_start,
                                    d_end,
                                    cache_mode=cache_mode,
                                )
                            )
                    if cnt_from_state < cmp_ratio:
                        # 需要拷贝的数据就在start_offset的前面
                        pre_start_offset = start_offset - (cmp_ratio - cnt_from_state)
                        pre_end_offset = start_offset
                        sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = (
                            new_kv_state[pre_start_offset:pre_end_offset, d_start:d_end]
                        )
                        sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = (
                            new_score_state[
                                pre_start_offset:pre_end_offset, d_start:d_end
                            ]
                        )

                # softmax_columns for score state
                sc_kv_state = sc_kv_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = sc_score_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = softmax_columns(sc_score_state)
                # kv * score
                sc_data = sc_kv_state * sc_score_state
                # reduce sum
                sc_cmp_kv = np.sum(sc_data, axis=0, keepdims=True)
                if not bs_combine_flag:
                    cmp_kv[b_idx, batch_out_sc_id, :] = sc_cmp_kv
                    cmp_kv_mask[b_idx, batch_out_sc_id, :] = 1
                else:
                    cmp_kv[out_sum_sc_cnt, :] = sc_cmp_kv
                    cmp_kv_mask[out_sum_sc_cnt, :] = 1
                batch_out_sc_id = batch_out_sc_id + 1
                out_sum_sc_cnt = out_sum_sc_cnt + 1

            # update loop idx
            batch_seq_idx = end_seq_idx - batch_start_pos
        out_cu_seqlen[b_idx + 1] = out_sum_sc_cnt
        out_seqused[b_idx] = batch_out_sc_id

    print(f"out_cu_seqlen: {out_cu_seqlen}")
    print(f"out_seqused: {out_seqused}")
    print(f"cmp_kv.shape: {cmp_kv.shape}")
    cmp_kv_torch = torch.tensor(cmp_kv).to(torch.bfloat16)
    return cmp_kv_torch, cmp_kv_mask


def run_compressor_eager(
    B,
    S_max,
    head_dim,
    coff,
    cmp_ratio,
    bs_combine_flag,
    S=0,
    start_pos=None,
    seqused=None,
    cu_seqlens=None,
    block_size=128,
    data_type=torch.bfloat16,
    hidden_size=4096,
    save_state_seqlens=None,
    cache_mode=quant_compressor.CacheMode.LINEAR_BUFFER,
    quant_mode=quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL,
    x_datarange=[-10, 10],
    wkv_datarange=[-10, 10],
    wgate_datarange=[-10, 10],
    ape_datarange=[-10, 10],
    kv_state_datarange=[-10, 10],
    score_state_datarange=[-10, 10],
    x_descale_datarange=[0.001, 1.0],
    wkv_descale_datarange=[0.001, 1.0],
    wgate_descale_datarange=[0.001, 1.0],
):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== set input params finish ========================
    if bs_combine_flag:
        # cu_seqlens = [0, 1] # (B+1,), None时表示非BSh，否则为Th
        if cu_seqlens is None:
            print("Error: layout of x is [T, hidden_size], cu_seqlens is required!!!")
            return
        old_S = S
        if seqused is not None:
            S = max(seqused)
        else:
            S = 0
            for i in range(B):
                if (cu_seqlens[i + 1] - cu_seqlens[i]) > S:
                    S = cu_seqlens[i + 1] - cu_seqlens[i]
        print(
            f"Warning: layout of x is [T, hidden_size], S={old_S}, it is modified to S={S}!!!"
        )
    else:
        if cu_seqlens is not None:
            print(
                "Warning: layout of x is [B, S, hidden_size], but cu_seqlens is not None, it is modified to None!!!"
            )
        cu_seqlens = None
    exist_start_pos = True
    if start_pos is None:
        exist_start_pos = False
        start_pos = [0] * B
    # ======================== set input params finish ========================
    # ======================== check input params start ========================
    if start_pos is not None:
        if len(start_pos) != B:
            print(
                f"Error: the len of start_pos is {len(start_pos)}, it should be B({B})"
            )
            return
    if seqused is not None:
        if len(seqused) != B:
            print(f"Error: the len of seqused is {len(seqused)}, it should be B({B})")
            return
    if cu_seqlens is not None:
        if len(cu_seqlens) != (B + 1):
            print(
                f"Error: the len of cu_seqlens is {len(cu_seqlens)}, it should be equal to B({B}) + 1"
            )
            return
    if bs_combine_flag:
        for i in range(B):
            if start_pos[i] + (cu_seqlens[i + 1] - cu_seqlens[i]) > S_max:
                print(
                    f"Error: for batch {i} when shape of x is (T, hidden_size), start_pos[{i}] + (cu_seqlens[{i + 1}] - cu_seqlens[{i}]) > S_max, "
                    f"start_pos[{i}]={start_pos[i]}, cu_seqlens[{i + 1}]={cu_seqlens[i + 1]}, cu_seqlens[{i}]={cu_seqlens[i]}, S_max={S_max}"
                )
                return
            if seqused is not None:
                if seqused[i] > (cu_seqlens[i + 1] - cu_seqlens[i]):
                    print(
                        f"Error: for batch {i} when shape of x is (T, hidden_size), seqused[{i}] > (cu_seqlens[{i + 1}] - cu_seqlens[{i}]), "
                        f"seqused[{i}]={seqused[i]}, cu_seqlens[{i + 1}]={cu_seqlens[i + 1]}, cu_seqlens[{i}]={cu_seqlens[i]}"
                    )
                    return
    else:
        for i in range(B):
            if start_pos[i] + S > S_max:
                print(
                    f"Error: for batch {i} when shape of x is (B, S, hidden_size), start_pos[{i}] + S > S_max, start_pos[{i}]={start_pos[i]}, S={S}, S_max={S_max}"
                )
                return
            if seqused is not None:
                if seqused[i] > S:
                    print(
                        f"Error: for batch {i} when shape of x is (B, S, hidden_size), seqused[{i}] > S, seqused[{i}]={seqused[i]}, S={S}"
                    )
                    return
    if save_state_seqlens is not None:
        if len(save_state_seqlens) != B:
            print(
                f"Error: the len of save_state_seqlens is {len(save_state_seqlens)}, it should be equal to B({B})"
            )
            return
        for i in range(B):
            b_seqused = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
            if b_seqused < save_state_seqlens[i]:
                print(
                    f"Error: for batch {i}, b_seqused < save_state_seqlens[{i}], b_seqused={b_seqused}, save_state_seqlens[{i}]={save_state_seqlens[i]}"
                )
                return

    # ======================== check input params finish ========================
    # ======================== gen input data start =============================
    # page state
    if cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER:
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = B * max_block_num_per_batch
        next_block_id = 1
        print(f"max_block_num_per_batch: {max_block_num_per_batch}")
        block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
        for i in range(B):
            # 需要读取state的范围
            cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
            cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
            if start_pos[i] % cmp_ratio == 0:
                cur_end = start_pos[i]
            cur_end = min(cur_end, start_pos[i] + S)
            cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
            cur_end_block_id = (cur_end - 1) // block_size
            for j in range(cur_start_block_id, cur_end_block_id + 1):
                if next_block_id < block_num:
                    block_table[i][j] = next_block_id
                    next_block_id = next_block_id + 1
            # 需要写入state的范围
            end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
            if save_state_seqlens is not None:
                next_start = start_pos[i] + end_pos - save_state_seqlens[i]
                next_end = start_pos[i] + end_pos
            else:
                next_start = (
                    start_pos[i] + end_pos
                ) // cmp_ratio * cmp_ratio - cmp_ratio
                next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
                if (start_pos[i] + end_pos) % cmp_ratio == 0:
                    next_end = start_pos[i] + end_pos
            next_end = min(next_end, start_pos[i] + end_pos)
            next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
            next_end_block_id = (next_end - 1) // block_size
            for j in range(next_start_block_id, next_end_block_id + 1):
                if next_block_id < block_num and block_table[i][j] == 0:
                    block_table[i][j] = next_block_id
                    next_block_id = next_block_id + 1

        actual_block_num = next_block_id
        if B == 0:
            kv_state = torch.tensor(
                np.random.uniform(
                    kv_state_datarange[0],
                    kv_state_datarange[1],
                    (0, block_size, coff * head_dim),
                )
            ).to(torch.float32)
            score_state = torch.tensor(
                np.random.uniform(
                    score_state_datarange[0],
                    score_state_datarange[1],
                    (0, block_size, coff * head_dim),
                )
            ).to(torch.float32)
        else:
            kv_state = torch.tensor(
                np.random.uniform(
                    kv_state_datarange[0],
                    kv_state_datarange[1],
                    (actual_block_num, block_size, coff * head_dim),
                )
            ).to(torch.float32)
            score_state = torch.tensor(
                np.random.uniform(
                    score_state_datarange[0],
                    score_state_datarange[1],
                    (actual_block_num, block_size, coff * head_dim),
                )
            ).to(torch.float32)
    else:
        block_table = torch.tensor(random.sample(list(range(B)), B), dtype=torch.int32)
        block_size = min(
            (2 * cmp_ratio + S - 1) if coff == 2 else (cmp_ratio + S - 1), 1024
        )
        if B == 0:
            kv_state = torch.tensor(
                np.random.uniform(
                    kv_state_datarange[0],
                    kv_state_datarange[1],
                    (0, block_size, coff * head_dim),
                )
            ).to(torch.float32)
            score_state = torch.tensor(
                np.random.uniform(
                    score_state_datarange[0],
                    score_state_datarange[1],
                    (0, block_size, coff * head_dim),
                )
            ).to(torch.float32)
        else:
            kv_state = torch.tensor(
                np.random.uniform(
                    kv_state_datarange[0],
                    kv_state_datarange[1],
                    (B, block_size, coff * head_dim),
                )
            ).to(torch.float32)
            score_state = torch.tensor(
                np.random.uniform(
                    score_state_datarange[0],
                    score_state_datarange[1],
                    (B, block_size, coff * head_dim),
                )
            ).to(torch.float32)

    # other input
    if bs_combine_flag:
        x_shape = (cu_seqlens[-1], hidden_size)
    else:
        x_shape = (B, S, hidden_size)

    x = torch.tensor(
        sample_hif8_value_uniform(x_datarange[0], x_datarange[1], x_shape).astype(
            np.uint8
        )
    )
    wkv = torch.tensor(
        sample_hif8_value_uniform(
            wkv_datarange[0], wkv_datarange[1], (coff * head_dim, hidden_size)
        ).astype(np.uint8)
    )
    wgate = torch.tensor(
        sample_hif8_value_uniform(
            wgate_datarange[0], wgate_datarange[1], (coff * head_dim, hidden_size)
        ).astype(np.uint8)
    )

    x_descale = torch.tensor(
        np.random.uniform(x_descale_datarange[0], x_descale_datarange[1], (1,))
    ).to(torch.float32)
    wkv_descale = torch.tensor(
        np.random.uniform(
            wkv_descale_datarange[0], wkv_descale_datarange[1], (coff * head_dim,)
        )
    ).to(torch.float32)
    wgate_descale = torch.tensor(
        np.random.uniform(
            wgate_descale_datarange[0], wgate_descale_datarange[1], (coff * head_dim,)
        )
    ).to(torch.float32)
    ape = torch.tensor(
        np.random.uniform(
            ape_datarange[0], ape_datarange[1], (cmp_ratio, coff * head_dim)
        )
    ).to(torch.float32)
    print(f"start_pos={start_pos}")
    print(f"seqused={seqused}")
    print(f"cu_seqlens={cu_seqlens}")

    # ======================== gen input data finish =============================

    # ======================== execute cpu start =================================
    cpu_kv_state = kv_state.clone()
    cpu_score_state = score_state.clone()

    update_kv = torch.zeros(
        (cpu_kv_state.shape[0], cpu_kv_state.shape[1], cpu_kv_state.shape[2]),
        dtype=torch.bool,
    )
    update_score = torch.zeros(
        (cpu_score_state.shape[0], cpu_score_state.shape[1], cpu_score_state.shape[2]),
        dtype=torch.bool,
    )

    cpu_out, cmp_kv_mask = cpu_compressor(
        x,
        wkv,
        wgate,
        x_descale,
        wkv_descale,
        wgate_descale,
        cpu_kv_state,
        cpu_score_state,
        update_kv,
        update_score,
        ape,
        block_table=block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        cmp_ratio=cmp_ratio,
        coff=coff,
        cache_mode=cache_mode,
        quant_mode=quant_mode,
    )

    # ======================== execute cpu finish ================================

    # ======================== execute npu start =================================

    x = x.to("npu:%s" % DEVICE_ID)
    wkv = wkv.to("npu:%s" % DEVICE_ID)
    wgate = wgate.to("npu:%s" % DEVICE_ID)
    x_descale = x_descale.to("npu:%s" % DEVICE_ID)
    wkv_descale = wkv_descale.to("npu:%s" % DEVICE_ID)
    wgate_descale = wgate_descale.to("npu:%s" % DEVICE_ID)
    is_contiguous_flag = True
    if (
        cache_mode == quant_compressor.CacheMode.LINEAR_BUFFER
    ):  # 连续buffer(支持在第二维非连续)
        if is_contiguous_flag:
            state_cache = torch.zeros(
                (kv_state.shape[0], kv_state.shape[1], 2 * kv_state.shape[2])
            )
            state_cache[:, :, : state_cache.shape[2] // 2] = kv_state.clone()
            state_cache[:, :, state_cache.shape[2] // 2 :] = score_state.clone()
            state_cache = state_cache.to("npu:%s" % DEVICE_ID)
        else:
            layer_stride = random.randint(1, 10)
            print(f"layer_stride: {layer_stride}")
            # state_cache_pad: [batch, (num_layers + stride) * head_dim * 2]
            state_cache_pad = torch.zeros(
                (
                    kv_state.shape[0],
                    (kv_state.shape[1] + layer_stride) * kv_state.shape[2] * 2,
                )
            )
            print(f"state_cache_pad: shape {state_cache_pad.shape}")
            # 使用 as_strided 创建非连续视图
            # stride(0) = (b + stride) * c, stride(1) = c, stride(2) = 1
            state_cache = torch.as_strided(
                state_cache_pad,
                size=(kv_state.shape[0], kv_state.shape[1], kv_state.shape[2] * 2),
                stride=(
                    (kv_state.shape[1] + layer_stride) * kv_state.shape[2] * 2,
                    kv_state.shape[2] * 2,
                    1,
                ),
            )
            # 填充数据
            state_cache[:, :, : kv_state.shape[2]] = kv_state.clone()
            state_cache[:, :, kv_state.shape[2] :] = score_state.clone()
            state_cache_pad = state_cache_pad.to("npu:%s" % DEVICE_ID)
            print(
                f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride: {state_cache.stride()}"
            )
    else:
        layer_pad = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_pad - 1)
        print(f"layer_pad: {layer_pad}")
        print(f"layer_start_idx: {layer_start_idx}")
        state_cache_pad = torch.zeros(
            (kv_state.shape[0], kv_state.shape[1] * kv_state.shape[2] * 2 + layer_pad)
        )
        print(f"state_cache_pad: shape {state_cache_pad.shape}")
        state_cache = state_cache_pad[
            :,
            layer_start_idx : layer_start_idx
            + kv_state.shape[1] * kv_state.shape[2] * 2,
        ].view(-1, kv_state.shape[1], kv_state.shape[2] * 2)
        state_cache[:, :, : state_cache.shape[2] // 2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2] // 2 :] = score_state.clone()
        state_cache_pad = state_cache_pad.to("npu:%s" % DEVICE_ID)
        print(
            f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride0: {state_cache.stride(0)}"
        )

    ape = ape.to("npu:%s" % DEVICE_ID)
    block_table = block_table.to("npu:%s" % DEVICE_ID)
    start_pos = torch.tensor(start_pos).to(torch.int32).to("npu:%s" % DEVICE_ID)
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).to("npu:%s" % DEVICE_ID)
    if seqused is not None:
        seqused = torch.tensor(seqused).to(torch.int32).to("npu:%s" % DEVICE_ID)
    # ======================== execute npu finish ================================
    # start run custom ops
    print(f"x: shape {x.shape}, dtype: {x.dtype}")
    print(f"wkv: shape {wkv.shape}, dtype: {wkv.dtype}")
    print(f"wgate: shape {wgate.shape}, dtype: {wgate.dtype}")
    print(f"x_descale: shape {x_descale.shape}, dtype: {x_descale.dtype}")
    print(f"wkv_descale: shape {wkv_descale.shape}, dtype: {wkv_descale.dtype}")
    print(f"wgate_descale: shape {wgate_descale.shape}, dtype: {wgate_descale.dtype}")
    print(f"kv_state: shape {kv_state.shape}, dtype: {kv_state.dtype}")
    print(f"score_state: shape {score_state.shape}, dtype: {score_state.dtype}")
    print(f"ape: shape {ape.shape}, dtype: {ape.dtype}")
    print(f"block_table: shape {block_table.shape}, dtype: {block_table.dtype}")

    npu_out = torch.ops.cann_ops_transformer.quant_compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        x_descale=x_descale,
        wkv_descale=wkv_descale,
        wgate_descale=wgate_descale,
        state_block_table=block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos if exist_start_pos else None,
        quant_mode=quant_mode,
        cmp_ratio=cmp_ratio,
        coff=coff,
        cache_mode=cache_mode,
    )
    print(f"cmp_kv: shape {npu_out.shape}, dtype: {npu_out.dtype}")

    # # 计算两个张量的差值
    # diff = torch.sub(cpu_out[cmp_kv_mask], npu_out.cpu()[cmp_kv_mask])

    # # 计算差值的最大值
    # max_diff = torch.max(torch.abs(diff))

    # # 计算差值的最小值
    # min_diff = torch.min(torch.abs(diff))

    # # 计算差值的均值
    # mean_diff = torch.mean(torch.abs(diff))

    # print("最大差异值:", max_diff.item())
    # print("最小差异值:", min_diff.item())
    # print("平均差异值:", mean_diff.item())

    # 结果精度对比
    data_type = str(npu_out.dtype).replace("torch.", "")
    # 提前将 state_cache 搬到 CPU, 后续 slice/indexing/dtype 全在 CPU 完成, 避免 NPU 上不必要的切片操作
    state_cache_cpu = state_cache.cpu()
    check_results = []

    print(
        "--------------------------------------------------------------check result-------------------------------------------------------------"
    )
    _, r1 = check_result(
        cpu_out[cmp_kv_mask].to(torch.float32),
        npu_out.cpu()[cmp_kv_mask].to(torch.float32),
        data_type,
    )
    check_results.append(("cmp_kv", r1))
    print(
        "--------------------------------------------------------------check kv state update-------------------------------------------------------------"
    )
    _, r2 = check_result(
        cpu_kv_state[update_kv].to(torch.float32),
        state_cache_cpu[:, :, : state_cache.shape[2] // 2][update_kv].to(torch.float32),
        data_type,
    )
    check_results.append(("kv_state_update", r2))
    print(
        "--------------------------------------------------------------check score state update-------------------------------------------------------------"
    )
    _, r3 = check_result(
        cpu_score_state[update_score].to(torch.float32),
        state_cache_cpu[:, :, state_cache.shape[2] // 2 :][update_score].to(
            torch.float32
        ),
        data_type,
    )
    check_results.append(("score_state_update", r3))
    print(
        "--------------------------------------------------------------check kv state origin-------------------------------------------------------------"
    )
    _, r4 = check_result(
        cpu_kv_state[~update_kv].to(torch.float32),
        state_cache_cpu[:, :, : state_cache.shape[2] // 2][~update_kv].to(
            torch.float32
        ),
        data_type,
        0.0,
    )
    check_results.append(("kv_state_origin", r4))
    print(
        "--------------------------------------------------------------check score state origin-------------------------------------------------------------"
    )
    _, r5 = check_result(
        cpu_score_state[~update_score].to(torch.float32),
        state_cache_cpu[:, :, state_cache.shape[2] // 2 :][~update_score].to(
            torch.float32
        ),
        data_type,
        0.0,
    )
    check_results.append(("score_state_origin", r5))

    failed = [name for name, r in check_results if r != "Pass"]
    assert len(failed) == 0, f"Check failed: {failed}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", required=True, type=int, help="batch size")
    parser.add_argument(
        "--S_max",
        required=True,
        type=int,
        help="max sequence length, for gen block_table",
    )
    parser.add_argument("--head_dim", required=True, type=int)
    parser.add_argument(
        "--coff", required=True, type=int, choices=[1, 2], help="1:no overlap 2:overlap"
    )
    parser.add_argument(
        "--cmp_ratio",
        required=True,
        type=int,
        choices=[2, 4, 8, 16, 32, 64, 128],
        help="compress cmp_ratio token to one",
    )
    parser.add_argument(
        "--bs_combine_flag",
        action="store_true",
        help="Include this flag to set x.shape to [T, hidden_size]",
    )
    parser.add_argument(
        "--S", type=int, default=0, help="cur sequence length dim of x when BSh"
    )
    parser.add_argument(
        "--start_pos", type=int, nargs="*", help="start_pos of every batch, len is B"
    )
    parser.add_argument(
        "--seqused", type=int, nargs="*", help="sequence which is used, len is B"
    )
    parser.add_argument(
        "--cu_seqlens",
        type=int,
        nargs="*",
        help="when x is [T, h], it is required, len is B+1",
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument(
        "--cache_mode", type=int, default=int(quant_compressor.CacheMode.LINEAR_BUFFER)
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="bfloat16 or float16",
    )
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument(
        "--quant_mode",
        type=int,
        choices=[
            int(quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL)
        ],
        default=int(
            quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL
        ),
        help="quantization mode, only support 1(HiF8)",
    )
    parser.add_argument(
        "--save_state_seqlens",
        type=int,
        nargs="*",
        help="save to kv_state and score_state of seq id in [start_pos + seqused - save_state_seqlens, start_pos + seqused)",
    )
    args = parser.parse_args()

    if args.data_type == "float16":
        data_type = torch.float16
    elif args.data_type == "bfloat16":
        data_type = torch.bfloat16
    else:
        raise ValueError("Error: data_type only support bfloat16 and float16")
        sys.exit(1)

    run_compressor_eager(
        args.B,
        args.S_max,
        args.head_dim,
        args.coff,
        args.cmp_ratio,
        args.bs_combine_flag,
        args.S,
        args.start_pos,
        args.seqused,
        args.cu_seqlens,
        args.block_size,
        data_type,
        args.hidden_size,
        args.save_state_seqlens,
        args.cache_mode,
        args.quant_mode,
    )
