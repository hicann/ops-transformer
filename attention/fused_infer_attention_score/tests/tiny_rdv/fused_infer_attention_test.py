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

import random
import torch
import torch_npu
import torchair
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import logging
import datetime
import os
import sys
import argparse
import math
import csv
import traceback

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
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
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
    overflows_count = (
        data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
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

    has_nan_inf = False
    if (
        "nan" in str(real_data)
        or "inf" in str(real_data)
        or "nan" in str(data_compe)
        or "inf" in str(data_compe)
    ):
        has_nan_inf = True

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
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

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


def rand_range(shape, data_range=[-10, 10], dtype=torch.bfloat16, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(
        shape, dtype=dtype, device=device
    )


def get_query_layout(input_layout):
    if (
        input_layout == "BSH"
        or input_layout == "BSH_BNSD"
        or input_layout == "BSH_NBSD"
    ):
        return "BSH"
    elif (
        input_layout == "BSND"
        or input_layout == "BSND_BNSD"
        or input_layout == "BSND_NBSD"
    ):
        return "BSND"
    elif (
        input_layout == "BNSD"
        or input_layout == "BNSD_BSND"
        or input_layout == "BNSD_NBSD"
    ):
        return "BNSD"
    elif input_layout == "TND" or input_layout == "TND_NTD":
        return "TND"
    elif input_layout == "NTD" or input_layout == "NTD_TND":
        return "NTD"
    else:
        return None


def get_attn_out_layout(input_layout):
    if input_layout == "BSH":
        return "BSH"
    elif input_layout == "BSND" or input_layout == "BNSD_BSND":
        return "BSND"
    elif (
        input_layout == "BNSD"
        or input_layout == "BSH_BNSD"
        or input_layout == "BSND_BNSD"
    ):
        return "BNSD"
    elif (
        input_layout == "BSH_NBSD"
        or input_layout == "BSND_NBSD"
        or input_layout == "BNSD_NBSD"
    ):
        return "NBSD"
    elif input_layout == "TND" or input_layout == "NTD_TND":
        return "TND"
    elif input_layout == "NTD" or input_layout == "TND_NTD":
        return "NTD"
    else:
        return None


def get_softmax_lse_layout(input_layout):
    if (
        input_layout == "TND"
        or input_layout == "NTD_TND"
        or input_layout == "NTD"
        or input_layout == "TND_NTD"
    ):
        return "TND"
    else:
        return "BNSD"


def get_shape(layout, b, n, s, d, t):
    if layout == "BSH":
        return (b, s, n * d)
    elif layout == "BSND":
        return (b, s, n, d)
    elif layout == "BNSD":
        return (b, n, s, d)
    elif layout == "TND":
        return (t, n, d)
    elif layout == "NTD":
        return (n, t, d)
    else:
        return None


def get_dtype(data_type):
    if data_type == "float16":
        return torch.float16
    elif data_type == "bfloat16":
        return torch.bfloat16
    elif data_type == "int8":
        return torch.int8


# torch不支持tensorlist
# def gen_tensorlist_kv(query_layout, kv_dtype, b, n, s, d, t):
#     kv_shape = get_shape(query_layout, 1, n, s, d, t)


def dtype_sizeof(data_type):
    if data_type == "float16" or data_type == "bfloat16":
        return 2
    elif data_type == "int8" or data_type == "float8":
        return 1


def get_t(b, act_seq_lens):
    if act_seq_lens is None:
        return 0
    if len(act_seq_lens) == 1:
        return b * act_seq_lens[0]
    sum = 0
    for i in range(b):
        sum += act_seq_lens[i]
    return sum


def update_act_seq_lens_for_tnd(layout, b, act_seq_lens):
    if layout == "TND" or layout == "NTD":
        sum = 0
        for i in range(b):
            sum += act_seq_lens[i]
            act_seq_lens[i] = sum
    return act_seq_lens


def TO_NPU(tensor):
    if tensor is None:
        return None
    else:
        return tensor.to("npu:%s" % DEVICE_ID)


def tensor_to_npu_inplace(*tensors: torch.Tensor) -> list[torch.Tensor]:
    for t in tensors:
        if t is None:
            continue
        else:
            t.data = t.to("npu:%s" % DEVICE_ID).data


def cpu_golden_fused_infer_attention_score(
    query,
    key,
    value,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
):
    return None, None


def get_act_seq_len_by_batch(b_idx, default_s, act_seq_lens):
    if act_seq_lens is None:
        return default_s
    elif len(act_seq_lens) == 1:
        return act_seq_lens[0]
    else:
        return act_seq_lens[b_idx]


def bnsd_to_bsh(bnsd_tensor):
    return bnsd_tensor.permute(0, 2, 1, 3).flatten(
        start_dim=2
    )  # BNSD->BSND, 再将ND合并为H


def bnsd_to_bsnd(bnsd_tensor):
    return bnsd_tensor.permute(0, 2, 1, 3)


def bnsd_to_tnd(bnsd_tensor, b, act_seq_lens):
    if act_seq_lens is None:
        return bnsd_tensor.permute(0, 2, 1, 3).flatten(
            start_dim=0, end_dim=1
        )  # 合并[start_dim, end_dim]维度
    elif len(act_seq_lens) == 1:
        # narrow是从指定维度截取数据
        return (
            torch.narrow(bnsd_tensor, dim=2, start=0, length=act_seq_lens[0])
            .permute(0, 2, 1, 3)
            .flatten(start_dim=0, end_dim=1)
        )
    else:
        t = get_t(b, act_seq_lens)
        tnd_tensor = torch.empty(
            t, bnsd_tensor.shape[1], bnsd_tensor.shape[3], dtype=bnsd_tensor.dtype
        )
        t_idx = 0
        for i in range(b):
            if act_seq_lens[i] > 0:
                tnd_tensor[t_idx : (t_idx + act_seq_lens[i]), :, :] = bnsd_tensor[
                    i, :, 0 : act_seq_lens[i], :
                ].permute(1, 0, 2)
                t_idx = t_idx + act_seq_lens[i]
        return tnd_tensor


def bnsd_to_ntd(bnsd_tensor, b, act_seq_lens):
    if act_seq_lens is None:
        return bnsd_tensor.permute(1, 0, 2, 3).flatten(
            start_dim=0, end_dim=1
        )  # 合并[start_dim, end_dim]维度
    elif len(act_seq_lens) == 1:
        # narrow是从指定维度截取数据
        return (
            torch.narrow(bnsd_tensor, dim=2, start=0, length=act_seq_lens[i])
            .permute(1, 0, 2, 3)
            .flatten(start_dim=1, end_dim=2)
        )
    else:
        t = get_t(b, act_seq_lens)
        ntd_tensor = torch.empty(
            bnsd_tensor.shape[1], t, bnsd_tensor.shape[3], dtype=bnsd_tensor.dtype
        )
        t_idx = 0
        for i in range(b):
            if act_seq_lens[i] > 0:
                ntd_tensor[:, t_idx : (t_idx + act_seq_lens[i]), :] = bnsd_tensor[
                    i, :, 0 : act_seq_lens[i], :
                ]
                t_idx = t_idx + act_seq_lens[i]
        return ntd_tensor


def get_block_table(b, act_seq_lens_kv, block_size):
    s2_max = max(act_seq_lens_kv)
    max_block_num_per_batch = (s2_max + block_size - 1) // block_size
    # print(f"max_block_num_per_batch: {max_block_num_per_batch}")
    block_table = torch.full((b, max_block_num_per_batch), -1, dtype=torch.int32)
    # get block_num
    block_num = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        block_num += (b_seq + block_size - 1) // block_size
    # page cache
    block_id_array = torch.randperm(block_num, dtype=torch.int32)
    index = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        for j in range(b_block_num):
            block_table[i][j] = block_id_array[index]
            index = index + 1
    return block_table


def page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max()) + 1
    kv_cache_bnsd_shape = (
        block_num,
        bnsd_tensor.shape[1],
        block_size,
        bnsd_tensor.shape[3],
    )
    page_cache_tensor = torch.zeros(size=kv_cache_bnsd_shape, dtype=bnsd_tensor.dtype)
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        for j in range(b_block_num):
            src_tensor = bnsd_tensor[i, :, (j * block_size) : ((j + 1) * block_size), :]
            page_cache_tensor[block_table[i][j], :, 0 : src_tensor.shape[1], :] = (
                src_tensor
            )
    return page_cache_tensor


def rearrange_by_layout(bnsd_tensor, layout, b, act_seq_lens):
    if layout == "BNSD":
        return bnsd_tensor
    elif layout == "BSH":
        return bnsd_to_bsh(bnsd_tensor)  # BNSD->BSND, 再将ND合并为H
    elif layout == "BSND":
        return bnsd_to_bsnd(bnsd_tensor)
    elif layout == "TND":
        return bnsd_to_tnd(bnsd_tensor, b, act_seq_lens)
    elif layout == "NTD":
        return bnsd_to_ntd(bnsd_tensor, b, act_seq_lens)
    else:
        return None


def rearrange_by_block_table(
    bnsd_tensor, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype
):
    page_cache_tensor = page_attn_for_bnsd(
        bnsd_tensor, b, act_seq_lens_kv, block_table, block_size
    )
    if kv_storage_mode == "pa_bbh":
        return bnsd_to_bsh(page_cache_tensor)
    elif kv_storage_mode == "pa_bnbd":
        return page_cache_tensor
    elif kv_storage_mode == "pa_nz":
        blk_elem = 32 // dtype_sizeof(kv_dtype)
        page_cache_tensor = page_cache_tensor.reshape(
            page_cache_tensor.shape[0],
            page_cache_tensor.shape[1],
            page_cache_tensor.shape[2],
            page_cache_tensor.shape[3] // blk_elem,
            blk_elem,
        ).permute(0, 1, 3, 2, 4)
        return page_cache_tensor
    else:
        return None


def create_select_mask(m_shape, pre_tokens, next_tokens):
    next_masks = torch.triu(
        torch.ones(m_shape, dtype=torch.bool), diagonal=1 + next_tokens
    )
    pre_masks = torch.tril(
        torch.ones(m_shape, dtype=torch.bool), diagonal=-1 - pre_tokens
    )
    select_mask = next_masks + pre_masks
    return select_mask


def softmax_v1_stable(x, dim=-1):
    """
    数值稳定的Softmax实现
    通过减去最大值避免指数溢出
    """
    # 1. 减去最大值（保持数值稳定）
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    # 2. 计算指数
    exp_x = torch.exp(x_stable)

    # 3. 归一化
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cpu_golden_base(
    query,
    key,
    value,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    quant_scale2=None,
    quant_offset2=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    sparse_mode=0,
    inner_precise=0,
    softmax_lse_flag=False,
    src_date_type=torch.float16,
    compute_date_type=torch.float32,
):
    # 注意: 不能修改输入数据
    b, n1, s1, _ = query.shape
    n2 = key.shape[1]
    s2 = key.shape[2]
    vd = value.shape[-1]
    g = n1 // n2
    mask_value = float("-inf")  # -1e12
    invalid_rows_out_value = 0
    invalid_rows_lse_value = float("inf")

    attn_out = torch.zeros([b, n1, s1, vd])
    softmax_lse = torch.full((b, n1, s1, 1), float("-inf"), dtype=compute_date_type)
    for i in range(b):
        b_s1 = get_act_seq_len_by_batch(i, s1, actual_seq_lengths)
        b_s2 = get_act_seq_len_by_batch(i, s2, actual_seq_lengths_kv)
        q = query[i, :, 0:b_s1, :].to(src_date_type).to(compute_date_type)
        k = key[i, :, 0:b_s2, :].to(src_date_type).to(compute_date_type)
        v = value[i, :, 0:b_s2, :].to(src_date_type).to(compute_date_type)
        q = q.reshape(n2, g, q.shape[1], q.shape[2])
        k = k.reshape(n2, 1, k.shape[1], k.shape[2])
        v = v.reshape(n2, 1, v.shape[1], v.shape[2])
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if query_rope is not None and key_rope is not None:
            q_r = query_rope[i, :, 0:b_s1, :].to(src_date_type).to(compute_date_type)
            k_r = key_rope[i, :, 0:b_s2, :].to(src_date_type).to(compute_date_type)
            q_r = q_r.reshape(n2, g, q_r.shape[1], q_r.shape[2])
            k_r = k_r.reshape(n2, 1, k_r.shape[1], k_r.shape[2])
            rope_attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1))
            attn_scores = attn_scores + rope_attn_scores
        attn_scores = attn_scores * scale
        # Mask
        if atten_mask is not None:
            if atten_mask.dim() == 2:
                attn_scores = attn_scores.masked_fill(
                    atten_mask[0:b_s1, 0:b_s2], mask_value
                )
            elif atten_mask.dim() == 3:
                attn_scores = attn_scores.masked_fill(
                    atten_mask[i, 0:b_s1, 0:b_s2], mask_value
                )
        # invalid_rows
        invalid_rows_flag = (attn_scores == mask_value).all(dim=-1)
        # Softmax
        scores_max = attn_scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(attn_scores - scores_max)
        scores_sum = exp_scores.sum(dim=-1, keepdim=True) + 1e-12
        attn_weights = exp_scores / scores_sum
        attn_out_tmp = torch.matmul(attn_weights, v)
        attn_out_tmp = attn_out_tmp.masked_fill(
            invalid_rows_flag.unsqueeze(-1), invalid_rows_out_value
        )
        attn_out_tmp = attn_out_tmp.reshape(
            n2 * g, attn_out_tmp.shape[2], attn_out_tmp.shape[3]
        )
        attn_out[i, :, 0:b_s1, :] = attn_out_tmp
        attn_out = attn_out.to(src_date_type)
        if softmax_lse_flag:
            softmax_lse_tmp = scores_max + torch.log(scores_sum)
            softmax_lse_tmp = softmax_lse_tmp.masked_fill(
                invalid_rows_flag.unsqueeze(-1), invalid_rows_lse_value
            )
            softmax_lse[i, :, 0:b_s1, :] = softmax_lse_tmp.reshape(
                n2 * g, softmax_lse_tmp.shape[2], softmax_lse_tmp.shape[3]
            )
            softmax_lse = softmax_lse.to(src_date_type)
        else:
            softmax_lse = None
    return attn_out, softmax_lse


def tensor_set_to_contiguous(tensor):
    if not tensor.is_contiguous():
        return tensor.contiguous()
    else:
        return tensor


def generate_cpu_input_and_output(
    b,
    n2,
    g,
    s1,
    s2,
    qk_d,
    v_d,
    rope_d,
    input_layout,
    kv_storage_mode,
    q_dtype,
    kv_dtype,
    out_dtype,
    rope_dtype,
    block_size,
    act_seq_lens_q,
    act_seq_lens_kv,
    enable_softmax_lse,
    enable_mask,
    mask_shape,
    sparse_mode,
    pre_tokens,
    next_tokens,
    enable_learnable_sink,
    inner_precise,
    out_scale_shape,
    out_scale_dtype,
    with_postquant_offset,
    enable_pse,
    pse_mode,
    pse_shape,
    pse_dtype,
    enable_q_padding,
    q_padding_size,
    enable_kv_padding,
    kv_padding_size,
    enable_shared_prefix,
    shared_prefix_s2,
    actual_shared_prefix_len,
    q_quant_mode,
    q_scale_shape,
    q_scale_dtype,
    k_quant_mode,
    k_scale_shape,
    k_scale_dtype,
    v_quant_mode,
    v_scale_shape,
    v_scale_dtype,
    with_antiquant_offset,
):
    if v_d is None:
        v_d = qk_d
    if kv_dtype is None:
        kv_dtype = q_dtype
    if out_dtype is None:
        out_dtype = q_dtype
    if rope_dtype is None:
        rope_dtype = q_dtype

    # ======================== generate source data start ========================
    scale = 1 / math.sqrt(qk_d)
    num_heads = n2 * g
    num_key_value_heads = n2
    softmax_lse_flag = enable_softmax_lse

    query_bnsd_shape = (b, n2 * g, s1, qk_d)
    query = rand_range(query_bnsd_shape, data_range=[-10, 10], dtype=torch.float32)

    key_bnsd_shape = (b, n2, s2, qk_d)
    key = rand_range(key_bnsd_shape, data_range=[-10, 10], dtype=torch.float32)

    value_bnsd_shape = (b, n2, s2, v_d)
    value = rand_range(value_bnsd_shape, data_range=[-10, 10], dtype=torch.float32)

    query_rope_bnsd_shape = (b, n2 * g, s1, rope_d)
    query_rope = (
        rand_range(query_rope_bnsd_shape, data_range=[-10, 10], dtype=torch.float32)
        if rope_d != 0
        else None
    )

    key_rope_bnsd_shape = (b, n2, s2, rope_d)
    key_rope = (
        rand_range(key_rope_bnsd_shape, data_range=[-10, 10], dtype=torch.float32)
        if rope_d != 0
        else None
    )

    atten_mask = None
    if enable_mask:
        if sparse_mode == 0:
            mask_shape = (b, s1, s2)
            atten_mask = torch.rand(mask_shape) < 0.5
            select_mask = create_select_mask((s1, s2), pre_tokens, next_tokens)
            for i in range(b):
                atten_mask[i, :, :] = torch.where(
                    select_mask, select_mask, atten_mask[i, :, :]
                )
        elif sparse_mode == 1:
            mask_shape = (b, s1, s2)
            atten_mask = torch.rand(mask_shape) < 0.5
        elif sparse_mode == 2:
            mask_shape = (s1, s2)
            atten_mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            )
        elif sparse_mode == 3:
            left_up_pre_tokens = 214748647
            left_up_next_tokens = s2 - s1
            atten_mask = create_select_mask(
                (s1, s2), left_up_pre_tokens, left_up_next_tokens
            )
        elif sparse_mode == 4:
            left_up_pre_tokens = s1 - s2 + pre_tokens
            left_up_next_tokens = s2 - s1 + next_tokens
            atten_mask = create_select_mask(
                (s1, s2), left_up_pre_tokens, left_up_next_tokens
            )

    pse_shift = None

    quant_scale2 = None
    quant_offset2 = None

    key_shared_prefix = None
    value_shared_prefix = None

    query_padding_size_tensor = None
    kv_padding_size_tensor = None

    # ======================== generate source data finish ========================

    # ======================== excute base golden start ========================
    cpu_attn_out_base, cpu_softmax_lse_base = cpu_golden_base(
        query,
        key,
        value,
        pse_shift=None,
        atten_mask=atten_mask,
        actual_seq_lengths=act_seq_lens_q,
        actual_seq_lengths_kv=act_seq_lens_kv,
        quant_scale2=None,
        quant_offset2=None,
        query_padding_size=None,
        kv_padding_size=None,
        key_shared_prefix=None,
        value_shared_prefix=None,
        actual_shared_prefix_len=None,
        query_rope=query_rope,
        key_rope=key_rope,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        softmax_lse_flag=softmax_lse_flag,
        src_date_type=get_dtype(q_dtype),
    )
    attn_out_layout = get_attn_out_layout(input_layout)
    cpu_attn_out_base = rearrange_by_layout(
        cpu_attn_out_base, attn_out_layout, b, act_seq_lens_q
    )
    if cpu_softmax_lse_base is not None:
        softmax_lse_layout = get_softmax_lse_layout(input_layout)
        cpu_softmax_lse_base = rearrange_by_layout(
            cpu_softmax_lse_base, softmax_lse_layout, b, act_seq_lens_q
        )
    # ======================== excute base golden finish ========================

    # ======================== generate cpu and npu golden data start ========================
    # query dtype
    if q_dtype == "int8":
        pass
    else:
        query = query.to(get_dtype(q_dtype))
        if query_rope is not None:
            query_rope = query_rope.to(get_dtype(rope_dtype))

    # query shape
    query_layout = get_query_layout(input_layout)
    query = rearrange_by_layout(query, query_layout, b, act_seq_lens_q)
    if query_rope is not None:
        query_rope = rearrange_by_layout(query_rope, query_layout, b, act_seq_lens_q)

    # kv dtype
    if kv_dtype == "int8":
        pass
    else:
        key = key.to(get_dtype(kv_dtype))
        value = value.to(get_dtype(kv_dtype))
        if key_rope is not None:
            key_rope = key_rope.to(get_dtype(rope_dtype))

    block_table = None
    if kv_storage_mode == "continue":
        key = rearrange_by_layout(key, query_layout, b, act_seq_lens_kv)
        value = rearrange_by_layout(value, query_layout, b, act_seq_lens_kv)
        if key_rope is not None:
            key_rope = rearrange_by_layout(key_rope, query_layout, b, act_seq_lens_kv)
    else:
        block_table = get_block_table(b, act_seq_lens_kv, block_size)
        key = rearrange_by_block_table(
            key, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype
        )
        value = rearrange_by_block_table(
            value,
            block_table,
            block_size,
            b,
            act_seq_lens_kv,
            kv_storage_mode,
            kv_dtype,
        )
        if key_rope is not None:
            key_rope = rearrange_by_block_table(
                key_rope,
                block_table,
                block_size,
                b,
                act_seq_lens_kv,
                kv_storage_mode,
                kv_dtype,
            )

    # update act_seq_lens
    act_seq_lens_q = update_act_seq_lens_for_tnd(query_layout, b, act_seq_lens_q)
    if kv_storage_mode == "continue":
        act_seq_lens_kv = update_act_seq_lens_for_tnd(query_layout, b, act_seq_lens_kv)

    # update mask
    if enable_mask:
        if sparse_mode == 2 or sparse_mode == 3 or sparse_mode == 4:
            mask_shape = (2048, 2048)
            atten_mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            )

    dequant_scale1 = None
    quant_scale1 = None
    dequant_scale2 = None
    key_antiquant_scale = None
    key_antiquant_offset = None
    value_antiquant_scale = None
    value_antiquant_offset = None

    # tensor set contiguous
    key = tensor_set_to_contiguous(key)
    value = tensor_set_to_contiguous(value)
    if key_rope is not None:
        key_rope = tensor_set_to_contiguous(key_rope)

    # ======================== generate cpu and npu golden data finish ========================
    return (
        query,
        key,
        value,
        pse_shift,
        atten_mask,
        act_seq_lens_q,
        act_seq_lens_kv,
        dequant_scale1,
        quant_scale1,
        dequant_scale2,
        quant_scale2,
        quant_offset2,
        block_table,
        query_padding_size_tensor,
        kv_padding_size_tensor,
        key_antiquant_scale,
        key_antiquant_offset,
        value_antiquant_scale,
        value_antiquant_offset,
        key_shared_prefix,
        value_shared_prefix,
        actual_shared_prefix_len,
        query_rope,
        key_rope,
        num_heads,
        scale,
        pre_tokens,
        next_tokens,
        input_layout,
        num_key_value_heads,
        sparse_mode,
        inner_precise,
        block_size,
        q_quant_mode,
        softmax_lse_flag,
        k_quant_mode,
        v_quant_mode,
        cpu_attn_out_base,
        cpu_softmax_lse_base,
    )


def run_fia_eager(
    case_name,
    b,
    n2,
    g,
    s1,
    s2,
    qk_d,
    v_d,
    rope_d,
    input_layout,
    kv_storage_mode,
    q_dtype,
    kv_dtype,
    out_dtype,
    rope_dtype,
    block_size,
    act_seq_lens_q,
    act_seq_lens_kv,
    enable_softmax_lse,
    enable_mask,
    mask_shape,
    sparse_mode,
    pre_tokens,
    next_tokens,
    enable_learnable_sink,
    inner_precise,
    out_scale_shape,
    out_scale_dtype,
    with_postquant_offset,
    enable_pse,
    pse_mode,
    pse_shape,
    pse_dtype,
    enable_q_padding,
    q_padding_size,
    enable_kv_padding,
    kv_padding_size,
    enable_shared_prefix,
    shared_prefix_s2,
    actual_shared_prefix_len,
    q_quant_mode,
    q_scale_shape,
    q_scale_dtype,
    k_quant_mode,
    k_scale_shape,
    k_scale_dtype,
    v_quant_mode,
    v_scale_shape,
    v_scale_dtype,
    with_antiquant_offset,
    npu_loop,
):
    try:
        (
            query,
            key,
            value,
            pse_shift,
            atten_mask,
            act_seq_lens_q,
            act_seq_lens_kv,
            dequant_scale1,
            quant_scale1,
            dequant_scale2,
            quant_scale2,
            quant_offset2,
            block_table,
            query_padding_size_tensor,
            kv_padding_size_tensor,
            key_antiquant_scale,
            key_antiquant_offset,
            value_antiquant_scale,
            value_antiquant_offset,
            key_shared_prefix,
            value_shared_prefix,
            actual_shared_prefix_len,
            query_rope,
            key_rope,
            num_heads,
            scale,
            pre_tokens,
            next_tokens,
            input_layout,
            num_key_value_heads,
            sparse_mode,
            inner_precise,
            block_size,
            q_quant_mode,
            softmax_lse_flag,
            k_quant_mode,
            v_quant_mode,
            cpu_attn_out_base,
            cpu_softmax_lse_base,
        ) = generate_cpu_input_and_output(
            b,
            n2,
            g,
            s1,
            s2,
            qk_d,
            v_d,
            rope_d,
            input_layout,
            kv_storage_mode,
            q_dtype,
            kv_dtype,
            out_dtype,
            rope_dtype,
            block_size,
            act_seq_lens_q,
            act_seq_lens_kv,
            enable_softmax_lse,
            enable_mask,
            mask_shape,
            sparse_mode,
            pre_tokens,
            next_tokens,
            enable_learnable_sink,
            inner_precise,
            out_scale_shape,
            out_scale_dtype,
            with_postquant_offset,
            enable_pse,
            pse_mode,
            pse_shape,
            pse_dtype,
            enable_q_padding,
            q_padding_size,
            enable_kv_padding,
            kv_padding_size,
            enable_shared_prefix,
            shared_prefix_s2,
            actual_shared_prefix_len,
            q_quant_mode,
            q_scale_shape,
            q_scale_dtype,
            k_quant_mode,
            k_scale_shape,
            k_scale_dtype,
            v_quant_mode,
            v_scale_shape,
            v_scale_dtype,
            with_antiquant_offset,
        )
    except Exception as e:
        print_log(case_name + " exec cpu Failed!", level="ERROR")
        err_type = type(e).__name__
        err_msg = str(e)
        trace = traceback.format_exc()
        print_log(trace)
        fulfill_percent_all = [0]
        res_all = ["Cpu Failed"]
        return fulfill_percent_all, res_all

    # ======================== excute cpu and npu start ========================
    cpu_attn_out, cpu_softmax_lse = cpu_golden_fused_infer_attention_score(
        query,
        key,
        value,
        pse_shift=pse_shift,
        atten_mask=atten_mask,
        actual_seq_lengths=act_seq_lens_q,
        actual_seq_lengths_kv=act_seq_lens_kv,
        dequant_scale1=dequant_scale1,
        quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
        block_table=block_table,
        query_padding_size=query_padding_size_tensor,
        kv_padding_size=kv_padding_size_tensor,
        key_antiquant_scale=key_antiquant_scale,
        key_antiquant_offset=key_antiquant_offset,
        value_antiquant_scale=value_antiquant_scale,
        value_antiquant_offset=value_antiquant_offset,
        key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix,
        actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope,
        key_rope=key_rope,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=q_quant_mode,
        softmax_lse_flag=softmax_lse_flag,
        key_antiquant_mode=k_quant_mode,
        value_antiquant_mode=v_quant_mode,
    )

    try:
        tensor_to_npu_inplace(
            query,
            key,
            value,
            pse_shift,
            atten_mask,
            dequant_scale1,
            quant_scale1,
            dequant_scale2,
            quant_scale2,
            quant_offset2,
            block_table,
            query_padding_size_tensor,
            kv_padding_size_tensor,
            key_antiquant_scale,
            key_antiquant_offset,
            value_antiquant_scale,
            value_antiquant_offset,
            key_shared_prefix,
            value_shared_prefix,
            actual_shared_prefix_len,
            query_rope,
            key_rope,
        )

        for i in range(npu_loop):
            npu_attn_out, npu_softmax_lse = torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                pse_shift=pse_shift,
                atten_mask=atten_mask,
                actual_seq_lengths=act_seq_lens_q,
                actual_seq_lengths_kv=act_seq_lens_kv,
                dequant_scale1=dequant_scale1,
                quant_scale1=quant_scale1,
                dequant_scale2=dequant_scale2,
                quant_scale2=quant_scale2,
                quant_offset2=quant_offset2,
                block_table=block_table,
                query_padding_size=query_padding_size_tensor,
                kv_padding_size=kv_padding_size_tensor,
                key_antiquant_scale=key_antiquant_scale,
                key_antiquant_offset=key_antiquant_offset,
                value_antiquant_scale=value_antiquant_scale,
                value_antiquant_offset=value_antiquant_offset,
                key_shared_prefix=key_shared_prefix,
                value_shared_prefix=value_shared_prefix,
                actual_shared_prefix_len=actual_shared_prefix_len,
                query_rope=query_rope,
                key_rope=key_rope,
                num_heads=num_heads,
                scale=scale,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                input_layout=input_layout,
                num_key_value_heads=num_key_value_heads,
                sparse_mode=sparse_mode,
                inner_precise=inner_precise,
                block_size=block_size,
                antiquant_mode=q_quant_mode,
                softmax_lse_flag=softmax_lse_flag,
                key_antiquant_mode=k_quant_mode,
                value_antiquant_mode=v_quant_mode,
            )
    except Exception as e:
        print_log(case_name + " exec npu Failed!", level="ERROR")
        err_type = type(e).__name__
        err_msg = str(e)
        trace = traceback.format_exc()
        print_log(trace)
        fulfill_percent_all = [0]
        res_all = ["Npu Failed"]
        return fulfill_percent_all, res_all
    # ======================== excute cpu and npu finish ========================

    try:
        # ======================== start check all result ========================
        fulfill_percent_all = []
        res_all = []
        # ======================== cpu_base vs npu start ========================
        if cpu_attn_out_base is not None and npu_attn_out is not None:
            print_log(
                "--------------------------------------------------------------cpu_base vs npu attn_out-------------------------------------------------------------"
            )
            fulfill_percent, res = check_result(
                cpu_attn_out_base.to(torch.float32),
                npu_attn_out.cpu().to(torch.float32),
                get_dtype(q_dtype),
            )
            fulfill_percent_all.append(fulfill_percent)
            res_all.append(res)
        if cpu_softmax_lse_base is not None and npu_softmax_lse is not None:
            print_log(
                "--------------------------------------------------------------cpu_base vs npu softmax_lse-------------------------------------------------------------"
            )
            fulfill_percent, res = check_result(
                cpu_softmax_lse_base.to(torch.float32),
                npu_softmax_lse.cpu().to(torch.float32),
                get_dtype(q_dtype),
            )
            fulfill_percent_all.append(fulfill_percent)
            res_all.append(res)
        # ======================== cpu_base vs npu finish ========================

        # ======================== cpu vs npu start ========================
        if cpu_attn_out is not None and npu_attn_out is not None:
            print_log(
                "--------------------------------------------------------------cpu_base vs npu attn_out-------------------------------------------------------------"
            )
            fulfill_percent, res = check_result(
                cpu_attn_out.to(torch.float32),
                npu_attn_out.cpu().to(torch.float32),
                get_dtype(q_dtype),
            )
            fulfill_percent_all.append(fulfill_percent)
            res_all.append(res)
        if cpu_softmax_lse is not None and npu_softmax_lse is not None:
            print_log(
                "--------------------------------------------------------------cpu_base vs npu softmax_lse-------------------------------------------------------------"
            )
            fulfill_percent, res = check_result(
                cpu_softmax_lse.to(torch.float32),
                npu_softmax_lse.cpu().to(torch.float32),
                get_dtype(q_dtype),
            )
            fulfill_percent_all.append(fulfill_percent)
            res_all.append(res)
        print_log(
            "--------------------------------------------------------------check result end-------------------------------------------------------------"
        )
        # ======================== cpu vs npu finish ========================
        # ======================== end check all result ========================
    except Exception as e:
        print_log(case_name + " exec compare Failed!", level="ERROR")
        err_type = type(e).__name__
        err_msg = str(e)
        trace = traceback.format_exc()
        print_log(trace)
        fulfill_percent_all = [0]
        res_all = ["Compare Failed"]
        return fulfill_percent_all, res_all
    return fulfill_percent_all, res_all


def parse_shape(s):
    try:
        # 支持输入 3,4 或 [3,4] 或 (3,4)
        s = s.strip("[]()")
        return tuple(map(int, s.split(",")))
    except:
        raise argparse.ArgumentTypeError("Shape 格式错误，示例：3,4 或 [3,4]")


def write_res_to_csv(csv_path, case_name, fulfill_percent_all, res_all):
    fulfill_percent_all_join = "/".join(str(item) for item in fulfill_percent_all)
    res_all_join = "/".join(str(item) for item in res_all)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["case_name", "precision", "result"])
        writer.writerow([case_name, fulfill_percent_all_join, res_all_join])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required base params
    parser.add_argument(
        "--case_name", default="fia_case", type=str, help="name of testcase"
    )
    parser.add_argument("--b", required=True, type=int, help="batch size")
    parser.add_argument("--n2", required=True, type=int, help="head_num of value")
    parser.add_argument(
        "--g", required=True, type=int, help="g = key_head_num / value_head_num"
    )
    parser.add_argument(
        "--s1", required=True, type=int, help="max sequence length of query"
    )
    parser.add_argument(
        "--s2", required=True, type=int, help="max sequence length of key and value"
    )
    parser.add_argument(
        "--qk_d", required=True, type=int, default=128, help="head_dim of query and key"
    )
    # other base params
    parser.add_argument(
        "--v_d", type=int, default=None, help="head_dim of value, default is qk_d"
    )
    parser.add_argument(
        "--rope_d",
        type=int,
        default=0,
        help="head_dim of query_rope and key_rope, 0: no exist query_rope and key_rope",
    )
    parser.add_argument(
        "--input_layout",
        type=str,
        default="BSH",
        choices=[
            "BSH",
            "BSND",
            "BNSD",
            "BNSD_BSND",
            "BSH_BNSD",
            "BSND_BNSD",
            "TND",
            "NTD",
            "TND_NTD",
            "NTD_TND",
            "BSH_NBSD",
            "BSND_NBSD",
            "BNSD_NBSD",
        ],
        help="layout of query and attention_out",
    )
    parser.add_argument(
        "--kv_storage_mode",
        type=str,
        default="continue",
        choices=["continue", "pa_bbh", "pa_bnbd", "pa_nz"],
        help="for get layout of key and value",
    )
    parser.add_argument(
        "--q_dtype", type=str, default="bfloat16", help="dtype of query"
    )
    parser.add_argument(
        "--kv_dtype",
        type=str,
        default=None,
        help="dtype of key and value, default is q_dtype",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default=None,
        help="dtype of attention_out, default is q_dtype",
    )
    parser.add_argument(
        "--rope_dtype",
        type=str,
        default=None,
        help="dtype of query_rope and key_rope, default is q_dtype",
    )
    # page attention
    parser.add_argument(
        "--block_size",
        type=int,
        default=0,
        help="when paga_attention, block_size of kv cache",
    )
    # q and kv actual_seq_lens
    parser.add_argument(
        "--act_seq_lens_q",
        type=int,
        nargs="*",
        default=None,
        help="actual sequence of query for every batch, should not greated than s1, len is 1/B/>B",
    )
    parser.add_argument(
        "--act_seq_lens_kv",
        type=int,
        nargs="*",
        default=None,
        help="sequence of key and value for every batch, should not greated than s2, len is 1/B/>B",
    )
    # lse
    parser.add_argument(
        "--enable_softmax_lse", action="store_true", help="output softmax_lse"
    )
    # mask
    parser.add_argument(
        "--enable_mask", action="store_true", help="enable attention mask"
    )
    parser.add_argument(
        "--mask_shape", type=parse_shape, default=None, help="shape of mask"
    )
    parser.add_argument(
        "--sparse_mode", type=int, default=0, choices=[0, 1, 2, 3, 4], help=""
    )
    parser.add_argument("--pre_tokens", type=int, default=2147483647, help="")
    parser.add_argument("--next_tokens", type=int, default=2147483647, help="")
    # learnable sink
    parser.add_argument(
        "--enable_learnable_sink", action="store_true", help="enable learnable_sink"
    )
    # high precise/high pref, row invalid
    parser.add_argument(
        "--inner_precise", type=int, default=0, choices=[0, 1, 2, 3], help="0/1/2/3"
    )
    # post quant
    parser.add_argument(
        "--out_scale_shape",
        type=parse_shape,
        default=None,
        help="shape of quant scale of attention out",
    )
    parser.add_argument(
        "--out_scale_dtype",
        type=str,
        default=None,
        help="dtype of quant scale of attention out",
    )
    parser.add_argument("--with_postquant_offset", action="store_true", help="")
    # pse
    parser.add_argument("--enable_pse", action="store_true", help="")
    parser.add_argument("--pse_mode", type=int, default=0, choices=[0, 2, 3], help="")
    parser.add_argument(
        "--pse_shape", type=parse_shape, default=None, help="shape of pse shift"
    )
    parser.add_argument(
        "--pse_dtype", type=str, default=None, help="dtype of pse shift"
    )
    # left padding
    parser.add_argument(
        "--enable_q_padding", action="store_true", help="enable query left padding"
    )
    parser.add_argument("--q_padding_size", type=int, default=None, help="")
    parser.add_argument(
        "--enable_kv_padding", action="store_true", help="enable kv left padding"
    )
    parser.add_argument("--kv_padding_size", type=int, default=None, help="")
    # system prefix
    parser.add_argument(
        "--enable_shared_prefix", action="store_true", help="enable kv shared prefix"
    )
    parser.add_argument("--shared_prefix_s2", type=int, default=None, help="")
    parser.add_argument("--actual_shared_prefix_len", type=int, default=None, help="")
    # antiquant and fullquant
    parser.add_argument("--q_quant_mode", type=int, default=0, help="")
    parser.add_argument(
        "--q_scale_shape",
        type=parse_shape,
        default=None,
        help="shape of query dequant scale",
    )
    parser.add_argument(
        "--q_scale_dtype", type=str, default=None, help="dtype of query dequant scale"
    )
    parser.add_argument("--k_quant_mode", type=int, default=0, help="")
    parser.add_argument(
        "--k_scale_shape",
        type=parse_shape,
        default=None,
        help="shape of key dequant scale",
    )
    parser.add_argument(
        "--k_scale_dtype", type=str, default=None, help="dtype of key dequant scale"
    )
    parser.add_argument("--v_quant_mode", type=int, default=0, help="")
    parser.add_argument(
        "--v_scale_shape",
        type=parse_shape,
        default=None,
        help="shape of value dequant scale",
    )
    parser.add_argument(
        "--v_scale_dtype", type=str, default=None, help="dtype of value dequant scale"
    )
    parser.add_argument("--with_antiquant_offset", action="store_true", help="")
    # script control params
    parser.add_argument(
        "--npu_loop",
        type=int,
        default=1,
        help="loop times of the operator executed on the NPU",
    )
    args = parser.parse_args()

    fulfill_percent_all, res_all = run_fia_eager(
        args.case_name,
        args.b,
        args.n2,
        args.g,
        args.s1,
        args.s2,
        args.qk_d,
        args.v_d,
        args.rope_d,
        args.input_layout,
        args.kv_storage_mode,
        args.q_dtype,
        args.kv_dtype,
        args.out_dtype,
        args.rope_dtype,
        args.block_size,
        args.act_seq_lens_q,
        args.act_seq_lens_kv,
        args.enable_softmax_lse,
        args.enable_mask,
        args.mask_shape,
        args.sparse_mode,
        args.pre_tokens,
        args.next_tokens,
        args.enable_learnable_sink,
        args.inner_precise,
        args.out_scale_shape,
        args.out_scale_dtype,
        args.with_postquant_offset,
        args.enable_pse,
        args.pse_mode,
        args.pse_shape,
        args.pse_dtype,
        args.enable_q_padding,
        args.q_padding_size,
        args.enable_kv_padding,
        args.kv_padding_size,
        args.enable_shared_prefix,
        args.shared_prefix_s2,
        args.actual_shared_prefix_len,
        args.q_quant_mode,
        args.q_scale_shape,
        args.q_scale_dtype,
        args.k_quant_mode,
        args.k_scale_shape,
        args.k_scale_dtype,
        args.v_quant_mode,
        args.v_scale_shape,
        args.v_scale_dtype,
        args.with_antiquant_offset,
        args.npu_loop,
    )

    write_res_to_csv(
        "./fia_test_result.csv", args.case_name, fulfill_percent_all, res_all
    )
