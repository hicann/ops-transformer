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

"""RFA (RainFusionAttention) 自定义 compare 逻辑

纯 Python 白盒实现 ATK cv_fused_double_benchmark 精度判定，不依赖任何 .so 文件。

三方对比模式:
  1. actual_stats    = compute_with_golden(NPU输出, 高精度golden)
  2. benchmark_stats = compute_with_golden(原dtype标杆输出, 高精度golden)
     标杆输出 = 用原dtype完整重新计算RFA算子 (rain_fusion_attention_forward_fp16)
  3. ratio = actual误差 / max(benchmark误差, error_thd)
  4. 4个条件全部满足才Pass: max_re_ratio<=10, avg_re_ratio<=2, rmse_ratio<=2, number_count_ratio<=2

基于 ATK_SC 源码 (ATK2TTK-main-ATK_test-source) 完整复现:
  - double_benchmark_config.py: DoubleBenchmarkCompareStandard + 各 Result 类
  - base_benchmark_config.py: PrecisionStatInfo + do_summary + get_result_msg
  - cv_fused_double_benchmark_compare.py: compute_with_golden + partition_value_range
"""

import importlib.util
import os
import torch


# ============================================================
# 详细ratio打印开关 — 通过环境变量 TTK_VERBOSE_RATIO=1 开启
# ============================================================

_VERBOSE_RATIO = os.environ.get("TTK_VERBOSE_RATIO", "0") == "1"


def _print_ratio_table(result):
    """打印四个 ratio 指标的表格，含计算方式和阈值"""
    detail = result["detail"]
    ratios = result["compare"]
    print(
        "[COMPARE] ┌─────────────────┬──────────────┬──────────────┬───────┐",
        flush=True,
    )
    print(
        "[COMPARE] │ 指标            │ Actual误差   │ Benchmark误差│ ratio │",
        flush=True,
    )
    print(
        "[COMPARE] ├─────────────────┼──────────────┼──────────────┼───────┤",
        flush=True,
    )
    print(
        f"[COMPARE] │ max_re_ratio    │ {detail['Actual_最大相对误差']:.8f} │ "
        f"{detail['Benchmark_最大相对误差']:.8f} │ {ratios['最大相对误差比例']:.4f} │",
        flush=True,
    )
    print(
        f"[COMPARE] │ avg_re_ratio    │ {detail['Actual_平均相对误差']:.8f} │ "
        f"{detail['Benchmark_平均相对误差']:.8f} │ {ratios['平均相对误差比例']:.4f} │",
        flush=True,
    )
    print(
        f"[COMPARE] │ rmse_ratio      │ {detail['Actual_均方根误差']:.8f} │ "
        f"{detail['Benchmark_均方根误差']:.8f} │ {ratios['均方根误差比例']:.4f} │",
        flush=True,
    )
    print(
        f"[COMPARE] │ num_count_ratio │ {detail['Actual_小值域数错数量']:<12} │ "
        f"{detail['Benchmark_小值域数错数量']:<12} │ {ratios['小值域数错误差比例']:.4f} │",
        flush=True,
    )
    print(
        "[COMPARE] └─────────────────┴──────────────┴──────────────┴───────┘",
        flush=True,
    )
    print(
        "[COMPARE] 阈值: max_re_ratio<=10, avg_re_ratio<=2, rmse_ratio<=2, num_count_ratio<=2",
        flush=True,
    )


# ============================================================
# 加载 golden 模块 (获取 rain_fusion_attention_forward_fp16)
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_golden_module():
    path = os.path.join(_THIS_DIR, "golden.py")
    spec = importlib.util.spec_from_file_location("_rfa_golden_for_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_golden_module()
rain_fusion_attention_forward_fp16 = _golden_mod.rain_fusion_attention_forward_fp16
rain_fusion_attention_forward = _golden_mod.rain_fusion_attention_forward


# ============================================================
# 阈值定义 — 与 ATK 源码 double_benchmark_config.py 完全一致
# ============================================================

_ERROR_THDS = {
    torch.float16: 2**-11,  # 0.00048828125
    torch.bfloat16: 2**-8,  # 0.00390625
    torch.float32: 2**-14,  # 0.00006103515625
}

_THRESHOLDS = {
    torch.float16: 2**-16,  # 1.52587890625e-05
    torch.bfloat16: 2**-16,  # 1.52587890625e-05
    torch.float32: 2**-30,  # 9.313225746154785e-10
}

_EB_THDS = {
    torch.float16: 2**-10,  # 0.0009765625
    torch.bfloat16: 2**-7,  # 0.0078125
    torch.float32: 2**-14,  # 0.00006103515625
}

_MAX_RE_RATIO = 10
_AVG_RE_RATIO = 2
_ROOT_MEAN_SQUARED_RATIO = 2
_NUMBER_COUNT_RADIO = 2

# compute_relative_error 中的 eps — 反射 .so 验证为 1e-7
_REL_EPS = 1e-7

# inf/nan 预检常量 — 与 ATK atk/tasks/post_process/utils.py 一致
_INF_NAN_STATE_NAMES = ("normal", "positive_inf", "negative_inf", "nan")
_INF_NAN_FAIL_PREVIEW_LIMIT = 10
_INF_NAN_FAIL_EARLY_STOP_LIMIT = 1000

# ATK INF_NAN_PASS_COMBINATIONS — 28种通过组合 (golden_state, benchmark_state, npu_state)
# 来源: atk.tasks.post_process.utils (运行时导出)
INF_NAN_PASS_COMBINATIONS = frozenset(
    {
        ("nan", "negative_inf", "nan"),
        ("normal", "positive_inf", "normal"),
        ("positive_inf", "normal", "positive_inf"),
        ("negative_inf", "negative_inf", "negative_inf"),
        ("normal", "nan", "normal"),
        ("nan", "negative_inf", "negative_inf"),
        ("normal", "negative_inf", "negative_inf"),
        ("nan", "normal", "nan"),
        ("positive_inf", "positive_inf", "positive_inf"),
        ("negative_inf", "normal", "negative_inf"),
        ("nan", "positive_inf", "nan"),
        ("positive_inf", "nan", "nan"),
        ("negative_inf", "positive_inf", "positive_inf"),
        ("nan", "positive_inf", "positive_inf"),
        ("positive_inf", "nan", "positive_inf"),
        ("normal", "nan", "nan"),
        ("negative_inf", "positive_inf", "negative_inf"),
        ("normal", "positive_inf", "positive_inf"),
        ("positive_inf", "normal", "normal"),
        ("negative_inf", "nan", "nan"),
        ("normal", "negative_inf", "normal"),
        ("normal", "normal", "normal"),
        ("negative_inf", "nan", "negative_inf"),
        ("negative_inf", "normal", "normal"),
        ("nan", "nan", "nan"),
        ("nan", "normal", "normal"),
        ("positive_inf", "negative_inf", "positive_inf"),
        ("positive_inf", "negative_inf", "negative_inf"),
    }
)


def _get_thd(dtype):
    if dtype not in _ERROR_THDS:
        raise ValueError(f"不支持的 dtype: {dtype}")
    return _ERROR_THDS[dtype], _THRESHOLDS[dtype], _EB_THDS[dtype]


# ============================================================
# inf/nan 预检 — 复现 ATK base_benchmark_compare.py 三层流程
# 1. check_invalid_value: 检测是否含 inf/nan
# 2. compute_inf_nan_result: 逐元素分类状态, 用 INF_NAN_PASS_COMBINATIONS 判定
# 3. filter_tensors_by_union_special_mask: 过滤掉含 inf/nan 的元素
# ============================================================


def _check_invalid_value(tensor):
    """检测 tensor 是否含有 inf 或 nan — 复现 atk check_invalid_value"""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return False
    return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()


def _classify_inf_nan_states(values):
    """逐元素分类 inf/nan 状态 — 复现 ATK _classify_inf_nan_states

    返回 int8 tensor: 0=normal, 1=positive_inf, 2=negative_inf, 3=nan
    """
    states = torch.zeros_like(values, dtype=torch.int8)
    states = torch.where(torch.isposinf(values), torch.ones_like(states), states)
    states = torch.where(torch.isneginf(values), torch.full_like(states, 2), states)
    states = torch.where(torch.isnan(values), torch.full_like(states, 3), states)
    return states


def _compute_inf_nan_result(local_output, remote_output, golden):
    """inf/nan 逐元素预检 — 复现 ATK _compute_inf_nan_result_by_position

    Args:
        local_output: NPU输出 (actual)
        remote_output: benchmark输出
        golden: 高精度golden

    Returns:
        (passed, error_info)
    """
    if local_output.shape != remote_output.shape or local_output.shape != golden.shape:
        return False, (
            f"inf/nan check failed: shape mismatch "
            f"local={tuple(local_output.shape)}, remote={tuple(remote_output.shape)}, "
            f"golden={tuple(golden.shape)}"
        )

    local_flat = local_output.reshape(-1)
    remote_flat = remote_output.reshape(-1)
    golden_flat = golden.reshape(-1)

    local_states = _classify_inf_nan_states(local_flat)
    remote_states = _classify_inf_nan_states(remote_flat)
    golden_states = _classify_inf_nan_states(golden_flat)

    special_mask = (local_states != 0) | (remote_states != 0) | (golden_states != 0)
    if not special_mask.any().item():
        return True, "inf/nan check passed (no special values)"

    special_indices = torch.nonzero(special_mask, as_tuple=False).flatten()

    # 逐元素判定
    failed_positions = []
    failed_count = 0
    for idx in special_indices:
        g_state = _INF_NAN_STATE_NAMES[golden_states[idx].item()]
        b_state = _INF_NAN_STATE_NAMES[remote_states[idx].item()]
        n_state = _INF_NAN_STATE_NAMES[local_states[idx].item()]
        if (g_state, b_state, n_state) not in INF_NAN_PASS_COMBINATIONS:
            failed_count += 1
            if len(failed_positions) < _INF_NAN_FAIL_PREVIEW_LIMIT:
                failed_positions.append((idx.item(), (g_state, b_state, n_state)))
            if failed_count >= _INF_NAN_FAIL_EARLY_STOP_LIMIT:
                break

    if failed_positions:
        preview = ", ".join(
            f"idx={idx}:golden={s[0]},benchmark={s[1]},npu={s[2]}"
            for idx, s in failed_positions
        )
        if failed_count > _INF_NAN_FAIL_PREVIEW_LIMIT:
            preview += ", ..."
        return (
            False,
            f"inf/nan check failed: {failed_count} position(s) failed: {preview}",
        )

    return True, "inf/nan check passed (all special-value combinations in pass list)"


def _filter_tensors_by_union_special_mask(local_output, remote_output, golden):
    """过滤掉三方中任何一方含 inf/nan 的元素 — 复现 ATK filter_tensors_by_union_special_mask

    保留三方都有限的元素。如果 inf/nan 预检已通过，
    这里只是去掉特殊值元素，不影响判定结果。

    Returns:
        (filtered_local, filtered_remote, filtered_golden, valid_count)
    """
    local_flat = local_output.reshape(-1)
    remote_flat = remote_output.reshape(-1)
    golden_flat = golden.reshape(-1)

    local_finite = torch.isfinite(local_flat)
    remote_finite = torch.isfinite(remote_flat)
    golden_finite = torch.isfinite(golden_flat)

    all_finite = local_finite & remote_finite & golden_finite
    valid_count = int(all_finite.sum().item())

    filtered_local = local_flat[all_finite]
    filtered_remote = remote_flat[all_finite]
    filtered_golden = golden_flat[all_finite]

    return filtered_local, filtered_remote, filtered_golden, valid_count


# ============================================================
# 底层计算函数 — 复现 atk.tasks.post_process.utils 中的 .so 函数
# ============================================================


def _compute_relative_error(actual, golden):
    """相对误差: |actual - golden| / max(|golden|, 1e-7)"""
    diff = torch.abs(actual - golden)
    golden_abs = torch.abs(golden)
    return diff / torch.maximum(golden_abs, torch.tensor(_REL_EPS, dtype=golden.dtype))


def _compute_root_mean_squared_error(actual, golden):
    """均方根误差: sqrt(mean((actual - golden)^2))"""
    diff = actual - golden
    return torch.sqrt(torch.mean(diff**2)).item()


def _compute_error_balance(actual, golden):
    """误差均衡性: mean((actual - golden) / max(1, |golden|))"""
    tensor_max = torch.maximum(torch.ones_like(golden), torch.abs(golden))
    diff = torch.subtract(actual, golden)
    return torch.mean(torch.div(diff, tensor_max)).item()


# ============================================================
# 核心计算 — 复现 cv_fused_double_benchmark_compare.py compute_with_golden
# ============================================================


def _compute_with_golden(actual, golden, orig_dtype=None):
    """计算 actual vs golden 的各项误差指标

    Args:
        actual: NPU输出或标杆输出
        golden: bm_data (可能是高精度版本)
        orig_dtype: 原始dtype (用于获取error_thd)，None时用golden.dtype
    """
    dtype_for_thd = orig_dtype if orig_dtype is not None else golden.dtype
    error_thd, threshold, eb_thd = _get_thd(dtype_for_thd)

    # Step 1: dtype 对齐 (源码: actual = actual.to(golden.dtype))
    actual = actual.to(golden.dtype)

    # Step 2: flatten
    actual_f = actual.contiguous().flatten()
    golden_f = golden.contiguous().flatten()

    # Step 3: 值域分区 (源码: partition_value_range)
    golden_abs = torch.abs(golden_f)
    big_mask = golden_abs > error_thd
    small_mask = ~big_mask

    # Step 4: 大值域计算 (源码: get_result)
    actual_big = actual_f[big_mask]
    golden_big = golden_f[big_mask]

    if golden_big.numel() == 0:
        max_relative_error = 0
        avg_relative_error = 0
        root_mean_squared_error = 0
        error_balance = 0
    else:
        rel_err = _compute_relative_error(actual_big, golden_big)
        max_relative_error = torch.max(rel_err).item()
        avg_relative_error = torch.mean(rel_err).item()
        root_mean_squared_error = _compute_root_mean_squared_error(
            actual_big, golden_big
        )
        error_balance = _compute_error_balance(actual_big, golden_big)

    # Step 5: 小值域计算 (源码: get_result_small_domain)
    actual_small = actual_f[small_mask]
    golden_small = golden_f[small_mask]

    if golden_small.numel() == 0:
        number_count_error = 0
    else:
        diff_small = torch.abs(actual_small - golden_small)
        number_count_error = int(torch.sum(diff_small > threshold).item())

    return {
        "max_relative_error": max_relative_error,
        "avg_relative_error": avg_relative_error,
        "root_mean_squared_error": root_mean_squared_error,
        "error_balance": error_balance,
        "number_count_error": number_count_error,
    }


def _compute_with_golden_flat(actual_flat, golden_flat, orig_dtype=None):
    """计算已 flatten 且已过滤的 actual vs golden 的各项误差指标

    用于 inf/nan 预检过滤后的数据, 输入已经是 1D tensor 且不含 inf/nan。
    """
    dtype_for_thd = orig_dtype if orig_dtype is not None else golden_flat.dtype
    error_thd, threshold, eb_thd = _get_thd(dtype_for_thd)

    # dtype 对齐
    actual_flat = actual_flat.to(golden_flat.dtype)

    # 值域分区
    golden_abs = torch.abs(golden_flat)
    big_mask = golden_abs > error_thd
    small_mask = ~big_mask

    # 大值域计算
    actual_big = actual_flat[big_mask]
    golden_big = golden_flat[big_mask]

    if golden_big.numel() == 0:
        max_relative_error = 0
        avg_relative_error = 0
        root_mean_squared_error = 0
        error_balance = 0
    else:
        rel_err = _compute_relative_error(actual_big, golden_big)
        max_relative_error = torch.max(rel_err).item()
        avg_relative_error = torch.mean(rel_err).item()
        root_mean_squared_error = _compute_root_mean_squared_error(
            actual_big, golden_big
        )
        error_balance = _compute_error_balance(actual_big, golden_big)

    # 小值域计算
    actual_small = actual_flat[small_mask]
    golden_small = golden_flat[small_mask]

    if golden_small.numel() == 0:
        number_count_error = 0
    else:
        diff_small = torch.abs(actual_small - golden_small)
        number_count_error = int(torch.sum(diff_small > threshold).item())

    return {
        "max_relative_error": max_relative_error,
        "avg_relative_error": avg_relative_error,
        "root_mean_squared_error": root_mean_squared_error,
        "error_balance": error_balance,
        "number_count_error": number_count_error,
    }


# ============================================================
# 双标杆裁决 — 复现 do_summary + get_result_msg
# ============================================================


def _do_summary_and_judge(actual_stats, benchmark_stats, error_thd):
    """
    双标杆裁决: ratio = actual误差 / max(benchmark误差, error_thd)
    4 个条件全部满足才 Pass:
      max_re_ratio <= 10, avg_re_ratio <= 2, rmse_ratio <= 2, number_count_ratio <= 2
    EBResult (error_balance) 用 warn_value, 不参与 Pass/Fail 裁决
    """
    max_re_ratio = actual_stats["max_relative_error"] / max(
        benchmark_stats["max_relative_error"], error_thd
    )
    avg_re_ratio = actual_stats["avg_relative_error"] / max(
        benchmark_stats["avg_relative_error"], error_thd
    )
    rmse_ratio = actual_stats["root_mean_squared_error"] / max(
        benchmark_stats["root_mean_squared_error"], error_thd
    )

    # number_count_ratio: actual数量 / max(benchmark数量, 1)
    bm_count = max(benchmark_stats["number_count_error"], 1)
    number_count_ratio = actual_stats["number_count_error"] / bm_count

    errors = []
    if max_re_ratio > _MAX_RE_RATIO:
        errors.append(
            f"ERROR: 最大相对误差比例({max_re_ratio})超过阈值({_MAX_RE_RATIO})，\n"
        )
    if avg_re_ratio > _AVG_RE_RATIO:
        errors.append(
            f"ERROR: 平均相对误差比例({avg_re_ratio})超过阈值({_AVG_RE_RATIO})，\n"
        )
    if rmse_ratio > _ROOT_MEAN_SQUARED_RATIO:
        errors.append(
            f"ERROR: 均方根误差比例({rmse_ratio})超过阈值({_ROOT_MEAN_SQUARED_RATIO})，\n"
        )
    if number_count_ratio > _NUMBER_COUNT_RADIO:
        errors.append(
            f"ERROR: 小值域数错误差比例({number_count_ratio})超过阈值({_NUMBER_COUNT_RADIO})，\n"
        )

    passed = len(errors) == 0
    msg = "".join(errors)
    return (
        passed,
        msg,
        {
            "最大相对误差比例": max_re_ratio,
            "平均相对误差比例": avg_re_ratio,
            "均方根误差比例": rmse_ratio,
            "小值域数错误差比例": number_count_ratio,
        },
    )


# ============================================================
# 标杆输出计算 — 用原dtype完整重新计算RFA算子
# ============================================================


def _compute_benchmark_output(compare_context, golden):
    """通过 compare_context 获取输入数据, 用fp32输入完整重新计算RFA算子

    ATK框架中 CPU标杆 (remote_output) 是CPU后端第二次调用__call__时,
    输入数据被提升到fp32后计算的结果。因此benchmark比golden精度更高。

    TTK中通过 rain_fusion_attention_forward 用fp32输入完整重新计算来模拟。

    Args:
        compare_context: CompareContext, 包含 input_tensors 和 attributes
        golden: 高精度golden, 用于确定原dtype (取NPU输出dtype)

    Returns:
        benchmark_output: 原dtype的标杆输出tensor
    """
    if compare_context is None or not hasattr(compare_context, "input_tensors"):
        orig_dtype = golden.dtype if golden.dtype != torch.float32 else torch.float16
        return golden.to(orig_dtype)

    input_tensors = compare_context.input_tensors
    attributes = compare_context.attributes if compare_context.attributes else {}

    # input_tensors 布局 (CSV tensor_view_shapes, 排除 output_tensor_indexes):
    # [0] query, [1] key, [2] value, [3] selectIdx, [4] selectNumIdx,
    # [5] attenMaskOptional (None), [6] blockShape (None — 实际在 attributes 中)
    # 注意: output tensor (attentionOut, softmaxLse) 也可能在 input_tensors 中
    # 需要排除 output_tensor_indexes
    output_indexes = set()
    csv_fields = (
        compare_context.csv_fields if hasattr(compare_context, "csv_fields") else {}
    )
    if csv_fields and hasattr(csv_fields, "get"):
        oi = csv_fields.get("output_tensor_indexes", None)
        if oi is not None:
            import ast

            try:
                output_indexes = set(ast.literal_eval(str(oi)))
            except (ValueError, SyntaxError):
                pass

    # 过滤掉输出 tensor，只保留输入
    input_only = [t for i, t in enumerate(input_tensors) if i not in output_indexes]

    query = input_only[0] if len(input_only) > 0 else None
    key = input_only[1] if len(input_only) > 1 else None
    value = input_only[2] if len(input_only) > 2 else None
    select_idx = input_only[3] if len(input_only) > 3 else None
    select_num_idx = input_only[4] if len(input_only) > 4 else None

    # 获取 attributes 中的参数
    scale_value = attributes.get("scaleValue", 1.0)
    if isinstance(scale_value, (list, tuple)):
        scale_value = float(scale_value[0]) if len(scale_value) > 0 else 1.0
    else:
        scale_value = float(scale_value)

    q_input_layout = attributes.get("qInputLayout", "TND")
    kv_input_layout = attributes.get("kvInputLayout", "TND")
    inner_precise = attributes.get("innerPrecise", 0)

    # blockShape 在 attributes 中
    block_shape = attributes.get("blockShape", [64, 128])
    if isinstance(block_shape, (list, tuple)):
        block_shape = [int(block_shape[0]), int(block_shape[1])]
    else:
        block_shape = [64, 128]

    # actualSeqLengths 在 attributes 中
    actual_seq_lengths = attributes.get("actualSeqLengthsOptional", None)
    actual_seq_lengths_kv = attributes.get("actualSeqLengthsKvOptional", None)
    q_seqlen_list = list(actual_seq_lengths) if actual_seq_lengths is not None else []
    kv_seqlen_list = (
        list(actual_seq_lengths_kv) if actual_seq_lengths_kv is not None else []
    )

    # 转为 CPU tensor
    query = query.detach().cpu() if isinstance(query, torch.Tensor) else query
    key = key.detach().cpu() if isinstance(key, torch.Tensor) else key
    value = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    if isinstance(select_idx, torch.Tensor):
        select_idx = select_idx.detach().cpu()
    if isinstance(select_num_idx, torch.Tensor):
        select_num_idx = select_num_idx.detach().cpu()

    # ATK benchmark: 第一次__call__(保存到cpu_benchmark目录)时输入是原dtype(bf16)
    # rain_fusion_attention_forward 内部: query.dtype!=float32 → online_softmax_attention_torch (bf16 matmul)
    # 返回: ref_output.to(float32), 转回原dtype保存
    benchmark_out = rain_fusion_attention_forward_fp16(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        block_shape,
        q_seqlen_list,
        kv_seqlen_list,
        scale_value,
        q_input_layout,
        kv_input_layout,
        inner_precise,
    )
    return benchmark_out


# ============================================================
# 公开接口
# ============================================================


def _atk_precision_check_three_way(actual, golden, benchmark_output):
    """三方对比精度判定 — ATK cv_fused_double_benchmark

    完整复现 ATK base_benchmark_compare.py 的三层流程:
      1. inf/nan 预检 (check_invalid_value → compute_inf_nan_result)
      2. 过滤特殊值 (filter_tensors_by_union_special_mask)
      3. ratio 计算

    Args:
        actual: NPU输出 (原dtype)
        golden: 高精度golden (fp32 for fp16输入)
        benchmark_output: 原dtype标杆输出 (fp16完整计算)

    Returns:
        dict: pass, msg, detail, compare
    """
    orig_dtype = actual.dtype
    error_thd, threshold, eb_thd = _get_thd(orig_dtype)

    # === Step 1: inf/nan 预检 — 复现 ATK compute_accuracy_result 入口 ===
    # ATK: if check_invalid_value(bm_data) or check_invalid_value(local_output):
    #          result, error_info = compute_inf_nan_result(local_output, remote_output, bm_data)
    has_invalid = (
        _check_invalid_value(golden)
        or _check_invalid_value(actual)
        or _check_invalid_value(benchmark_output)
    )

    if has_invalid:
        inf_nan_passed, inf_nan_error_info = _compute_inf_nan_result(
            actual.detach().cpu().to(golden.dtype),
            benchmark_output.detach().cpu().to(golden.dtype),
            golden.detach().cpu(),
        )
        print(
            f"[COMPARE] inf/nan check: {'PASS' if inf_nan_passed else 'FAIL'} - {inf_nan_error_info}",
            flush=True,
        )
        if not inf_nan_passed:
            # inf/nan 预检不通过, 直接 FAIL, 不进入 ratio 计算
            return {
                "pass": False,
                "msg": inf_nan_error_info,
                "detail": {
                    "Actual_最大相对误差": float("nan"),
                    "Actual_平均相对误差": float("nan"),
                    "Actual_均方根误差": float("nan"),
                    "Actual_误差均衡性": float("nan"),
                    "Actual_小值域数错数量": -1,
                    "Benchmark_最大相对误差": float("nan"),
                    "Benchmark_平均相对误差": float("nan"),
                    "Benchmark_均方根误差": float("nan"),
                    "Benchmark_误差均衡性": float("nan"),
                    "Benchmark_小值域数错数量": -1,
                },
                "compare": {
                    "最大相对误差比例": float("nan"),
                    "平均相对误差比例": float("nan"),
                    "均方根误差比例": float("nan"),
                    "小值域数错误差比例": float("nan"),
                },
            }

    # === Step 2: 过滤特殊值 — 复现 ATK filter_tensors_by_union_special_mask ===
    # ATK: filtered_tensors, filter_success, valid_count = filter_tensors_by_union_special_mask(...)
    #       if filter_success and valid_count == 0: return Pass (all elements were special values)
    #       if filter_success: use filtered tensors
    actual_cpu = actual.detach().cpu().to(golden.dtype)
    benchmark_cpu = benchmark_output.detach().cpu().to(golden.dtype)
    golden_cpu = golden.detach().cpu()

    filtered_actual, filtered_benchmark, filtered_golden, valid_count = (
        _filter_tensors_by_union_special_mask(actual_cpu, benchmark_cpu, golden_cpu)
    )

    if has_invalid and valid_count == 0:
        # ATK: all elements were filtered out by the special-value mask → Pass
        print(
            "[COMPARE] all elements filtered out by special-value mask, valid_count=0 → PASS",
            flush=True,
        )
        return {
            "pass": True,
            "msg": "inf/nan check passed, all elements were special values",
            "detail": {
                "Actual_最大相对误差": 0,
                "Actual_平均相对误差": 0,
                "Actual_均方根误差": 0,
                "Actual_误差均衡性": 0,
                "Actual_小值域数错数量": 0,
                "Benchmark_最大相对误差": 0,
                "Benchmark_平均相对误差": 0,
                "Benchmark_均方根误差": 0,
                "Benchmark_误差均衡性": 0,
                "Benchmark_小值域数错数量": 0,
            },
            "compare": {
                "最大相对误差比例": 0,
                "平均相对误差比例": 0,
                "均方根误差比例": 0,
                "小值域数错误差比例": 0,
            },
        }

    # === Step 3: ratio 计算 — 在过滤后的元素上计算 ===
    if has_invalid:
        print(
            f"[COMPARE] filtered out special values, computing ratio on {valid_count} valid elements",
            flush=True,
        )
        actual_stats = _compute_with_golden_flat(
            filtered_actual, filtered_golden, orig_dtype=orig_dtype
        )
        benchmark_stats = _compute_with_golden_flat(
            filtered_benchmark, filtered_golden, orig_dtype=orig_dtype
        )
    else:
        actual_stats = _compute_with_golden(actual, golden, orig_dtype=orig_dtype)
        benchmark_stats = _compute_with_golden(
            benchmark_output, golden, orig_dtype=orig_dtype
        )

    # 双标杆裁决
    passed, msg, ratios = _do_summary_and_judge(
        actual_stats, benchmark_stats, error_thd
    )

    return {
        "pass": passed,
        "msg": msg,
        "detail": {
            "Actual_最大相对误差": actual_stats["max_relative_error"],
            "Actual_平均相对误差": actual_stats["avg_relative_error"],
            "Actual_均方根误差": actual_stats["root_mean_squared_error"],
            "Actual_误差均衡性": actual_stats["error_balance"],
            "Actual_小值域数错数量": actual_stats["number_count_error"],
            "Benchmark_最大相对误差": benchmark_stats["max_relative_error"],
            "Benchmark_平均相对误差": benchmark_stats["avg_relative_error"],
            "Benchmark_均方根误差": benchmark_stats["root_mean_squared_error"],
            "Benchmark_误差均衡性": benchmark_stats["error_balance"],
            "Benchmark_小值域数错数量": benchmark_stats["number_count_error"],
        },
        "compare": ratios,
    }


# ============================================================
# compare 入口
# ============================================================


def compare(*outputs, compare_context=None, **kwargs):
    """精度比较 — 纯 Python 白盒 cv_fused_double_benchmark 三方对比

    RFA 有两个输出: attentionOut, softmaxLseOptional
    只比较 attentionOut (output[0] vs golden[0])
    softmaxLseOptional 不比较 (golden 设为 None)

    三方对比:
      - Actual:    NPU输出 vs 高精度golden
      - Benchmark: 原dtype标杆输出 vs 高精度golden
      - 标杆输出通过 compare_context.input_tensors 获取输入, 用原dtype完整重新计算

    Args:
        *outputs: [output0, output1, golden0, golden1]
                 前半为 NPU 输出，后半为 golden
        compare_context: CompareContext (包含 input_tensors, attributes)

    Returns:
        list[dict]: 每个输出的比较结果
    """
    n_outputs = len(outputs) // 2
    results = []

    for i in range(n_outputs):
        output = outputs[i]
        golden = outputs[i + n_outputs]

        # RFA 有两个输出: attentionOut, softmaxLseOptional
        # 只比较 attentionOut (output[0])
        if i != 0:
            results.append({"pass": True, "precision": 100.0, "error_info": None})
            continue

        if golden is None or output is None:
            results.append({"pass": True, "precision": 100.0, "error_info": None})
            continue

        # 计算fp16标杆输出: 用原dtype完整重新计算RFA算子
        benchmark_output = _compute_benchmark_output(compare_context, golden)

        print(
            f"[COMPARE] output[{i}] actual dtype={output.dtype}, "
            f"golden dtype={golden.dtype}, benchmark dtype={benchmark_output.dtype}",
            flush=True,
        )

        result = _atk_precision_check_three_way(
            output.detach().cpu().to(output.dtype),
            golden.detach().cpu(),
            benchmark_output.detach().cpu().to(output.dtype),
        )
        pass_i = result["pass"]

        if not pass_i:
            print(f"[COMPARE] output[{i}] FAIL: {result['msg'].strip()}", flush=True)
            print(f"[COMPARE] detail: {result['detail']}", flush=True)
            print(f"[COMPARE] compare: {result['compare']}", flush=True)
        else:
            print(f"[COMPARE] output[{i}] PASS", flush=True)

        if _VERBOSE_RATIO:
            _print_ratio_table(result)

        results.append(
            {
                "pass": pass_i,
                "precision": 100.0 if pass_i else 0.0,
                "error_info": None if pass_i else "precision mismatch",
            }
        )

    return results
