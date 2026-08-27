# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
昇腾算子精度比对脚本 (method=12 风格)

实现要点 (与 aclnn_fuzz/libs/tools.py 的 precision_method=12 对齐):
  1. 双标杆比对: NPU / benchmark (GPU/同dtype) / golden (高精度)
     - 两路独立计算指标 (rst_npu, rst_gpu)
     - rst_npu.check_result(rst_gpu, new=True) 三态判定 (Pass/warning/Failed)
  2. 指标体系: 大值域 + 小值域 + Inf/Nan 统计
     - 大值: diff_big_max/avg/sum, diff_big_ratio_max/avg/rmse,
            err_w1/k1/k5/h1 (按 red_range 阈值)
     - 小值: total_small_num, err_small_num (> small_value_atol)
     - 全局: diff_rmse, rst_eb, diff_eb
  3. 判等公式 (check_result_debug):
        X > bench * X_rtol → error
        X > bench         → warning
     new=True 时对 GPU 端 diff_big_ratio_max/avg, diff_rmse 做下界抬高
     (fp16=2^-11, bf16=2^-8) 抑制极小值噪声
  4. FP8 dtype (e5m2/e4m3fn/hifloat8): numpy 转 fp32 再 .to(qDtype) 对齐输入 dtype
  5. 阈值来自硬编码默认表 (与 aclnn_op_bm_cmp_std.json / aclnn_op_red_range.json 默认值一致)

API:
  - check_result(golden, npu_result)               单标杆模式
  - check_result(golden, benchmark, npu_result)    CV 双标杆模式 (method=12)
  - check_result_cv(golden, benchmark, npu_result) 显式 CV 双标杆
  - compute_cv_report(...)                         返回 CVPrecisionReport
  - display_cv_report(report)                      打印报告
"""

import logging
import math
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 阈值表 (与 aclnn_op_bm_cmp_std.json / aclnn_op_red_range.json 默认值一致)
# ============================================================

# bm_cmp_std: 大小值域划分 + 各项 Ratio 容忍倍率
#   max_re_rtol : diff_big_ratio_max 容忍倍率
#   avg_re_rtol : diff_big_ratio_avg / err_small_num / diff_big_sum 容忍倍率
#   rmse_rtol   : diff_big_ratio_rmse / diff_rmse 容忍倍率
#   small_value : |golden| < small_value → 归入小值域
#   small_value_atol : 小值域内 |actual-golden| > 此值计 err_small_num
BM_CMP_STD_DEFAULT: Dict[str, Dict[str, float]] = {
    "fp32": {
        "max_re_rtol": 10.0,
        "avg_re_rtol": 2.0,
        "rmse_rtol": 2.0,
        "small_value": 0.000001,
        "small_value_atol": 0.000000,
    },
    "fp16": {
        "max_re_rtol": 10.0,
        "avg_re_rtol": 2.0,
        "rmse_rtol": 2.0,
        "small_value": 0.001,
        "small_value_atol": 0.001,
    },
    "bf16": {
        "max_re_rtol": 10.0,
        "avg_re_rtol": 2.0,
        "rmse_rtol": 2.0,
        "small_value": 0.0000001,
        "small_value_atol": 0.004,
    },
    "hf32": {
        "max_re_rtol": 10.0,
        "avg_re_rtol": 2.0,
        "rmse_rtol": 2.0,
        "small_value": 0.000001,
        "small_value_atol": 0.000000,
    },
}

# red_range: err_w1/k1/k5/h1 阈值, 以 "/" 分隔 (与 aclnn_op_red_range.json 默认值一致)
RED_RANGE_DEFAULT: Dict[str, str] = {
    "fp32": "0.000001/0.00001/0.0001/0.0005",
    "fp16": "0.001/0.002/0.005/0.01",
    "bf16": "0.001/0.002/0.005/0.01",
    "hf32": "0.001/0.002/0.005/0.01",
}


def _normalize_dtype_key(dtype) -> str:
    """torch.dtype / numpy dtype / 字符串 → 'fp16' / 'bf16' / 'fp32' / 'hf32' 等内部 key"""
    s = str(dtype).lower()
    mapping = {
        "torch.float16": "fp16",
        "float16": "fp16",
        "fp16": "fp16",
        "torch.bfloat16": "bf16",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "torch.float32": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "torch.float64": "fp32",
        "float64": "fp32",
        "fp64": "fp32",
        "torch.half": "fp16",
        "torch.float8_e4m3fn": "fp16",
        "torch.float8_e5m2": "fp16",
        "float8_e4m3fn": "fp16",
        "float8_e5m2": "fp16",
        "torch_npu.float8_e8m0fnu": "fp16",
        "float8_e8m0": "fp16",
        "torch_npu.hifloat8": "fp16",
        "hifloat8": "fp16",
        "torch.int8": "fp16",
        "int8": "fp16",
    }
    return mapping.get(s, "fp16")


def _get_bm_cmp_std(output_dtype) -> Dict[str, float]:
    key = _normalize_dtype_key(output_dtype)
    return BM_CMP_STD_DEFAULT.get(key, BM_CMP_STD_DEFAULT["fp16"])


def _get_red_list(output_dtype) -> List[float]:
    key = _normalize_dtype_key(output_dtype)
    red_str = RED_RANGE_DEFAULT.get(key, RED_RANGE_DEFAULT["fp16"])
    return [float(x) for x in red_str.split("/")]


# ============================================================
# 2. Result 类 (字段完全对齐 aclnn_fuzz/libs/tools.py 的 Result)
# ============================================================


class Result:
    """单路 (NPU 或 benchmark) 精度统计结果"""

    def __init__(
        self,
        result_name,
        total_big_num,
        total_big_ratio,
        diff_big_max,
        diff_big_avg,
        diff_big_sum,
        err_w1_num,
        err_w1_ratio,
        err_k1_num,
        err_k1_ratio,
        err_k5_num,
        err_k5_ratio,
        err_h1_num,
        err_h1_ratio,
        total_small_num,
        total_small_ratio,
        err_small_num,
        err_small_ratio,
        diff_rmse,
        rst_eb,
        diff_eb,
        diff_big_ratio_max,
        diff_big_ratio_avg,
        diff_big_ratio_rmse,
        bm_cmp_std,
        num_total_nan=0,
        err_total_nan=0,
        num_total_inf=0,
        err_total_inf=0,
        num_total_ninf=0,
        err_total_ninf=0,
    ):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.err_w1_num = err_w1_num
        self.err_w1_ratio = err_w1_ratio
        self.err_k1_num = err_k1_num
        self.err_k1_ratio = err_k1_ratio
        self.err_k5_num = err_k5_num
        self.err_k5_ratio = err_k5_ratio
        self.err_h1_num = err_h1_num
        self.err_h1_ratio = err_h1_ratio
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.bm_cmp_std = bm_cmp_std
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf
        self.diff_big_ratio_max = diff_big_ratio_max
        self.diff_big_ratio_avg = diff_big_ratio_avg
        self.diff_big_ratio_rmse = diff_big_ratio_rmse

    def print_result(self):
        logger.info(f"---- Result: {self.result_name} ----")
        logger.info(f" big_value_total: {self.total_big_num}")
        logger.info(f" big_value_ratio: {self.total_big_ratio:.2%}")
        logger.info(f" big_value_max_abs_error: {self.diff_big_max:.8f}")
        logger.info(f" big_value_avg_abs_error: {self.diff_big_avg:.8f}")
        logger.info(f" big_value_abs_error_sum: {self.diff_big_sum:.2f}")
        logger.info(f" big_value_max_rel_error: {self.diff_big_ratio_max:.8f}")
        logger.info(f" big_value_avg_rel_error: {self.diff_big_ratio_avg:.8f}")
        logger.info(f" big_value_rel_error_rmse (RMSE): {self.diff_big_ratio_rmse:.8f}")
        logger.info(
            f" big_value_1e-4_error_count: {self.err_w1_num}, ratio {self.err_w1_ratio:.2%}"
        )
        logger.info(
            f" big_value_1e-3_error_count: {self.err_k1_num}, ratio {self.err_k1_ratio:.2%}"
        )
        logger.info(
            f" big_value_5e-3_error_count: {self.err_k5_num}, ratio {self.err_k5_ratio:.2%}"
        )
        logger.info(
            f" big_value_1e-2_error_count: {self.err_h1_num}, ratio {self.err_h1_ratio:.2%}"
        )
        logger.info(f" small_value_total: {self.total_small_num}")
        logger.info(f" small_value_ratio: {self.total_small_ratio:.2%}")
        logger.info(
            f" small_value_error_count: {self.err_small_num}, ratio {self.err_small_ratio:.2%}"
        )
        logger.info(f" error_rmse (RMSE): {self.diff_rmse:.8f}")
        logger.info(f" balance_deviation_count: {self.rst_eb}")
        logger.info(f" balance_diff_sum: {self.diff_eb:.8f}")
        logger.info(
            f" golden nan total: {self.num_total_nan}, nan error_count: {self.err_total_nan}"
        )
        logger.info(
            f" golden inf total: {self.num_total_inf}, inf error_count: {self.err_total_inf}"
        )
        logger.info(
            f" golden -inf total: {self.num_total_ninf}, -inf error_count: {self.err_total_ninf}"
        )

    def check_result_debug(self, benchmark, new=False, output_dtype="fp16"):
        """method=12 判等:
        X > bench * X_rtol → error
        X > bench         → warning
        new=True 时对 GPU 端 diff_big_ratio_max/avg, diff_rmse 做下界抬高
        """
        if new:
            lo_bound = 0.0
            if output_dtype == "fp16":
                lo_bound = 2.0 ** (-11)
            if output_dtype == "bf16":
                lo_bound = 2.0 ** (-8)
            lo_tensor = torch.tensor(lo_bound, dtype=torch.float32)
            benchmark.diff_big_ratio_max = benchmark.diff_big_ratio_max.maximum(
                lo_tensor
            )
            benchmark.diff_big_ratio_avg = benchmark.diff_big_ratio_avg.maximum(
                lo_tensor
            )
            benchmark.diff_rmse = benchmark.diff_rmse.maximum(lo_tensor)

        reason_str = ""
        # diff_big_ratio_max
        if (
            self.diff_big_ratio_max
            > benchmark.diff_big_ratio_max * self.bm_cmp_std["max_re_rtol"]
        ):
            reason_str += " diff_big_ratio_max error/"
        elif self.diff_big_ratio_max > benchmark.diff_big_ratio_max:
            reason_str += " diff_big_ratio_max warning/"
        # diff_big_ratio_avg
        if (
            self.diff_big_ratio_avg
            > benchmark.diff_big_ratio_avg * self.bm_cmp_std["avg_re_rtol"]
        ):
            reason_str += " diff_big_ratio_avg error/"
        elif self.diff_big_ratio_avg > benchmark.diff_big_ratio_avg:
            reason_str += " diff_big_ratio_avg warning/"
        # diff_big_sum (new=False 才比)
        if not new:
            if (
                self.diff_big_sum
                > benchmark.diff_big_sum * self.bm_cmp_std["avg_re_rtol"]
            ):
                reason_str += " diff_big_sum error/"
            elif self.diff_big_sum > benchmark.diff_big_sum:
                reason_str += " diff_big_sum warning/"
            # diff_big_ratio_rmse (new=False 才比)
            if (
                self.diff_big_ratio_rmse
                > benchmark.diff_big_ratio_rmse * self.bm_cmp_std["rmse_rtol"]
            ):
                reason_str += " diff_big_ratio_rmse error/"
            elif self.diff_big_ratio_rmse > benchmark.diff_big_ratio_rmse:
                reason_str += " diff_big_ratio_rmse warning/"
        # err_small_num
        if (
            self.err_small_num
            > benchmark.err_small_num * self.bm_cmp_std["avg_re_rtol"]
        ):
            reason_str += " err_small_num error/"
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += " err_small_num warning/"
        # diff_rmse
        if self.diff_rmse > benchmark.diff_rmse * self.bm_cmp_std["rmse_rtol"]:
            reason_str += " diff_rmse error/"
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += " diff_rmse warning/"
        # nan / inf
        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += " err_total_nan error/"
        elif self.err_total_nan > 0:
            reason_str += " err_total_nan warning/"
        if (
            self.err_total_inf > benchmark.err_total_inf
            or self.err_total_ninf > benchmark.err_total_ninf
        ):
            reason_str += " err_total_inf error/"
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += " err_total_inf warning"
        return reason_str

    def check_result(self, benchmark, new=False, output_dtype="fp16"):
        """与竞品对比精度结果, 返回 (status, reason)
        status: 'Pass' / 'warning' / 'Failed'
        """
        logger.info(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        reason_str = self.check_result_debug(benchmark, new, output_dtype)
        if "error" in reason_str:
            logger.error(f"{self.result_name} compare result: error")
            return "Failed", reason_str
        elif "warning" in reason_str:
            logger.warning(f"{self.result_name} compare result: warning")
            return "warning", reason_str
        else:
            logger.info(f"{self.result_name} compare result: ok")
            return "Pass", ""


# ============================================================
# 3. 单路指标计算 (替代 aclnn_fuzz 的 checkResult / checkResultNew)
# ============================================================


def _to_numpy_1d(data):
    """torch tensor / numpy array → 1D numpy array (保持原 dtype 语义)
    fp8 numpy dtype (ml_dtypes 的 float8_e5m2/e4m3fn) 原样保留, 由 _to_torch_tensor 做转换
    """
    import numpy as np

    if isinstance(data, torch.Tensor):
        t = data.detach().cpu()
        try:
            return t.numpy().flatten()
        except (TypeError, RuntimeError):
            # fp8 / 不支持的 dtype → 转 fp32
            return t.to(torch.float32).numpy().flatten()
    arr = np.asarray(data)
    return arr.flatten()


def _to_torch_tensor(data, qdtype_hint: Optional[torch.dtype] = None) -> torch.Tensor:
    """1D numpy array → torch tensor (fp8/bf16 特殊处理, 对齐 method=12)
    qdtype_hint: fp8 输入场景下, 把数据升 fp32 后 .to(qdtype_hint) 对齐输入 dtype
    """
    np_dtype_str = str(data.dtype).lower()
    if np_dtype_str in ("float8_e5m2", "float8_e4m3fn", "hifloat8"):
        t = torch.from_numpy(data.astype("float32")).detach()
        if qdtype_hint is not None:
            t = t.to(qdtype_hint)
        return t
    elif np_dtype_str == "bfloat16":
        return torch.from_numpy(data.astype("float32")).to(torch.bfloat16).detach()
    else:
        return torch.from_numpy(data).detach()


def compute_result(
    value: torch.Tensor,
    golden: torch.Tensor,
    name: str,
    output_dtype: str,
    bm_cmp_std: Optional[Dict[str, float]] = None,
    red_list: Optional[List[float]] = None,
) -> Optional[Result]:
    """计算单路 (value vs golden) 的全部指标, 返回 Result
    output_dtype: 内部 normalize 后的 key (fp16/bf16/fp32/hf32)
    """
    if bm_cmp_std is None:
        bm_cmp_std = _get_bm_cmp_std(output_dtype)
    if red_list is None:
        red_list = _get_red_list(output_dtype)

    if value.shape != golden.shape:
        logger.error(
            f"error: {name} shape mismatch, value={value.shape} golden={golden.shape}"
        )
        return None

    # ---- Inf/Nan 统计 (简单 xor, 对齐 method=12) ----
    mask_g_nan = torch.isnan(golden)
    mask_v_nan = torch.isnan(value)
    num_total_nan = int(torch.sum(mask_g_nan).item())
    err_total_nan = int(torch.sum(mask_g_nan.logical_xor(mask_v_nan)).item())

    mask_g_inf = torch.isinf(golden) & (golden > 0)
    mask_v_inf = torch.isinf(value) & (value > 0)
    num_total_inf = int(torch.sum(mask_g_inf).item())
    err_total_inf = int(torch.sum(mask_g_inf.logical_xor(mask_v_inf)).item())

    mask_g_ninf = torch.isinf(golden) & (golden < 0)
    mask_v_ninf = torch.isinf(value) & (value < 0)
    num_total_ninf = int(torch.sum(mask_g_ninf).item())
    err_total_ninf = int(torch.sum(mask_g_ninf.logical_xor(mask_v_ninf)).item())

    # Inf/Nan 位置统一赋 1, 不参与数值指标
    value = value.clone()
    golden = golden.clone()
    value[torch.isinf(value)] = 1
    golden[torch.isinf(golden)] = 1
    value[torch.isnan(value)] = 1
    golden[torch.isnan(golden)] = 1

    # ---- 大值对比 ----
    total_big_num = int(torch.sum(golden.abs() >= bm_cmp_std["small_value"]).item())
    total_big_ratio = (
        float(total_big_num / golden.numel()) if golden.numel() > 0 else 0.0
    )

    # 小值位置统一赋 1, 忽略影响
    value_big = value.clone()
    value_big[golden.abs() < bm_cmp_std["small_value"]] = 1
    golden_big = golden.clone()
    golden_big[golden.abs() < bm_cmp_std["small_value"]] = 1

    diff_big = torch.abs(value_big.sub(golden_big))
    diff_big_max = diff_big.max() if diff_big.numel() > 0 else torch.tensor(0.0)
    diff_big_sum = diff_big.sum()
    diff_big_avg = (
        diff_big_sum / total_big_num if total_big_num > 0 else torch.tensor(0.0)
    )

    diff_big_ratio = diff_big / golden_big.abs()
    diff_big_ratio_max = (
        diff_big_ratio.max() if diff_big_ratio.numel() > 0 else torch.tensor(0.0)
    )
    diff_big_ratio_avg = (
        diff_big_ratio.sum() / total_big_num if total_big_num > 0 else torch.tensor(0.0)
    )
    diff_big_ratio_rmse = (
        torch.sqrt(torch.mean(torch.square(diff_big_ratio)))
        if diff_big_ratio.numel() > 0
        else torch.tensor(0.0)
    )

    err_w1_num = int(torch.sum(diff_big_ratio > red_list[0]).item())
    err_w1_ratio = (
        err_w1_num / total_big_num if total_big_num > 0 else torch.tensor(0.0)
    )
    err_k1_num = int(torch.sum(diff_big_ratio > red_list[1]).item())
    err_k1_ratio = (
        err_k1_num / total_big_num if total_big_num > 0 else torch.tensor(0.0)
    )
    err_k5_num = int(torch.sum(diff_big_ratio > red_list[2]).item())
    err_k5_ratio = (
        err_k5_num / total_big_num if total_big_num > 0 else torch.tensor(0.0)
    )
    err_h1_num = int(torch.sum(diff_big_ratio > red_list[3]).item())
    err_h1_ratio = (
        err_h1_num / total_big_num if total_big_num > 0 else torch.tensor(0.0)
    )

    # ---- 小值对比 ----
    total_small_num = int(torch.sum(golden.abs() < bm_cmp_std["small_value"]).item())
    total_small_ratio = (
        total_small_num / golden.numel() if golden.numel() > 0 else torch.tensor(0.0)
    )

    # 大值位置统一赋 1, 忽略影响
    value_small = value.clone()
    value_small[golden.abs() > bm_cmp_std["small_value"]] = 1
    golden_small = golden.clone()
    golden_small[golden.abs() > bm_cmp_std["small_value"]] = 1

    diff_small = torch.abs(value_small.sub(golden_small))
    err_small_num = int(torch.sum(diff_small > bm_cmp_std["small_value_atol"]).item())
    err_small_ratio = (
        err_small_num / total_small_num if total_small_num > 0 else torch.tensor(0.0)
    )

    # ---- 全局 RMSE (method=12 用 diff_big_rmse 覆写, 这里保持一致) ----
    diff_big_rmse = (
        torch.sqrt(torch.mean(torch.square(diff_big)))
        if diff_big.numel() > 0
        else torch.tensor(0.0)
    )
    diff_rmse = diff_big_rmse  # 对齐 method=12: tools.py:3676

    # ---- 误差均衡性 (eb) ----
    eb_bigger = torch.sum(value > golden)
    eb_smaller = torch.sum(value < golden)
    rst_eb = torch.abs(eb_bigger.sub(eb_smaller))
    diff_eb = torch.sum(value.sub(golden))

    return Result(
        name,
        total_big_num,
        total_big_ratio,
        diff_big_max,
        diff_big_avg,
        diff_big_sum,
        err_w1_num,
        err_w1_ratio,
        err_k1_num,
        err_k1_ratio,
        err_k5_num,
        err_k5_ratio,
        err_h1_num,
        err_h1_ratio,
        total_small_num,
        total_small_ratio,
        err_small_num,
        err_small_ratio,
        diff_rmse,
        rst_eb,
        diff_eb,
        diff_big_ratio_max,
        diff_big_ratio_avg,
        diff_big_ratio_rmse,
        bm_cmp_std,
        num_total_nan,
        err_total_nan,
        num_total_inf,
        err_total_inf,
        num_total_ninf,
        err_total_ninf,
    )


# ============================================================
# 4. 双标杆比对 (替代 data_compare_benchmark_new)
# ============================================================


def _pick_qdtype(input_dtype_hint: Optional[torch.dtype]) -> Optional[torch.dtype]:
    """fp8 输入时, 把数据升 fp32 再 .to(input_dtype_hint) 对齐输入 dtype"""
    return input_dtype_hint


def data_compare_benchmark_new(
    golden: torch.Tensor,
    benchmark: Optional[torch.Tensor],
    actual: torch.Tensor,
    output_dtype: str = "fp16",
    test_name: str = "",
    input_dtype_hint: Optional[torch.dtype] = None,
) -> Tuple[str, str, List]:
    """method=12 双标杆比对
    返回 (status, reason, data) — data 为报表行字段 (与 method=12 顺序对齐)
    """
    real_data = _to_numpy_1d(actual)
    data_compe = _to_numpy_1d(benchmark) if benchmark is not None else None
    cpu_golden = _to_numpy_1d(golden)

    if (
        real_data.size == 0
        and (data_compe is None or real_data.size == data_compe.size)
        and real_data.size == cpu_golden.size
    ):
        logger.info("The npu_output is [], and it matches bm/golden, result=Pass")
        return "Pass", "", []

    if real_data.size != cpu_golden.size:
        logger.error(
            f"Error, size of npu output[{real_data.size}] and golden[{cpu_golden.size}] not equal."
        )
        return "Failed", "size mismatch", []

    if data_compe is not None and real_data.size != data_compe.size:
        logger.error(
            f"Error, size of npu output[{real_data.size}] and benchmark[{data_compe.size}] not equal."
        )
        return "Failed", "size mismatch", []

    # numpy 1D → torch tensor (fp8 / bf16 特殊处理)
    qdtype_hint = _pick_qdtype(input_dtype_hint)
    npu_res = _to_torch_tensor(real_data, qdtype_hint)
    golden_t = _to_torch_tensor(cpu_golden, qdtype_hint)

    bm_cmp_std = _get_bm_cmp_std(output_dtype)
    red_list = _get_red_list(output_dtype)

    # NPU vs golden
    rst_npu = compute_result(
        npu_res, golden_t, test_name + "_npu", output_dtype, bm_cmp_std, red_list
    )
    if rst_npu is None:
        return "Failed", "npu shape mismatch", []
    rst_npu.print_result()

    # benchmark vs golden (单标杆模式时, 用 golden 自己当 bench, check_result 永远 Pass)
    if data_compe is not None:
        benchmark_res = _to_torch_tensor(data_compe, qdtype_hint)
        rst_gpu = compute_result(
            benchmark_res,
            golden_t.clone(),
            test_name + "_bench",
            output_dtype,
            bm_cmp_std,
            red_list,
        )
        if rst_gpu is None:
            return "Failed", "bench shape mismatch", []
        rst_gpu.print_result()
        # new=True: 下界抬高 + 去掉 diff_big_sum / diff_big_ratio_rmse 判等
        status, reason = rst_npu.check_result(
            rst_gpu, new=True, output_dtype=output_dtype
        )
    else:
        # 单标杆模式: 没有 benchmark, 无法做 Ratio 判等, 直接判 Pass (仅展示指标)
        rst_gpu = rst_npu
        status, reason = "Pass", ""

    # 报表行 (与 method=12 data_compare_benchmark_new 返回的 data 顺序对齐)
    data = [
        test_name,
        test_name,
        f"{rst_npu.total_big_num}",
        f"{rst_npu.total_big_ratio:.2%}",
        f"{rst_npu.err_w1_ratio:.2%}",
        f"{rst_npu.err_k1_ratio:.2%}",
        f"{rst_npu.err_k5_ratio:.2%}",
        f"{rst_npu.err_h1_ratio:.2%}",
        f"{rst_npu.diff_big_max:.8f}",
        f"{rst_npu.diff_big_avg:.8f}",
        f"{rst_npu.diff_big_sum:.2f}",
        f"{rst_npu.diff_big_ratio_max:.8f}",
        f"{rst_npu.diff_big_ratio_avg:.8f}",
        f"{rst_npu.diff_big_ratio_rmse:.8f}",
        f"{rst_npu.total_small_num}",
        f"{rst_npu.total_small_ratio:.2%}",
        f"{rst_npu.err_small_num}",
        f"{rst_npu.err_small_ratio:.2%}",
        f"{rst_npu.num_total_nan:.2f}",
        f"{rst_npu.err_total_nan:.2f}",
        f"{rst_npu.num_total_inf:.2f}",
        f"{rst_npu.err_total_inf:.2f}",
        f"{rst_npu.num_total_ninf:.2f}",
        f"{rst_npu.err_total_ninf:.2f}",
        f"{rst_npu.diff_rmse:.8f}",
        f"{rst_npu.rst_eb}",
        f"{rst_npu.diff_eb:.8f}",
        f"{rst_gpu.total_big_num}",
        f"{rst_gpu.total_big_ratio:.2%}",
        f"{rst_gpu.err_w1_ratio:.2%}",
        f"{rst_gpu.err_k1_ratio:.2%}",
        f"{rst_gpu.err_k5_ratio:.2%}",
        f"{rst_gpu.err_h1_ratio:.2%}",
        f"{rst_gpu.diff_big_max:.8f}",
        f"{rst_gpu.diff_big_avg:.8f}",
        f"{rst_gpu.diff_big_sum:.2f}",
        f"{rst_gpu.diff_big_ratio_max:.8f}",
        f"{rst_gpu.diff_big_ratio_avg:.8f}",
        f"{rst_gpu.diff_big_ratio_rmse:.8f}",
        f"{rst_gpu.total_small_num}",
        f"{rst_gpu.total_small_ratio:.2%}",
        f"{rst_gpu.err_small_num}",
        f"{rst_gpu.err_small_ratio:.2%}",
        f"{rst_gpu.num_total_nan:.2f}",
        f"{rst_gpu.err_total_nan:.2f}",
        f"{rst_gpu.num_total_inf:.2f}",
        f"{rst_gpu.err_total_inf:.2f}",
        f"{rst_gpu.num_total_ninf:.2f}",
        f"{rst_gpu.err_total_ninf:.2f}",
        f"{rst_gpu.diff_rmse:.8f}",
        f"{rst_gpu.rst_eb}",
        f"{rst_gpu.diff_eb:.8f}",
        f"{status}",
        f"{reason}",
    ]
    return status, reason, data


# ============================================================
# 5. CV 报告 (兼容旧 API)
# ============================================================


@dataclass
class CVPrecisionReport:
    """CV 模式综合报告 (字段兼容旧 API)"""

    test_name: str = ""
    level: str = "L1"
    output_dtype: str = "fp16"
    passed: bool = False
    status: str = "Failed"  # Pass / warning / Failed
    failure_reasons: List[str] = field(default_factory=list)

    # NPU 误差指标
    npu_total_big_num: int = 0
    npu_total_big_ratio: float = 0.0
    npu_diff_big_max: float = 0.0
    npu_diff_big_avg: float = 0.0
    npu_diff_big_sum: float = 0.0
    npu_diff_big_ratio_max: float = 0.0
    npu_diff_big_ratio_avg: float = 0.0
    npu_diff_big_ratio_rmse: float = 0.0
    npu_err_w1_ratio: float = 0.0
    npu_err_k1_ratio: float = 0.0
    npu_err_k5_ratio: float = 0.0
    npu_err_h1_ratio: float = 0.0
    npu_total_small_num: int = 0
    npu_total_small_ratio: float = 0.0
    npu_err_small_num: int = 0
    npu_err_small_ratio: float = 0.0
    npu_diff_rmse: float = 0.0
    npu_rst_eb: int = 0
    npu_diff_eb: float = 0.0
    npu_num_total_nan: int = 0
    npu_err_total_nan: int = 0
    npu_num_total_inf: int = 0
    npu_err_total_inf: int = 0
    npu_num_total_ninf: int = 0
    npu_err_total_ninf: int = 0

    # Benchmark 误差指标
    bench_total_big_num: int = 0
    bench_total_big_ratio: float = 0.0
    bench_diff_big_max: float = 0.0
    bench_diff_big_avg: float = 0.0
    bench_diff_big_sum: float = 0.0
    bench_diff_big_ratio_max: float = 0.0
    bench_diff_big_ratio_avg: float = 0.0
    bench_diff_big_ratio_rmse: float = 0.0
    bench_err_w1_ratio: float = 0.0
    bench_err_k1_ratio: float = 0.0
    bench_err_k5_ratio: float = 0.0
    bench_err_h1_ratio: float = 0.0
    bench_total_small_num: int = 0
    bench_total_small_ratio: float = 0.0
    bench_err_small_num: int = 0
    bench_err_small_ratio: float = 0.0
    bench_diff_rmse: float = 0.0
    bench_rst_eb: int = 0
    bench_diff_eb: float = 0.0
    bench_num_total_nan: int = 0
    bench_err_total_nan: int = 0
    bench_num_total_inf: int = 0
    bench_err_total_inf: int = 0
    bench_num_total_ninf: int = 0
    bench_err_total_ninf: int = 0

    # bm_cmp_std / red_list (用于展示)
    bm_cmp_std: Dict[str, float] = field(default_factory=dict)
    red_list: List[float] = field(default_factory=list)

    # 旧 API 字段 (兼容)
    total_elements: int = 0
    fail_count: int = 0
    fail_ratio: float = 0.0
    npu_mare: float = 0.0
    mare_rate: float = 0.0
    mere_rate: float = 0.0
    rmse_rate: float = 0.0

    # 子结果 (透传)
    rst_npu: Optional[Result] = field(default=None, repr=False)
    rst_bench: Optional[Result] = field(default=None, repr=False)


def _scalar(v) -> float:
    """0-d tensor → python float; 其他数值原样返回 float"""
    if isinstance(v, torch.Tensor) and v.dim() == 0:
        return float(v.item())
    return float(v)


def _fill_report_from_result(report: CVPrecisionReport, rst: Result, is_npu: bool):
    """把 Result 字段映射到 CVPrecisionReport (前缀 npu_ / bench_)
    Result 内部字段是 0-d tensor, 这里转 python 标量赋给 dataclass
    """
    prefix = "npu" if is_npu else "bench"
    mapping = {
        f"{prefix}_total_big_num": int(rst.total_big_num),
        f"{prefix}_total_big_ratio": _scalar(rst.total_big_ratio),
        f"{prefix}_diff_big_max": _scalar(rst.diff_big_max),
        f"{prefix}_diff_big_avg": _scalar(rst.diff_big_avg),
        f"{prefix}_diff_big_sum": _scalar(rst.diff_big_sum),
        f"{prefix}_diff_big_ratio_max": _scalar(rst.diff_big_ratio_max),
        f"{prefix}_diff_big_ratio_avg": _scalar(rst.diff_big_ratio_avg),
        f"{prefix}_diff_big_ratio_rmse": _scalar(rst.diff_big_ratio_rmse),
        f"{prefix}_err_w1_ratio": _scalar(rst.err_w1_ratio),
        f"{prefix}_err_k1_ratio": _scalar(rst.err_k1_ratio),
        f"{prefix}_err_k5_ratio": _scalar(rst.err_k5_ratio),
        f"{prefix}_err_h1_ratio": _scalar(rst.err_h1_ratio),
        f"{prefix}_total_small_num": int(rst.total_small_num),
        f"{prefix}_total_small_ratio": _scalar(rst.total_small_ratio),
        f"{prefix}_err_small_num": int(rst.err_small_num),
        f"{prefix}_err_small_ratio": _scalar(rst.err_small_ratio),
        f"{prefix}_diff_rmse": _scalar(rst.diff_rmse),
        f"{prefix}_rst_eb": int(rst.rst_eb),
        f"{prefix}_diff_eb": _scalar(rst.diff_eb),
        f"{prefix}_num_total_nan": int(rst.num_total_nan),
        f"{prefix}_err_total_nan": int(rst.err_total_nan),
        f"{prefix}_num_total_inf": int(rst.num_total_inf),
        f"{prefix}_err_total_inf": int(rst.err_total_inf),
        f"{prefix}_num_total_ninf": int(rst.num_total_ninf),
        f"{prefix}_err_total_ninf": int(rst.err_total_ninf),
    }
    for k, v in mapping.items():
        setattr(report, k, v)


def compute_cv_report(
    golden: torch.Tensor,
    benchmark: Optional[torch.Tensor],
    actual: torch.Tensor,
    test_name: str = "",
    output_dtype: str = "fp16",
    input_dtype_hint: Optional[torch.dtype] = None,
    bm_cmp_std: Optional[Dict[str, float]] = None,
    red_list: Optional[List[float]] = None,
) -> CVPrecisionReport:
    """CV 双标杆比对 (method=12 风格)
    返回 CVPrecisionReport, 内含 rst_npu / rst_bench / status / reason
    """
    if bm_cmp_std is None:
        bm_cmp_std = _get_bm_cmp_std(output_dtype)
    if red_list is None:
        red_list = _get_red_list(output_dtype)

    report = CVPrecisionReport(
        test_name=test_name,
        level="L1",
        output_dtype=output_dtype,
        bm_cmp_std=bm_cmp_std,
        red_list=red_list,
    )

    real_data = _to_numpy_1d(actual)
    golden_flat = _to_numpy_1d(golden)
    bench_flat = _to_numpy_1d(benchmark) if benchmark is not None else None

    report.total_elements = int(golden_flat.size)

    if (
        real_data.size == 0
        and real_data.size == golden_flat.size
        and (bench_flat is None or real_data.size == bench_flat.size)
    ):
        report.passed = True
        report.status = "Pass"
        return report

    if real_data.size != golden_flat.size:
        report.failure_reasons.append(
            f"Shape mismatch: actual={real_data.size} golden={golden_flat.size}"
        )
        report.status = "Failed"
        return report
    if bench_flat is not None and real_data.size != bench_flat.size:
        report.failure_reasons.append(
            f"Shape mismatch: actual={real_data.size} bench={bench_flat.size}"
        )
        report.status = "Failed"
        return report

    qdtype_hint = _pick_qdtype(input_dtype_hint)
    npu_res = _to_torch_tensor(real_data, qdtype_hint)
    golden_t = _to_torch_tensor(golden_flat, qdtype_hint)

    rst_npu = compute_result(
        npu_res, golden_t, test_name + "_npu", output_dtype, bm_cmp_std, red_list
    )
    if rst_npu is None:
        report.failure_reasons.append("NPU shape mismatch")
        report.status = "Failed"
        return report
    _fill_report_from_result(report, rst_npu, is_npu=True)
    report.rst_npu = rst_npu
    report.npu_mare = float(_scalar(rst_npu.diff_big_ratio_max))
    report.fail_count = int(rst_npu.err_h1_num)
    report.fail_ratio = float(_scalar(rst_npu.err_h1_ratio))

    if bench_flat is not None:
        bench_res = _to_torch_tensor(bench_flat, qdtype_hint)
        rst_bench = compute_result(
            bench_res,
            golden_t.clone(),
            test_name + "_bench",
            output_dtype,
            bm_cmp_std,
            red_list,
        )
        if rst_bench is None:
            report.failure_reasons.append("Benchmark shape mismatch")
            report.status = "Failed"
            return report
        _fill_report_from_result(report, rst_bench, is_npu=False)
        report.rst_bench = rst_bench
        # new=True: 下界抬高 + 去掉 diff_big_sum / diff_big_ratio_rmse 判等
        status, reason = rst_npu.check_result(
            rst_bench, new=True, output_dtype=output_dtype
        )
    else:
        # 单标杆: 无法 Ratio 判等, 直接判 Pass
        report.rst_bench = rst_npu
        status, reason = "Pass", ""

    report.status = status
    report.passed = status == "Pass"
    if reason:
        report.failure_reasons = [
            r.strip().rstrip("/") for r in reason.split("/") if r.strip()
        ]
    return report


def display_cv_report(report: CVPrecisionReport):
    SEP = "=" * 72
    SUB = "-" * 72
    logger.info(SEP)
    logger.info("  Operator precision comparison report (method=12 style)")
    if report.test_name:
        logger.info("  Test: %s", report.test_name)
    logger.info("  Dtype: %s  |  Total: %d", report.output_dtype, report.total_elements)
    logger.info(SUB)

    bm = report.bm_cmp_std
    logger.info(
        f"  [bm_cmp_std]  max_re_rtol={bm.get('max_re_rtol', 0):.1f} "
        f"avg_re_rtol={bm.get('avg_re_rtol', 0):.1f} rmse_rtol={bm.get('rmse_rtol', 0):.1f} "
        f"small_value={bm.get('small_value', 0):g} small_value_atol={bm.get('small_value_atol', 0):g}"
    )
    logger.info(
        f"  [red_range]   w1={report.red_list[0]:g} k1={report.red_list[1]:g} "
        f"k5={report.red_list[2]:g} h1={report.red_list[3]:g}"
    )

    def _print_one(label, rst: Result):
        logger.info(SUB)
        logger.info(f"  [{label}]")
        logger.info(
            f"    big value: num={rst.total_big_num} ratio={rst.total_big_ratio:.2%} "
            f"max_abs={rst.diff_big_max:.8f} avg_abs={rst.diff_big_avg:.8f} sum={rst.diff_big_sum:.2f}"
        )
        logger.info(
            f"    big value rel error: max={rst.diff_big_ratio_max:.8f} "
            f"avg={rst.diff_big_ratio_avg:.8f} rmse={rst.diff_big_ratio_rmse:.8f}"
        )
        logger.info(
            f"    big value error ratio: w1={rst.err_w1_ratio:.2%} k1={rst.err_k1_ratio:.2%} "
            f"k5={rst.err_k5_ratio:.2%} h1={rst.err_h1_ratio:.2%}"
        )
        logger.info(
            f"    small value: num={rst.total_small_num} ratio={rst.total_small_ratio:.2%} "
            f"err_num={rst.err_small_num} err_ratio={rst.err_small_ratio:.2%}"
        )
        logger.info(
            f"    RMSE={rst.diff_rmse:.8f}  eb={rst.rst_eb}  diff_eb={rst.diff_eb:.8f}"
        )
        logger.info(
            f"    nan: total={rst.num_total_nan} err={rst.err_total_nan} | "
            f"inf: total={rst.num_total_inf} err={rst.err_total_inf} | "
            f"-inf: total={rst.num_total_ninf} err={rst.err_total_ninf}"
        )

    if report.rst_npu is not None:
        _print_one("NPU vs Golden", report.rst_npu)
    if report.rst_bench is not None and report.rst_bench is not report.rst_npu:
        _print_one("Benchmark vs Golden", report.rst_bench)

    logger.info(SUB)
    if report.status == "Pass":
        logger.info("  >>> Verdict: PASS")
    elif report.status == "warning":
        logger.warning("  >>> Verdict: WARNING")
        for i, r in enumerate(report.failure_reasons, 1):
            logger.info("      [%d] %s", i, r)
    else:
        logger.error("  >>> Verdict: FAIL")
        for i, r in enumerate(report.failure_reasons, 1):
            logger.info("      [%d] %s", i, r)
    logger.info(SEP)


# ============================================================
# 6. 对外 API
# ============================================================


def check_result_cv(
    golden: torch.Tensor,
    benchmark: Optional[torch.Tensor],
    npu_result: torch.Tensor,
    test_name: str = "",
    output_dtype: str = "fp16",
    input_dtype_hint: Optional[torch.dtype] = None,
    bm_cmp_std: Optional[Dict[str, float]] = None,
    red_list: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """CV 模式精度比对 (显式双标杆)
    返回 (status, info_dict)
      status: "Pass" / "warning" / "Failed"
    """
    cv = compute_cv_report(
        golden=golden,
        benchmark=benchmark,
        actual=npu_result,
        test_name=test_name,
        output_dtype=output_dtype,
        input_dtype_hint=input_dtype_hint,
        bm_cmp_std=bm_cmp_std,
        red_list=red_list,
    )
    if verbose:
        display_cv_report(cv)

    return cv.status, {
        "npu_diff_big_ratio_max": cv.npu_diff_big_ratio_max,
        "npu_diff_big_ratio_avg": cv.npu_diff_big_ratio_avg,
        "npu_diff_rmse": cv.npu_diff_rmse,
        "bench_diff_big_ratio_max": cv.bench_diff_big_ratio_max,
        "bench_diff_big_ratio_avg": cv.bench_diff_big_ratio_avg,
        "bench_diff_rmse": cv.bench_diff_rmse,
        "npu_err_small_num": cv.npu_err_small_num,
        "bench_err_small_num": cv.bench_err_small_num,
        "npu_err_h1_ratio": cv.npu_err_h1_ratio,
        "status": cv.status,
        "reasons": list(cv.failure_reasons),
    }


def check_result(
    expect: torch.Tensor,
    npu_result: torch.Tensor,
    benchmark: Optional[torch.Tensor] = None,
    config=None,
    verbose: bool = False,
    output_dtype: Optional[str] = None,
    input_dtype_hint: Optional[torch.dtype] = None,
    level: str = "L1",
) -> Tuple[str, float, float]:
    """兼容旧版 result_compare_method.check_result 接口
    旧版两参数: check_result(golden, npu_result)              → 单标杆 (无 Ratio, 直接 Pass)
    新版三参数: check_result(golden, benchmark, npu_result)   → CV 双标杆 (method=12)

    返回: (status, fulfill_percent, max_rel_err)
      status:          "Pass" / "warning" / "Failed"
      fulfill_percent: 通过元素百分比 (0-100)
      max_rel_err:     NPU 最大相对误差 (diff_big_ratio_max)
    """
    if output_dtype is None:
        output_dtype = _normalize_dtype_key(
            npu_result.dtype if isinstance(npu_result, torch.Tensor) else "fp16"
        )

    cv = compute_cv_report(
        golden=expect,
        benchmark=benchmark,
        actual=npu_result,
        test_name="",
        output_dtype=output_dtype,
        input_dtype_hint=input_dtype_hint,
    )
    if verbose:
        display_cv_report(cv)

    # 兼容旧 API: fulfill_percent 用 1 - err_h1_ratio
    fulfill_percent = (1.0 - cv.npu_err_h1_ratio) * 100.0
    max_rel_err = cv.npu_diff_big_ratio_max
    return cv.status, fulfill_percent, max_rel_err


# ============================================================
# 7. 多输出批量比对
# ============================================================


def compare_multi_output(
    outputs: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]],
    output_dtype: str = "fp16",
    input_dtype_hint: Optional[torch.dtype] = None,
) -> Dict[str, CVPrecisionReport]:
    """对多个输出进行批量精度比较
    outputs: {name: (golden, benchmark, actual)} 字典; benchmark 可为 None
    """
    reports = {}
    for name, (golden, benchmark, actual) in outputs.items():
        reports[name] = compute_cv_report(
            golden=golden,
            benchmark=benchmark,
            actual=actual,
            test_name=name,
            output_dtype=output_dtype,
            input_dtype_hint=input_dtype_hint,
        )
    return reports
