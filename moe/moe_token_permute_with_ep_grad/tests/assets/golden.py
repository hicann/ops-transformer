#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoeTokenPermuteWithEpGrad golden（精度标杆）。

纯 TensorFlow 小算子拼接实现（无 for 循环）：
  - token_grad：sorted_indices 值在 [start, end) 内的逐行取回（行号 = 值 - start），
    float32 中间精度按 k 从 0 到 numTopk-1 串行累加（tf.scan 顺序前缀和保序），
    范围外贡献 0，最后 round-to-nearest-even 舍入回输入 dtype
  - probs_grad：纯索引搬运（无计算、无累加），范围外置 0，保持输入 dtype
  - fp32 输入路径不做中间 Cast；bf16/fp16 输入一律 Cast float32 累加
  - permuted_probs_output_grad 为空(None)时 probs_grad 返回 None（比对跳过该输出）

输入行语义（以 kernel 实际编址为准）：
  - permuted_tokens_output_grad: (R, H)，R = range[1] - range[0]，
    第 j 行对应全局 permuted 位置 range[0] + j
  - sorted_indices: (num_tokens * topk,)，值为全局 permuted 位置
  - permuted_probs_output_grad: (R,)，可选
"""

import tensorflow as tf

__spec__ = {
    "moe_token_permute_with_ep_grad": "MoeTokenPermuteWithEpGradTfTestSpec",
    "aclnnMoeTokenPermuteWithEpGrad": "AclnnMoeTokenPermuteWithEpGradTfTestSpec",
}

_TOLERANCE = {
    "float32": {"standard": "cross_check", "rtol": 1e-4, "atol": 1e-8},
    "float16": {"standard": "cross_check", "rtol": 1e-3, "atol": 1e-8},
    "bfloat16": {"standard": "cross_check", "rtol": 1e-3, "atol": 1e-8},
}

# tf.bfloat16 哨兵（用于识别 numpy ml_dtypes.bfloat16 输入并桥接）
try:
    import ml_dtypes

    _BF16_NUMPY = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover - TTK 环境必然提供
    _BF16_NUMPY = None


class _Prepared:
    """_prepare_inputs 的产物：可直接参与计算的预备数据（dtype 已就位、索引已预计算）。"""

    __slots__ = (
        "tokens",
        "probs",
        "in_range",
        "safe_idx",
        "in_dtype",
        "num_tokens",
        "topk",
    )

    def __init__(self, tokens, probs, in_range, safe_idx, in_dtype, num_tokens, topk):
        self.tokens = tokens
        self.probs = probs
        self.in_range = in_range
        self.safe_idx = safe_idx
        self.in_dtype = in_dtype
        self.num_tokens = num_tokens
        self.topk = topk


def _numpy_to_tf(arr):
    """numpy -> tf.Tensor；ml_dtypes.bfloat16 经 fp32 无损桥接。"""
    if arr is None:
        return None
    if _BF16_NUMPY is not None and arr.dtype == _BF16_NUMPY:
        return tf.constant(arr.astype("float32"), dtype=tf.bfloat16)
    return tf.constant(arr)


def _tf_to_numpy(tensor):
    """tf.Tensor -> numpy；bfloat16 经 fp32 无损桥接（cast 本身为 RNT，与 CAST_RNT 一致）。"""
    if tensor is None:
        return None
    arr = tensor.numpy()
    if tensor.dtype == tf.bfloat16:
        arr = arr.astype("float32")
        if _BF16_NUMPY is not None:
            return arr.astype(_BF16_NUMPY)
    return arr


def _normalize_range(range_value):
    """把 CSV/ACLNN 传入的 range 属性规整为 (start, end) 或 None。

    支持 list/tuple；空列表/None 均视为回退语义（返回 None）。
    """
    if range_value is None:
        return None
    if hasattr(range_value, "numpy"):  # tf.Tensor
        range_value = range_value.numpy().flatten().tolist()
    if isinstance(range_value, (int, float)):
        return None
    range_list = [int(v) for v in range_value]
    if len(range_list) == 0:
        return None
    if len(range_list) != 2:
        raise ValueError("range must be a ListInt of size 2")
    return range_list


def _optional_probs(probs):
    """可选 probs 输入归一化：空 tensor / 0 元素 / None 统一为 None（缺席语义）。"""
    if probs is None:
        return None
    if hasattr(probs, "shape") and probs.shape.num_elements() == 0:
        return None
    if hasattr(probs, "numpy") and probs.numpy().size == 0:
        return None
    return probs


def _prepare_inputs(
    permuted_tokens_output_grad,
    sorted_indices,
    permuted_probs_output_grad,
    num_topk,
    range_value,
    padded_mode=False,
):
    """输入预处理：入参校验 + dtype 转换 + 索引预计算。

    所有 dtype 转换集中于此（golden 流在 golden 入口调用、third_party 流在
    __init__ 调用），保证 _compute_core 调用链内零转换：
      - sorted_indices 转 int64（tf.gather 索引要求）
      - bf16/fp16 输入提升 float32（kernel fp32 累加语义）；fp32/fp64 保持原 dtype
    返回 _Prepared，probs 输入为 None 时 probs_grad 亦为 None。
    """
    if padded_mode:
        raise ValueError("padded_mode only supports False")
    if len(permuted_tokens_output_grad.shape) != 2:
        raise ValueError("permuted_tokens_output_grad must be 2D")
    if len(sorted_indices.shape) != 1:
        raise ValueError("sorted_indices must be 1D")
    topk = int(num_topk)
    if not (1 <= topk <= 512):
        raise ValueError("num_topk must be in [1, 512]")

    in_dtype = permuted_tokens_output_grad.dtype
    total_len = sorted_indices.shape[0]
    num_tokens = total_len // topk

    if range_value is None:
        # range 为空：aclnn 层回退 aclnnMoeTokenPermuteGrad 语义（等效全量范围、忽略 probs）
        start, end = 0, total_len
        permuted_probs_output_grad = None
    else:
        start, end = int(range_value[0]), int(range_value[1])
    r_len = end - start
    if permuted_tokens_output_grad.shape[0] != r_len:
        raise ValueError(
            "permuted_tokens_output_grad rows (%d) must equal range[1]-range[0] (R=%d)"
            % (permuted_tokens_output_grad.shape[0], r_len)
        )

    # 索引预计算：范围内掩码 + 防越界安全索引（范围外取值不参与结果，由 where 置零，
    # 避免 0*inf=nan 污染）
    idx = tf.cast(sorted_indices, tf.int64)
    in_range = tf.logical_and(tf.greater_equal(idx, start), tf.less(idx, end))
    safe_idx = idx - start
    safe_idx = tf.clip_by_value(
        safe_idx, tf.constant(0, tf.int64), tf.constant(max(r_len - 1, 0), tf.int64)
    )

    # 累加精度提升：bf16->fp32 转换无损，对整个输入先转后取与逐行取等价
    compute_dtype = tf.float32 if in_dtype in (tf.bfloat16, tf.float16) else in_dtype
    tokens = tf.cast(permuted_tokens_output_grad, compute_dtype)

    if permuted_probs_output_grad is not None:
        if len(permuted_probs_output_grad.shape) != 1:
            raise ValueError("permuted_probs_output_grad must be 1D")
        if permuted_probs_output_grad.shape[0] != r_len:
            raise ValueError(
                "permuted_probs_output_grad numel (%d) must equal R=%d"
                % (permuted_probs_output_grad.shape[0], r_len)
            )

    return _Prepared(
        tokens,
        permuted_probs_output_grad,
        in_range,
        safe_idx,
        in_dtype,
        num_tokens,
        topk,
    )


def _compute_core(prepared):
    """纯计算核心（零 dtype 转换）：gather + tf.scan 顺序前缀和 + probs 搬运。

    tf.scan(lambda a, b: a + b, ...) 按 axis 0 顺序执行（每步一次 fp32 舍入），
    与 kernel k=0,1,...,topk-1 逐路 Add bitwise 一致（含极值域溢出时机）；
    禁用 reduce_sum/cumsum：并行/树形归约改变 fp32 舍入链导致分叉。
    返回 (token_grad, probs_grad)。
    """
    # gather tokens by safe_idx → (num_tokens * topk, H) → (num_tokens, topk, H)
    gathered = tf.gather(prepared.tokens, prepared.safe_idx, axis=0)
    gathered = tf.reshape(gathered, [prepared.num_tokens, prepared.topk, -1])
    zero = tf.zeros_like(gathered)
    masked = tf.where(
        tf.reshape(prepared.in_range, [prepared.num_tokens, prepared.topk, 1]),
        gathered,
        zero,
    )

    # k 序串行累加：转置 topk 到 axis 0 → tf.scan 顺序前缀和 → 取末位
    scan_in = tf.transpose(masked, [1, 0, 2])  # (topk, num_tokens, H)
    scan_out = tf.scan(lambda a, b: a + b, scan_in)  # 顺序前缀和，每步一次舍入
    token_grad = scan_out[-1]  # (num_tokens, H) = 全和

    # probs_grad：纯索引搬运，范围外置 0（无计算、无累加，保 dtype）
    probs_grad = None
    if prepared.probs is not None:
        pg = tf.gather(prepared.probs, prepared.safe_idx, axis=0)
        pg = tf.reshape(pg, [prepared.num_tokens, prepared.topk])
        in_range_2d = tf.reshape(
            prepared.in_range, [prepared.num_tokens, prepared.topk]
        )
        probs_grad = tf.where(in_range_2d, pg, tf.zeros_like(pg))

    # CAST_RNT 回原 dtype（结果 cast，B4 例外放行）
    return tf.cast(token_grad, prepared.in_dtype), probs_grad


_TORCH_DT_MAP = {
    "torch.bfloat16": tf.bfloat16,
    "torch.float16": tf.float16,
    "torch.float32": tf.float32,
    "torch.float64": tf.float64,
    "torch.int32": tf.int32,
    "torch.int64": tf.int64,
}


def _to_tf(t):
    """框架 Tensor / numpy -> tf.Tensor（duck typing，保留原始 dtype）。

    兼容 torch.Tensor / tf.Tensor / numpy（含 ml_dtypes.bfloat16）。
    """
    if t is None:
        return None
    if hasattr(t, "detach"):  # torch.Tensor：保留原始 dtype 到 tf
        arr = t.detach().to("cpu")
        np_arr = (
            arr.float().numpy()
            if str(arr.dtype) in ("torch.bfloat16", "torch.float16")
            else arr.numpy()
        )
        return tf.constant(np_arr, dtype=_TORCH_DT_MAP.get(str(arr.dtype), tf.float32))
    if hasattr(t, "numpy"):  # tf.Tensor
        return tf.constant(t.numpy())
    # numpy（ml_dtypes.bfloat16 桥接回 fp32 再标 bf16 dtype）
    if _BF16_NUMPY is not None and hasattr(t, "dtype") and t.dtype == _BF16_NUMPY:
        return tf.constant(t.astype("float32"), dtype=tf.bfloat16)
    return tf.constant(t)


class MoeTokenPermuteWithEpGradTfTestSpec:
    """Kernel/GEIR 共用 TestSpec（CSV op_name = moe_token_permute_with_ep_grad）。

    输入按 def.cpp 顺序位置传入，属性按名传入（num_topk / range / padded_mode）。
    golden 接收 numpy，返回 numpy list。
    """

    @staticmethod
    def golden(
        permuted_tokens_output_grad,
        sorted_indices,
        permuted_probs_output_grad=None,
        *,
        num_topk=1,
        range=None,
        padded_mode=False,
        **kwargs,
    ):
        # 单/多输出均返回 list；probs 输入缺席时 probs_grad 为 None 哨兵（比对跳过）
        prepared = _prepare_inputs(
            _numpy_to_tf(permuted_tokens_output_grad),
            _numpy_to_tf(sorted_indices),
            _optional_probs(_numpy_to_tf(permuted_probs_output_grad)),
            num_topk,
            _normalize_range(range),
            padded_mode,
        )
        token_grad, probs_grad = _compute_core(prepared)
        return [_tf_to_numpy(token_grad), _tf_to_numpy(probs_grad)]

    class ThirdPartyImpl:
        """tf provider：输入转换前置 __init__（B4），__call__ 链内零转换。"""

        def __init__(
            self,
            permuted_tokens_output_grad=None,
            sorted_indices=None,
            permuted_probs_output_grad=None,
            num_topk=1,
            range=None,
            padded_mode=False,
            **kwargs,
        ):
            self._prepared = _prepare_inputs(
                _to_tf(permuted_tokens_output_grad),
                _to_tf(sorted_indices),
                _optional_probs(_to_tf(permuted_probs_output_grad)),
                num_topk,
                _normalize_range(range),
                padded_mode,
            )

        def __call__(self, **kwargs):
            token_grad, probs_grad = _compute_core(self._prepared)
            return [token_grad, probs_grad]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = _TOLERANCE


class AclnnMoeTokenPermuteWithEpGradTfTestSpec:
    """ACLNN TestSpec（api_name = aclnnMoeTokenPermuteWithEpGrad）。

    参数顺序与 aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize 一致
    （去掉 workspaceSize/executor）：3 输入 + 3 属性 + 2 输出。
    golden 收框架 Tensor（torch/tf 均可，经 numpy 桥接），返回 numpy list。
    """

    @staticmethod
    def golden(
        permutedTokensOutputGrad,
        sortedIndices,
        permutedProbsOutputGradOptional=None,
        numTopk=1,
        rangeOptional=None,
        paddedMode=False,
        tokenGradOut=None,
        probsGradOut=None,
        **kwargs,
    ):
        prepared = _prepare_inputs(
            _to_tf(permutedTokensOutputGrad),
            _to_tf(sortedIndices),
            _optional_probs(_to_tf(permutedProbsOutputGradOptional)),
            numTopk,
            _normalize_range(rangeOptional),
            paddedMode,
        )
        token_grad, probs_grad = _compute_core(prepared)
        return [_tf_to_numpy(token_grad), _tf_to_numpy(probs_grad)]

    class ThirdPartyImpl:
        """tf provider：输入转换前置 __init__（B4），__call__ 链内零转换。"""

        def __init__(
            self,
            permutedTokensOutputGrad=None,
            sortedIndices=None,
            permutedProbsOutputGradOptional=None,
            numTopk=1,
            rangeOptional=None,
            paddedMode=False,
            **kwargs,
        ):
            self._prepared = _prepare_inputs(
                _to_tf(permutedTokensOutputGrad),
                _to_tf(sortedIndices),
                _optional_probs(_to_tf(permutedProbsOutputGradOptional)),
                numTopk,
                _normalize_range(rangeOptional),
                paddedMode,
            )

        def __call__(self, **kwargs):
            token_grad, probs_grad = _compute_core(self._prepared)
            return [token_grad, probs_grad]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = _TOLERANCE
