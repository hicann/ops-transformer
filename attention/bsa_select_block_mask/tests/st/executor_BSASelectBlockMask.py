# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ctypes
import math
import os
import random
import struct

import numpy as np
import torch
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare

logging = Logger().get_logger()

SEED = 42
BSA_META_FILE = "output_0.bsa_meta.pth"
BSA_SOFTMAX_FILE = "output_0.bsa_softmax_fp32.pth"
# BOUNDARY-TOL: 1e-4 < 1 个 fp16 ULP 的相对间距(~9.8e-4)——post 小网格下任何
# 1~2 ULP 的合法边界翻转都会被误拒（post 全量实锤 274/274 均为纯边界互换）。
# 放宽到 1e-3 ≈ 1 fp16 ULP，覆盖实测最大抖动 4.76e-4
BSA_BOUNDARY_REL_TOL = 1e-3
BSA_HEAD_RTOL = 0.005
BSA_HEAD_ATOL = 2.5e-5
BSA_HEAD_PASS_RATE = 0.995
# Golden 与比较器之间通过这些文件传递有效形状和 FP32 softmax。
valid_shape_list = []
BSA_SOFTMAX_OUT = None
B_MASK = 1
N_MASK = 1
X_MASK = 1
Y_MASK = 1


@ACCURACY_REGISTRY.register("standard_basmask")
class BSAMASKSingleBenchmarkAccuracyCompare(BaseAccuracyCompare):
    """BSA mask 的 Golden/NPU 比较器。

    比较对象是形状为 ``[B, N, X, Y]`` 的 block mask：

    * ``local_output`` 是 NPU 算子输出；
    * ``remote_output`` 是 CPU Golden 输出；
    * Golden 目录还保存了每个 block 的 FP32 softmax，用于判断 TopK 边界附近的替换
      是否只是浮点排序差异。

    每个 head 按以下顺序判定：先检查 mask 合法性和选中数量，再比较有效区域的
    ``1`` 集合；集合不一致时，依次尝试 TopK 边界相对误差和 head 级逐元素容错。
    """

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        """比较一个用例并返回 ATK ``AccuracyConfig`` 结果。"""

        # ------------------------------------------------------------------
        # 1. 读取 Golden 的辅助文件。
        #
        # mask 本身由 ATK 传入 remote_output；额外的 metadata 和 FP32 softmax
        # 由 Golden executor 保存到同一个目录。读取失败时无法可靠比较，直接失败。
        # ------------------------------------------------------------------
        _, golden_dir = self.remote_manager.get_local_and_remote_dir()
        meta_path = os.path.join(golden_dir, BSA_META_FILE)
        softmax_path = os.path.join(golden_dir, BSA_SOFTMAX_FILE)
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
            shape = tuple(int(value) for value in meta["shape"])
            valid_shapes = [tuple(map(int, item)) for item in meta["valid_shapes"]]
            npu_sparsity = float(meta["npu_sparsity"])
            golden_sparsity = float(meta["golden_sparsity"])
            softmax_out = torch.load(
                softmax_path, map_location="cpu", weights_only=True
            )
        except (
            FileNotFoundError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=f"Invalid BSA golden metadata: {error}",
            )

        # ------------------------------------------------------------------
        # 2. 校验输出和 Golden 辅助数据的形状。
        # ------------------------------------------------------------------
        if len(shape) != 4 or len(valid_shapes) != shape[0]:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=f"Invalid BSA metadata shape={shape}, valid_shapes={valid_shapes}",
            )
        expected_numel = math.prod(shape)
        if (
            local_output.numel() != expected_numel
            or remote_output.numel() != expected_numel
        ):
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=(
                    f"BSA output size does not match metadata: local={local_output.numel()}, "
                    f"remote={remote_output.numel()}, shape={shape}"
                ),
            )

        if not torch.is_tensor(softmax_out) or tuple(softmax_out.shape) != shape:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=f"Invalid FP32 softmax golden shape: {getattr(softmax_out, 'shape', None)}, expected={shape}",
            )

        # ------------------------------------------------------------------
        # 3. 规范化 Tensor，并构造每个 batch 的有效 block 范围。
        #
        # BNSD/TND 输入最终都会生成 [B, N, X, Y] mask，但变长序列只覆盖
        # [0:valid_x[b], 0:valid_y[b]]。padding 区域不参与数量、集合和误差比较。
        # ------------------------------------------------------------------
        npu = local_output.detach().cpu().reshape(shape)
        golden = remote_output.detach().cpu().reshape(shape)
        softmax_out = softmax_out.detach().cpu().to(torch.float32).reshape(shape)
        batch, heads, x_blocks, y_blocks = shape
        valid_x = torch.tensor([item[0] for item in valid_shapes], dtype=torch.int64)
        valid_y = torch.tensor([item[1] for item in valid_shapes], dtype=torch.int64)
        valid_shape_error = (
            (valid_x <= 0)
            | (valid_x > x_blocks)
            | (valid_y <= 0)
            | (valid_y > y_blocks)
        )
        if bool(valid_shape_error.any()):
            invalid_batches = (
                torch.nonzero(valid_shape_error, as_tuple=False).flatten().tolist()
            )
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info=f"Invalid valid_shapes for batches {invalid_batches[:20]}",
            )

        # rows/cols 通过广播生成 [B, X, Y] 有效区域，再扩展 head 维度。
        rows = torch.arange(x_blocks).reshape(1, x_blocks, 1)
        cols = torch.arange(y_blocks).reshape(1, 1, y_blocks)
        valid = (
            (rows < valid_x.reshape(batch, 1, 1))
            & (cols < valid_y.reshape(batch, 1, 1))
        ).unsqueeze(1)

        # mask 语义只能是 0/1；先记录有效区域内的非法数值，后续统一汇总错误。
        invalid_golden_mask = valid & (golden != 0) & (golden != 1)
        invalid_npu_mask = valid & (npu != 0) & (npu != 1)
        has_invalid_golden = bool(invalid_golden_mask.any())
        has_invalid_npu = bool(invalid_npu_mask.any())

        # 每个 batch 的有效 block 总数，用于计算该 batch 应选中的 mask 数量。
        elements = valid_x * valid_y
        expected_ones = expected_mask_ones(npu_sparsity, elements).reshape(batch, 1)

        # 统计每个 [batch, head] 的实际选中数，并和算子规则计算出的期望值比较。
        golden_ones = ((golden == 1) & valid).sum(dim=(-2, -1))
        npu_ones = ((npu == 1) & valid).sum(dim=(-2, -1))
        npu_count_error = npu_ones != expected_ones
        golden_count_error = golden_ones != expected_ones
        # compared 是有效 mask 元素总数；valid 当前没有 head 维，故乘以 head 数。
        compared = int(valid.sum()) * heads

        errors = []
        if has_invalid_golden or has_invalid_npu:
            invalid_golden = (
                torch.unique(golden[invalid_golden_mask]).tolist()
                if has_invalid_golden
                else []
            )
            invalid_npu = (
                torch.unique(npu[invalid_npu_mask]).tolist() if has_invalid_npu else []
            )
            errors.append(
                f"invalid mask values golden={invalid_golden} npu={invalid_npu}"
            )

        npu_count_error_indices = torch.nonzero(npu_count_error, as_tuple=False)
        for batch_index, head_index in npu_count_error_indices[:20].tolist():
            errors.append(
                f"B{batch_index}H{head_index}: NPU selected count "
                f"expected={int(expected_ones[batch_index, 0])} "
                f"npu={int(npu_ones[batch_index, head_index])}"
            )
        if npu_count_error_indices.shape[0] > 20:
            errors.append(
                f"additional NPU selected-count errors={npu_count_error_indices.shape[0] - 20}"
            )

        golden_count_error_indices = torch.nonzero(golden_count_error, as_tuple=False)
        for batch_index, head_index in golden_count_error_indices[:20].tolist():
            errors.append(
                f"B{batch_index}H{head_index}: Golden selected count "
                f"expected={int(expected_ones[batch_index, 0])} "
                f"golden={int(golden_ones[batch_index, head_index])}"
            )
        if golden_count_error_indices.shape[0] > 20:
            errors.append(
                f"additional Golden selected-count errors={golden_count_error_indices.shape[0] - 20}"
            )

        # ------------------------------------------------------------------
        # 4. 逐 head 比较。
        # ------------------------------------------------------------------
        # mismatches 统计集合差异元素数；tolerated 统计被边界/容错接受的差异数；
        # failed_mismatches 只统计最终导致失败的差异数。
        head_passes = 0
        failed_heads = 0
        mismatches = 0
        failed_mismatches = 0
        tolerated = 0
        golden_only_count = 0
        samples = []
        for b in range(batch):
            for h in range(heads):
                vx, vy = int(valid_x[b]), int(valid_y[b])
                # 将当前 head 的有效 mask 展平为 block index 集合，忽略 padding。
                npu_idx = torch.nonzero(
                    npu[b, h, :vx, :vy].reshape(-1) == 1, as_tuple=False
                ).flatten()
                golden_idx = torch.nonzero(
                    golden[b, h, :vx, :vy].reshape(-1) == 1, as_tuple=False
                ).flatten()
                npu_set = set(npu_idx.tolist())
                golden_set = set(golden_idx.tolist())
                # only_npu：NPU 多选的 block；only_golden：NPU 漏选的 Golden block。
                only_npu = npu_set - golden_set
                only_golden = golden_set - npu_set
                diff_count = len(only_npu) + len(only_golden)
                mismatches += diff_count
                golden_only_count += len(only_golden)
                # 集合完全相同即通过，不要求 mask 中 1 的排列顺序一致。
                if not only_npu and not only_golden:
                    head_passes += 1
                    continue

                # ----------------------------------------------------------
                # 4.1 TopK 边界容差。
                # 按 Golden softmax 从大到小排序，最后一个 Golden 入选 block
                # 的分数是 TopK 边界。只要所有替换 block 与该边界的相对误差
                # 不超过 BSA_BOUNDARY_REL_TOL，就把本 head 视为排序边界抖动。
                # ----------------------------------------------------------
                score_flat = softmax_out[b, h, :vx, :vy].reshape(-1)
                ordered_golden = sorted(
                    golden_set, key=lambda idx: (-float(score_flat[idx]), idx)
                )
                boundary = (
                    float(score_flat[ordered_golden[-1]]) if ordered_golden else 0.0
                )
                edge_pass = True
                for idx in only_npu | only_golden:
                    candidate = float(score_flat[idx])
                    if boundary == 0.0:
                        relative_error = 0.0 if candidate == 0.0 else float("inf")
                    else:
                        relative_error = abs(candidate - boundary) / abs(boundary)
                    if relative_error > BSA_BOUNDARY_REL_TOL:
                        edge_pass = False
                        break

                if edge_pass:
                    head_passes += 1
                    tolerated += diff_count
                    continue

                # ----------------------------------------------------------
                # 4.2 head 级 mask 容错。
                # 边界分数差异过大时，再比较整个有效 head 的 0/1 mask。
                # mask_close 是逐元素布尔结果，close_rate 必须达到 99.5%。
                # ----------------------------------------------------------
                npu_res_head = npu[b, h, :vx, :vy].reshape(-1).to(torch.float32)
                golden_res_head = golden[b, h, :vx, :vy].reshape(-1).to(torch.float32)
                if npu_res_head.numel() > 0:
                    # rtol/atol 保留统一的数值比较接口；对 0/1 mask 实际等价于
                    # 逐元素相等判断，同时支持按比例统计而非整 head 一票否决。
                    mask_close = torch.isclose(
                        npu_res_head,
                        golden_res_head,
                        rtol=BSA_HEAD_RTOL,
                        atol=BSA_HEAD_ATOL,
                        equal_nan=True,
                    )
                    close_rate = float(mask_close.to(torch.float32).mean())
                else:
                    close_rate = 0.0
                if close_rate >= BSA_HEAD_PASS_RATE:
                    head_passes += 1
                    tolerated += diff_count
                else:
                    failed_heads += 1
                    failed_mismatches += diff_count
                    for idx in sorted(only_npu | only_golden):
                        row, col = divmod(idx, vy)
                        if len(samples) < 20:
                            samples.append((b, h, row, col))
                    errors.append(
                        f"B{b}H{h}: index set mismatch={diff_count}, "
                        f"boundary_rel_tol={BSA_BOUNDARY_REL_TOL}, close_rate={close_rate:.6g}"
                    )

        # ------------------------------------------------------------------
        # 5. 汇总整个用例结果。
        # 非法值、选中数量错误或任一 head 最终失败，都会使整个用例失败。
        # ------------------------------------------------------------------
        passed = not (
            has_invalid_golden
            or has_invalid_npu
            or bool(npu_count_error.any())
            or bool(golden_count_error.any())
            or failed_heads
        )

        # exact 仅记录是否逐元素完全相等，不覆盖上面的容错判定结果。
        exact = torch.equal(npu, golden)
        errors.insert(
            0,
            f"boundary_rel_tol={BSA_BOUNDARY_REL_TOL:.6g}, head_pass_rate={BSA_HEAD_PASS_RATE:.6g}, "
            f"npu_sparsity={npu_sparsity:.6g}, golden_sparsity={golden_sparsity:.6g}, "
            f"passed_heads={head_passes}, failed_heads={failed_heads}",
        )

        return AccuracyConfig(
            result=passed,
            filename=data_file,
            error_info="; ".join(errors[:40]) if not passed else None,
            exact=exact,
            mismatches=mismatches,
            compared=compared,
            tolerated=tolerated,
            failed_mismatches=failed_mismatches,
            mismatch_samples=samples,
            golden_only=golden_only_count,
            error_ratio=(failed_mismatches / max(compared, 1)),
        )


def _to_float32(value):
    """按算子计算习惯将 Python 数值舍入为 IEEE FP32。"""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def op_selected_count(sparsity, elem_count, *, bottom_k=False):
    """计算一个 batch 有效区域应被选中的 block 数。"""
    s = _to_float32(sparsity)
    e = _to_float32(elem_count)
    r = _to_float32(_to_float32(1.0) - s) if bottom_k else s
    return max(math.floor(_to_float32(r * e) + 0.5), 1)


def expected_mask_ones(sparsity, elements):
    """Return the operator-style expected number of mask ones per batch."""
    bottom_k = sparsity > 0.5
    sparsity_fp32 = torch.tensor(float(sparsity), dtype=torch.float32)
    ratio = 1.0 - sparsity_fp32 if bottom_k else sparsity_fp32
    selected = torch.floor(ratio * elements.to(torch.float32) + 0.5)
    selected = selected.to(torch.int64).clamp_min(1)
    return torch.where(
        elements == 1,
        torch.ones_like(elements),
        elements - selected if bottom_k else selected,
    )


def golden_actual_block_lengths(
    batch_size, max_blocks, block_size, layout, actual_seq_lengths
):
    """Generate the same flat values as ACLNN and map TND's compact block indexing."""
    flat_values = [
        random.randint(0, block_size + 1) for _ in range(max_blocks * batch_size)
    ]
    if str(layout).upper() != "TND" or actual_seq_lengths is None:
        return [
            flat_values[batch_index * max_blocks : (batch_index + 1) * max_blocks]
            for batch_index in range(batch_size)
        ]

    nested_values = []
    compact_offset = 0
    for sequence_length in actual_seq_lengths:
        valid_blocks = _ceil_div(int(sequence_length), int(block_size))
        row = flat_values[compact_offset : compact_offset + valid_blocks]
        row.extend([0] * (max_blocks - valid_blocks))
        nested_values.append(row)
        compact_offset += valid_blocks
    return nested_values


def select_indices_with_index_tie(flat, count, *, largest):
    """按分数选择 block，并用原始扁平 index 作为稳定 tie-break。"""
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if count >= flat.numel():
        return torch.arange(flat.numel(), dtype=torch.long)
    try:
        ordered = torch.argsort(flat, descending=largest, stable=True)
    except TypeError:
        ol = sorted(
            range(flat.numel()),
            key=lambda i: (-float(flat[i]) if largest else float(flat[i]), i),
        )
        ordered = torch.tensor(ol, dtype=torch.long)
    return ordered[:count]


def _ceil_div(n, divisor):
    """整数向上除法，用于计算有效 block 数。"""
    return (n + divisor - 1) // divisor if divisor else 0


# ============================================================
# 主 golden
# ============================================================
def golden_bsa_mask(
    query,
    key,
    block_shape_x,
    block_shape_y,
    sparsity,
    scale,
    actual_block_len_query=None,
    actual_block_len_key=None,
    layout="BNSD",
    actual_seq_lens_q=None,
    actual_seq_lens_kv=None,
):
    global valid_shape_list
    global B_MASK, N_MASK, X_MASK, Y_MASK, BSA_SOFTMAX_OUT

    is_tnd_q, is_tnd_kv = layout in ("TND", "tnd"), layout in ("TND", "tnd")
    dev = query.device

    if not is_tnd_q:
        B, N, Sq, D = query.shape
        Skv = key.shape[2]
    else:
        N, D = query.shape[1], query.shape[2]
        B = len(actual_seq_lens_q) if actual_seq_lens_q else 1
        Sq = max(actual_seq_lens_q) if actual_seq_lens_q else query.shape[0]
        Skv = max(actual_seq_lens_kv) if actual_seq_lens_kv else key.shape[0]

    X = _ceil_div(Sq, block_shape_x)
    Y = _ceil_div(Skv, block_shape_y)
    mask = torch.zeros(B, N, X, Y, dtype=torch.int8).to(query.device)
    softmax_out = torch.zeros(B, N, X, Y, dtype=torch.float32, device=dev)
    B_MASK, N_MASK, X_MASK, Y_MASK = mask.shape

    q_seq_prefix, k_seq_prefix = 0, 0
    q_flat = query.flatten() if is_tnd_q else None
    k_flat = key.flatten() if is_tnd_kv else None
    per_seq = N * D

    valid_shape_list = []

    for b in range(B):
        cur_sq = actual_seq_lens_q[b] if (is_tnd_q and actual_seq_lens_q) else Sq
        cur_skv = actual_seq_lens_kv[b] if (is_tnd_kv and actual_seq_lens_kv) else Skv
        valid_x = _ceil_div(cur_sq, block_shape_x)
        valid_y = _ceil_div(cur_skv, block_shape_y)
        valid_shape_list.append([valid_x, valid_y])

        for n in range(N):
            # ==== Q Pooling (批量或逐块) ====
            q_blk = torch.zeros(X, D, dtype=torch.float16, device=dev)
            if is_tnd_q:
                # TND: 逐块 (变长 stride 无法简单 reshape)
                for x in range(valid_x):
                    ts = x * block_shape_x
                    al = block_shape_x
                    if actual_block_len_query is not None:
                        al = min(int(actual_block_len_query[b][x]), block_shape_x)
                    al = min(al, max(0, cur_sq - ts))
                    if al <= 0:
                        continue
                    base = q_seq_prefix * per_seq + ts * per_seq + n * D
                    off = torch.arange(al, device=dev) * per_seq + base
                    idx = off.unsqueeze(1) + torch.arange(D, device=dev).unsqueeze(0)
                    q_blk[x] = (
                        q_flat[idx.reshape(-1)].reshape(al, D).float().mean(0).half()
                    )
            elif actual_block_len_query is not None:
                # BNSD with actualBlockLen: 逐块
                for x in range(valid_x):
                    ts = x * block_shape_x
                    al = min(int(actual_block_len_query[b][x]), block_shape_x)
                    al = min(al, max(0, cur_sq - ts))
                    if al <= 0:
                        continue
                    q_blk[x] = query[b, n, ts : ts + al].float().mean(0).half()
            else:
                # BNSD uniform: 批量 reshape+mean (处理非整除)
                full_blocks = cur_sq // block_shape_x
                if full_blocks > 0:
                    src = query[b, n, : full_blocks * block_shape_x].float()
                    q_blk[:full_blocks] = (
                        src.reshape(full_blocks, block_shape_x, D).mean(1).half()
                    )
                if full_blocks < valid_x:
                    ts = full_blocks * block_shape_x
                    al = cur_sq - ts
                    if al > 0:
                        q_blk[full_blocks] = (
                            query[b, n, ts : ts + al].float().mean(0).half()
                        )

            # ==== K Pooling ====
            k_blk = torch.zeros(Y, D, dtype=torch.float16, device=dev)
            if is_tnd_kv:
                for y in range(valid_y):
                    ts = y * block_shape_y
                    al = block_shape_y
                    if actual_block_len_key is not None:
                        al = min(int(actual_block_len_key[b][y]), block_shape_y)
                    al = min(al, max(0, cur_skv - ts))
                    if al <= 0:
                        continue
                    base = k_seq_prefix * per_seq + ts * per_seq + n * D
                    off = torch.arange(al, device=dev) * per_seq + base
                    idx = off.unsqueeze(1) + torch.arange(D, device=dev).unsqueeze(0)
                    k_blk[y] = (
                        k_flat[idx.reshape(-1)].reshape(al, D).float().mean(0).half()
                    )
            elif actual_block_len_key is not None:
                for y in range(valid_y):
                    ts = y * block_shape_y
                    al = min(int(actual_block_len_key[b][y]), block_shape_y)
                    al = min(al, max(0, cur_skv - ts))
                    if al <= 0:
                        continue
                    k_blk[y] = key[b, n, ts : ts + al].float().mean(0).half()
            else:
                full_blocks = cur_skv // block_shape_y
                if full_blocks > 0:
                    src = key[b, n, : full_blocks * block_shape_y].float()
                    k_blk[:full_blocks] = (
                        src.reshape(full_blocks, block_shape_y, D).mean(1).half()
                    )
                if full_blocks < valid_y:
                    ts = full_blocks * block_shape_y
                    al = cur_skv - ts
                    if al > 0:
                        k_blk[full_blocks] = (
                            key[b, n, ts : ts + al].float().mean(0).half()
                        )

            # ==== Matmul + Softmax + TopK ====
            qv = q_blk[:valid_x].float()
            kv = k_blk[:valid_y].float()
            score = qv @ kv.T
            attn_fp32 = torch.softmax(score * _to_float32(scale), dim=-1)
            softmax_out[b, n, :valid_x, :valid_y] = attn_fp32
            # 选择仍使用原始 FP16 softmax，FP32 softmax 仅用于比较边界。
            attn = attn_fp32.half()

            flat = attn.flatten()
            ec = valid_x * valid_y
            if ec == 1:
                mask[b, n, 0, 0] = 1
                continue

            k_value = op_selected_count(sparsity, ec, bottom_k=(sparsity > 0.5))
            if sparsity <= 0.5:
                if k_value > 0:
                    idx = select_indices_with_index_tie(flat, k_value, largest=True)
                    mask[
                        b,
                        n,
                        torch.div(idx, valid_y, rounding_mode="floor"),
                        idx % valid_y,
                    ] = 1
            else:
                mask[b, n, :valid_x, :valid_y] = 1
                if k_value > 0:
                    idx = select_indices_with_index_tie(flat, k_value, largest=False)
                    mask[
                        b,
                        n,
                        torch.div(idx, valid_y, rounding_mode="floor"),
                        idx % valid_y,
                    ] = 0

        if is_tnd_q:
            q_seq_prefix += cur_sq
        if is_tnd_kv:
            k_seq_prefix += cur_skv

    BSA_SOFTMAX_OUT = softmax_out
    return mask


@register("function_aclnnBSASelectBlockMask")
class AiBSASelectBlockMaskFunctionnGoldenExecutor(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        OpsDataset.seed_everything(SEED)

        query = input_data.kwargs.get("query")
        key = input_data.kwargs.get("key")
        blockShapeOptional = input_data.kwargs.get("blockShapeOptional")
        qInputLayout = input_data.kwargs.get("qInputLayout")
        kvInputLayout = input_data.kwargs.get("kvInputLayout")
        scaleValue = input_data.kwargs.get("scaleValue")
        sparsity = input_data.kwargs.get("sparsity")
        actualBlockLenQueryOptional = input_data.kwargs.get(
            "actualBlockLenQueryOptional"
        )
        actualBlockLenKeyOptional = input_data.kwargs.get("actualBlockLenKeyOptional")
        actualSeqLengths = input_data.kwargs.get("actualSeqLengthsOptional")
        actualSeqLengthsKv = input_data.kwargs.get("actualSeqLengthsKvOptional")
        maxQSeqLength = input_data.kwargs.get("maxQSeqLength")
        maxKSeqLength = input_data.kwargs.get("maxKSeqLength")
        B = input_data.kwargs.get("B")

        if blockShapeOptional is not None:
            block_shape_x, block_shape_y = blockShapeOptional[0], blockShapeOptional[1]
        else:
            block_shape_x, block_shape_y = 128, 128

        x_blocks = (maxQSeqLength + block_shape_x - 1) // block_shape_x
        y_blocks = (maxKSeqLength + block_shape_y - 1) // block_shape_y

        if actualSeqLengths[0] is None:
            actualSeqLengths = None

        if actualSeqLengthsKv[0] is None:
            actualSeqLengthsKv = None

        if actualBlockLenQueryOptional[0] is not None:
            actualBlockLenQueryOptional = golden_actual_block_lengths(
                B, x_blocks, block_shape_x, qInputLayout, actualSeqLengths
            )
        else:
            actualBlockLenQueryOptional = None

        if actualBlockLenKeyOptional[0] is not None:
            actualBlockLenKeyOptional = golden_actual_block_lengths(
                B, y_blocks, block_shape_y, kvInputLayout, actualSeqLengthsKv
            )
        else:
            actualBlockLenKeyOptional = None

        # Golden 使用算子的原始 sparsity；容差只在比较器的 TopK 边界判断中生效。
        golden_sparsity = float(sparsity)
        out = golden_bsa_mask(
            query,
            key,
            block_shape_x,
            block_shape_y,
            golden_sparsity,
            scaleValue,
            actualBlockLenQueryOptional,
            actualBlockLenKeyOptional,
            qInputLayout,
            actualSeqLengths,
            actualSeqLengthsKv,
        )
        torch.npu.synchronize()
        meta = {
            "format": "bsa_mask_boundary_meta_v3",
            "shape": [
                int(B_MASK),
                int(N_MASK),
                int(X_MASK),
                int(Y_MASK),
            ],
            "valid_shapes": valid_shape_list,
            "npu_sparsity": float(sparsity),
            "golden_sparsity": float(golden_sparsity),
            "softmax_file": BSA_SOFTMAX_FILE,
            "softmax_dtype": "float32",
        }

        output_dir = self.task_result.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        meta_path = os.path.join(output_dir, BSA_META_FILE)
        tmp_path = meta_path + f".{os.getpid()}.tmp"

        torch.save(meta, tmp_path)
        os.replace(tmp_path, meta_path)

        softmax_path = os.path.join(output_dir, BSA_SOFTMAX_FILE)
        softmax_tmp_path = softmax_path + f".{os.getpid()}.tmp"
        torch.save(BSA_SOFTMAX_OUT.detach().cpu().to(torch.float32), softmax_tmp_path)
        os.replace(softmax_tmp_path, softmax_path)

        return out


@register("aclnn_function_aclnnBSASelectBlockMask")
class AiBSASelectBlockMaskFunctionGold(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """准备 ACLNN 参数，将 Python 可选参数转换为空指针或 ACL 数组。"""
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        AclIntArrayPtr = ctypes.POINTER(AclIntArray)  # tensor指针类型
        null_array_ptr = ctypes.cast(
            null_void_ptr, AclIntArrayPtr
        )  # 把这个空指针类型转换为tensor指针类型

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        OpsDataset.seed_everything(SEED)

        blockShapeOptional = input_data.kwargs.get("blockShapeOptional")
        actualBlockLenQueryOptional = input_data.kwargs.get(
            "actualBlockLenQueryOptional"
        )
        actualBlockLenKeyOptional = input_data.kwargs.get("actualBlockLenKeyOptional")
        actualSeqLengths = input_data.kwargs.get("actualSeqLengthsOptional")
        actualSeqLengthsKv = input_data.kwargs.get("actualSeqLengthsKvOptional")
        maxQSeqLength = input_data.kwargs.get("maxQSeqLength")
        maxKSeqLength = input_data.kwargs.get("maxKSeqLength")
        B = input_data.kwargs.get("B")

        if blockShapeOptional is None:
            input_data.kwargs["blockShapeOptional"] = [128, 128]
            blockShapeOptional = [128, 128]
        block_shape_x, block_shape_y = blockShapeOptional[0], blockShapeOptional[1]

        x_blocks = (maxQSeqLength + block_shape_x - 1) // block_shape_x
        y_blocks = (maxKSeqLength + block_shape_y - 1) // block_shape_y

        genQ = actualBlockLenQueryOptional[0] is not None
        genK = actualBlockLenKeyOptional[0] is not None
        if genQ:
            actualBlockLenQueryOptional = [
                random.randint(0, block_shape_x + 1) for _ in range(x_blocks * B)
            ]
            input_data.kwargs["actualBlockLenQueryOptional"] = (
                actualBlockLenQueryOptional
            )
        else:
            input_data.kwargs["actualBlockLenQueryOptional"] = null_array_ptr
        if genK:
            actualBlockLenKeyOptional = [
                random.randint(0, block_shape_y + 1) for _ in range(y_blocks * B)
            ]
            input_data.kwargs["actualBlockLenKeyOptional"] = actualBlockLenKeyOptional
        else:
            input_data.kwargs["actualBlockLenKeyOptional"] = null_array_ptr
        # DEBUG-GOLDEN: 打印执行器生成的 abl 参数（仅日志，不改逻辑），验完删除
        # 单行自包含（并发安全）：case id + 形状签名 + 完整 abl，供离线对账导出
        _cid = getattr(input_data, "id", None)
        _qshp = (
            input_data.kwargs.get("query").shape
            if hasattr(input_data.kwargs.get("query"), "shape")
            else None
        )
        print(
            f"[GOLDEN-PARAM] id={_cid} B={B} bsx={block_shape_x} bsy={block_shape_y} "
            f"xb={x_blocks} yb={y_blocks} qshape={_qshp} genQ={genQ} genK={genK} "
            f"ablQ={actualBlockLenQueryOptional if genQ else 'null'} "
            f"ablK={actualBlockLenKeyOptional if genK else 'null'}",
            flush=True,
        )

        if actualSeqLengths[0] is None:
            input_data.kwargs["actualSeqLengthsOptional"] = null_array_ptr

        if actualSeqLengthsKv[0] is None:
            input_data.kwargs["actualSeqLengthsKvOptional"] = null_array_ptr

        input_data.kwargs["numKeyValueHeads"] = input_data.kwargs.get("N")

        # 当前算子暂不支持 postBlockShape，只能传空指针。
        input_data.kwargs["postBlockShapeOptional"] = null_array_ptr

        # 删除仅用于数据生成、但不属于 ACLNN 接口的参数。
        del input_data.kwargs["B"]
        del input_data.kwargs["N"]
        del input_data.kwargs["maxQSeqLength"]
        del input_data.kwargs["maxKSeqLength"]

        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnBSASelectBlockMaskGetWorkspaceSize("
            "const aclTensor *query, "
            "const aclTensor *key, "
            "const aclIntArray *blockShapeOptional, "
            "const aclIntArray *postBlockShapeOptional, "
            "const aclIntArray *actualSeqLengthsOptional, "
            "const aclIntArray *actualSeqLengthsKvOptional, "
            "const aclIntArray *actualBlockLenQueryOptional, "
            "const aclIntArray *actualBlockLenKeyOptional, "
            "char *qInputLayoutOptional, "
            "char *kvInputLayoutOptional, "
            "int64_t numKeyValueHeads, "
            "double scaleValue, "
            "double sparsity, "
            "const aclTensor *out, "
            "uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )
