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
"""aclnnSparseFlashMlaGrad ATK executor.

Same split as executor_aclnnFlashAttentionScoreGradV3:
  execute_aclnnSparseFlashMlaGrad       CPU formula. ATK runs it twice for
                                        cv_fused_double_benchmark: once in fp32
                                        (golden) and once in fp16/bf16 (benchmark).
  pyaclnn_execute_aclnnSparseFlashMlaGrad  aclnn adapter. The device node
                                        must use backend pyaclnn so this path,
                                        not the CPU formula, is what runs.

Register names match the existing normal-case JSON. Topology is rebuilt from
the case id seed in both classes, so they share cu_seqlens and cmp indices
without a golden_path cache.
"""

from itertools import accumulate

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def tsoftmax(x, sinks):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    #### max(sink, max(S)) ####
    N1 = sinks.shape[0]
    x_max = torch.max(x_max, sinks.view(N1, 1))
    ###########################

    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)  # sum(exp(x - m))

    #### sum += exp(sink-max) ####
    exp_sink = torch.exp(sinks.view(N1, 1) - x_max)
    x_sum += exp_sink
    ###########################
    ans = y.div(x_sum)
    lse = torch.log(x_sum) + x_max
    return ans, lse


def simpleSoftmax(x, lse):
    x_sub = x.sub(lse)
    softmax_res = torch.exp(x_sub)
    return softmax_res


def tsoftmax_grad(p, dp, out, outGrad):
    muls = outGrad * out
    muls_res = muls.sum(dim=-1, keepdims=True)
    sub_res = dp - muls_res
    res = sub_res * p
    return res


def get_tnd_idx(actual_q_len, t_idx):
    b_idx = 0
    while t_idx >= actual_q_len[b_idx]:
        b_idx += 1
    if b_idx == 0:
        s1_offset = 0
    else:
        s1_offset = actual_q_len[b_idx - 1]
    s1_idx = t_idx - s1_offset
    return b_idx, s1_idx


# mask mode: 3=causal, 4=swa/band
MASK_MODE_CAUSAL = 3
MASK_MODE_SWA = 4
# causal: attend all history tokens (effectively infinite left window)
CAUSAL_WIN_LEFT = 1 << 30


def resolve_ori_windows(ori_mask_mode, ori_win_left, ori_win_right):
    """Map oriMaskMode to effective (win_left, win_right) for golden."""
    if ori_mask_mode == MASK_MODE_CAUSAL:
        return CAUSAL_WIN_LEFT, 0
    # SWA / band: use configured sliding window
    return ori_win_left, ori_win_right


def is_nonempty_tensor(t):
    return t is not None and t.shape[0] != 0


def _match_dtype(t, dtype):
    """Keep matmul operands on one dtype. torch.cat promotes half/bf16 with float."""
    if t is None or not torch.is_tensor(t) or t.dtype == dtype:
        return t
    return t.to(dtype)


def is_empty_tensor(t):
    """A3 empty tensor: missing, numel 0, or any dimension is 0. K=0 indices are empty."""
    if t is None or not torch.is_tensor(t):
        return True
    if t.numel() == 0:
        return True
    return any(int(dim) == 0 for dim in t.shape)


def resolve_cmp_scene(
    cmp_kv, cmp_sparse_indices, ori_sparse_indices=None, ori_mask_mode=None
):
    """A3 使用说明, independent of cmpRatio (cmpRatio is only the compression rate, 1~128).

    oriMaskMode 0 + oriSparseIndices nonempty + cmpKv empty -> ORI (MODE=3)
    cmpKv empty -> SWA
    cmpKv nonempty and cmpSparseIndices empty -> CFA
    both nonempty -> SCFA (Gather by topk; cmpSoftmaxL1Norm is produced)
    """
    if (
        ori_mask_mode == 0
        and not is_empty_tensor(ori_sparse_indices)
        and is_empty_tensor(cmp_kv)
    ):
        return "ORI"
    if is_empty_tensor(cmp_kv):
        return "SWA"
    if is_empty_tensor(cmp_sparse_indices):
        return "CFA"
    return "SCFA"


def _as_int_list(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        val = val.detach().cpu().tolist()
    if not isinstance(val, (list, tuple)):
        return None
    out = []
    for x in val:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, np.integer)):
            out.append(int(x))
        elif isinstance(x, (float, np.floating)) and float(x).is_integer():
            out.append(int(x))
        else:
            return None
    return out


def _config_slots(case_config, name, total, batch):
    # ATK may treat range_values as a sampling range rather than literal data.
    prefix = _as_int_list(_inp_range(_input_by_name(case_config, name)))
    if (
        prefix is None
        or len(prefix) != batch + 1
        or prefix[0] != 0
        or prefix[-1] != total
        or any(left >= right for left, right in zip(prefix, prefix[1:]))
    ):
        raise ValueError(
            f"case {_case_id(case_config)}: {name} must contain {batch + 1} "
            f"strictly increasing offsets from 0 to {total}, got {prefix}"
        )
    return [right - left for left, right in zip(prefix, prefix[1:])]


def _cu_to_slots(cu, batch):
    items = _as_int_list(cu)
    if items is None:
        return None
    if len(items) == batch + 1:
        return [items[i + 1] - items[i] for i in range(batch)]
    if len(items) == batch:
        return items
    return None


def _input_by_name(case_config, name):
    for inp in getattr(case_config, "inputs", None) or []:
        inp_name = (
            inp.get("name") if isinstance(inp, dict) else getattr(inp, "name", None)
        )
        if inp_name == name:
            return inp
    return None


def _inp_range(inp):
    if inp is None:
        return None
    if isinstance(inp, dict):
        return inp.get("range_values")
    return getattr(inp, "range_values", None)


def parse_eod_spec(case_config, batch, t1, t2, t3, cmp_ratio):
    """Read EOD slot/seqused from generate strategy or concrete range_values.

    Non-EOD cases keep range_values like [-5, 5] and are rejected here.
    """
    strat = getattr(case_config, "strategy", None)
    if not isinstance(strat, dict):
        strat = {}

    def _from_strat():
        if not (strat.get("eod") or strat.get("seqused_q") or strat.get("slot_q")):
            return None
        slot_q = _as_int_list(strat.get("slot_q"))
        slot_kv = _as_int_list(strat.get("slot_kv"))
        slot_cmp = _as_int_list(strat.get("slot_cmp"))
        used_q = _as_int_list(strat.get("seqused_q"))
        used_kv = _as_int_list(strat.get("seqused_ori_kv"))
        used_cmp = _as_int_list(strat.get("seqused_cmp_kv"))
        residuals = _as_int_list(strat.get("cmp_residuals"))
        if slot_q is None or slot_kv is None or used_q is None or used_kv is None:
            return None
        if slot_cmp is None:
            slot_cmp = (
                [0] * batch if cmp_ratio == 1 else [uk // cmp_ratio for uk in slot_kv]
            )
        if used_cmp is None:
            used_cmp = (
                [0] * batch if cmp_ratio == 1 else [uk // cmp_ratio for uk in used_kv]
            )
        if residuals is None:
            residuals = (
                [0] * batch if cmp_ratio == 1 else [uk % cmp_ratio for uk in used_kv]
            )
        return slot_q, slot_kv, slot_cmp, used_q, used_kv, used_cmp, residuals

    def _from_range_values():
        cu_q = _input_by_name(case_config, "cuSeqlensQ")
        cu_kv = _input_by_name(case_config, "cuSeqlensOriKv")
        cu_cmp = _input_by_name(case_config, "cuSeqlensCmpKv")
        seq_q = _input_by_name(case_config, "sequsedQ")
        seq_kv = _input_by_name(case_config, "sequsedOriKv")
        seq_cmp = _input_by_name(case_config, "sequsedCmpKv")
        res_in = _input_by_name(case_config, "cmpResidualKv")
        slot_q = _cu_to_slots(_inp_range(cu_q), batch)
        slot_kv = _cu_to_slots(_inp_range(cu_kv), batch)
        used_q = _as_int_list(_inp_range(seq_q))
        used_kv = _as_int_list(_inp_range(seq_kv))
        if slot_q is None or slot_kv is None or used_q is None or used_kv is None:
            return None
        if slot_q[0] < 0 or used_q[0] < 0:
            return None
        if len(used_q) != batch or len(used_kv) != batch:
            return None
        slot_cmp = _cu_to_slots(_inp_range(cu_cmp), batch)
        used_cmp = _as_int_list(_inp_range(seq_cmp))
        residuals = _as_int_list(_inp_range(res_in))
        if slot_cmp is None:
            slot_cmp = (
                [0] * batch
                if cmp_ratio == 1
                else [uk // max(cmp_ratio, 1) for uk in slot_kv]
            )
        if used_cmp is None:
            used_cmp = (
                [0] * batch
                if cmp_ratio == 1
                else [uk // max(cmp_ratio, 1) for uk in used_kv]
            )
        if residuals is None:
            residuals = (
                [0] * batch
                if cmp_ratio == 1
                else [uk % max(cmp_ratio, 1) for uk in used_kv]
            )
        return slot_q, slot_kv, slot_cmp, used_q, used_kv, used_cmp, residuals

    parsed = _from_strat() or _from_range_values()
    if parsed is None:
        return None
    slot_q, slot_kv, slot_cmp, used_q, used_kv, used_cmp, residuals = parsed
    if not (len(slot_q) == len(slot_kv) == len(used_q) == len(used_kv) == batch):
        return None
    if len(slot_cmp) != batch or len(used_cmp) != batch or len(residuals) != batch:
        return None
    if any(u < 0 or u > s for u, s in zip(used_q, slot_q)):
        return None
    if any(u < 0 or u > s for u, s in zip(used_kv, slot_kv)):
        return None
    if any(u < 0 or u > s for u, s in zip(used_cmp, slot_cmp)):
        return None
    if sum(slot_q) != t1 or sum(slot_kv) != t2:
        print(
            f"EOD spec slot sum mismatch: slot_q={sum(slot_q)} vs T1={t1}, "
            f"slot_kv={sum(slot_kv)} vs T2={t2}; ignore"
        )
        return None
    if t3 is not None and sum(slot_cmp) != t3:
        print(f"EOD spec slot_cmp sum {sum(slot_cmp)} vs T3={t3}; ignore")
        return None
    return {
        "slot_q": slot_q,
        "slot_kv": slot_kv,
        "slot_cmp": slot_cmp,
        "seqused_q": used_q,
        "seqused_ori_kv": used_kv,
        "seqused_cmp_kv": used_cmp,
        "residuals": residuals,
    }


class SparseFlashMla:
    def __init__(
        self,
        T1,
        T2,
        T3,
        N1,
        N2,
        D,
        B,
        G,
        query,
        ori_kv,
        cmp_kv,
        out,
        dy,
        cmp_sparse_indices,
        dtype,
        cuSeqlensQ,
        cuSeqlensOriKv,
        cuSeqlensCmpKv,
        scaleValue=1.0,
        blockCount=512,
        cmp_ratio=1,
        ori_win_right=0,
        ori_win_left=127,
        sinks=None,
        lse=None,
        cmp_residuals=None,
        is_golden=False,
        ori_mask_mode=MASK_MODE_SWA,
        cmp_mask_mode=MASK_MODE_CAUSAL,
        used_q_len=None,
        used_kv_len=None,
        used_cmp_kv_len=None,
        ori_sparse_indices=None,
        ori_topk_length=None,
    ):
        self.query = query
        self.ori_kv = ori_kv
        self.cmp_kv = cmp_kv
        self.out = out
        self.dy = dy
        self.cmp_sparse_indices = cmp_sparse_indices
        self.sinks = sinks
        self.lse = lse
        self.orgDtype = dtype
        self.cu_seqlens_q = cuSeqlensQ
        self.cu_seqlens_ori_k = cuSeqlensOriKv
        self.cu_seqlens_cmp_k = cuSeqlensCmpKv
        self.next_token = ori_win_right
        self.pre_token = ori_win_left
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.scaleValue = scaleValue
        self.T1, self.T2, self.T3, self.N1, self.N2, self.D, self.B, self.G = (
            T1,
            T2,
            T3,
            N1,
            N2,
            D,
            B,
            G,
        )
        self.is_golden = is_golden
        self.cmp_residuals = cmp_residuals
        # EOD: used_* is seqused (valid prefix); default equals physical slot (cu delta)
        slot_q = _cu_to_slots(cuSeqlensQ, B) or [0] * B
        slot_kv = _cu_to_slots(cuSeqlensOriKv, B) or [0] * B
        slot_cmp = _cu_to_slots(cuSeqlensCmpKv, B) or [0] * B
        self.used_q_len = _as_int_list(used_q_len) or slot_q
        self.used_kv_len = _as_int_list(used_kv_len) or slot_kv
        self.used_cmp_kv_len = _as_int_list(used_cmp_kv_len) or slot_cmp
        self.ori_sparse_indices = ori_sparse_indices
        self.ori_topk_length = ori_topk_length
        self.cmp_scene = resolve_cmp_scene(
            cmp_kv, cmp_sparse_indices, ori_sparse_indices, ori_mask_mode
        )

    def _token_index(self, b_idx, s1_index):
        return int(self.cu_seqlens_q[b_idx]) + int(s1_index)

    def _ori_cols(self, b_idx, s1_index, cur_s2, n2_idx=0):
        """Valid ori_kv columns for one query token.

        actualK = min(ori_topk_length, K, usedS2). length < 0 is 0.
        index < 0 or index >= usedS2, and columns at/after actualK, are dropped:
        Gather would write 0 and Softmax would mask them as -inf, so they add
        nothing to the probability or the gradient. Scatter skips them.
        """
        tok = self._token_index(b_idx, s1_index)
        indices = self.ori_sparse_indices[tok, n2_idx]
        length = int(self.ori_topk_length[tok, n2_idx])
        if length < 0:
            length = 0
        k = int(indices.shape[-1])
        actual = min(length, k, int(cur_s2))
        cols = []
        for j in range(actual):
            idx = int(indices[j])
            if 0 <= idx < cur_s2:
                cols.append(idx)
        return cols

    def selectKV(
        self, ori_kv, cmp_kv, s1_index, topk, cur_s1, cur_s2, cur_s3, b_idx, n2_idx=0
    ):
        if self.cmp_scene == "ORI":
            cols = self._ori_cols(b_idx, s1_index, cur_s2, n2_idx)
            if cols:
                sel_ori_kv = ori_kv[cols, :]
            else:
                sel_ori_kv = ori_kv[:0, :]
            return sel_ori_kv, None

        diag_offset = cur_s2 - cur_s1
        right = s1_index + self.next_token + 1 + diag_offset
        left = s1_index - self.pre_token + diag_offset
        start = max(0, left)
        end = min(cur_s2, right)
        end = max(end, 0)
        sel_ori_kv = ori_kv[start:end, :]

        if self.cmp_scene == "SCFA":
            s2_sparse = list()
            for sparse_id in topk:
                if sparse_id == -1:
                    break
                begin_idx = sparse_id
                end_idx = begin_idx + 1 if begin_idx + 1 <= cur_s3 else cur_s3
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            sel_cmp_kv = cmp_kv[s2_sparse, :]
        elif self.cmp_scene == "CFA":
            ########### cmpMaskMode causal-style / sparseMode=3 ###########
            diag_offset_cmp = (
                cur_s3 * self.cmp_ratio + self.cmp_residuals[b_idx]
            ) - cur_s1
            threshold = max(diag_offset_cmp + s1_index + 1, 0) // self.cmp_ratio
            sel_cmp_kv = cmp_kv[:threshold, :]
        else:
            sel_cmp_kv = None
        return sel_ori_kv, sel_cmp_kv

    def scatterKV(
        self,
        dori_out,
        dcmp_out,
        dkv,
        s1_index,
        topk,
        cur_s1,
        cur_s2,
        cur_s3,
        b_idx,
        n2_idx=0,
    ):
        """
        dori_out: [cur_s2, D]
        ocmp_out: [cur_s3, D]
        dkv_out: [sel_s2, D]
        topk: [blockCount]
        """
        if self.cmp_scene == "ORI":
            cols = self._ori_cols(b_idx, s1_index, cur_s2, n2_idx)
            if cols and is_nonempty_tensor(dkv):
                idx = torch.tensor(cols, dtype=torch.long)
                dori_out.index_add_(0, idx, dkv[: len(cols)].to(dori_out.dtype))
            return dori_out, dcmp_out

        # ori_kv scatter
        diag_offset = cur_s2 - cur_s1
        right = s1_index + self.next_token + 1 + diag_offset
        left = s1_index - self.pre_token + diag_offset

        start = max(0, left)
        end = min(cur_s2, right)
        end = max(end, 0)
        actual_len = end - start
        if actual_len > 0:
            dori_out[start:end, :] += dkv[:actual_len, :]
            dkv = dkv[actual_len:]

        if self.cmp_scene == "SCFA":
            dkv_start = 0
            for sparse_id in topk:
                if sparse_id == -1:
                    break
                begin_idx = sparse_id
                end_idx = begin_idx + 1 if begin_idx + 1 <= cur_s3 else cur_s3
                dkv_end = dkv_start + (end_idx - begin_idx)
                dcmp_out[begin_idx:end_idx, :] += dkv[dkv_start:dkv_end, :]
                dkv_start = dkv_end
        elif self.cmp_scene == "CFA":
            ########### cmpMaskMode causal-style / sparseMode=3 ###########
            diag_offset_cmp = (
                cur_s3 * self.cmp_ratio + self.cmp_residuals[b_idx]
            ) - cur_s1
            threshold = max(diag_offset_cmp + s1_index + 1, 0) // self.cmp_ratio
            dcmp_out[:threshold] += dkv
        return dori_out, dcmp_out

    def forward(self):
        query = self.query.reshape(self.T1, self.N2, self.G, self.D)
        ori_kv = self.ori_kv.reshape(self.T2, self.N2, self.D)

        T1 = self.T1
        N1 = self.N2 * self.G
        T2, N2, D_qk = ori_kv.shape

        if self.cmp_scene not in ("SWA", "ORI"):
            cmp_kv = self.cmp_kv.reshape(self.T3, self.N2, self.D)
            T3, N2, D_qk = cmp_kv.shape
        else:
            cmp_kv = None

        if self.is_golden:
            query = query.float()
            ori_kv = ori_kv.float()
            if cmp_kv is not None:
                cmp_kv = cmp_kv.float()
        else:
            ori_kv = _match_dtype(ori_kv, query.dtype)
            if cmp_kv is not None:
                cmp_kv = _match_dtype(cmp_kv, query.dtype)

        print("forward query dtype:", query.dtype)
        print("forward ori_kv dtype:", ori_kv.dtype)
        print("111 forward orgDtype:", self.orgDtype)

        out_golden = torch.zeros_like(query)
        lse = torch.zeros(T1, N1)

        for i in range(T1):
            n2_idx = 0
            topk = (
                self.cmp_sparse_indices[i][n2_idx]
                if self.cmp_sparse_indices != None
                else None
            )
            qi = query[i][n2_idx]  # [G, D1]

            b_idx, s1_idx = get_tnd_idx(self.cu_seqlens_q[1:], i)
            curS1 = self.used_q_len[b_idx]
            curS2 = self.used_kv_len[b_idx]
            if int(s1_idx) >= curS1:
                continue
            s2_start = int(self.cu_seqlens_ori_k[b_idx])
            ori_kvi = ori_kv[s2_start : s2_start + curS2, n2_idx, :]

            if self.cmp_scene not in ("SWA", "ORI"):
                s3_start = int(self.cu_seqlens_cmp_k[b_idx])
                curS3 = self.used_cmp_kv_len[b_idx]
                cmp_kvi = cmp_kv[s3_start : s3_start + curS3, n2_idx, :]
            else:
                cmp_kvi = None
                curS3 = 0

            ori_kvi_calc, cmp_kvi_calc = self.selectKV(
                ori_kvi, cmp_kvi, s1_idx, topk, curS1, curS2, curS3, b_idx, n2_idx
            )

            kv = (
                torch.cat(
                    (ori_kvi_calc, _match_dtype(cmp_kvi_calc, ori_kvi_calc.dtype)),
                    dim=-2,
                )
                if is_nonempty_tensor(cmp_kvi_calc)
                else ori_kvi_calc
            )
            kv = _match_dtype(kv, qi.dtype)

            if kv.shape[0] == 0 and self.cmp_scene == "ORI":
                # No legal column. Kernel skips the row; lse falls back to sinks.
                lse[i] = self.sinks.reshape(-1)[:N1].float()
            elif kv.shape[0] != 0:
                qk = torch.matmul(qi, kv.permute(1, 0)).mul(
                    self.scaleValue
                )  # [G, catS2]
                qk = qk.float()
                softmax_res, lsei = tsoftmax(qk, self.sinks)  # [G, catS2], [G, 1]

                outi_golden = torch.matmul(softmax_res.to(kv.dtype), kv)
                out_golden[i] = outi_golden

                lsei = lsei.permute(1, 0)
                lse[i] = lsei

        self.out = out_golden.to(self.orgDtype)
        self.lse = lse

        return self.out, self.lse

    def backward(self):
        query = self.query.reshape(self.T1, self.N2, self.G, self.D)
        ori_kv = self.ori_kv.reshape(self.T2, self.N2, self.D)
        dy = self.dy.reshape(self.T1, self.N2, self.G, self.D)

        # shape
        T1 = self.T1
        N1 = self.N2 * self.G
        T2, N2, D_qk = ori_kv.shape
        G = N1 // N2

        if self.is_golden:
            query = query.float()
            ori_kv = ori_kv.float()
            dy = dy.float()
        else:
            # fp16/bf16 golden stays in the input dtype. A float cmp/ori cache
            # would otherwise cat-promote KV and fail matmul (Half != float).
            ori_kv = _match_dtype(ori_kv, query.dtype)
            dy = _match_dtype(dy, query.dtype)

        print("111 backward query dtype:", query.dtype)
        print("111 backward ori_kv dtype:", ori_kv.dtype)
        print("111 backward orgDtype:", self.orgDtype)

        dq = torch.zeros_like(query).float()
        dori_kv = torch.zeros_like(ori_kv).float()
        dsinks = torch.zeros_like(self.sinks).float()

        if self.cmp_scene not in ("SWA", "ORI"):
            cmp_kv = self.cmp_kv.reshape(self.T3, self.N2, self.D)
            T3, N2, D_qk = cmp_kv.shape
            if self.is_golden:
                cmp_kv = cmp_kv.float()
            else:
                cmp_kv = _match_dtype(cmp_kv, query.dtype)
            dcmp_kv = torch.zeros_like(cmp_kv).float()
        else:
            dcmp_kv = None

        if self.cmp_scene == "SCFA":
            cmp_softmax_l1 = torch.zeros_like(self.cmp_sparse_indices).to(torch.float)
        else:
            cmp_softmax_l1 = None

        out = self.out.reshape(T1, N2, G, D_qk)
        if self.is_golden:
            out = out.float()
        else:
            out = _match_dtype(out, query.dtype)
        for i in range(T1):
            # print("backward i:", i)
            n2_idx = 0
            topk = (
                self.cmp_sparse_indices[i][n2_idx]
                if self.cmp_sparse_indices != None
                else None
            )
            qi = query[i][n2_idx]  # [G, D1]
            dyi = dy[i][n2_idx]
            outi = out[i][n2_idx]

            b_idx, s1_idx = get_tnd_idx(self.cu_seqlens_q[1:], i)
            curS1 = self.used_q_len[b_idx]
            curS2 = self.used_kv_len[b_idx]
            if int(s1_idx) >= curS1:
                continue
            s2_start = int(self.cu_seqlens_ori_k[b_idx])
            ori_kvi = ori_kv[s2_start : s2_start + curS2, n2_idx, :]

            if self.cmp_scene not in ("SWA", "ORI"):
                s3_start = int(self.cu_seqlens_cmp_k[b_idx])
                curS3 = self.used_cmp_kv_len[b_idx]
                cmp_kvi = cmp_kv[s3_start : s3_start + curS3, n2_idx, :]
            else:
                cmp_kvi = None
                curS3 = 0

            ori_kvi_calc, cmp_kvi_calc = self.selectKV(
                ori_kvi, cmp_kvi, s1_idx, topk, curS1, curS2, curS3, b_idx, n2_idx
            )
            ori_s2 = ori_kvi_calc.shape[0] if is_nonempty_tensor(ori_kvi_calc) else 0

            kv = (
                torch.cat(
                    (ori_kvi_calc, _match_dtype(cmp_kvi_calc, ori_kvi_calc.dtype)),
                    dim=-2,
                )
                if is_nonempty_tensor(cmp_kvi_calc)
                else ori_kvi_calc
            )
            kv = _match_dtype(kv, qi.dtype)

            if kv.shape[0] != 0:
                qk = torch.matmul(qi, kv.permute(1, 0)).mul(
                    self.scaleValue
                )  # [G, catS2]
                qk = qk.float()
                softmax_res = simpleSoftmax(
                    qk, self.lse[i, :].reshape(N1, 1)
                )  # [G, catS2], [G, 1]
                softmax_res = softmax_res.float()

                if self.cmp_scene == "SCFA":
                    cmp_softmax = softmax_res[:, ori_s2:]  # [G, cmpS2]
                    cmp_softmax = cmp_softmax.sum(dim=0, keepdim=True) / G  # [1, cmpS2]
                    cmp_softmax_l1[i, :, : cmp_softmax.shape[-1]] = cmp_softmax

                dp = torch.matmul(dyi, kv.permute(1, 0))

                softmax_grad_res = tsoftmax_grad(
                    softmax_res, dp, outi.float(), dyi.float()
                )  # [G, catS2]
                softmax_grad_res = softmax_grad_res.to(qi.dtype)

                #### dsinks ####
                dsink_sum = softmax_res * dp  # [G, catS2]
                dsink_sum *= simpleSoftmax(
                    self.sinks.reshape(N1, 1), self.lse[i, :].reshape(N1, 1)
                )  # [G, catS2] * [G, 1] = [G, catS2]
                dsink = -dsink_sum.sum(dim=(-1))  # [G, catS2] --> [G]
                dsinks += dsink

                dqi = torch.matmul(
                    softmax_grad_res, kv
                )  # [G, catS2] * [catS2, D] = [G, D]
                dkv = (
                    torch.matmul(softmax_grad_res.permute(1, 0), qi) * self.scaleValue
                )  # [catS2, G] * [G, D] = [catS2, D]
                dkv += torch.matmul(
                    softmax_res.permute(1, 0).to(qi.dtype), dyi
                )  # [catS2, G] * [G, D] = [catS2, D]

                dq[i : i + 1, :, :] = dqi
                if self.cmp_scene in ("SWA", "ORI"):
                    dori_kv[s2_start : s2_start + curS2, n2_idx, :], _ = self.scatterKV(
                        dori_kv[s2_start : s2_start + curS2, n2_idx, :],
                        None,
                        dkv,
                        s1_idx,
                        topk,
                        curS1,
                        curS2,
                        curS3,
                        b_idx,
                        n2_idx,
                    )
                else:
                    (
                        dori_kv[s2_start : s2_start + curS2, n2_idx, :],
                        dcmp_kv[s3_start : s3_start + curS3, n2_idx, :],
                    ) = self.scatterKV(
                        dori_kv[s2_start : s2_start + curS2, n2_idx, :],
                        dcmp_kv[s3_start : s3_start + curS3, n2_idx, :],
                        dkv,
                        s1_idx,
                        topk,
                        curS1,
                        curS2,
                        curS3,
                        b_idx,
                        n2_idx,
                    )

        dq = dq * self.scaleValue

        # print("111 dq dtype:", dq.dtype)
        # print("111 dori_kv dtype:", dori_kv.dtype)
        # print("111 dcmp_kv dtype:", dcmp_kv.dtype)

        dq = dq.to(self.orgDtype)
        dori_kv = dori_kv.to(self.orgDtype)
        dcmp_kv = dcmp_kv.to(self.orgDtype) if dcmp_kv != None else dcmp_kv

        return dq, dori_kv, dcmp_kv, dsinks, cmp_softmax_l1


def get_cmp_topk(q_slots, used_q, used_kv, used_cmp, n2, selected, cmp_ratio, case_id):
    rng = np.random.RandomState(10007 + case_id)
    indices = torch.full((sum(q_slots), n2, selected), -1, dtype=torch.int32)
    offset = 0
    for batch, (slot_q, q_len, ori_len, cmp_len) in enumerate(
        zip(q_slots, used_q, used_kv, used_cmp)
    ):
        visible = (
            (torch.arange(q_len) + ori_len - q_len + 1).clamp(min=0) // cmp_ratio
        ).clamp(max=cmp_len)
        rows = indices[offset : offset + q_len]
        for row, limit in zip(rows, visible.tolist()):
            count = min(selected, limit)
            if count:
                for head in row:
                    head[:count] = torch.from_numpy(
                        rng.choice(limit, count, replace=False)
                    )
        if ((rows < -1) | (rows >= visible[:, None, None])).any() or (
            (rows[..., :-1] == -1) & (rows[..., 1:] >= 0)
        ).any():
            raise ValueError(
                f"case {case_id}: invalid cmpSparseIndices in batch {batch}"
            )
        offset += slot_q
    return indices


def _on_cpu(value):
    """Formula tensors stay on CPU. ATK may already have moved inputs to NPU."""
    if torch.is_tensor(value) and value.device.type != "cpu":
        return value.detach().cpu()
    return value


# The helpers and SparseFlashMla class above are the CPU formula.
# Registration below follows aclnnFlashAttentionScoreGradV3:
#   api_type       CPU formula only (fp32 golden task and fp16/bf16 benchmark)
#   aclnn_api_type aclnn adapter only
# cv_fused_double_benchmark compares kernel vs fp32 formula, divided by
# the same formula at the case dtype. The NPU node must not reuse the kernel
# as that low-precision benchmark.


def _case_id(case_config):
    value = getattr(case_config, "id", 0)
    return 0 if value is None else int(value)


def _declared_dtype(case_config):
    name = "fp32"
    inputs = getattr(case_config, "inputs", None) or []
    if inputs:
        item = inputs[0]
        name = (
            item.get("dtype")
            if isinstance(item, dict)
            else getattr(item, "dtype", name)
        )
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        name, torch.float32
    )


def _formula_dtype(task_result, query):
    """fp32 when ATK's golden task promotes inputs or sets is_benchmark_task."""
    if getattr(task_result, "is_benchmark_task", False) or query.dtype == torch.float32:
        return torch.float32
    return query.dtype


def _ori_topk_input(case_config, name, tensor):
    values = _as_int_list(_inp_range(_input_by_name(case_config, name)))
    # ATK samples range_values. A full list with more than two entries
    # declares exact indices or lengths for each row.
    if values is not None and len(values) > 2 and len(values) == tensor.numel():
        return torch.tensor(values, dtype=torch.int32).reshape_as(tensor)
    return tensor


def build_case(input_data, case_config):
    """Rebuild cu_seqlens, residuals and cmp indices. Seeded by case id.

    CPU golden and the NPU adapter call this independently and must see the
    same topology. Do not cache it on disk.
    """
    case_id = _case_id(case_config)

    kw = input_data.kwargs
    query = _on_cpu(kw["query"])
    ori_kv = _on_cpu(kw["ori_kv"])
    cmp_kv = _on_cpu(kw["cmp_kv"])
    dy = _on_cpu(kw["dOut"])
    sinks = _on_cpu(kw["sinks"])
    ori_indices = _on_cpu(kw["oriSparseIndices"])
    cmp_indices = _on_cpu(kw["cmpSparseIndices"])
    cu_q = _on_cpu(kw["cuSeqlensQ"])
    layout = kw["layoutQ"]
    layout_kv = kw["layoutKv"]
    cmp_ratio = int(kw["cmpRatio"])
    if not 1 <= cmp_ratio <= 128:
        raise ValueError(
            f"case {case_id}: cmpRatio must be in [1, 128], got {cmp_ratio}"
        )
    ori_mask_mode = int(kw["oriMaskMode"])
    cmp_mask_mode = int(kw["cmpMaskMode"])
    ori_win_left = int(kw["oriWinLeft"])
    ori_win_right = int(kw["oriWinRight"])
    scale = float(kw["scaleValue"])

    has_cmp = not is_empty_tensor(cmp_kv)
    has_idx = has_cmp and not is_empty_tensor(cmp_indices)
    has_ori = ori_mask_mode == 0 and not is_empty_tensor(ori_indices) and not has_cmp
    if not has_cmp:
        cmp_kv = None
        cmp_indices = None

    if layout == "TND":
        if cu_q is None or cu_q.ndim != 1 or cu_q.numel() < 2:
            raise ValueError(
                f"case {case_id}: TND requires a one-dimensional cuSeqlensQ with B+1 entries"
            )
        batch = cu_q.numel() - 1
        t1, n1, dim = [int(v) for v in query.shape]
        t2, n2, _ = [int(v) for v in ori_kv.shape]
        t3 = int(cmp_kv.shape[0]) if has_cmp else 0
        s1 = s2 = s3 = None
    else:
        batch = int(query.shape[0])
        s1, n1, dim = [int(v) for v in query.shape[1:]]
        s2, n2 = int(ori_kv.shape[1]), int(ori_kv.shape[2])
        s3 = int(cmp_kv.shape[1]) if has_cmp else 0
        t1, t2, t3 = s1 * batch, s2 * batch, s3 * batch

    if has_ori:
        ori_indices = _ori_topk_input(case_config, "oriSparseIndices", ori_indices)
        ori_indices = ori_indices.reshape(t1, n2, -1).contiguous()
        ori_topk = _ori_topk_input(
            case_config, "oriTopkLength", _on_cpu(kw["oriTopkLength"])
        )
        ori_topk = ori_topk.reshape(t1, n2).contiguous()
        selected = int(ori_indices.shape[-1])
    else:
        ori_indices = None
        ori_topk = None
        selected = int(cmp_indices.shape[-1]) if cmp_indices is not None else 0

    eff_left, eff_right = resolve_ori_windows(
        ori_mask_mode, ori_win_left, ori_win_right
    )
    group = n1 // n2
    eod = (
        parse_eod_spec(case_config, batch, t1, t2, t3, cmp_ratio)
        if layout == "TND"
        else None
    )
    used_q = used_kv = used_cmp = None
    residuals = None

    if layout == "TND" and eod is not None:
        q_list = eod["slot_q"]
        ori_list = eod["slot_kv"]
        cmp_list = eod["slot_cmp"] if has_cmp else None
        residuals = eod["residuals"] if has_cmp else None
        used_q = eod["seqused_q"]
        used_kv = eod["seqused_ori_kv"]
        used_cmp = eod["seqused_cmp_kv"] if has_cmp else None
    elif layout == "TND":
        q_list = _config_slots(case_config, "cuSeqlensQ", t1, batch)
        ori_list = _config_slots(case_config, "cuSeqlensOriKv", t2, batch)
        cmp_list = (
            _config_slots(case_config, "cuSeqlensCmpKv", t3, batch) if has_cmp else None
        )
    else:
        q_list = [s1] * batch
        ori_list = [s2] * batch
        cmp_list = [s3] * batch if has_cmp else None

    if has_cmp and eod is None:
        residuals = []
        for batch_idx, (ori_len, cmp_len) in enumerate(zip(ori_list, cmp_list)):
            expected_cmp, residual = divmod(ori_len, cmp_ratio)
            if cmp_len != expected_cmp:
                raise ValueError(
                    f"case {case_id}: batch {batch_idx} compressed KV length must be "
                    f"{ori_len} // {cmp_ratio} = {expected_cmp}, got {cmp_len}"
                )
            residuals.append(residual)

    cu_q = [0, *accumulate(q_list)]
    cu_ori = [0, *accumulate(ori_list)]
    cu_cmp = [0, *accumulate(cmp_list)] if has_cmp else None
    cmp_indices = (
        get_cmp_topk(
            q_list,
            used_q or q_list,
            used_kv or ori_list,
            used_cmp or cmp_list,
            n2,
            selected,
            cmp_ratio,
            case_id,
        )
        if has_idx
        else None
    )

    return {
        "query": query,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "dy": dy,
        "sinks": sinks,
        "ori_indices": ori_indices,
        "ori_topk": ori_topk,
        "cmp_indices": cmp_indices,
        "cu_q": cu_q,
        "cu_ori": cu_ori,
        "cu_cmp": cu_cmp,
        "residuals": residuals,
        "used_q": used_q,
        "used_kv": used_kv,
        "used_cmp": used_cmp,
        "layout": layout,
        "layout_kv": layout_kv,
        "scale": scale,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": ori_mask_mode,
        "cmp_mask_mode": cmp_mask_mode,
        "ori_win_left": ori_win_left,
        "ori_win_right": ori_win_right,
        "eff_left": eff_left,
        "eff_right": eff_right,
        "has_cmp": has_cmp,
        "has_idx": has_idx,
        "has_ori": has_ori,
        "B": batch,
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "N1": n1,
        "N2": n2,
        "D": dim,
        "G": group,
        "selected": selected,
    }


def _model(case, compute_dtype, out, lse):
    return SparseFlashMla(
        case["T1"],
        case["T2"],
        case["T3"],
        case["N1"],
        case["N2"],
        case["D"],
        case["B"],
        case["G"],
        case["query"],
        case["ori_kv"],
        case["cmp_kv"],
        out,
        case["dy"],
        case["cmp_indices"],
        compute_dtype,
        case["cu_q"],
        case["cu_ori"],
        case["cu_cmp"],
        case["scale"],
        case["selected"],
        case["cmp_ratio"],
        case["eff_right"],
        case["eff_left"],
        case["sinks"],
        lse,
        case["residuals"],
        compute_dtype == torch.float32,
        case["ori_mask_mode"],
        case["cmp_mask_mode"],
        used_q_len=case["used_q"],
        used_kv_len=case["used_kv"],
        used_cmp_kv_len=case["used_cmp"],
        ori_sparse_indices=case["ori_indices"],
        ori_topk_length=case["ori_topk"],
    )


def run_forward(case, compute_dtype):
    out_raw, lse_raw = _model(case, compute_dtype, case["query"], None).forward()
    if case["layout"] == "TND":
        out = out_raw.reshape(case["T1"], case["N1"], case["D"])
        lse = (
            lse_raw.reshape(case["T1"], case["N2"], case["G"])
            .permute(1, 0, 2)
            .contiguous()
        )
    else:
        out = out_raw.reshape(case["B"], case["S1"], case["N1"], case["D"])
        lse = (
            lse_raw.reshape(case["B"], case["S1"], case["N2"], case["G"])
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    return out.to(compute_dtype).contiguous(), lse.float().contiguous(), lse_raw


def run_backward(case, out, lse_raw, compute_dtype):
    dq, dori, dcmp, dsinks, cmp_l1 = _model(
        case, compute_dtype, out, lse_raw
    ).backward()
    if case["layout"] == "BSND":
        dq = dq.reshape(case["B"], case["S1"], case["N1"], case["D"])
        dori = dori.reshape(case["B"], case["S2"], case["N2"], case["D"])
        if dcmp is not None:
            dcmp = dcmp.reshape(case["B"], case["S3"], case["N2"], case["D"])
        if cmp_l1 is not None:
            cmp_l1 = cmp_l1.reshape(case["B"], case["S1"], case["N2"], case["selected"])
    else:
        dq = dq.reshape(case["T1"], case["N1"], case["D"])
    # None keeps optional outputs out of the compare list. The aclnn adapter
    # puts those same optionals on null pointers.
    return dq, dori, dcmp, dsinks, cmp_l1


def _tensor_i32(values):
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.to(torch.int32).contiguous()
    return torch.tensor(values, dtype=torch.int32)


def _to_npu(value, dtype=None):
    if value is None:
        return None
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor.contiguous().npu()


def _bind_kernel_inputs(input_data, case, out, lse, compute_dtype):
    """Same optional-null rules as the previous adapter, without a file cache."""
    kw = input_data.kwargs
    layout = case["layout"]
    has_cmp = case["has_cmp"]
    has_idx = case["has_idx"]
    kw["query"] = _to_npu(case["query"], compute_dtype)
    kw["dOut"] = _to_npu(case["dy"], compute_dtype)
    kw["ori_kv"] = _to_npu(case["ori_kv"], compute_dtype)
    kw["out"] = _to_npu(out, compute_dtype)
    kw["lse"] = _to_npu(lse, torch.float32)
    kw["sinks"] = _to_npu(case["sinks"], torch.float32)
    kw["cmp_kv"] = _to_npu(case["cmp_kv"], compute_dtype) if has_cmp else None
    kw["cuSeqlensQ"] = _to_npu(_tensor_i32(case["cu_q"])) if layout == "TND" else None
    kw["cuSeqlensOriKv"] = (
        _to_npu(_tensor_i32(case["cu_ori"])) if layout == "TND" else None
    )
    kw["cuSeqlensCmpKv"] = (
        _to_npu(_tensor_i32(case["cu_cmp"])) if layout == "TND" and has_cmp else None
    )
    kw["sequsedQ"] = (
        _to_npu(_tensor_i32(case["used_q"]))
        if layout == "TND" and case["used_q"] is not None
        else None
    )
    kw["sequsedOriKv"] = (
        _to_npu(_tensor_i32(case["used_kv"]))
        if layout == "TND" and case["used_kv"] is not None
        else None
    )
    kw["sequsedCmpKv"] = (
        _to_npu(_tensor_i32(case["used_cmp"]))
        if layout == "TND" and has_cmp and case["used_cmp"] is not None
        else None
    )
    kw["cmpResidualKv"] = _to_npu(_tensor_i32(case["residuals"])) if has_cmp else None
    if case["has_ori"]:
        ori_idx = case["ori_indices"]
        ori_len = case["ori_topk"]
        if layout == "BSND":
            ori_idx = ori_idx.reshape(case["B"], case["S1"], case["N2"], -1)
            ori_len = ori_len.reshape(case["B"], case["S1"], case["N2"])
        kw["oriSparseIndices"] = _to_npu(ori_idx)
        kw["oriTopkLength"] = _to_npu(ori_len)
    else:
        kw["oriSparseIndices"] = None
        kw["oriTopkLength"] = None
    cmp_idx = case["cmp_indices"]
    if has_idx and cmp_idx is not None and layout == "BSND" and cmp_idx.dim() == 3:
        cmp_idx = cmp_idx.reshape(case["B"], case["S1"], case["N2"], -1)
    kw["cmpSparseIndices"] = _to_npu(cmp_idx) if has_idx else None
    # A2/A3 do not take these two. 950 ignores them for this case set.
    kw["cmpTopkLength"] = None
    kw["metadata"] = None


# GetWorkspaceSize order: 19 tensors + 8 attributes, then 6 outputs.
FORMAL_INPUTS = (
    "query",
    "dOut",
    "out",
    "lse",
    "ori_kv",
    "cmp_kv",
    "oriSparseIndices",
    "cmpSparseIndices",
    "cuSeqlensQ",
    "cuSeqlensOriKv",
    "cuSeqlensCmpKv",
    "sequsedQ",
    "sequsedOriKv",
    "sequsedCmpKv",
    "cmpResidualKv",
    "oriTopkLength",
    "cmpTopkLength",
    "sinks",
    "metadata",
    "scaleValue",
    "cmpRatio",
    "oriMaskMode",
    "cmpMaskMode",
    "oriWinLeft",
    "oriWinRight",
    "layoutQ",
    "layoutKv",
)
OUTPUT_SLOTS = (
    "dQuery",
    "dOriKv",
    "dCmpKv",
    "dSinks",
    "oriSoftmaxL1Norm",
    "cmpSoftmaxL1Norm",
)
_PRIVATE_KEYS = ("_smlag_case", "_smlag_lse_raw", "_smlag_dtype")


def _null_tensor():
    import ctypes
    from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

    return ctypes.cast(ctypes.c_void_p(None), ctypes.POINTER(AclTensor))


def _drop_private_state(kwargs):
    """ATK binds every kwargs entry. A dict here becomes arg 28 and has no from_param."""
    for key in _PRIVATE_KEYS:
        kwargs.pop(key, None)


def _check_formal_inputs(kwargs):
    _drop_private_state(kwargs)
    names = list(kwargs.keys())
    if names != list(FORMAL_INPUTS):
        raise RuntimeError(
            "aclnnSparseFlashMlaGrad expects 27 inputs in signature order, "
            f"got {len(names)}: {names}"
        )


def _bind_workspace_args(input_args, has_cmp, has_idx):
    """Expand ATK's compact outputs into the six GetWorkspaceSize slots."""
    output_at = len(FORMAL_INPUTS)
    active_outputs = (True, True, has_cmp, True, False, has_idx)
    expected = output_at + sum(active_outputs)
    if len(input_args) != expected:
        raise RuntimeError(
            f"aclnnSparseFlashMlaGradGetWorkspaceSize expects {output_at} inputs "
            f"and {sum(active_outputs)} allocated outputs from ATK, got {len(input_args)} arguments"
        )

    null = _null_tensor()
    workspace_args = []
    for index, arg in enumerate(input_args[:output_at]):
        if isinstance(arg, dict):
            raise RuntimeError(
                f"arg[{index}] is a dict; executor state leaked into GetWorkspaceSize"
            )
        workspace_args.append(null if type(arg).__name__ == "c_void_p" else arg)

    outputs = iter(input_args[output_at:])
    workspace_args.extend(
        next(outputs) if active else null for active in active_outputs
    )
    expected = output_at + len(OUTPUT_SLOTS)
    if len(workspace_args) != expected:
        raise RuntimeError(
            f"aclnnSparseFlashMlaGradGetWorkspaceSize expects {expected} bound arguments, "
            f"got {len(workspace_args)}"
        )
    return workspace_args


@register("execute_aclnnSparseFlashMlaGrad")
class SparseFlashMlaGradApi(BaseApi):
    """CPU formula used as both the fp32 golden and the fp16/bf16 benchmark."""

    def init_by_input_data(self, input_data: InputDataset):
        case = build_case(input_data, self.task_result.case_config)
        compute_dtype = _formula_dtype(self.task_result, case["query"])
        out, lse, lse_raw = run_forward(case, compute_dtype)
        # out/lse are formal inputs. The case itself stays on this instance so
        # ATK does not append it after the 27 ACLNN parameters.
        input_data.kwargs["out"] = out
        input_data.kwargs["lse"] = lse
        _drop_private_state(input_data.kwargs)
        self._case = case
        self._lse_raw = lse_raw
        self._dtype = compute_dtype

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _drop_private_state(input_data.kwargs)
        case = getattr(self, "_case", None)
        if case is None:
            case = build_case(input_data, self.task_result.case_config)
            compute_dtype = _formula_dtype(self.task_result, case["query"])
            out, _, lse_raw = run_forward(case, compute_dtype)
        else:
            compute_dtype = self._dtype
            out = input_data.kwargs["out"]
            lse_raw = self._lse_raw
        return run_backward(case, out, lse_raw, compute_dtype)


@register("pyaclnn_execute_aclnnSparseFlashMlaGrad")
class AclnnSparseFlashMlaGradApi(AclnnBaseApi):
    """aclnnSparseFlashMlaGrad adapter. This is the only path that runs the kernel."""

    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        _drop_private_state(input_data.kwargs)
        case = build_case(input_data, self.task_result.case_config)
        compute_dtype = _declared_dtype(self.task_result.case_config)
        # Compute LSE before rounding logits to the case dtype. The binder casts out only.
        out, lse, _ = run_forward(case, torch.float32)
        _bind_kernel_inputs(input_data, case, out, lse, compute_dtype)
        _check_formal_inputs(input_data.kwargs)
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args = _bind_workspace_args(input_args, case["has_cmp"], case["has_idx"])
        return input_args, output_packages

    def after_call(self, output_packages):
        return super().after_call(output_packages)
