#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""SparseLightningIndexer(candidate consumer) ATK 执行插件(A2: ascend910b / ascend910_93)。

内容(参考 sli_exec.py / pool_key_indexer gdd_exec 组织方式, 合入单文件):
  1. 统一标杆实现(纯 torch, 按输入设备自动跑 CPU/NPU):
       sli_reference_full() —— consumer 语义(无 leak 两档掩码: 非候选块一律 -inf,
                                候选内有效 token 数 < topk 剩余槽位 -1; rv=1 增 values)。
     语义对齐 pytest 标杆(ops-transformer tests/pytest)与仓 README:
       candidate_block_indices 值为 batch 内压缩后 K 位置空间的相对块号
       (块号 × candidate_block_size 为起始位置)或 -1(无效槽), 槽位无序, 允许
       重复/越界/-1(按"非候选"处理);
       mask_mode=3 时行 i 可见压缩位
         valid(i) = floor((actK*cmpRatio + residual - actQ + i + 1) / cmpRatio)。
  2. 标杆节点 API: @register("sli")(BaseApi, CPU 真值 / NPU 标杆)。
  3. pyaclnn 适配器(与仓 op_host/sparse_lightning_indexer_def.cpp 声明顺序逐位
     一致; A2 不消费入参 seqused_q/output_idx_offset/metadata 恒空指针):
       @register("aclnn_sparse_lightning_indexer") —— 12 输入(含 REQUIRED
       candidate_block_indices + 预留 candidate_block_length) + 8 属性 + 2 输出。
  4. 自定义 L2 双标杆精度比较器: @ACCURACY_REGISTRY.register("sli_double_benchmark")
     —— standard 名字含 "benchmark" 触发 ATK 真值任务(CPU 金标 fp32 升精度, 输出
     落盘 cpu_benchmark/); 三方比对: 算子(pyaclnn) vs 真值(cpu) / 标杆(npu) vs 真值:
       - sparseIndices(整型): 集合比对(逐行有效索引排序集合相等, 100% 行通过),
         无 leak 语义: 非候选块索引一律不入选, 候选内有效 token 数 < topk 时剩余
         槽位填 -1(全无效行全 -1); 并列翻转两级仲裁(rv=1 values 分数仲裁 /
         rv=0 结构仲裁); 有分数预言机(score_oracle)时升级为 tie 感知 top-k
         合法性校验;
       - sparseValues(浮点, rv 双分支): rv=0 恒空 (0,) 跳过; rv=1 与 indices 同形
         (fp32), 按 (行, idx) 配对 isclose 同族判据(values_rtol/values_atol, 参考
         仓 tests/pytest/result_compare_method.py), idx=-1 槽 value 精确 == 0,
         无哨兵污染(|value| < 1e29);
       - 拦截铁律(一票否决, 不参与任何翻转容忍): sparseIndices 行内有效索引必须
         连续前置, -1 只允许出现在行尾补齐段(不允许 -1 之后再出现有效索引)。
     长行 oracle 落盘门控: max(k_counts) > ORACLE_MAX_K 时跳过 oracle 落盘
     (磁盘成本), 退回集合比对 + 结构仲裁口径。

输入规整约定(与 pool_key_indexer prepare 一致):
  - TND cu_seqlens: ATK 随机数据仅作 shape 载体, 按 tnd_unequal 标志在 prepare 侧
    确定性重建前缀和(算子/标杆/真值三方一致);
  - PA block_table: 随机数据仅作 shape 载体, prepare 侧重建逐 batch 连续物理块映射;
  - BSND seqused_k: 使用 ATK 随机值([s2//2, s2] 合法覆盖);
  - TND 不做 seqused_k 覆盖(规避 A2 kernel TND+seqused_k 覆盖的已知语义分歧)。
"""

# ---------------------------------------------------------------------------
# ST 用例说明（aclnnSparseLightningIndexer 泛化精度 ST 用例（取 ATK 泛化资产前 100 条））
#
# 本文件为配套执行器，与同目录 ``atk_aclnnSparseLightningIndexer.json``（76 条用例）联合使用：
#
#   atk task -n <node.yaml> -c <atk_aclnnSparseLightningIndexer.json> -p <executor_aclnnSparseLightningIndexer.py> -s 0 -e 76 -mt 1
#
# 用例截取自转测泛化精度资产前 100 条（shape 从小到大递增）中
# 按实测耗时剔除慢例后的 76 条（910B 实测总时长 275s），覆盖各 layout 与 dtype 组合；节点配置需含 pyaclnn accuracy 节点与 npu accuracy 标杆节点。
# --------------------------------------------------------------------------

import ctypes
import math
import os

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

# ---------------------------------------------------------------------------
# 统一标杆实现(纯 torch, 无 numpy 依赖, 输入在哪设备计算就在哪设备)
# ---------------------------------------------------------------------------

NEG_INF = float("-inf")
VALUES_ABS_FLOOR = 1.0e-4  # rv=1 values 近零对绝对地板(fp16 抵消行噪声量级)
ORACLE_MAX_K = 32768  # (已废弃的一刀切行宽门控, 保留常量供历史口径参考)
ORACLE_MAX_POS_BYTES = (
    256 * 1024 * 1024
)  # oracle 分数矩阵落盘字节预算(位置级 tie 仲裁覆盖):
# 实测全量 1400 case 单 case pos ≤ ~75MB, 256MB 预算
# 全覆盖(含 64K~1M 长 KV 档), /tmp 旁车目录总量 ~5GB 可容纳


def _split_counts(total, b, unequal):
    """把 total 拆成 b 份非负整数(和为 total); unequal 时首份偏大。"""
    if b <= 1:
        return [total]
    if not unequal:
        base = total // b
        counts = [base] * b
        counts[0] += total - base * b
        return counts
    first = min(total, max(1, int(total * 0.6)))
    rest = total - first
    per = rest // (b - 1)
    counts = [first] + [per] * (b - 1)
    counts[-1] += rest - per * (b - 1)
    return counts


def _prefix(counts):
    """数量列表 -> 前缀和(首项为首批数量, 与 op 契约的 (B+1,) 前缀和对齐)。"""
    out = [0]
    for c in counts:
        out.append(out[-1] + int(c))
    return out[1:]


def _as_int_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in x]


def _stable_sort_desc(scores):
    """行级降序排序; CPU 稳定(同分按下标升序, 金标约定), NPU 非稳定(集合比对吸收)。"""
    stable = scores.device.type == "cpu"
    try:
        return torch.sort(scores, dim=-1, descending=True, stable=stable)
    except (RuntimeError, TypeError):
        return torch.sort(scores, dim=-1, descending=True)


def _split_query(q, layout_q, cu_counts):
    """query -> 逐 batch (Sb, N1, D) fp32 列表 + counts + S1_dim(BSN 补齐维)。"""
    if layout_q == "BSND":
        b, s1, _, _ = q.shape
        return (
            [q[b_idx].to(torch.float32) for b_idx in range(b)],
            [s1] * b,
            s1,
        )
    if layout_q == "TND":
        q_list, start = [], 0
        for c in cu_counts:
            q_list.append(q[start : start + c].to(torch.float32))
            start += c
        return q_list, list(cu_counts), q.shape[0]
    raise ValueError(f"unsupported layout_q: {layout_q}")


def _split_key(k, layout_k, act_k_counts, block_table, block_size):
    """key -> 逐 batch (Pb, D) fp32 列表(act_k_counts = 每 batch 压缩 K 有效长度)。"""
    if layout_k == "BSND":
        return [k[b, :c, 0, :].to(torch.float32) for b, c in enumerate(act_k_counts)]
    if layout_k == "TND":
        k_list, start = [], 0
        for c in act_k_counts:
            k_list.append(k[start : start + c, 0, :].to(torch.float32))
            start += c
        return k_list
    if layout_k == "PA_BBND":
        block_num, bs_phys, _, d = k.shape
        bt = block_table.detach().cpu().reshape(block_table.shape[0], -1)
        k_list = []
        for b_idx, need in enumerate(act_k_counts):
            rows, pos = [], 0
            for t in range(bt.shape[1]):
                if pos >= need:
                    break
                blk = int(bt[b_idx, t])
                take = min(bs_phys, need - pos)
                rows.append(k[blk, :take, 0, :].to(torch.float32))
                pos += take
            k_list.append(torch.cat(rows, dim=0) if rows else k.new_zeros((0, d)))
        return k_list
    raise ValueError(f"unsupported layout_k: {layout_k}")


def _liv2_scores_batch(q_b, w_b, k_b, mask_mode, cmp_ratio, residual):
    """单 batch 打分 + 掩码 -> scores (Sb, Pb) fp32(不可达位 -inf)。

    score[i, j] = Σ_h w[i,h] * relu(q[i,h,:]·k[j,:]);
    mask3: 行 i 可见压缩位数 = floor((P*cr + residual - Sb + i + 1)/cr) 截断到 [0, P]。
    """
    s_len, _, _ = q_b.shape
    p_len = k_b.shape[0]
    dot = torch.relu(torch.einsum("snd,pd->snp", q_b, k_b))
    scores = torch.einsum("sn,snp->sp", w_b, dot)
    if mask_mode == 3:
        rows = torch.arange(s_len, device=scores.device, dtype=torch.float32)
        valid = torch.div(
            p_len * cmp_ratio + residual - s_len + rows + 1.0,
            cmp_ratio,
            rounding_mode="floor",
        ).clamp(min=0)
        valid = torch.minimum(valid, torch.full_like(valid, float(p_len)))
        valid = valid.to(torch.int64)
        cols = torch.arange(p_len, device=scores.device)[None, :]
        scores = scores.masked_fill(cols >= valid[:, None], NEG_INF)
    return scores


def _topk_rows(scores, topk):
    """位置级 TopK -> (indices (Sb, topk) int32, values (Sb, topk) fp32)。

    排序窗 min(topk, P); -inf 位(不可达)输出 -1; 尾槽 -1 / -inf。
    """
    _, p_len = scores.shape
    take = min(topk, p_len)
    vals, idx = _stable_sort_desc(scores)
    vals, idx = vals[:, :take], idx[:, :take]
    idx_out = torch.where(vals > NEG_INF, idx, torch.full_like(idx, -1)).to(torch.int32)
    pad_n = topk - take
    if pad_n > 0:
        idx_out = torch.nn.functional.pad(idx_out, (0, pad_n), value=-1)
        vals = torch.nn.functional.pad(vals, (0, pad_n), value=NEG_INF)
    return idx_out, vals


def _assemble_bnsd(rows, batch, s1_dim, width, fill, dtype):
    """逐 batch 行块 -> BSND (B, S1_dim, 1, width)(padding 行填 fill)。"""
    out = torch.full((batch, s1_dim, 1, width), fill, dtype=dtype)
    for b_idx, r in enumerate(rows):
        if r is not None and r.shape[0] > 0:
            out[b_idx, : r.shape[0], 0, :] = r
    return out


def _concat_rows(rows, n2, width, dtype, fill=None):
    """TND 行块拼接 -> (T, 1, width); 全空时 (0, 1, width)。"""
    valid = [r for r in rows if r is not None and r.shape[0] > 0]
    if valid:
        return torch.cat(valid, dim=0).unsqueeze(1)
    return torch.zeros((0, n2, width), dtype=dtype)


def _w_batch_slice(w, layout_q, q_counts, b_idx):
    """按 layout 取第 b_idx batch 的权重切片 (act_q, N1) fp32。

    BSND: w (B, S1, N1) -> w[b]; TND: w (T, N1) -> 按 token 前缀切片。
    """
    if layout_q == "BSND":
        return w[b_idx].to(torch.float32)
    start = sum(q_counts[:b_idx])
    return w[start : start + q_counts[b_idx]].to(torch.float32)


def sli_reference_full(
    q,
    k,
    w,
    candidate_block_indices,
    *,
    layout_q,
    layout_k,
    topk,
    mask_mode,
    cmp_ratio,
    candidate_block_size=8,
    return_value=0,
    q_counts,
    k_counts,
    block_table=None,
    block_size=0,
    residuals=None,
    oracle_out=None,
):
    """SparseLightningIndexer 统一标杆完整入口(candidate consumer, 无 leak 两档掩码)。

    candidateBlockIndices: BSND (B,S1,N2,K) / TND (T,N2,K) int32(脏数据免疫:
    越界/-1 按非候选处理, 重复无影响)。
    score'(b,i,p) = score      若 p < vl 且 floor(p/bs) ∈ candSet
                  = -inf       其它(可达但不在候选内的位置 与 p >= vl 不可达位
                                合并为同一无效档; 候选外位置一律不入选)
    候选内有效 token 数 < topk 时剩余槽位填 -1(全无效行全 -1)。
    返回 (indices, values):
      values: return_value=1 时与 indices 同形 fp32(有效槽=入选位置真实分,
              idx=-1 槽 0); return_value=0 时恒 (0,)。
    """
    q_list, q_counts, s1_dim = _split_query(q, layout_q, q_counts)
    k_list = _split_key(k, layout_k, k_counts, block_table, block_size)
    batch = len(q_list)

    # 候选张量按 batch/layout 切行: (act_q, K)
    if layout_q == "BSND":
        cand_rows_in = [candidate_block_indices[b, :, 0, :] for b in range(batch)]
    else:
        cand_rows_in, start = [], 0
        for c in q_counts:
            cand_rows_in.append(candidate_block_indices[start : start + c, 0, :])
            start += c

    idx_rows = []
    val_rows = []
    oracle_rows = [] if oracle_out is not None else None
    oracle_maxk = max(k_counts) if (oracle_rows is not None and k_counts) else 0
    for b_idx in range(batch):
        act_q, act_k = q_counts[b_idx], k_counts[b_idx]
        residual = residuals[b_idx] if residuals is not None else 0
        if oracle_rows is not None and act_q > 0:
            oracle_rows.append(
                torch.full((act_q, oracle_maxk), NEG_INF, dtype=torch.float32)
            )
        if act_q == 0 or act_k == 0:
            idx_rows.append(torch.full((act_q, topk), -1, dtype=torch.int32))
            # 全无效行: idx 全 -1, values 全 0(rv=1 时 -1 槽恒 0)
            val_rows.append(torch.zeros((act_q, topk), dtype=torch.float32))
            continue
        scores = _liv2_scores_batch(
            q_list[b_idx][:act_q],
            _w_batch_slice(w, layout_q, q_counts, b_idx)[:act_q],
            k_list[b_idx],
            mask_mode,
            cmp_ratio,
            residual,
        )
        cand_in = cand_rows_in[b_idx].cpu().to(torch.int64)  # (act_q, K)
        nb = max(1, math.ceil(act_k / candidate_block_size))
        cand_valid = (cand_in >= 0) & (cand_in < nb)
        cand_mask = torch.zeros((act_q, nb), dtype=torch.bool)
        if bool(cand_valid.any()):
            rows_idx = torch.arange(act_q).unsqueeze(1).expand_as(cand_in)[cand_valid]
            cols_idx = cand_in[cand_valid]
            cand_mask[rows_idx, cols_idx] = True
        pos_blk = torch.arange(act_k) // candidate_block_size
        in_cand = cand_mask[:, pos_blk].to(scores.device)  # (act_q, act_k)

        finite = scores > NEG_INF
        # 无 leak 两档掩码: 非候选块位置(含可达但未入选 与 不可达位)统一 -inf,
        # 候选外索引一律不进入 topk 输出
        scores2 = torch.where(
            finite & in_cand,
            scores,
            torch.full_like(scores, NEG_INF),
        )
        if oracle_rows is not None and oracle_rows[-1].shape[0] == act_q:
            oracle_rows[-1][:, :act_k] = scores2.detach().to(torch.float32).cpu()
        idx_b, val_b = _topk_rows(scores2, topk)
        # rv=1: 有效槽取真实分(掩码位 score' == score), idx=-1 槽 0
        val_b = torch.where(idx_b >= 0, val_b, torch.zeros_like(val_b)).to(
            torch.float32
        )
        idx_rows.append(idx_b.cpu())
        val_rows.append(val_b.cpu())

    if layout_q == "BSND":
        indices = _assemble_bnsd(idx_rows, batch, s1_dim, topk, -1, torch.int32)
        values = (
            _assemble_bnsd(val_rows, batch, s1_dim, topk, 0.0, torch.float32)
            if return_value
            else torch.zeros(0, dtype=torch.float32)
        )
    else:
        indices = _concat_rows(idx_rows, 1, topk, torch.int32)
        values = (
            _concat_rows(val_rows, 1, topk, torch.float32)
            if return_value
            else torch.zeros(0, dtype=torch.float32)
        )
    if oracle_out is not None and oracle_rows:
        oracle_out["pos"] = torch.cat(oracle_rows, dim=0)
        oracle_out["row_k"] = [
            k_counts[b] for b in range(len(q_counts)) for _ in range(q_counts[b])
        ]
        oracle_out["cbs"] = int(candidate_block_size)
    return indices, values


# ---------------------------------------------------------------------------
# 输入解析与规整(三方一致消费)
# ---------------------------------------------------------------------------


def _tensor_or_none(t):
    if t is None:
        return None
    if torch.is_tensor(t) and t.numel() == 0:
        return None
    return t


def _parse_bool(raw, default=False):
    if raw is None:
        return default
    if isinstance(raw, (bool, int)):
        return bool(raw)
    return str(raw).lower() in ("true", "1")


def _is_abnormal_case(task_result):
    """case 配置了 expected_error_msg => 异常值用例（ATK 走错误信息匹配判定）。

    异常用例下：算子侧必须收到"原始非法输入"（跳过一切伴随参数补全/重建），
    由算子 host 校验拒绝；标杆侧不参与精度比对（错误信息匹配一票定案），
    仅需返回占位输出避免测试侧崩溃。
    """
    try:
        case_cfg = getattr(task_result, "case_config", None)
        return bool(getattr(case_cfg, "expected_error_msg", None))
    except Exception:
        return False


def _make_noncontiguous_view(tensor, axis):
    """别名保真的轴 stride 翻倍视图：逻辑内容不变，仅目标轴 stride 非标准。

    构造：双倍该轴长度的 buffer 上做步长 2 切片 —— 与"分页/缝隙排布"的非连续
    输入同构（如 PA 场景按块缝隙排布的 k）。仅在算子侧应用；标杆/真值侧继续
    用原连续 tensor（逻辑内容一致），由此比对出算子对 stride 的处理是否正确。
    """
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return tensor
    if not isinstance(axis, int) or axis < 0 or axis >= tensor.dim():
        return tensor
    if tensor.shape[axis] == 0:
        return tensor
    t = tensor.contiguous()
    big_shape = list(t.shape)
    big_shape[axis] *= 2
    big = torch.empty(big_shape, dtype=t.dtype, device=t.device)
    view = big[
        tuple(
            slice(None, None, 2) if i == axis else slice(None) for i in range(t.dim())
        )
    ]
    view.copy_(t)
    return view


def _parse_noncontig(spec):
    """解析 noncontig 属性("tensor:axis,...") -> {tensor_name: axis}。"""
    axes = {}
    if not spec:
        return axes
    for item in str(spec).split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        name, _, axis = item.rpartition(":")
        try:
            axes[name.strip()] = int(axis)
        except ValueError:
            continue
    return axes


def _normalize_optional(prep, keys):
    for key in keys:
        prep[key] = _tensor_or_none(prep.get(key))
    return prep


def _tensor_or_none_placeholder(t):
    """异常用例口径的可选输入规整。

    仅把 [1,0] 占位（设计文档未提供该输入，转换器固定产物）规整为 None(NULL 下发)；
    其余空 tensor（如末维 0 的 [1,8,1,0]）保真下发，由算子按 shape 值校验拒绝，
    与独立 ctypes 口径(空 tensor 传实参、缺失传 NULL)一致。
    """
    if t is None:
        return None
    if torch.is_tensor(t) and list(t.shape) == [1, 0]:
        return None
    return t


def prepare_sli_inputs(input_data: InputDataset, abnormal: bool = False):
    """解析 SLI 用例输入(与 sparse_lightning_indexer*.yaml inputs 顺序一致)。

    abnormal=True 时按异常用例口径规整可选输入（见 _tensor_or_none_placeholder）。
    """
    args = list(input_data.args) if input_data.args else []
    kw = dict(input_data.kwargs) if input_data.kwargs else {}

    def pick(idx, name):
        if idx < len(args) and args[idx] is not None:
            return args[idx]
        return kw.get(name)

    def pick_attr(idx, name, default):
        v = pick(idx, name)
        return default if v is None else v

    prep = dict(
        q=pick(0, "q"),
        k=pick(1, "k"),
        w=pick(2, "w"),
        candidate=pick(3, "candidate_block_indices"),
        cu_seqlens_q=pick(4, "cu_seqlens_q"),
        cu_seqlens_k=pick(5, "cu_seqlens_k"),
        seqused_k=pick(6, "seqused_k"),
        cmp_residual_k=pick(7, "cmp_residual_k"),
        block_table=pick(8, "block_table"),
        topk=int(pick_attr(9, "topk", 64)),
        max_seqlen_q=int(pick_attr(10, "max_seqlen_q", -1)),
        layout_q=str(pick_attr(11, "layout_q", "BSND")),
        layout_k=str(pick_attr(12, "layout_k", "BSND")),
        mask_mode=int(pick_attr(13, "mask_mode", 0)),
        cmp_ratio=int(pick_attr(14, "cmp_ratio", 1)),
        return_value=int(pick_attr(15, "return_value", 0)),
        candidate_block_size=int(pick_attr(16, "candidate_block_size", 8)),
        tnd_unequal=_parse_bool(pick_attr(17, "tnd_unequal", False)),
        # 预留接口入参(yaml 不含, 仅异常用例提供; 按名从 kwargs 兜底读取)
        candidate_block_length=pick(18, "candidate_block_length"),
        # 非连续输入场景标志(仅非连续套件提供): "tensor:axis,..." 仅算子侧消费
        noncontig=str(pick_attr(19, "noncontig", "") or ""),
    )
    # candidate / candidate_block_length: 正常口径空 tensor 占位规整为 None(NULL 下发);
    # 异常口径仅 [1,0] 占位转 None, 其余空 tensor 保真(缺失->NULL / 空 shape->实参)
    opt_keys = (
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_k",
        "cmp_residual_k",
        "block_table",
        "candidate",
        "candidate_block_length",
    )
    if abnormal:
        for key in opt_keys:
            prep[key] = _tensor_or_none_placeholder(prep.get(key))
    else:
        _normalize_optional(prep, opt_keys)
    return prep


def normalize_liv3_shapes(prep):
    """TND 前缀和 / PA block_table / act_k 序列确定性重建(算子/标杆/真值三方一致)。"""
    q, k = prep["q"], prep["k"]
    layout_q, layout_k = prep["layout_q"], prep["layout_k"]
    b = int(q.shape[0]) if layout_q == "BSND" else None

    # q 侧 batch 行数
    if layout_q == "TND":
        q_counts = _split_counts(
            int(q.shape[0]),
            max(1, int(prep["cu_seqlens_q"].shape[0]) - 1),
            prep["tnd_unequal"],
        )
    else:
        q_counts = [int(q.shape[1])] * b

    # k 侧 act_k(压缩 K 有效长度)
    if layout_k == "BSND":
        s2 = int(k.shape[1])
        if prep["seqused_k"] is not None:
            k_counts = [
                max(0, min(int(v), s2)) for v in _as_int_list(prep["seqused_k"])
            ]
        else:
            k_counts = [s2] * b
    elif layout_k == "TND":
        # TND 不做 seqused_k 覆盖(A2 已知语义分歧), act_k = cu 差分
        t2 = int(k.shape[0])
        kb = max(1, int(prep["cu_seqlens_k"].shape[0]) - 1)
        k_counts = _split_counts(t2, kb, prep["tnd_unequal"])
        prep["cu_seqlens_k"] = torch.tensor(
            [0] + list(_prefix(k_counts)), dtype=torch.int32
        )
    else:  # PA_BBND
        block_num, bs_phys = int(k.shape[0]), int(k.shape[1])
        if prep["seqused_k"] is not None:
            pb = int(prep["seqused_k"].shape[0])
        else:
            pb = int(q.shape[0])
            prep["seqused_k"] = torch.zeros(pb, dtype=torch.int32)
        # PA 满覆盖: 每 batch act_k = blocks_per_batch * bs_phys
        blocks_per_batch = block_num // max(1, pb)
        k_counts = [blocks_per_batch * bs_phys] * pb
        prep["seqused_k"] = torch.tensor(k_counts, dtype=torch.int32)
        # block_table 确定性重建: 逐 batch 连续物理块
        bt = torch.zeros((pb, blocks_per_batch), dtype=torch.int32)
        for b_idx in range(pb):
            bt[b_idx] = torch.arange(
                b_idx * blocks_per_batch,
                (b_idx + 1) * blocks_per_batch,
                dtype=torch.int32,
            )
        prep["block_table"] = bt
        prep["pa_block_size"] = bs_phys

    if layout_q == "TND":
        prep["cu_seqlens_q"] = torch.tensor(
            [0] + list(_prefix(q_counts)), dtype=torch.int32
        )
    prep["q_counts"] = q_counts
    prep["k_counts"] = k_counts
    residuals = (
        [int(v) for v in _as_int_list(prep["cmp_residual_k"])]
        if prep["cmp_residual_k"] is not None
        else None
    )
    prep["residuals"] = residuals
    return prep


# ---------------------------------------------------------------------------
# 输出规格推导(is_get_output_task)
# ---------------------------------------------------------------------------


def sli_output_stubs(prep):
    """SLI 输出 stub(2 输出: indices / rv=1 全形 values, rv=0 恒空)。"""
    q = prep["q"]
    layout_q = prep["layout_q"]
    topk = int(prep["topk"])
    # 推导输出规格时把异常 topk(0/越界)收敛到合法窗: 全空输出会触发框架"空输出
    # 短路成功"(不下发 tiling), 非法值就绕过了算子 host 校验 —— 输出仅是规格载体,
    # 真正的 topk 以属性原样下发, 由 tiling 校验拒绝
    topk = topk if 1 <= topk <= 8192 else 8
    if layout_q == "BSND":
        indices = torch.empty(
            (int(q.shape[0]), int(q.shape[1]), 1, topk), dtype=torch.int32
        )
    else:
        indices = torch.empty((int(q.shape[0]), 1, topk), dtype=torch.int32)
    values = (
        torch.empty(indices.shape, dtype=torch.float32)
        if prep.get("return_value", 0)
        else torch.zeros(0, dtype=torch.float32)
    )
    return indices, values


def _decode_uniform_fast_reference(
    q,
    k,
    w,
    candidate_block_indices,
    *,
    layout_q,
    layout_k,
    topk,
    mask_mode,
    cmp_ratio,
    return_value,
    candidate_topk_blocks,
    candidate_block_size,
    q_counts,
    k_counts,
    block_table=None,
    block_size=0,
    residuals=None,
    oracle_out=None,
    is_sli=False,
):
    """decode 均匀档快速标杆(大 Batch 避免逐 batch Python 循环)。

    启用条件(API 侧守门): 所有 batch act_q/act_k 一致、act_q<=4、batch 数>=16。
    与逐 batch 慢路径(liv3_reference_full/sli_reference_full)同语义:
      - 打分/掩码公式逐位一致(mask3 的 valid 在均匀档下为 (B, act_q) 网格广播);
      - CPU 稳定排序同分按下标升序, 双路径逐位一致; NPU 非稳定差异由集合比对吸收;
      - PA 块表为执行器确定性重建的逐 batch 连续映射, 等价于按 batch 重排。
    LIV3 返回 (indices, values, candidate, cand_len); SLI 返回 (indices, values)。
    """
    B = len(q_counts)
    act_q, act_k = q_counts[0], k_counts[0]
    cand_on = candidate_topk_blocks != -1

    # ---- 组装 batch 化输入 (B, act_q, n1, D) / (B, act_k, D) / (B, act_q, n1) ----
    n1 = q.shape[-2]
    if layout_q == "BSND":
        q4 = q.to(torch.float32)
        w3 = w.to(torch.float32)
    else:  # TND: (T, n1, D) -> (B, act_q, n1, D)
        q4 = q.to(torch.float32).reshape(B, act_q, n1, q.shape[-1])
        w3 = w.to(torch.float32).reshape(B, act_q, n1)
    if layout_k == "BSND":
        k3 = k.to(torch.float32).reshape(B, act_k, k.shape[-1])
    elif layout_k == "TND":
        k3 = k.to(torch.float32).reshape(B, act_k, k.shape[-1])
    else:  # PA_BBND: 确定性连续块表 -> 按 batch 重排
        bpb = act_k // block_size
        k3 = (
            k.to(torch.float32)
            .reshape(-1, block_size, k.shape[-1])[: B * bpb]
            .reshape(B, bpb, block_size, k.shape[-1])
            .reshape(B, act_k, k.shape[-1])
        )

    # ---- 打分 + 掩码(与 _liv2_scores_batch 同公式; bmm/逐元素实现,
    #      避免 einsum 在大 B 小分数场景的病态慢路径) ----
    qd = q4.reshape(B, act_q * n1, -1)
    dot = torch.relu(torch.bmm(qd, k3.transpose(1, 2)))  # (B, act_q*n1, act_k)
    dot = dot.reshape(B, act_q, n1, act_k)
    scores = (w3.unsqueeze(-1) * dot).sum(dim=2).reshape(B * act_q, act_k)
    if mask_mode == 3:
        dev = scores.device
        res = torch.tensor(
            residuals if residuals is not None else [0] * B,
            dtype=torch.float32,
            device=dev,
        ).reshape(B, 1)
        rows = torch.arange(act_q, dtype=torch.float32, device=dev)
        valid = torch.div(
            act_k * cmp_ratio + res - act_q + rows[None, :] + 1.0,
            cmp_ratio,
            rounding_mode="floor",
        ).clamp(min=0)
        valid = torch.minimum(valid, torch.full_like(valid, float(act_k))).to(
            torch.int64
        )
        scores = scores.masked_fill(
            torch.arange(act_k, device=dev)[None, :] >= valid.reshape(-1, 1), NEG_INF
        )

    oracle_pos = scores.detach().to(torch.float32).cpu()

    # ---- SLI: 无 leak 候选掩码(与慢路径逐位同公式, 行独立可批化) ----
    if is_sli:
        cand_in = candidate_block_indices.cpu().reshape(B * act_q, -1).to(torch.int64)
        nb = max(1, math.ceil(act_k / candidate_block_size))
        cand_valid = (cand_in >= 0) & (cand_in < nb)
        cand_mask = torch.zeros((B * act_q, nb), dtype=torch.bool)
        if bool(cand_valid.any()):
            rows_idx = (
                torch.arange(B * act_q).unsqueeze(1).expand_as(cand_in)[cand_valid]
            )
            cand_mask[rows_idx, cand_in[cand_valid]] = True
        pos_blk = torch.arange(act_k) // candidate_block_size
        in_cand = cand_mask[:, pos_blk].to(scores.device)
        finite = scores > NEG_INF
        scores = torch.where(finite & in_cand, scores, torch.full_like(scores, NEG_INF))
        oracle_pos = scores.detach().to(torch.float32).cpu()

    # ---- 位置级 TopK + 块级候选(行独立, 直接复用行级实现) ----
    idx_flat, val_flat = _topk_rows(scores, topk)
    if is_sli:
        val_flat = torch.where(idx_flat >= 0, val_flat, torch.zeros_like(val_flat)).to(
            torch.float32
        )
    cand_flat = (
        _candidate_rows(scores, candidate_topk_blocks, candidate_block_size)
        if cand_on
        else None
    )

    # ---- 输出组装(与慢路径同布局) ----
    if layout_q == "BSND":
        indices = idx_flat.reshape(B, act_q, 1, topk)
        values = (
            val_flat.reshape(B, act_q, 1, topk)
            if return_value
            else torch.zeros(0, dtype=torch.float32)
        )
        candidate = (
            cand_flat.reshape(B, act_q, 1, candidate_topk_blocks)
            if cand_on
            else torch.zeros(0, dtype=torch.int32)
        )
    else:
        indices = idx_flat.reshape(B * act_q, 1, topk)
        values = (
            val_flat.reshape(B * act_q, 1, topk)
            if return_value
            else torch.zeros(0, dtype=torch.float32)
        )
        candidate = (
            cand_flat.reshape(B * act_q, 1, candidate_topk_blocks)
            if cand_on
            else torch.zeros(0, dtype=torch.int32)
        )
    if oracle_out is not None:
        oracle_out["pos"] = oracle_pos
        oracle_out["row_k"] = [act_k] * (B * act_q)
        oracle_out["cbs"] = int(candidate_block_size)
    return (
        (
            indices.cpu(),
            values.cpu(),
            candidate.cpu(),
            torch.zeros(0, dtype=torch.int32),
        )
        if not is_sli
        else (indices.cpu(), values.cpu())
    )


# ---------------------------------------------------------------------------
# 标杆节点 API(CPU 真值 / NPU 组合标杆, 按输入设备路由)
# ---------------------------------------------------------------------------


def _save_score_oracle(task_result, oracle):
    """把分数预言机写入本节点 case 输出目录(score_oracle.pt), 供比较器做 tie 合法性校验。

    仅 benchmark/真值节点调用(算子节点不跑标杆 API); 失败静默(不阻塞测试)。
    """
    try:
        import os

        from atk.common.utils import get_output_path_by_case
        from atk.configs.nodes_config import BENCHMARK_PATH_PATTERN

        try:
            bd = getattr(task_result, "benchmark_device", None)
            bd_name = getattr(bd, "value", "cpu") if bd is not None else "cpu"
            out_dir = get_output_path_by_case(
                task_result.get_output_path(),
                BENCHMARK_PATH_PATTERN.format(bd_name),
                task_result.case_config,
            )
            os.makedirs(out_dir, exist_ok=True)
            torch.save(oracle, os.path.join(out_dir, "score_oracle.pt"))
        except Exception:
            # 主节点路径不可得(benchmark 节点 TaskResult 无 output_path)时退回旁车目录
            torch.save(oracle, _oracle_side_path(task_result))
    except Exception as e:
        import logging

        logging.getLogger("atk").warning(
            f"score_oracle save failed: {e!r} | type={type(task_result).__name__} "
            f"bm_output_path={getattr(task_result, 'bm_output_path', 'NA')!r} "
            f"output_dir={getattr(task_result, '_output_dir', 'NA')!r}"
        )


def _oracle_side_path(task_result):
    """oracle 旁车路径(临时目录, 按套件名+用例 id 键控; 数据由 case json 种子决定, 跨次一致)。"""
    import os
    import tempfile

    cc = getattr(task_result, "case_config", None)
    name = getattr(cc, "name", "case") if cc is not None else "case"
    cid = getattr(cc, "id", 0) if cc is not None else 0
    d = os.path.join(tempfile.gettempdir(), "sli_atk_score_oracle")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}_{cid}.pt")


@register("sli")
class SLIApi(BaseApi):
    """SparseLightningIndexer 统一标杆执行器(candidate consumer)。"""

    def __init__(self, task_result):
        super().__init__(task_result)
        self.run_modes = []

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # performance_device 任务以 with_output=False 循环调用本标杆: 必须照常
        # 执行参考计算(profiler 才能采集到 acl->npu 事件), 仅输出落盘/返回值
        # 按需关闭; 原实现此处直接 return None 导致标杆侧 Device 性能恒 0。
        prep = prepare_sli_inputs(
            input_data, abnormal=_is_abnormal_case(self.task_result)
        )
        if getattr(self.task_result, "is_get_output_task", False):
            return sli_output_stubs(prep)
        if _is_abnormal_case(self.task_result):
            # 异常用例：非法输入会让标杆重建/参考计算在测试侧崩溃, 而精度比对
            # 对异常用例不生效（ATK 以错误信息匹配一票定案）, 返回占位输出即可
            return sli_output_stubs(prep)
        prep = normalize_liv3_shapes(prep)
        prep = normalize_liv3_shapes(prep)
        oracle = {}
        # decode 均匀档快速路径(大 Batch 避免逐 batch 循环): act_q/act_k 全 batch
        # 一致且 act_q<=4、batch 数>=16 时启用, 与慢路径同语义(CPU 下逐位一致)
        if (
            len(set(prep["q_counts"])) == 1
            and len(set(prep["k_counts"])) == 1
            and prep["q_counts"][0] <= 4
            and len(prep["q_counts"]) >= 16
        ):
            outs = _decode_uniform_fast_reference(
                prep["q"],
                prep["k"],
                prep["w"],
                prep.get("candidate"),
                layout_q=prep["layout_q"],
                layout_k=prep["layout_k"],
                topk=prep["topk"],
                mask_mode=prep["mask_mode"],
                cmp_ratio=prep["cmp_ratio"],
                return_value=prep["return_value"],
                candidate_topk_blocks=-1,  # SLI 无 candidate 输出, 固定关闭
                candidate_block_size=prep["candidate_block_size"],
                q_counts=prep["q_counts"],
                k_counts=prep["k_counts"],
                block_table=prep["block_table"],
                block_size=prep.get("pa_block_size", 0),
                residuals=prep["residuals"],
                oracle_out=oracle,
                is_sli=True,
            )
        else:
            outs = outs = sli_reference_full(
                prep["q"],
                prep["k"],
                prep["w"],
                prep["candidate"],
                layout_q=prep["layout_q"],
                layout_k=prep["layout_k"],
                topk=prep["topk"],
                mask_mode=prep["mask_mode"],
                cmp_ratio=prep["cmp_ratio"],
                candidate_block_size=prep["candidate_block_size"],
                return_value=prep["return_value"],
                q_counts=prep["q_counts"],
                k_counts=prep["k_counts"],
                block_table=prep["block_table"],
                block_size=prep.get("pa_block_size", 0),
                residuals=prep["residuals"],
                oracle_out=oracle,
            )
        # 铁律·无泄漏校验上下文恒落盘: 逐行候选块布尔掩码(R, nb_max) + 行宽 row_k + cbs。
        # 体积恒 ≤ 同行数分数矩阵 pos 的 1/4(nb = ceil(act_k/cbs) ≤ act_k), 大 Batch
        # 小 KV 场景为 KB 级; pos(逐行分数矩阵, R×act_k)超字节预算时仅剔除 pos 本身
        # ——tie 仲裁退回集合比对口径, 但铁律(非候选块索引不允许输出)不依赖 pos。
        # 掩码构建与 sli_reference_full 的 cand_valid 同口径(脏值免疫: 仅 [0, nb_r)
        # 内的槽位计入候选)。
        if prep.get("candidate") is not None:
            _cand = prep["candidate"].cpu().to(torch.int64)
            if _cand.dim() >= 2:
                _cand = _cand.reshape(-1, _cand.shape[-1])
            _row_k = [
                prep["k_counts"][b]
                for b in range(len(prep["q_counts"]))
                for _ in range(prep["q_counts"][b])
            ]
            _cbs = int(prep["candidate_block_size"])
            if _cand.shape[0] == len(_row_k) and _cbs > 0 and _row_k:
                _nb = [(rk + _cbs - 1) // _cbs for rk in _row_k]
                _nb_max = max(max(_nb), 1)
                _mask = torch.zeros((_cand.shape[0], _nb_max), dtype=torch.bool)
                _valid = (_cand >= 0) & (_cand < torch.tensor(_nb).unsqueeze(1))
                if bool(_valid.any()):
                    _ri = (
                        torch.arange(_cand.shape[0])
                        .unsqueeze(1)
                        .expand_as(_cand)[_valid]
                    )
                    _mask[_ri, _cand[_valid]] = True
                oracle["cand_mask"] = _mask
                oracle["row_k"] = _row_k
                oracle["cbs"] = _cbs
        # C-1 整改(记录整改建议①): pos 落盘门控由"行宽≤32768 一刀切"改为字节预算制
        # ——原门控把 64K~1M 长 KV case 的位置级 tie 仲裁全部拒之门外, 恰是并列池
        # 误报的重灾区(rv=0 无 values 仲裁通道, 集合比对 + 微小翻转预算必然误报)。
        if "pos" in oracle:
            _pos_bytes = sum(prep["q_counts"] or [0]) * max(prep["k_counts"] or [0]) * 4
            if _pos_bytes > ORACLE_MAX_POS_BYTES:
                oracle.pop("pos")
        if oracle and with_output:
            _save_score_oracle(self.task_result, oracle)
        return outs if with_output else None


# ---------------------------------------------------------------------------
# pyaclnn 适配器(与仓 op_host/sparse_lightning_indexer_def.cpp 声明顺序一致)
# ---------------------------------------------------------------------------


def _to_npu_tensor(backend, tensor, device_id, nc_axis=None):
    if tensor is None:
        return None
    dev = f"npu:{device_id}"
    if tensor.device.type == "npu" and tensor.device.index == device_id:
        t = tensor
    else:
        t = tensor.to(dev)
    # 非连续视图必须在设备搬移之后构造: .to(dev) 会把非连续 tensor 物化为连续副本
    if nc_axis is not None:
        t = _make_noncontiguous_view(t, nc_axis)
    result = backend.convert_input_data(t, name="tensor")
    if isinstance(result, list):
        result = result[0]
    return result


def _out_dtype(raw, default):
    mapping = {
        "torch.int32": torch.int32,
        "int32": torch.int32,
        "torch.float32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(str(raw), default)


def _build_output_packages(backend, task_result, device_id, dtype_plan):
    dev = f"npu:{device_id}"
    saved, packages = [], []
    for index, out in enumerate(task_result.output_info_list):
        shape = (
            list(out.shape) if hasattr(out, "shape") else list(out.get("shape", [1]))
        )
        default = dtype_plan[index] if index < len(dtype_plan) else torch.float32
        raw_dtype = out.dtype if hasattr(out, "dtype") else out.get("dtype", "")
        torch_dtype = _out_dtype(raw_dtype, default)
        pytensor = torch.empty(shape, dtype=torch_dtype, device=dev)
        acl_struct = backend.torch_tensor_to_acl(pytensor)
        saved.append(acl_struct)
        packages.append(acl_struct)
    return saved, packages


class _AclnnDeviceGuardMixin:
    def _get_device_id(self):
        device_id = 0
        if hasattr(self.backend, "device_id") and self.backend.device_id is not None:
            device_id = self.backend.device_id
        elif getattr(self.task_result, "device_id", None) is not None:
            device_id = self.task_result.device_id
        return int(device_id)

    def __call__(self):
        device_id = self._get_device_id()
        torch.npu.set_device(device_id)
        super().__call__()
        torch.npu.synchronize()

    def after_call(self, output_packages):
        outputs = []
        for output_pack in output_packages:
            if hasattr(output_pack, "pytensor"):
                outputs.append(output_pack.pytensor)
        return outputs


@register("aclnn_sparse_lightning_indexer")
class SLIAclnnApi(_AclnnDeviceGuardMixin, AclnnBaseApi):
    """aclnnSparseLightningIndexer(candidate consumer) 两段式 pyaclnn 适配器。

    签名与仓 op_host/sparse_lightning_indexer_def.cpp 逐位一致:
    输入 0-10 与 LIV2 相同, 11 = candidate_block_indices(REQUIRED),
    12 = candidate_block_length(预留, 仅空 tensor); 属性 topk/max_seqlen_q/
    layout_q/layout_k/mask_mode/cmp_ratio/return_value/candidate_block_size;
    输出 sparse_indices/sparse_values。
    """

    def __init__(self, task_result, backend):
        super().__init__(task_result, backend)
        self.run_modes = []

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnSparseLightningIndexerGetWorkspaceSize("
            "const aclTensor* q, const aclTensor* k, const aclTensor* w, "
            "const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, "
            "const aclTensor* sequsedQOptional, const aclTensor* sequsedKOptional, "
            "const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional, "
            "const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional, "
            "const aclTensor* candidateBlockIndices, "
            "const aclTensor* candidateBlockLengthOptional, "
            "int64_t topk, int64_t maxSeqlenQ, char* layoutQOptional, char* layoutKOptional, "
            "int64_t maskMode, int64_t cmpRatio, int64_t returnValue, "
            "int64_t candidateBlockSize, "
            "const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut, "
            "uint64_t* workspaceSize, aclOpExecutor** executor)"
        )

    def init_by_input_data(self, input_data: InputDataset):
        device_id = self._get_device_id()
        torch.npu.set_device(device_id)
        torch.npu.synchronize()
        prep = prepare_sli_inputs(
            input_data, abnormal=_is_abnormal_case(self.task_result)
        )
        # TND 前缀和 / PA block_table / seqused_k 确定性重建(算子与标杆同源)。
        # 异常用例跳过重建：cu_seqlens/block_table/seqused_k 等非法输入必须
        # 原样下发，由算子 host 校验拒绝（否则重建会"修复"掉目标异常）
        if not _is_abnormal_case(self.task_result):
            prep = normalize_liv3_shapes(prep)

        # 非连续输入场景: 按 noncontig 规约在算子侧构造别名保真的非连续视图
        nc_axes = _parse_noncontig(prep.get("noncontig"))
        input_args = []
        input_args.append(
            _to_npu_tensor(self.backend, prep["q"], device_id, nc_axis=nc_axes.get("q"))
        )
        input_args.append(
            _to_npu_tensor(self.backend, prep["k"], device_id, nc_axis=nc_axes.get("k"))
        )
        input_args.append(
            _to_npu_tensor(self.backend, prep["w"], device_id, nc_axis=nc_axes.get("w"))
        )
        # 顺序与签名一致: cuSeqlensQ/K -> sequsedQ/K -> cmpResidualK -> blockTable
        # -> outputIdxOffset -> metadata -> candidateBlockIndices -> candidateBlockLength
        for key in ("cu_seqlens_q", "cu_seqlens_k"):
            if prep[key] is not None:
                input_args.append(_to_npu_tensor(self.backend, prep[key], device_id))
            else:
                input_args.append(TensorPtr())
        input_args.append(TensorPtr())  # sequsedQOptional(A2 拒绝)
        for key in ("seqused_k", "cmp_residual_k", "block_table"):
            if prep[key] is not None:
                input_args.append(_to_npu_tensor(self.backend, prep[key], device_id))
            else:
                input_args.append(TensorPtr())
        input_args.append(TensorPtr())  # outputIdxOffsetOptional(A2 拒绝)
        input_args.append(TensorPtr())  # metadataOptional(A2 拒绝)
        # candidateBlockIndices(REQUIRED): 异常用例"缺失"以 NULL 下发触发算子校验;
        # 非连续场景支持 candidate_block_indices 轴 stride 翻倍视图
        if prep["candidate"] is not None:
            input_args.append(
                _to_npu_tensor(
                    self.backend,
                    prep["candidate"],
                    device_id,
                    nc_axis=nc_axes.get("candidate_block_indices"),
                )
            )
        else:
            input_args.append(TensorPtr())
        # candidateBlockLengthOptional(预留, 仅空; 异常用例提供非空时原样下发)
        if prep.get("candidate_block_length") is not None:
            input_args.append(
                _to_npu_tensor(self.backend, prep["candidate_block_length"], device_id)
            )
        else:
            input_args.append(TensorPtr())

        input_args.append(ctypes.c_int64(prep["topk"]))
        input_args.append(ctypes.c_int64(prep["max_seqlen_q"]))
        input_args.append(ctypes.c_char_p(prep["layout_q"].encode("utf-8")))
        input_args.append(ctypes.c_char_p(prep["layout_k"].encode("utf-8")))
        input_args.append(ctypes.c_int64(prep["mask_mode"]))
        input_args.append(ctypes.c_int64(prep["cmp_ratio"]))
        input_args.append(ctypes.c_int64(prep["return_value"]))
        input_args.append(ctypes.c_int64(prep["candidate_block_size"]))

        self._saved_output_structs, output_packages = _build_output_packages(
            self.backend, self.task_result, device_id, [torch.int32, torch.float32]
        )
        input_args.extend(output_packages)
        return input_args, output_packages


# ---------------------------------------------------------------------------
# 双标杆精度比较器
# ---------------------------------------------------------------------------

from atk.common.utils import get_output_path_by_case, torch_load_safe
from atk.configs.nodes_config import BENCHMARK_PATH_PATTERN
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare


def _case_input_value(case_config, name, default=None):
    for inp in case_config.inputs:
        if getattr(inp, "name", None) == name:
            rv = getattr(inp, "range_values", None)
            if rv is None:
                return default
            if isinstance(rv, (list, tuple)):
                return rv[0] if rv else default
            return rv
    return default


def _to_rows(tensor):
    """(B, S, N2, L)/(T, N2, L) -> (rows, L) 行视图(CPU): BSND 折叠 B*S, TND 即 T。"""
    t = tensor.cpu()
    if t.dim() == 4:
        t = t.reshape(-1, t.shape[-1])
    elif t.dim() == 3:
        t = t[:, 0, :]
    return t


def _check_valid_slots_contiguous(op_indices):
    """拦截铁律: 输出的有效索引必须连续前置, -1 只允许出现在行尾补齐段。

    逐行校验: 一旦某槽位为 -1, 其后所有槽位必须全为 -1
    (即不允许 "-1 之后再出现有效索引" 的中间空洞/乱序布局)。
    返回 (ok, info); 空输出视为通过。
    """
    op = _to_rows(op_indices)
    if op.numel() == 0:
        return True, "empty output, rule trivially pass"
    neg = op < 0
    # neg_before[:, i] = 行内 [0, i-1] 是否出现过 -1
    neg_before = torch.zeros_like(neg)
    if op.shape[1] > 1:
        neg_before[:, 1:] = torch.cummax(neg[:, :-1].to(torch.int8), dim=-1)[0].bool()
    bad_mask = neg_before & (op >= 0)
    bad_rows = bad_mask.any(dim=-1).nonzero(as_tuple=True)[0]
    if bad_rows.numel() == 0:
        return True, f"valid-slots-contiguous rule pass ({op.shape[0]} rows)"
    first_bad = []
    for r in bad_rows[:3].tolist():
        col = int(bad_mask[r].nonzero(as_tuple=True)[0][0])
        first_bad.append(
            f"row({r}): 槽 {col} 为有效索引 {int(op[r, col])} 但其前已出现 -1"
        )
    return False, (
        f"valid-slots-contiguous rule FAIL: {bad_rows.numel()}/{op.shape[0]} rows "
        f"存在 -1 之后的有效索引; first bad: " + "; ".join(first_bad)
    )


def _check_no_leak_indices(op_indices, oracle):
    """拦截铁律: 非候选块的索引不允许输出(一票否决, 独立于/先于任何并列翻转容忍)。

    逐行校验算子输出有效槽(idx >= 0): idx 必须 < 该行 act_k 且 idx // cbs 落在该行
    候选块掩码内(掩码由标杆节点按脏值免疫口径构建: 仅 [0, nb) 内的槽位计入候选,
    与 sli_reference_full 的 cand_valid 同口径)。发现任何非候选块/越域索引即精度失败。
    返回 (ok, info); oracle 缺失泄漏校验上下文时判不可校验(fail-closed, 铁律必须可保证)。
    """
    op = _to_rows(op_indices).to(torch.int64)
    if op.numel() == 0:
        return True, "empty output, rule trivially pass"
    if (
        not isinstance(oracle, dict)
        or "cand_mask" not in oracle
        or "row_k" not in oracle
        or "cbs" not in oracle
    ):
        if not bool((op >= 0).any()):
            return True, "no valid slots, rule trivially pass"
        return (
            False,
            "leak-check context missing (cand_mask/row_k/cbs); cannot verify no-leak rule",
        )
    cand_mask = oracle["cand_mask"]
    cand_mask = (
        cand_mask.cpu() if torch.is_tensor(cand_mask) else torch.tensor(cand_mask)
    )
    if cand_mask.dim() == 1:
        cand_mask = cand_mask.unsqueeze(0)
    row_k = oracle["row_k"]
    row_k = (
        row_k.cpu().to(torch.int64)
        if torch.is_tensor(row_k)
        else torch.tensor(list(row_k), dtype=torch.int64)
    )
    cbs = int(oracle["cbs"])
    rows = op.shape[0]
    if cand_mask.shape[0] != rows or row_k.numel() != rows or cbs <= 0:
        if not bool((op >= 0).any()):
            return True, "no valid slots, rule trivially pass"
        return False, (
            f"leak-check context mismatch: rows op={rows} cand_mask={cand_mask.shape[0]} "
            f"row_k={row_k.numel()} cbs={cbs}"
        )
    nb_max = cand_mask.shape[1]
    vmask = op >= 0
    oob = vmask & (op >= row_k.unsqueeze(1))
    blk = (op.clamp(min=0) // cbs).clamp(max=nb_max - 1)
    not_cand = vmask & ~oob & ~cand_mask.gather(1, blk)
    bad = oob | not_cand
    n_bad = int(bad.sum())
    if n_bad == 0:
        n_valid = int(vmask.sum())
        return True, (
            f"no-leak rule pass ({n_valid} valid slots all in candidate blocks, "
            f"{rows} rows, cbs={cbs})"
        )
    nz = bad.nonzero()[:8]
    ex = "; ".join(
        f"row({int(r)}) idx={int(op[r, c])} blk={int(op[r, c]) // cbs}"
        for r, c in nz.tolist()
    )
    return False, f"{n_bad} non-candidate/out-of-range indices leaked: {ex}"


def _load_score_oracle(truth_dir, case_config=None):
    """加载标杆节点落盘的分数预言机; 优先 truth_dir, 退回旁车目录; 缺失返回 None。"""
    candidates = [os.path.join(truth_dir, "score_oracle.pt")]
    if case_config is not None:
        import tempfile

        name = getattr(case_config, "name", "case")
        cid = getattr(case_config, "id", 0)
        candidates.append(
            os.path.join(
                tempfile.gettempdir(), "sli_atk_score_oracle", f"{name}_{cid}.pt"
            )
        )
    for path in candidates:
        if os.path.exists(path):
            try:
                oracle = torch_load_safe(path)
            except Exception:
                continue
            # pos(tie 仲裁)与 cand_mask(无泄漏铁律)任一存在即返回; 长行 case 仅含 cand_mask/row_k/cbs
            if isinstance(oracle, dict) and ("pos" in oracle or "cand_mask" in oracle):
                return oracle
    return None


def _oracle_topk_row_valid(op_set, truth_set, scores_row, tol_abs, tol_rel):
    """单行 top-K 合法性校验(同分自由 + 选中域硬校验; 向量化实现, 判据与原逐元素
    版本逐位等价):
    1) 选中域: op 选中索引的 oracle 分数必须有限(> -inf)且在行宽内 —— 选中
       非候选/不可达/越界位置(oracle 分数 -inf, oracle 已含无 leak 掩码)直接判
       不合法(无 leak 硬门禁);
    2) 同分自由: 合法 = op 选中集的最小有限分数 >= 全行中被 op 排除的最大有限
       分数 - tol(即 op 的集合是按分数取前 k 的某个同分打破方案; 任何严格逆序
       都判不合法), -inf(pad/不可达/非候选)不参与。
    返回 (ok, worst/invalid 描述)。"""
    if not op_set:
        return True, None
    row = (
        scores_row
        if torch.is_tensor(scores_row)
        else torch.tensor(list(scores_row), dtype=torch.float32)
    )
    row = row.to(torch.float32)
    width = int(row.numel())
    bad_sel = [i for i in op_set if i >= width or not (float(row[i]) > NEG_INF + 1.0)]
    if bad_sel:
        return False, (
            "selected invalid (non-candidate/unreachable/out-of-range) "
            f"positions: {bad_sel[:8]}"
        )
    sel = [i for i in op_set if 0 <= i < width]
    if not sel:
        return True, None
    sel_t = torch.tensor(sel, dtype=torch.long)
    v = row[sel_t]
    m_local = int(v.argmin())
    m_val, m_idx = float(v[m_local]), sel[m_local]
    tol = max(tol_abs, tol_rel * abs(m_val))
    sel_mask = torch.zeros(width, dtype=torch.bool)
    sel_mask[sel_t] = True
    excl = (row > NEG_INF + 1.0) & ~sel_mask
    if not bool(excl.any()):
        return True, None
    excl_idx = excl.nonzero().flatten()
    excl_scores = row[excl_idx]
    j = int(excl_scores.argmax())
    mx = float(excl_scores[j])
    if mx <= m_val + tol:
        return True, None
    return False, (mx, int(excl_idx[j]), m_val, m_idx)


def _oracle_set_compare(
    op_indices, truth_indices, oracle, row_pass_ratio, block_level=False, topk=None
):
    """tie 感知集合比对: 逐行用分数预言机校验 op 选择集是否为合法 top-k。
    - 同分打破自由: 合法同分子集差异判 tie-match(通过);
    - 严格逆序(排除了更高分元素)判失败, 并给出最坏逆序对。
    oracle: {"pos": [rows, max_k] fp32(-inf pad, 已含无 leak 掩码), "cbs": int}
    """
    op = _to_rows(op_indices)
    truth = _to_rows(truth_indices)
    if op.shape != truth.shape:
        return (
            False,
            f"shape mismatch: operator {tuple(op.shape)} vs truth {tuple(truth.shape)}",
        )

    neg = op[op < 0]
    if neg.numel() > 0 and not bool((neg == -1).all()):
        bad = neg[neg != -1][:8].tolist()
        return False, f"invalid slots must be exactly -1, got {bad}"

    pos_o = oracle["pos"].to(torch.float32)
    rows = min(op.shape[0], pos_o.shape[0])

    total_rows = op.shape[0]
    exact = tie = invalid = 0
    first_bad = []
    for r in range(rows):
        op_row = op[r]
        tr_row = truth[r]
        op_set = set(int(v) for v in op_row.tolist() if int(v) >= 0)
        tr_set = set(int(v) for v in tr_row.tolist() if int(v) >= 0)
        if op_set == tr_set:
            exact += 1
            continue
        scores_row = pos_o[r] if r < rows else None
        if scores_row is None:
            invalid += 1
            continue
        # 行可达域内集合大小一致性与 oracle 校验(oracle 已含无 leak 掩码:
        # 非候选位 -inf, 选中候选外索引必然触发严格逆序判失败)
        ok, worst = _oracle_topk_row_valid(op_set, tr_set, scores_row, 2.5e-5, 5.0e-3)
        if ok:
            tie += 1
        else:
            invalid += 1
            if len(first_bad) < 3 and worst:
                if isinstance(worst, str):
                    first_bad.append(f"row({r}): {worst}")
                else:
                    first_bad.append(
                        f"row({r}): 严格逆序 排除分 {worst[0]:.6g}@{worst[1]} > 选中最小分 {worst[2]:.6g}@{worst[3]}"
                    )
    ratio = (exact + tie) / total_rows if total_rows else 0.0
    ok = ratio >= row_pass_ratio
    info = (
        f"oracle topk compare: exact {exact}, tie-subset {tie}, invalid {invalid}"
        f" ({ratio:.2%})"
    )
    if not ok:
        info += "; first bad: " + "; ".join(first_bad)
    return ok, info


def _indices_set_compare(op_indices, truth_indices, row_pass_ratio, tie_ctx=None):
    """indices 集合比对(算子 vs 真值): 逐行有效索引排序集合相等 + 无效位精确 -1。

    并列翻转容忍(两级仲裁, 参考 pool_key_indexer pki_double_benchmark):
      1) 分数仲裁(has_values): 差异行 truth-only 索引 A 与 op-only 索引 B 按分数
         排序配对, 全部 |sA - sB| <= tie_rtol*max(|sA|, denom_floor) 判为并列翻转;
      2) 结构仲裁(无分数): 差集等量且 <= max_flip 判为翻转, 受 flip_row_ratio 约束。
    """
    op = _to_rows(op_indices)
    truth = _to_rows(truth_indices)
    if op.shape != truth.shape:
        return (
            False,
            f"indices shape mismatch: operator {tuple(op.shape)} vs truth {tuple(truth.shape)}",
        )

    neg = op[op < 0]
    if neg.numel() > 0 and not bool((neg == -1).all()):
        bad = neg[neg != -1][:8].tolist()
        return False, f"invalid slots must be exactly -1, got {bad}"

    tie_ctx = tie_ctx or {}
    op_vals = tie_ctx.get("op_values")
    truth_vals = tie_ctx.get("truth_values")
    has_values = (
        op_vals is not None
        and truth_vals is not None
        and op_vals.numel() > 0
        and truth_vals.numel() > 0
    )
    if has_values:
        op_vals = _to_rows(op_vals).to(torch.float32)
        truth_vals = _to_rows(truth_vals).to(torch.float32)
        if op_vals.shape != op.shape or truth_vals.shape != truth.shape:
            has_values = False
    tie_rtol = float(tie_ctx.get("tie_rtol", 0.0))
    denom_floor = float(tie_ctx.get("denom_floor", 1e-3))
    max_flip = int(tie_ctx.get("max_flip", 4))
    flip_row_ratio = float(tie_ctx.get("flip_row_ratio", 0.0))

    op_sorted, _ = op.sort(dim=-1)
    tr_sorted, _ = truth.sort(dim=-1)
    eq_rows = (op_sorted == tr_sorted).all(dim=-1).tolist()

    total_rows = op.shape[0]
    pass_rows = tie_rows = struct_rows = 0
    first_diffs = []
    for r in range(total_rows):
        if eq_rows[r]:
            pass_rows += 1
            continue
        g_row = truth[r]
        n_row = op[r]
        g_set = set(int(v) for v in g_row.tolist() if int(v) >= 0)
        n_set = set(int(v) for v in n_row.tolist() if int(v) >= 0)
        only_g, only_n = sorted(g_set - n_set), sorted(n_set - g_set)
        tolerated = False
        if len(only_g) == len(only_n) and 0 < len(only_g):
            if has_values and tie_rtol > 0:
                # 分数仲裁: 索引->分数映射(输出逐位对应: 第 k 位索引配第 k 位分数)
                t_map = {
                    int(i): float(v)
                    for i, v in zip(g_row.tolist(), truth_vals[r].tolist())
                    if int(i) >= 0
                }
                o_map = {
                    int(i): float(v)
                    for i, v in zip(n_row.tolist(), op_vals[r].tolist())
                    if int(i) >= 0
                }
                sa = sorted(t_map.get(p, float("-inf")) for p in only_g)
                sb = sorted(o_map.get(p, float("-inf")) for p in only_n)
                if all(math.isfinite(a) and math.isfinite(bb) for a, bb in zip(sa, sb)):
                    tolerated = all(
                        abs(a - bb) <= tie_rtol * max(abs(a), denom_floor)
                        for a, bb in zip(sa, sb)
                    )
            elif not has_values and max_flip > 0 and len(only_g) <= max_flip:
                tolerated = True
        if tolerated:
            if has_values:
                tie_rows += 1
            else:
                struct_rows += 1
        elif len(first_diffs) < 3:
            first_diffs.append(f"row({r}): golden-only{only_g[:6]} op-only{only_n[:6]}")
    ratio = (pass_rows + tie_rows + struct_rows) / total_rows if total_rows else 0.0
    struct_ratio = struct_rows / total_rows if total_rows else 0.0
    # 单行退化: flip_row_ratio*rows < 1 行时(如 1 行的 S1 下界边界用例), 允许
    # 至多 1 行结构翻转(仍受单行 max_flip 索引数上限约束)
    struct_budget = max(1, int(flip_row_ratio * total_rows))
    ok = ratio >= row_pass_ratio and struct_rows <= struct_budget
    info = (
        f"indices set compare: {pass_rows}/{total_rows} matched"
        f" (+{tie_rows} tie-flip, +{struct_rows} struct-flip) ({ratio:.2%})"
    )
    if not ok:
        info += "; first diffs: " + "; ".join(first_diffs)
    return ok, info


def _row_index_value_pairs(idx_row, val_row):
    """从 indices/values 单行提取 (index, value) 对(第 k 位索引配第 k 位分数)。

    TopK 并列翻转只改变槽位次序, 不改变 (index -> value) 映射,
    按索引配对比位置级配对比对更稳健(与 pool_key_indexer 口径一致)。
    """
    pairs = {}
    for i, v in zip(idx_row.tolist(), val_row.tolist()):
        i = int(i)
        if i >= 0 and math.isfinite(v):
            pairs[i] = float(v)
    return pairs


def _sli_values_isclose(
    op_indices,
    op_values,
    bm_indices,
    bm_values,
    truth_indices,
    truth_values,
    rtol,
    atol,
):
    """SLI rv=1 sparseValues 双标杆 isclose 口径。

    判据(对齐仓 tests/pytest 判据源 result_compare_method.py 的 isclose 族):
      1) shape: values 与 indices 同形;
      2) 无哨兵污染: |value| < 1e29(kernel 无 leak 化后不残留 -1e30 量级哨兵);
      3) idx=-1 槽 value 精确 == 0(算子/标杆/真值一致);
      4) 有效槽: 按 (行, idx) 配对(第 k 位索引配第 k 位分数, TopK 并列翻转只改
         槽位次序不改 (idx→value) 映射), 算子/标杆分别 vs 真值在配对交集上
         isclose(rtol, atol, equal_nan=True) 全通过。
    返回 (ok, info)。
    """
    op_idx = _to_rows(op_indices)
    bm_idx = _to_rows(bm_indices)
    truth_idx = _to_rows(truth_indices)
    op_vals = _to_rows(op_values).to(torch.float32)
    bm_vals = _to_rows(bm_values).to(torch.float32)
    truth_vals = _to_rows(truth_values).to(torch.float32)
    if op_vals.shape != truth_vals.shape:
        return False, (
            f"values shape mismatch: operator {tuple(op_vals.shape)} "
            f"vs truth {tuple(truth_vals.shape)}"
        )
    if op_vals.shape != op_idx.shape:
        return False, (
            f"values shape {tuple(op_vals.shape)} != indices shape {tuple(op_idx.shape)}"
        )

    for name, vals in (("operator", op_vals), ("benchmark", bm_vals)):
        if vals.numel() > 0 and bool((vals.abs() >= 1e29).any()):
            bad = vals[vals.abs() >= 1e29][:8].tolist()
            return False, f"values({name}) sentinel-magnitude (|v|>=1e29): {bad}"

    for name, vals, idxs in (
        ("operator", op_vals, op_idx),
        ("benchmark", bm_vals, bm_idx),
        ("truth", truth_vals, truth_idx),
    ):
        invalid = idxs < 0
        if invalid.any() and not bool((vals[invalid] == 0).all()):
            bad = vals[invalid][vals[invalid] != 0][:8].tolist()
            return False, f"values({name}) must be exactly 0 at idx=-1 slots, got {bad}"

    stats = {}
    for name, idxs, vals in (
        ("actual", op_idx, op_vals),
        ("benchmark", bm_idx, bm_vals),
    ):
        n_pairs, mismatches, first_bad = 0, 0, []
        for r in range(truth_idx.shape[0]):
            truth_pairs = _row_index_value_pairs(truth_idx[r], truth_vals[r])
            own_pairs = _row_index_value_pairs(idxs[r], vals[r])
            for idx, val in own_pairs.items():
                tv = truth_pairs.get(idx)
                if tv is None:
                    continue
                n_pairs += 1
                if not (
                    abs(val - tv) <= atol + rtol * abs(tv)
                    or abs(val - tv) <= VALUES_ABS_FLOOR
                    or (math.isnan(val) and math.isnan(tv))
                ):
                    mismatches += 1
                    if len(first_bad) < 3:
                        first_bad.append(
                            f"row({r}) idx={idx} op={val:.6g} truth={tv:.6g}"
                        )
        stats[name] = (n_pairs, mismatches, first_bad)

    ok = all(m == 0 for _, m, _ in stats.values())
    parts = []
    for name, (n, m, bad) in stats.items():
        info = f"{name}: {n - m}/{n} pairs isclose(rtol={rtol}, atol={atol})"
        if bad:
            info += "; first bad: " + "; ".join(bad)
        parts.append(info)
    return ok, "; ".join(parts)


@ACCURACY_REGISTRY.register("sli_double_benchmark")
class SLIDoubleBenchmarkCompare(BaseAccuracyCompare):
    """SparseLightningIndexer 双标杆精度比较。

    rv 双分支口径:
      - rv=0: indices 集合比对(结构仲裁), values 恒空跳过;
      - rv=1: indices 集合比对(values 分数仲裁) + sparseValues 按 idx 配对
        isclose 双标杆比对(-1 槽 value 精确 0 + 无哨兵污染)。
    """

    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)
        self.row_pass_ratio = float(kwargs.get("row_pass_ratio", 1.0))
        self.max_flip_index = int(kwargs.get("max_flip_index", 4))
        self.flip_row_ratio = float(kwargs.get("flip_row_ratio", 1.0e-2))
        # rv=1 values isclose 同族判据(参考仓 tests/pytest/result_compare_method.py)
        self.values_rtol = float(kwargs.get("values_rtol", 5.0e-3))
        self.values_atol = float(kwargs.get("values_atol", 2.5e-5))

    def _load_output(self, output_dir, filename):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return None
        return torch_load_safe(path)

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        return AccuracyConfig(filename=data_file, result=True)

    def precision_compare_for_case(
        self, main_node, remote_node, is_npu=False, device_id=0
    ):
        return_value = int(_case_input_value(self.case_config, "return_value", 0))

        operator_dir = get_output_path_by_case(
            main_node.get_output_path(), main_node.get_backend_name(), self.case_config
        )
        benchmark_dir = get_output_path_by_case(
            remote_node.get_output_path(),
            remote_node.get_backend_name(),
            self.case_config,
        )
        benchmark_device = getattr(
            getattr(self.executor, "task_result", None), "benchmark_device", "cpu"
        )
        truth_dir = get_output_path_by_case(
            main_node.get_output_path(),
            BENCHMARK_PATH_PATTERN.format(benchmark_device),
            self.case_config,
        )

        outputs = []
        missing = None
        for name in ("output_0.pt", "output_1.pt"):
            tensor = self._load_output(operator_dir, name)
            if tensor is None:
                missing = name
                break
            outputs.append(tensor)
        if missing is not None:
            return [
                AccuracyConfig(
                    filename=missing,
                    result=False,
                    error_info=f"operator output missing in {operator_dir}",
                )
            ]
        op_idx, op_val = outputs

        truth_idx = self._load_output(truth_dir, "output_0.pt")
        truth_val = self._load_output(truth_dir, "output_1.pt")
        if truth_idx is None or truth_val is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=(
                        f"truth(cpu_benchmark) output missing in {truth_dir}; "
                        f"check standard name contains 'benchmark' and bm task ran"
                    ),
                )
            ]
        bm_idx = self._load_output(benchmark_dir, "output_0.pt")
        bm_val = self._load_output(benchmark_dir, "output_1.pt")
        if bm_idx is None or bm_val is None:
            return [
                AccuracyConfig(
                    filename="output_0.pt",
                    result=False,
                    error_info=f"benchmark node output missing in {benchmark_dir}",
                )
            ]

        # ---- 0. 拦截铁律: 输出有效索引必须连续前置, -1 只允许行尾补齐 ----
        # 一票否决(不参与任何翻转容忍), 与无 leak 语义叠加看护。
        idx_rule_ok, idx_rule_info = _check_valid_slots_contiguous(op_idx)

        # ---- 0b. 拦截铁律: 非候选块索引不允许输出(一票否决, 先于任何翻转容忍) ----
        # 上下文来自标杆节点恒落盘的 cand_mask/row_k/cbs(长行 case 不落 pos 亦受看护)。
        oracle = _load_score_oracle(truth_dir, self.case_config)
        leak_ok, leak_info = _check_no_leak_indices(op_idx, oracle)

        # ---- 1. indices 集合比对(rv=1 启用 values 分数仲裁, rv=0 结构仲裁) ----
        tie_ctx = dict(
            op_values=op_val if return_value else None,
            truth_values=truth_val if return_value else None,
            tie_rtol=1.0e-2,
            denom_floor=1.0e-3,
            max_flip=self.max_flip_index,
            flip_row_ratio=self.flip_row_ratio,
        )
        if oracle is not None and "pos" in oracle:
            idx_ok, idx_info = _oracle_set_compare(
                op_idx, truth_idx, oracle, self.row_pass_ratio, block_level=False
            )
        else:
            idx_ok, idx_info = _indices_set_compare(
                op_idx, truth_idx, self.row_pass_ratio, tie_ctx=tie_ctx
            )
        idx_ok = idx_ok and idx_rule_ok and leak_ok
        if not idx_rule_ok:
            idx_info = f"{idx_info}; [铁律] {idx_rule_info}"
        if not leak_ok:
            idx_info = f"{idx_info}; [铁律·无泄漏] {leak_info}"
        results = [
            AccuracyConfig(filename="output_0.pt", result=idx_ok, error_info=idx_info)
        ]

        # ---- 2. values 双分支(rv=0 恒空跳过; rv=1 按 idx 配对 isclose 双标杆) ----
        if return_value == 0 or (op_val.numel() == 0 and truth_val.numel() == 0):
            results.append(
                AccuracyConfig(
                    filename="output_1.pt",
                    result=True,
                    error_info="values empty (return_value=0), skip compare",
                )
            )
        else:
            val_ok, val_info = _sli_values_isclose(
                op_idx,
                op_val,
                bm_idx,
                bm_val,
                truth_idx,
                truth_val,
                self.values_rtol,
                self.values_atol,
            )
            results.append(
                AccuracyConfig(
                    filename="output_1.pt", result=val_ok, error_info=val_info
                )
            )
        return results
