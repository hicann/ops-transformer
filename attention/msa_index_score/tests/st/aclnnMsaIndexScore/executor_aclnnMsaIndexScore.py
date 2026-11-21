#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnnMsaIndexScore 的 ATK 执行插件。比较交给 ATK 内置的 cv_fused_double_benchmark。

三路只负责产出结果，不在本文件里算 mare / mere / rmse，也不注册自定义比较器：
  golden     ATK 自动拉起的 CPU 任务（is_benchmark_task）上的 fp32 重放，作为真值。
  benchmark  本机 CPU（node_cpu.yaml，is_benchmark_task 为假）或远端 GPU 上按声明精度重放同一条公式。
  dut        本机 NPU 通过 cann_ops_transformer.ops.msa_index_score 调用 CANN 或自定义算子包里的算子。
内置比较器计算：NPU 算子输出相对 golden 的误差 / 本机 CPU 或远端 GPU 输出相对同一份 golden 的误差。

计算公式：
  score = Maxpool[(key_scale·) Q @ K^T] + local_mask
  非量化先在 fp32 点积，再落到 fp16 后做 max。不乘 head_dim**-0.5，不做 top-k。
A2/A3 没有融合 TopK。return_choice 固定 0，只比较 score。
query_scale 只占接口，计算不使用。
"""

from __future__ import annotations

import random
from typing import Optional

import torch

NEG_INF = -3.4028234663852886e38
LOCAL_SCORE_INIT = 1.0e30
LOCAL_SCORE_LOCAL = 1.0e29
BLOCK = 128
ROW_TILE_M = 128
M_ALIGN = 16
SCORE_ALIGN = 16
INPUT_SLOTS = (
    "query",
    "key",
    "start_loc",
    "block_table",
    "query_scale",
    "key_scale",
    "atten_mask",
    "actual_seq_qlen",
    "actual_seq_klen",
)
ATTR_COUNT = 7
PA_LAYOUTS = ("BBND", "BNBD")


def _as_tensor(value) -> Optional[torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        return None
    if value.dtype == torch.bool or value.ndim == 0 or value.numel() == 0:
        return None
    return value


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b if a > 0 else 0


def _visible_key_end(sparse_mode: int, kv: int, q_len: int, t_off: int) -> int:
    if sparse_mode == 3:
        return int(max(0, min(kv, kv - q_len + t_off + 1)))
    return int(max(kv, 0))


def _tile_visible_end_blk(
    row: int,
    total_rows: int,
    num_heads: int,
    sparse_mode: int,
    kv: int,
    q_len: int,
    num_blocks: int,
) -> int:
    """尾 tile 不足 16 行时向前重叠，后写的 tile 生效。local_mask 用这个块上界。"""
    if total_rows <= 0 or num_heads <= 0:
        return 0
    end_blk = 0
    n_tiles = (total_rows + ROW_TILE_M - 1) // ROW_TILE_M
    for tile in range(n_tiles):
        m_start = tile * ROW_TILE_M
        m_actual = min(ROW_TILE_M, total_rows - m_start)
        if m_actual <= 0:
            continue
        if m_actual < M_ALIGN and m_start >= M_ALIGN:
            m_start -= M_ALIGN - m_actual
            m_actual = M_ALIGN
        if row < m_start or row >= m_start + m_actual:
            continue
        t_hi = (m_start + m_actual - 1) // num_heads
        visible = _visible_key_end(sparse_mode, kv, q_len, t_hi)
        end_blk = min(_ceil_div(max(visible, 0), BLOCK), num_blocks)
    return end_blk


def _dot_max(
    q_row: torch.Tensor, k_page: torch.Tensor, golden: bool, quant: bool
) -> torch.Tensor:
    """q_row [N1, D]，k_page [valid, D]，返回 [N1] fp32。"""
    score = torch.matmul(q_row.float(), k_page.float().transpose(0, 1))
    if not golden:
        return score.max(dim=1).values
    if quant:
        return score.max(dim=1).values.to(torch.float16).float()
    return score.to(torch.float16).max(dim=1).values.float()


def _key_page(
    key: torch.Tensor, layout: str, page: int, key_lo: int, valid: int
) -> torch.Tensor:
    if layout == "TND":
        return key[key_lo : key_lo + valid, 0, :]
    if layout == "BNBD":
        return key[page, 0, :valid, :]
    return key[page, :valid, 0, :]


def _scale_rows(
    scale: torch.Tensor, layout: str, page: int, key_lo: int, valid: int
) -> torch.Tensor:
    if layout == "TND":
        flat = scale.reshape(-1) if scale.ndim == 1 else scale[:, 0]
        return flat[key_lo : key_lo + valid]
    return scale[page, 0, :valid]


def msa_index_score_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    block_table: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    actual_seq_qlen: torch.Tensor,
    actual_seq_klen: torch.Tensor,
    start_loc: torch.Tensor,
    layout: str,
    sparse_mode: int,
    init_blocks: int,
    local_blocks: int,
    precision: str,
    device=None,
) -> torch.Tensor:
    """precision='fp32' 为真值；'low' 为对齐 kernel 舍入的标杆。点积在 query 所在设备上算。"""
    if device is not None:
        query = query.to(device)
        key = key.to(device if key.dtype != torch.int8 else "cpu")
        if key.dtype == torch.int8 and str(device) != "cpu":
            key = key.to(device)
        if scale is not None:
            scale = scale.to(device)
    compute = query.device
    key = key.to(compute)
    if scale is not None:
        scale = scale.to(compute)

    golden = precision != "fp32"
    quant = key.dtype == torch.int8
    layout = str(layout)
    num_heads = int(query.shape[1])

    q_cu = [int(v) for v in actual_seq_qlen.detach().cpu().reshape(-1)]
    k_len = [int(v) for v in actual_seq_klen.detach().cpu().reshape(-1)]
    starts = [int(v) for v in start_loc.detach().cpu().reshape(-1)]
    prefix_k = layout == "TND"
    if prefix_k:
        batch = len(k_len) - 1
    else:
        batch = len(k_len)
    table = (
        None if block_table is None else block_table.detach().cpu().reshape(batch, -1)
    )
    if layout in PA_LAYOUTS:
        max_blocks = int(table.shape[1])
    else:
        max_blocks = 1
        for b in range(batch):
            kv = (k_len[b + 1] - k_len[b]) if prefix_k else k_len[b]
            max_blocks = max(max_blocks, _ceil_div(kv, BLOCK) if kv > 0 else 1)
    total_q = q_cu[-1] if q_cu else 0
    score_stride = _round_up(max_blocks, SCORE_ALIGN)
    out = torch.full(
        (num_heads, total_q, score_stride), NEG_INF, dtype=torch.float32, device=compute
    )
    boost_init = torch.tensor(LOCAL_SCORE_INIT, dtype=torch.float32, device=compute)
    boost_local = torch.tensor(LOCAL_SCORE_LOCAL, dtype=torch.float32, device=compute)

    for b in range(batch):
        q_begin, q_end = q_cu[b], q_cu[b + 1]
        q_len = q_end - q_begin
        if prefix_k:
            cu_k = k_len[b]
            kv = k_len[b + 1] - cu_k
        else:
            cu_k = 0
            kv = k_len[b]
        num_blocks = _ceil_div(kv, BLOCK) if kv > 0 else 0
        if layout in PA_LAYOUTS:
            num_blocks = min(num_blocks, max_blocks)
        q_block = starts[b]
        local_start = max(0, q_block + 1 - int(local_blocks))
        if q_len <= 0:
            continue
        for t_off in range(q_len):
            t = q_begin + t_off
            visible = _visible_key_end(sparse_mode, kv, q_len, t_off)
            q_row = query[t]
            for blk in range(num_blocks):
                key_lo = cu_k + blk * BLOCK if prefix_k else blk * BLOCK
                valid = min(visible - blk * BLOCK, BLOCK)
                if valid <= 0:
                    continue
                page = 0 if table is None else int(table[b, blk])
                k_page = _key_page(key, layout, page, key_lo if prefix_k else 0, valid)
                if scale is not None:
                    rows = _scale_rows(
                        scale, layout, page, key_lo if prefix_k else 0, valid
                    ).float()
                    k_page = k_page.float() * rows.unsqueeze(-1)
                out[:, t, blk] = _dot_max(q_row, k_page, golden, quant)
            total_rows = q_len * num_heads
            for h in range(num_heads):
                vis_end = _tile_visible_end_blk(
                    t_off * num_heads + h,
                    total_rows,
                    num_heads,
                    sparse_mode,
                    kv,
                    q_len,
                    num_blocks,
                )
                for blk in range(num_blocks):
                    if blk >= vis_end:
                        continue
                    boost = None
                    if blk < int(init_blocks):
                        boost = boost_init
                    if local_start <= blk <= q_block:
                        boost = boost_local
                    if boost is not None:
                        out[h, t, blk] = boost
    return out.detach().to("cpu")


def _split_total(total: int, parts: int) -> list:
    if parts <= 1:
        return [total]
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def bind_case_tensors(kwargs: dict, seed: int) -> None:
    """按 case seed 重写长度、block_table、start_loc，让真值、标杆、NPU 看到同一套下标。"""
    query = _as_tensor(kwargs.get("query"))
    key = _as_tensor(kwargs.get("key"))
    if query is None or key is None:
        return
    rng = random.Random(seed)
    layout = str(kwargs.get("layout_key", "BBND"))
    total_q = int(query.shape[0])
    qlen = _as_tensor(kwargs.get("actual_seq_qlen"))
    batch = (
        max(1, int(qlen.shape[0]) - 1)
        if qlen is not None and int(qlen.shape[0]) >= 2
        else 1
    )
    q_lens = _split_total(total_q, batch)
    cu = [0]
    for length in q_lens:
        cu.append(cu[-1] + length)
    kwargs["actual_seq_qlen"] = torch.tensor(cu, dtype=torch.int32)

    if layout == "TND":
        total_k = int(key.shape[0])
        k_lens = _split_total(total_k, batch)
        k_cu = [0]
        for length in k_lens:
            k_cu.append(k_cu[-1] + length)
        kwargs["actual_seq_klen"] = torch.tensor(k_cu, dtype=torch.int32)
        kwargs["block_table"] = None
        logical = [
            max(1, _ceil_div(length, BLOCK)) if length else 1 for length in k_lens
        ]
    else:
        table = _as_tensor(kwargs.get("block_table"))
        pages = int(key.shape[0])
        width = int(table.shape[1]) if table is not None else 1
        kwargs["block_table"] = torch.tensor(
            [[(b * 3 + k) % pages for k in range(width)] for b in range(batch)],
            dtype=torch.int32,
        )
        cap = width * BLOCK
        k_lens = [rng.randint(1, cap) for _ in range(batch)]
        kwargs["actual_seq_klen"] = torch.tensor(k_lens, dtype=torch.int32)
        logical = [max(1, _ceil_div(length, BLOCK)) for length in k_lens]
    kwargs["start_loc"] = torch.tensor(
        [rng.randrange(n) for n in logical], dtype=torch.int32
    )


def parse_case_seed(name: str, case_id) -> int:
    if case_id is not None:
        try:
            return int(case_id) + 1
        except (TypeError, ValueError):
            pass
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits[-6:]) if digits else 1


def _case_bits(task_result):
    case_config = getattr(task_result, "case_config", None)
    name = str(getattr(case_config, "name", "") or "")
    case_id = getattr(case_config, "id", None)
    return name, case_id


def _is_placeholder(value) -> bool:
    if value is None:
        return True
    if not isinstance(value, torch.Tensor):
        return False
    return value.dtype == torch.bool or value.numel() == 0


def _drop_illegal_optionals(kwargs: dict) -> None:
    kwargs.pop("query_scale", None)
    key = _as_tensor(kwargs.get("key"))
    if (
        key is None
        or key.dtype != torch.int8
        or _is_placeholder(kwargs.get("key_scale"))
    ):
        kwargs.pop("key_scale", None)
    if int(kwargs.get("sparse_mode", 3)) != 3 or _is_placeholder(
        kwargs.get("atten_mask")
    ):
        kwargs.pop("atten_mask", None)
    if str(kwargs.get("layout_key", "BBND")) not in PA_LAYOUTS or _is_placeholder(
        kwargs.get("block_table")
    ):
        kwargs.pop("block_table", None)


def _declared_low_dtype(task_result) -> torch.dtype:
    case_config = getattr(task_result, "case_config", None)
    inputs = getattr(case_config, "inputs", None) or []
    text = str(getattr(inputs[0], "dtype", "") if inputs else "").lower()
    if "bf" in text:
        return torch.bfloat16
    return torch.float16


def _score_from_kwargs(kwargs: dict, precision: str, device=None) -> torch.Tensor:
    return msa_index_score_ref(
        kwargs["query"],
        kwargs["key"],
        _as_tensor(kwargs.get("block_table")),
        _as_tensor(kwargs.get("key_scale")),
        kwargs["actual_seq_qlen"],
        kwargs["actual_seq_klen"],
        kwargs["start_loc"],
        str(kwargs.get("layout_key", "BBND")),
        int(kwargs.get("sparse_mode", 3)),
        int(kwargs.get("init_blocks", 0)),
        int(kwargs.get("local_blocks", 1)),
        precision,
        device=device,
    )


def _with_low_inputs(kwargs: dict, low_dtype: torch.dtype) -> dict:
    cloned = dict(kwargs)
    query = kwargs["query"]
    if query.dtype != low_dtype:
        cloned["query"] = query.to(low_dtype)
    key = kwargs["key"]
    if key.dtype.is_floating_point and key.dtype != low_dtype:
        cloned["key"] = key.to(low_dtype)
    return cloned


def _resolve_role(device: str, task_name: str, is_benchmark_task: bool) -> str:
    """dut=本机 NPU。benchmark=本机 CPU 或远端 GPU 的同精度重放。golden=自动 CPU 任务上的 fp32 真值。"""
    text = str(device).strip().lower()
    if text.startswith("cpu"):
        return "golden" if is_benchmark_task else "benchmark"
    if text.startswith("cuda") or text == "gpu":
        return "golden" if is_benchmark_task else "benchmark"
    if text.startswith("npu") or text == "pyaclnn":
        if is_benchmark_task:
            raise RuntimeError(
                f"NPU 不承载 golden：device={device!r} name={task_name!r}"
            )
        return "dut"
    raise RuntimeError(
        "无法识别 ATK 双标杆任务："
        f"device={device!r} name={task_name!r} is_benchmark_task={is_benchmark_task}"
    )


def _npu_device(query: torch.Tensor):
    if isinstance(query, torch.Tensor) and getattr(query.device, "type", "") == "npu":
        return query.device
    return torch.device("npu")


def _on_device(value, device):
    tensor = _as_tensor(value)
    if tensor is None:
        return None
    return tensor.to(device)


def _run_npu_op(kwargs: dict) -> torch.Tensor:
    """调用 CANN 包或自定义算子包注册的 torch 接口。"""
    import torch_npu  # noqa: F401
    from cann_ops_transformer.ops import msa_index_score as npu_msa_index_score

    query = kwargs["query"]
    device = _npu_device(query)
    output = npu_msa_index_score(
        _on_device(query, device),
        _on_device(kwargs["key"], device),
        _on_device(kwargs["start_loc"], device),
        block_table=_on_device(kwargs.get("block_table"), device),
        query_scale=None,
        key_scale=_on_device(kwargs.get("key_scale"), device),
        atten_mask=_on_device(kwargs.get("atten_mask"), device),
        actual_seq_qlen=_on_device(kwargs["actual_seq_qlen"], device),
        actual_seq_klen=_on_device(kwargs["actual_seq_klen"], device),
        layout_key=str(kwargs.get("layout_key", "BBND")),
        sparse_mode=int(kwargs.get("sparse_mode", 3)),
        init_blocks=int(kwargs.get("init_blocks", 0)),
        local_blocks=int(kwargs.get("local_blocks", 1)),
        return_choice=0,
        topk=int(kwargs.get("topk", 16)),
        compress_factor=int(kwargs.get("compress_factor", 1)),
    )
    return output.detach().to("cpu")


def _prepare_inputs(kwargs: dict, precision: str, low_dtype: torch.dtype) -> dict:
    if precision == "fp32":
        prepared = _with_low_inputs(kwargs, torch.float32)
        key = _as_tensor(kwargs.get("key"))
        if key is not None and key.dtype == torch.int8:
            prepared["key"] = kwargs["key"]
        return prepared
    return _with_low_inputs(kwargs, low_dtype)


try:
    from atk.configs.dataset_config import InputDataset
    from atk.configs.results_config import TaskResult
    from atk.tasks.api_execute import register
    from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
    from atk.tasks.api_execute.base_api import BaseApi
except ImportError:
    register = None
    BaseApi = object
    AclnnBaseApi = object
    InputDataset = object
    TaskResult = object


if register is not None:
    import ctypes

    from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

    def _null_tensor_ptr():
        return ctypes.POINTER(AclTensor)()

    def _expand_optional_slots(input_args, kwargs):
        """补上省掉的可选输入，并把唯一的 score 放进三个输出槽的第一格。"""
        declared = [
            name for name in INPUT_SLOTS if isinstance(kwargs.get(name), torch.Tensor)
        ]
        tail = list(input_args[len(declared) :])
        remaining = list(declared)
        expanded = []
        incoming = iter(input_args[: len(declared)])
        for slot in INPUT_SLOTS:
            if remaining and remaining[0] == slot:
                expanded.append(next(incoming))
                remaining.pop(0)
            else:
                expanded.append(_null_tensor_ptr())
        if len(tail) < ATTR_COUNT + 1:
            expanded.extend(tail)
            return expanded
        attrs = tail[:ATTR_COUNT]
        score = tail[ATTR_COUNT]
        rest = tail[ATTR_COUNT + 1 :]
        expanded.extend(attrs)
        expanded.append(score)
        expanded.append(_null_tensor_ptr())
        expanded.append(_null_tensor_ptr())
        expanded.extend(rest)
        return expanded

    class _MsaRoleApi(BaseApi):
        def __init__(self, task_result: TaskResult):
            super().__init__(task_result)
            self._task_name = str(getattr(task_result, "name", "") or "")
            self._is_benchmark_task = bool(
                getattr(task_result, "is_benchmark_task", False)
            )
            name, case_id = _case_bits(task_result)
            seed = parse_case_seed(name, case_id)
            random.seed(seed)
            torch.manual_seed(seed)

        def init_by_input_data(self, input_data: InputDataset):
            name, case_id = _case_bits(self.task_result)
            bind_case_tensors(input_data.kwargs, parse_case_seed(name, case_id))
            _drop_illegal_optionals(input_data.kwargs)

        def __call__(self, input_data: InputDataset, with_output: bool = False):
            role = _resolve_role(self.device, self._task_name, self._is_benchmark_task)
            _, case_id = _case_bits(self.task_result)
            print(
                f"[msa-index-score] case={case_id} role={role} "
                f"device={self.device} node={self._task_name or '-'} "
                f"is_benchmark_task={self._is_benchmark_task}",
                flush=True,
            )
            kwargs = input_data.kwargs
            if role == "dut":
                return (_run_npu_op(kwargs),)
            query = kwargs.get("query")
            device = query.device if isinstance(query, torch.Tensor) else "cpu"
            low_dtype = _declared_low_dtype(self.task_result)
            precision = "fp32" if role == "golden" else "low"
            prepared = _prepare_inputs(kwargs, precision, low_dtype)
            return (_score_from_kwargs(prepared, precision, device=device),)

    class _MsaAclnnApi(AclnnBaseApi):
        def init_by_input_data(self, input_data: InputDataset):
            name, case_id = _case_bits(self.task_result)
            bind_case_tensors(input_data.kwargs, parse_case_seed(name, case_id))
            _drop_illegal_optionals(input_data.kwargs)
            result = super().init_by_input_data(input_data)
            if not isinstance(result, tuple) or not result:
                return result
            expanded = _expand_optional_slots(result[0], input_data.kwargs)
            backend = getattr(self, "backend", None)
            if backend is not None:
                backend.input_args = expanded
            return (expanded, *result[1:])

        def get_cpp_func_signature_type(self):
            return (
                "aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize("
                "const aclTensor *query, const aclTensor *key, const aclTensor *startLoc, "
                "const aclTensor *blockTableOptional, const aclTensor *queryScaleOptional, "
                "const aclTensor *keyScaleOptional, const aclTensor *attenMaskOptional, "
                "const aclTensor *actualSeqQlenOptional, const aclTensor *actualSeqKlenOptional, "
                "char *layoutKeyOptional, int64_t sparseMode, int64_t initBlocks, int64_t localBlocks, "
                "int64_t returnChoice, int64_t topk, int64_t compressFactor, "
                "const aclTensor *scoreOptional, const aclTensor *sparseIndicesOptional, "
                "const aclTensor *sparseValuesOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"
            )

    @register("function_msa_index_score")
    class FunctionMsaIndexScore(_MsaRoleApi):
        """自动 CPU 任务出 fp32 golden；node_cpu.yaml 的 CPU 或远端 GPU 出同精度标杆。npu 调算子。"""

    @register("pyaclnn_function_msa_index_score")
    class PyAclnnMsaIndexScore(_MsaAclnnApi):
        """待测：aclnnMsaIndexScore。A2/A3 只写 score，不调用 V2。"""


def _self_check() -> None:
    torch.manual_seed(0)
    query = torch.randn(4, 2, 16)
    key = torch.randn(2, BLOCK, 1, 16)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    qlen = torch.tensor([0, 4], dtype=torch.int32)
    klen = torch.tensor([200], dtype=torch.int32)
    start = torch.tensor([1], dtype=torch.int32)
    truth = msa_index_score_ref(
        query, key, table, None, qlen, klen, start, "BBND", 3, 0, 1, "fp32"
    )
    golden = msa_index_score_ref(
        query.half(), key.half(), table, None, qlen, klen, start, "BBND", 3, 0, 1, "low"
    )
    assert truth.shape == golden.shape == (2, 4, 16)
    boost = torch.tensor(LOCAL_SCORE_LOCAL, dtype=torch.float32)
    assert truth[0, 0, 1] == boost
    assert _resolve_role("cpu", "golden", True) == "golden"
    assert _resolve_role("cpu", "cpu", False) == "benchmark"
    assert _resolve_role("cuda:0", "gpu", False) == "benchmark"
    assert _resolve_role("npu", "npu", False) == "dut"
    print("msa_index_score executor self-check ok", tuple(truth.shape))


if __name__ == "__main__":
    _self_check()
