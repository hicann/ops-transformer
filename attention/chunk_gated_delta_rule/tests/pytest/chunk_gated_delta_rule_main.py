# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)
import time
import torch
import torch_npu

# from cann_ops_transformer.ops import chunk_gated_delta_rule
from chunk_gated_delta_rule_benchmark import chunk_gdn_benchmark_opt
from chunk_gated_delta_rule_golden import chunk_gated_delta_rule_npu
import torch.nn.functional as F
import numpy as np
import logging
import os

_USE_GRAPH = os.environ.get("USE_GRAPH", "false").lower() in ("true", "1", "yes")
_ENABLE_PROF = os.environ.get("ENABLE_PROF", "false").lower() in ("true", "1", "yes")
_SAVE_PT = os.environ.get("SAVE_PT", "false").lower() in ("true", "1", "yes")
_LOAD_PT = os.environ.get("LOAD_PT", "false").lower() in ("true", "1", "yes")
_LOAD_PT_FILE = os.environ.get("LOAD_PT_FILE", "")

if _USE_GRAPH:
    import torchair
    import warnings
    from torchair.core.utils import logger
    import torch.nn.functional as F

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger.setLevel(logging.ERROR)

    os.environ["ENABLE_ACLNN"] = "false"
    # 配置图模式 aclgraph config
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, query, key, value, initial_state, beta, actual_seq_lengths, scale, gamma
    ):
        chunk_gated_delta_rule = torch_npu.npu_chunk_gated_delta_rule(
            # chunk_gated_delta_rule = torch.ops.cann_ops_transformer.chunk_gated_delta_rule(
            query,
            key,
            value,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            scale=scale,
            g=gamma,
        )

        return chunk_gated_delta_rule


np.random.seed(21)
np.set_printoptions(suppress=True)
DEVICE_ID = 0
torch.npu.config.allow_internal_format = True
eb_threshold = 2 ** (-8)
err_threshold = 2 ** (-8)
CV_MAX_RE = 5  # 最大相对误差
CV_AVER_RE = 1.5  # 平均相对误差
CV_RMSE = 1.5  # 均方根误差
CV_SMALL_VAL = 2  # 小值域错误占比
CV_ERR_BALANCE = 2  # 误差均衡性
MIN_ERR = 1e-3
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# 定义ANSI颜色常量（新增）
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"  # 绿色
    WARNING = "\033[93m"
    FAIL = "\033[91m"  # 红色
    ENDC = "\033[0m"  # 重置颜色
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_max_re(golden: torch.Tensor, actual: torch.Tensor):
    # 最大相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    max_re = torch.max(abs_error.flatten())
    return max_re


def get_avg_re(golden: torch.Tensor, actual: torch.Tensor):
    # 平均相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    avg_re = torch.mean(abs_error)
    return avg_re


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    # 均方根误差
    sqr_err = torch.pow((actual - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse


def get_smra(golden: torch.Tensor, actual: torch.Tensor):
    # 小值域错误占比
    abs_A = torch.abs(golden)
    mask_A = abs_A < 2 ** (-10)
    num_a = torch.sum(mask_A).item()

    # 统计对应位置 B 中元素绝对值大于 1e-16 的个数
    abs_B = torch.abs(golden - actual)
    mask_B = abs_B > 1e-16
    num_b = torch.sum(mask_A & mask_B).item()

    smra = num_b / num_a if num_a > 0 else 0
    return smra


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    # 误差均衡性
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual - golden
    error_balance = torch.mean(actual_error / golden_nmax)
    return error_balance


def compare_cv(
    golden: torch.Tensor,
    golden_high_type: torch.Tensor,
    actual: torch.Tensor,
    name=None,
):
    golden = golden.to(torch.float32)
    golden_high_type = golden_high_type.to(torch.float32)
    actual = actual.to(torch.float32)
    # 最大相对误差
    max_re_npu = get_max_re(golden, actual)
    max_re_high_type = get_max_re(golden, golden_high_type)
    logger.info(f"{max_re_npu=}, {max_re_high_type=}")
    # 平均相对误差
    avg_re_npu = get_avg_re(golden, actual)
    avg_re_high_type = get_avg_re(golden, golden_high_type)
    # 均方根误差
    rmse_npu = get_rmse(golden, actual)
    rmse_high_type = get_rmse(golden, golden_high_type)
    # 小值域错误占比
    smra_npu = get_smra(golden, actual)
    smra_high_type = get_smra(golden, golden_high_type)
    max_re_rate = max_re_npu / max(max_re_high_type, err_threshold)
    avg_re_rate = avg_re_npu / max(avg_re_high_type, err_threshold)
    rmse_rate = rmse_npu / max(rmse_high_type, err_threshold)
    smra_rate = smra_npu / max(smra_high_type, err_threshold)
    if name is not None:
        logger.info(f"compare_cv for {name}:")
    logger.info(
        f"\tmax_re_rate={max_re_rate:.3f} ({CV_MAX_RE}), max_re_high_type={max_re_high_type:.3e}"
    )
    logger.info(
        f"\tavg_re_rate={avg_re_rate:.3f} ({CV_AVER_RE}), avg_re_high_type={avg_re_high_type:.3e}"
    )
    logger.info(
        f"\trmse_rate={rmse_rate:.3f} ({CV_RMSE}), rmse_high_type={rmse_high_type:.3e}"
    )
    logger.info(
        f"\tsmra_rate={smra_rate:.3f} ({CV_SMALL_VAL}), smra_high_type={smra_high_type:.3e}"
    )
    result = (
        (max_re_rate < CV_MAX_RE)
        and (avg_re_rate < CV_AVER_RE)
        and (rmse_rate < CV_RMSE)
    )
    result = result and smra_rate < CV_SMALL_VAL
    if not result:
        epsilon = 2.0**-7
        if max_re_npu < epsilon:
            logger.info(f"\t max_re_npu={max_re_npu} less than {epsilon}.")
            result = True
    return result


def cgdr_golden(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    actual_seq_lengths,
    use_float64=False,
    chunk_size=64,
):
    t0 = time.time()
    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    if use_float64:
        v = v.cpu().to(torch.float64)
    else:
        v = v.to(torch.float32)
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(v.dtype)
    o_golden, state_golden = chunk_gated_delta_rule_npu(
        q.unsqueeze(0).to(v.device).to(v.dtype),
        k.unsqueeze(0).to(v.device).to(v.dtype),
        v.unsqueeze(0).to(v.device).to(v.dtype),
        g.unsqueeze(0).to(v.device).to(v.dtype),
        beta.unsqueeze(0).to(v.device).to(v.dtype),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone().to(v.device).to(v.dtype),
        cu_seqlens=cu_seqlens.to(v.device),
        chunk_size=chunk_size,
    )
    o_golden = o_golden[0]
    state_golden = state_golden.transpose(-1, -2)
    logger.info(f"cgdr_golden {use_float64=} time cost: {time.time() - t0} s")
    return o_golden.to(torch.float32).npu(), state_golden.to(torch.float32).npu()


def cgdr_benchmark(
    q, k, v, g, beta, scale, initial_state, actual_seq_lengths, chunk_size=64
):
    dtype = torch.bfloat16
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(torch.float32)
    o_bench, state_bench = chunk_gdn_benchmark_opt(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        beta.to(dtype),
        scale,
        initial_state.to(dtype),
        actual_seq_lengths,
        g,
        chunk_size=chunk_size,
    )
    o_bench = o_bench.to(torch.float32)
    state_bench = state_bench.to(torch.float32)
    return o_bench, state_bench


def cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    logger.info(
        f"[cgdr_npu] run mode: {'aclgraph' if _USE_GRAPH else 'torch direct call'}"
    )
    if _USE_GRAPH:
        model = MyModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        o_npu, state_npu = model(
            q, k, v, initial_state, beta, actual_seq_lengths, scale, g
        )
    else:
        # o_npu, state_npu = torch.ops.cann_ops_transformer.chunk_gated_delta_rule(
        o_npu, state_npu = torch_npu.npu_chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            scale=scale,
            g=g,
        )
    o_npu = o_npu.to(torch.float32)
    state_npu = state_npu.to(torch.float32)
    return o_npu, state_npu


def _pt_filename(
    B,
    seqlen,
    nk,
    nv,
    dk,
    dv,
    chunk_size,
    data_type,
    state_data_type,
    has_g,
    is_contiguous,
):
    sl_str = (
        "-".join(str(x) for x in seqlen)
        if isinstance(seqlen, (list, tuple))
        else str(seqlen)
    )
    return (
        f"input_B{B}_S{sl_str}_nk{nk}_nv{nv}_dk{dk}_dv{dv}_cs{chunk_size}"
        f"_{str(data_type).replace('torch.', '')}"
        f"_{str(state_data_type).replace('torch.', '')}"
        f"_g{int(has_g)}_contig{int(is_contiguous)}.pt"
    )


def _save_input_pt(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    actual_seq_lengths,
    B,
    seqlen,
    nk,
    nv,
    dk,
    dv,
    chunk_size,
    data_type,
    state_data_type,
    has_g,
    is_contiguous,
):
    save_dir = os.path.join("output", "pt")
    os.makedirs(save_dir, exist_ok=True)
    fname = _pt_filename(
        B,
        seqlen,
        nk,
        nv,
        dk,
        dv,
        chunk_size,
        data_type,
        state_data_type,
        has_g,
        is_contiguous,
    )
    fpath = os.path.join(save_dir, fname)
    data = {
        "q": q.cpu(),
        "k": k.cpu(),
        "v": v.cpu(),
        "beta": beta.cpu(),
        "scale": scale,
        "initial_state": initial_state.cpu(),
        "actual_seq_lengths": actual_seq_lengths.cpu(),
        "g": None if g is None else g.cpu(),
        "meta": {
            "B": B,
            "seqlen": seqlen,
            "nk": nk,
            "nv": nv,
            "dk": dk,
            "dv": dv,
            "chunk_size": chunk_size,
            "data_type": data_type,
            "state_data_type": state_data_type,
            "has_g": has_g,
            "is_contiguous": is_contiguous,
        },
    }
    torch.save(data, fpath)
    logger.info(f"[SAVE_PT] saved input data to {fpath}")


def _load_input_pt(
    B,
    seqlen,
    nk,
    nv,
    dk,
    dv,
    chunk_size,
    data_type,
    state_data_type,
    has_g,
    is_contiguous,
    pt_path="",
):
    if _LOAD_PT_FILE:
        fpath = _LOAD_PT_FILE
    elif pt_path:
        fpath = pt_path
    else:
        load_dir = os.path.join("output", "pt")
        fname = _pt_filename(
            B,
            seqlen,
            nk,
            nv,
            dk,
            dv,
            chunk_size,
            data_type,
            state_data_type,
            has_g,
            is_contiguous,
        )
        fpath = os.path.join(load_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"[LOAD_PT] pt file not found: {fpath}")
    data = torch.load(fpath, map_location="cpu")
    dev = f"npu:{DEVICE_ID}"
    loaded = {
        "q": data["q"].to(dev),
        "k": data["k"].to(dev),
        "v": data["v"].to(dev),
        "beta": data["beta"].to(dev),
        "scale": data["scale"],
        "initial_state": data["initial_state"].to(dev),
        "actual_seq_lengths": data["actual_seq_lengths"].to(dev),
        "g": None if data["g"] is None else data["g"].to(dev),
    }
    logger.info(f"[LOAD_PT] loaded input data from {fpath}")
    return loaded


def run_chunk_gated_delta_rule_eager(
    B,
    seqlen,
    nk,
    nv,
    dk,
    dv,
    chunk_size=64,
    data_type=torch.bfloat16,
    state_data_type=torch.bfloat16,
    has_g=True,
    is_contiguous=True,
    pt_path="",
):
    torch_npu.npu.set_device(int(DEVICE_ID))
    # ======================== gen input data start =============================
    if _LOAD_PT:
        loaded = _load_input_pt(
            B,
            seqlen,
            nk,
            nv,
            dk,
            dv,
            chunk_size,
            data_type,
            state_data_type,
            has_g,
            is_contiguous,
            pt_path=pt_path,
        )
        q = loaded["q"]
        k = loaded["k"]
        v = loaded["v"]
        g = loaded["g"]
        beta = loaded["beta"]
        scale = loaded["scale"]
        initial_state = loaded["initial_state"]
        actual_seq_lengths = loaded["actual_seq_lengths"]
        B = initial_state.shape[0]
        T = q.shape[0]
    else:
        if isinstance(seqlen, (list, tuple)):
            seqlen_list = list(seqlen)
            B = len(seqlen_list)
            T = sum(seqlen_list)
        else:
            seqlen_list = [seqlen] * B
            T = B * seqlen
        q = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
        k = torch.rand((T, nk, dk), dtype=data_type, device="npu:%s" % DEVICE_ID)
        v = torch.rand((T, nv, dv), dtype=data_type, device="npu:%s" % DEVICE_ID)
        if has_g:
            g = (
                torch.rand((T, nv), dtype=torch.float32, device="npu:%s" % DEVICE_ID)
                * -1.0
            )
        else:
            g = None
        beta = torch.rand((T, nv), dtype=data_type, device="npu:%s" % DEVICE_ID)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
        scale = 1 / (dk**0.5)
        initial_state = torch.rand(
            (B, nv, dv, dk), dtype=state_data_type, device="npu:%s" % DEVICE_ID
        )
        if not is_contiguous:
            state_pad = torch.zeros(
                (B, nv, dv + 1, dk), dtype=state_data_type, device="npu:%s" % DEVICE_ID
            )
            state_pad[:, :, :dv, :] = initial_state
            initial_state = state_pad[:, :, :dv, :]
        actual_seq_lengths = torch.tensor(
            seqlen_list, dtype=torch.int32, device="npu:%s" % DEVICE_ID
        )
    # ======================== gen input data finish =============================
    logger.info(
        f"initial_state: is_contiguous={initial_state.is_contiguous()}, stride={initial_state.stride()}"
    )

    if _SAVE_PT:
        _save_input_pt(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            B=B,
            seqlen=seqlen,
            nk=nk,
            nv=nv,
            dk=dk,
            dv=dv,
            chunk_size=chunk_size,
            data_type=data_type,
            state_data_type=state_data_type,
            has_g=has_g,
            is_contiguous=is_contiguous,
        )

    if _ENABLE_PROF:
        cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
        for _ in range(5):
            cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)
        torch.npu.synchronize()
        logger.info("PASSED (prof)")
        return True

    # ======================== execute golden/benchmark/npu ================================
    o_golden, state_golden = cgdr_golden(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        actual_seq_lengths,
        use_float64=False,
        chunk_size=chunk_size,
    )
    o_bench, state_bench = cgdr_benchmark(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        actual_seq_lengths,
        chunk_size=chunk_size,
    )
    o_npu, state_npu = cgdr_npu(
        q, k, v, g, beta, scale, initial_state, actual_seq_lengths
    )
    # ======================== check result ================================
    ret = True
    if not compare_cv(o_golden, o_bench, o_npu, name="o"):
        logger.error("compare o failed.")
        err_o = torch.abs(o_golden - o_npu).flatten()
        idx = torch.argmax(err_o)
        logger.info(
            f"idx={idx}, err_o={err_o[idx]}, o_golden={o_golden.flatten()[idx]}, "
            f"o_npu={o_npu.flatten()[idx]}, o_bench={o_bench.flatten()[idx]}"
        )
        ret = False
    if not compare_cv(state_golden, state_bench, state_npu, name="state"):
        logger.error("compare state failed.")
        err_s = torch.abs(state_golden - state_npu).flatten()
        idx = torch.argmax(err_s)
        logger.info(
            f"idx={idx}, err_s={err_s[idx]}, state_golden={state_golden.flatten()[idx]}, "
            f"state_npu={state_npu.flatten()[idx]}, state_bench={state_bench.flatten()[idx]}"
        )
        ret = False

    return ret


def run_precision_test(inputs):
    run_chunk_gated_delta_rule_eager(
        inputs["B"],
        inputs["seqlen"],
        inputs["nk"],
        inputs["nv"],
        inputs["dk"],
        inputs["dv"],
        inputs["chunk_size"],
        data_type=inputs["data_type"],
        state_data_type=inputs.get("state_data_type", torch.bfloat16),
        has_g=inputs.get("has_g", True),
        is_contiguous=inputs.get("is_contiguous", True),
        pt_path=inputs.get("pt_path", ""),
    )
