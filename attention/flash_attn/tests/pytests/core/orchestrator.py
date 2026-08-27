# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""编排器 — 串联数据生成 → 后端计算 → 精度对比 + golden 持久化。"""

import gc
import logging
import sys
from pathlib import Path
from typing import List

import torch

from core.data import InputSpec, flash_attn_inputs
from core.backends.base import Backend
from utils.compare import check_result, get_tolerance
from utils.data import trans_bnsd_to_layout, make_noncontiguous

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# ─── Golden 持久化 ───────────────────────────────────────────────────────────


def save_golden_case(
    case_name: str,
    inputs: dict,
    golden_out: dict,
    params: dict,
    golden_dir: str,
    primary_out: dict = None,
    primary_name: str = None,
):
    safe_name = case_name.replace("/", "_")
    case_dir = Path(golden_dir) / safe_name
    case_dir.mkdir(parents=True, exist_ok=True)

    inputs_save = {}
    for k, v in inputs.items():
        inputs_save[k] = v.cpu() if isinstance(v, torch.Tensor) else v
    for k, v in params.items():
        if k not in inputs_save:
            inputs_save[k] = v.cpu() if isinstance(v, torch.Tensor) else v
    torch.save(inputs_save, case_dir / "inputs.pt")

    golden_save = {}
    for k, v in golden_out.items():
        if isinstance(v, torch.Tensor):
            golden_save[k] = v.cpu().float()
    torch.save(golden_save, case_dir / "golden.pt")

    if primary_out and primary_name:
        dev_save = {}
        for k, v in primary_out.items():
            if isinstance(v, torch.Tensor):
                dev_save[k] = v.cpu().float()
        torch.save(dev_save, case_dir / f"{primary_name}_out.pt")


def load_golden_case(case_name: str, golden_dir: str):
    safe_name = case_name.replace("/", "_")
    case_dir = Path(golden_dir) / safe_name

    if not case_dir.exists():
        raise FileNotFoundError(f"Golden 目录不存在: {case_dir}")

    raw = torch.load(case_dir / "inputs.pt", map_location="cpu", weights_only=False)
    golden = torch.load(case_dir / "golden.pt", map_location="cpu", weights_only=False)

    inputs = {}
    params = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v
        else:
            params[k] = v

    return inputs, golden, params


# ─── 核心运行逻辑 ────────────────────────────────────────────────────────────


def run_case(
    params: dict,
    primary: Backend,
    golden: Backend,
    spec: InputSpec = None,
    comparators: List[Backend] = None,
    golden_dir: str = None,
    case_name: str = None,
    verbose_diff: bool = False,
    visualize: bool = False,
    viz_dir: str = "./viz_output",
    fail_analysis: bool = False,
    compare_dir: str = None,
) -> dict:
    spec = spec or flash_attn_inputs
    comparators = comparators or []
    layout = params.get("input_layout", params.get("layout_q", "BNSD"))
    out_layout = params.get("layout_out", layout)
    output_dtype = params.get("Dtype", "fp16")
    atol, rtol = get_tolerance(output_dtype)

    if compare_dir and case_name:
        safe_name = case_name.replace("/", "_")
        case_dir = Path(compare_dir) / safe_name
        if case_dir.exists():
            logger.info(f"  [load] loading inputs and golden from {case_dir}...")
            raw = torch.load(
                case_dir / "inputs.pt", map_location="cpu", weights_only=False
            )
            golden_out = torch.load(
                case_dir / "golden.pt", map_location="cpu", weights_only=False
            )
            cpu_inputs = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    cpu_inputs[k] = v
                else:
                    params[k] = v
        else:
            logger.info(f"  [warn] {case_dir} not found, falling back to regeneration")
            compare_dir = None

    if not (compare_dir and case_name):
        cpu_inputs = spec.generate(
            params,
            device=torch.device("cpu"),
            layout=layout,
            layout_kv=params.get("layout_kv", layout),
        )
        if str(params.get("layout_kv", "")).startswith("PA_"):
            cpu_inputs["block_table"] = params["block_table"]
        elif "block_table" in params and params["block_table"] is not None:
            cpu_inputs["block_table"] = params["block_table"]

        logger.info("  [golden] CPU reference computation...")
        golden_out = golden.compute(cpu_inputs, params)

    logger.info(f"  [primary] {primary.name} computing...")
    dev_inputs = {
        k: v.to(primary.device) if isinstance(v, torch.Tensor) else v
        for k, v in cpu_inputs.items()
    }
    primary_out_raw = primary.compute(dev_inputs, params)
    primary_out = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in primary_out_raw.items()
    }
    del primary_out_raw
    primary.clear_cache()

    # --- kvcache 非连续 self-consistency check ---
    nc_kv_dims = params.get("nc_kv_dims")
    nc_attn_result = None
    if nc_kv_dims is not None:
        # FIA做法: 连续tensor传NPU后, 在NPU上直接构造NC, 避免.to(npu)碾连续
        nc_dev_inputs = {}
        for k, v in dev_inputs.items():
            if isinstance(v, torch.Tensor) and k in ("k", "v"):
                nc_dev_inputs[k] = make_noncontiguous(v, nc_kv_dims)
            else:
                nc_dev_inputs[k] = v
        assert torch.equal(nc_dev_inputs["k"], dev_inputs["k"]), (
            "k data after NC is inconsistent with contiguous"
        )
        assert torch.equal(nc_dev_inputs["v"], dev_inputs["v"]), (
            "v data after NC is inconsistent with contiguous"
        )
        logger.info(
            f"  [nc_check] nc_kv_dims={nc_kv_dims} "
            f"k_contig={nc_dev_inputs['k'].is_contiguous()} "
            f"k_stride={nc_dev_inputs['k'].stride()} "
            f"v_contig={nc_dev_inputs['v'].is_contiguous()} "
            f"v_stride={nc_dev_inputs['v'].stride()}"
        )
        nc_raw = primary.compute(nc_dev_inputs, params)
        nc_cpu = (
            nc_raw["out"].cpu()
            if isinstance(nc_raw["out"], torch.Tensor)
            else nc_raw["out"]
        )
        del nc_dev_inputs, nc_raw
        primary.clear_cache()
        contig_f = primary_out["out"].float()
        nc_f = nc_cpu.float()
        exact_match = torch.equal(contig_f, nc_f)
        diff = (contig_f - nc_f).abs()
        max_abs = diff.max().item()
        logger.info(
            f"  [nc_check] NC vs Contiguous  exact_match={exact_match}  max_abs={max_abs:.8e}"
        )
        nc_attn_result = {
            "passed": exact_match,
            "max_abs": max_abs,
            "mean_abs": diff.mean().item(),
            "fail_cnt": 0 if exact_match else int((diff > 0).sum().item()),
            "total": contig_f.numel(),
            "fail_ratio": 0.0
            if exact_match
            else float((diff > 0).sum().item()) / contig_f.numel(),
        }
        del nc_cpu

    del dev_inputs

    dev_raw = primary_out["out"]
    dev_out = dev_raw

    cpu_raw = golden_out["out"]
    cpu_out = (
        trans_bnsd_to_layout(cpu_raw, out_layout, **params)
        if out_layout != "BNSD"
        else cpu_raw
    ).float()

    if compare_dir and case_name:
        other_name = "gpu" if primary.name == "npu" else "npu"
        safe_name = case_name.replace("/", "_")
        other_path = Path(compare_dir) / safe_name / f"{other_name}_out.pt"
        if other_path.exists():
            from utils.compare import data_compare_benchmark_new

            other_out = torch.load(other_path, map_location="cpu", weights_only=False)
            other_raw = other_out["out"].float()
            output_dtype = params.get("Dtype", "fp16")
            cmp_params = {
                "case_name": case_name,
                "op_name": "flash_attn",
                "dtype_output": [output_dtype],
                "dtype_input": [output_dtype],
                "red_range": {
                    "fp32": "0.000001/0.00001/0.0001/0.0005",
                    "fp16": "0.001/0.002/0.005/0.01",
                    "bf16": "0.001/0.002/0.005/0.01",
                },
                "bm_cmp_std": {
                    "fp32": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 1e-06,
                        "small_value_atol": 0.0,
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
                        "small_value": 1e-07,
                        "small_value_atol": 0.004,
                    },
                },
            }
            logger.info(
                f"\n{'=' * 40} three-way comparison mode (detailed precision stats) {'=' * 40}"
            )
            str1, str2, data = data_compare_benchmark_new(
                cmp_params,
                dev_out.float().numpy(),
                other_raw.numpy(),
                cpu_out.numpy(),
                output_dtype,
                0,
            )
            logger.info("\n====== three-way comparison results ======")
            logger.info(f"status: {str1}, reason: {str2}")
            passed = str1 == "Pass" or str1 == "warning"
            attn_result = {
                "passed": passed,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "fail_cnt": 0 if passed else 1,
                "total": 1,
                "fail_ratio": 0.0 if passed else 1.0,
                "verdict_npu": str1,
                "verdict_gpu": str2,
            }
        else:
            logger.info(
                f"  [warn] {other_path} not found, falling back to CPU comparison"
            )
            logger.info(f"  [compare] CPU vs {primary.name}")
            attn_result = check_result(
                f"CPU_vs_{primary.name}",
                cpu_out,
                dev_out.float(),
                except_label="CPU",
                comp_label=primary.name,
                verbose_diff=verbose_diff,
                atol=atol,
                rtol=rtol,
            )
    else:
        logger.info(f"  [compare] CPU vs {primary.name}")
        attn_result = check_result(
            f"CPU_vs_{primary.name}",
            cpu_out,
            dev_out.float(),
            except_label="CPU",
            comp_label=primary.name,
            verbose_diff=verbose_diff,
            atol=atol,
            rtol=rtol,
        )

    lse_result = {
        "passed": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "fail_cnt": 0,
        "total": 0,
        "fail_ratio": 0.0,
    }
    if (
        params.get("return_softmax_lse", False)
        and "lse" in primary_out
        and "lse" in golden_out
    ):
        dev_lse = primary_out["lse"]
        if isinstance(dev_lse, torch.Tensor):
            dev_lse = dev_lse.cpu().float()
            cpu_lse = golden_out["lse"]
            if isinstance(cpu_lse, torch.Tensor):
                cpu_lse = cpu_lse.float()
                cpu_lse = cpu_lse.clone()
                dev_lse = dev_lse.clone()
                cpu_lse[cpu_lse.isinf() & (cpu_lse > 0)] = float("-inf")
                dev_lse[dev_lse.isinf() & (dev_lse > 0)] = float("-inf")
                logger.info("  [compare] LSE check")
                lse_result = check_result(
                    "LSE",
                    cpu_lse,
                    dev_lse,
                    except_label="CPU_lse",
                    comp_label=f"{primary.name}_lse",
                    verbose_diff=verbose_diff,
                    atol=atol,
                    rtol=rtol,
                )

    for comp in comparators:
        comp_inputs = {
            k: v.to(comp.device) if isinstance(v, torch.Tensor) else v
            for k, v in cpu_inputs.items()
        }
        comp_out = comp.compute(comp_inputs, params)
        comp_cpu = comp_out["out"].cpu()
        logger.info(f"  [compare] {primary.name} vs {comp.name}")
        _ = check_result(
            f"{primary.name}_vs_{comp.name}",
            dev_out.float(),
            comp_cpu.float(),
            except_label=primary.name,
            comp_label=comp.name,
            atol=atol,
            rtol=rtol,
        )

    ret = {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}
    if nc_attn_result is not None:
        ret["nc_attn"] = _to_stats(nc_attn_result)

    if golden_dir and case_name:
        save_golden_case(
            case_name,
            cpu_inputs,
            golden_out,
            params,
            golden_dir,
            primary_out=primary_out,
            primary_name=primary.name,
        )

    attn_passed = (
        ret["attn"].get("passed", True) if isinstance(ret["attn"], dict) else True
    )
    if not attn_passed:
        if fail_analysis:
            try:
                from utils.compare import analyze_fail_distribution

                analyze_fail_distribution(
                    test_name=case_name or "unknown",
                    expect=cpu_out.float(),
                    result=dev_out.float(),
                    dump_dir=viz_dir,
                )
            except Exception as e:
                import traceback

                logger.error(f"  [warn] fail_analysis error: {e}")
                traceback.print_exc()
        if visualize:
            try:
                from utils.precision_visual import visualize_from_tensors

                visualize_from_tensors(
                    cpu_out.float().numpy(),
                    dev_out.float().numpy(),
                    case_name=case_name or "unknown",
                    out_dir=viz_dir,
                )
            except Exception as e:
                import traceback

                logger.error(f"  [warn] visualize error: {e}")
                traceback.print_exc()

    del cpu_inputs, primary_out, golden_out
    del dev_raw, cpu_out, dev_out
    gc.collect()

    return ret


def run_case_load(
    case_name: str,
    golden_dir: str,
    primary: Backend,
    device_id: int = 0,
    timeout: int = 300,
    verbose_diff: bool = False,
    visualize: bool = False,
    viz_dir: str = "./viz_output",
    fail_analysis: bool = False,
) -> dict:
    inputs, golden_data, params = load_golden_case(case_name, golden_dir)
    output_dtype = params.get("Dtype", "fp16")
    atol, rtol = get_tolerance(output_dtype)

    dev_inputs = {k: v.to(primary.device) for k, v in inputs.items()}
    npu_out_raw = primary.compute(dev_inputs, params)
    npu_out = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in npu_out_raw.items()
    }

    out_layout = params.get("layout_out", params.get("layout_q", "BNSD"))

    cpu_raw = golden_data["out"]
    dev_raw = npu_out["out"]

    cpu_out = (
        trans_bnsd_to_layout(cpu_raw, out_layout, **params)
        if out_layout != "BNSD"
        else cpu_raw
    )

    attn_result = check_result(
        "CPU_vs_NPU",
        cpu_out.float(),
        dev_raw.float(),
        except_label="CPU",
        comp_label=primary.name,
        verbose_diff=verbose_diff,
        atol=atol,
        rtol=rtol,
    )

    lse_result = {
        "passed": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "fail_cnt": 0,
        "total": 0,
        "fail_ratio": 0.0,
    }
    if (
        params.get("return_softmax_lse", False)
        and "lse" in golden_data
        and "lse" in npu_out
    ):
        dev_lse = npu_out["lse"]
        cpu_lse = golden_data["lse"]
        if isinstance(dev_lse, torch.Tensor) and isinstance(cpu_lse, torch.Tensor):
            cpu_lse = cpu_lse.float().clone()
            dev_lse = dev_lse.float().clone()
            cpu_lse[cpu_lse.isinf() & (cpu_lse > 0)] = float("-inf")
            dev_lse[dev_lse.isinf() & (dev_lse > 0)] = float("-inf")
            lse_result = check_result(
                "LSE",
                cpu_lse,
                dev_lse,
                except_label="CPU_lse",
                comp_label=f"{primary.name}_lse",
                verbose_diff=verbose_diff,
                atol=atol,
                rtol=rtol,
            )

    ret = {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}

    attn_passed = (
        ret["attn"].get("passed", True) if isinstance(ret["attn"], dict) else True
    )
    if not attn_passed:
        if fail_analysis:
            try:
                from utils.compare import analyze_fail_distribution

                analyze_fail_distribution(
                    test_name=case_name,
                    expect=cpu_out.float(),
                    result=dev_raw.float(),
                    dump_dir=viz_dir,
                )
            except Exception as e:
                import traceback

                logger.error(f"  [warn] fail_analysis error: {e}")
                traceback.print_exc()
        if visualize:
            try:
                from utils.precision_visual import visualize_from_tensors

                visualize_from_tensors(
                    cpu_out.float().numpy(),
                    dev_raw.float().numpy(),
                    case_name=case_name,
                    out_dir=viz_dir,
                )
            except Exception as e:
                import traceback

                logger.error(f"  [warn] visualize error: {e}")
                traceback.print_exc()

    del inputs, golden_data, npu_out
    del cpu_raw, dev_raw, cpu_out
    primary.clear_cache()
    gc.collect()

    return ret


def _to_stats(val) -> dict:
    if isinstance(val, dict):
        return val
    if val:
        return {
            "passed": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "fail_cnt": 0,
            "total": 0,
            "fail_ratio": 0.0,
        }
    return {
        "passed": False,
        "max_abs": float("nan"),
        "mean_abs": float("nan"),
        "fail_cnt": -1,
        "total": -1,
        "fail_ratio": float("nan"),
    }
