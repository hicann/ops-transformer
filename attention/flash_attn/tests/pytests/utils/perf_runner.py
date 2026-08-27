# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Performance runner: per-case msprof invocation, surviving crashes."""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def build_case_script(case: Dict, runs: int, case_name: str, device_id: int = 0) -> str:
    """Build a single-case Python script for msprof."""
    is_hot = case.get("S1", 9999) > 16
    mode = "hot" if is_hot else "cold"
    params_repr = _serialize_params(case)

    lines = [
        "import sys, os",
        f"_pytests_dir = {repr(str(Path(__file__).resolve().parent.parent))}",
        "sys.path.insert(0, _pytests_dir)",
        "import torch, torch_npu",
        f"torch.npu.set_device({device_id})",
        "from core.data import flash_attn_inputs",
        "from cann_ops_transformer.ops import flash_attn, flash_attn_metadata",
        "device = torch.device('npu:0')",
        f"c = {params_repr}",
        f"print('CASE {case_name}  {mode}  runs={runs}')",
        "",
        "def _to_int32_tensor(v):",
        "    if v is None: return None",
        "    return torch.tensor(v, dtype=torch.int32, device=device)",
        "",
    ]

    lines.extend(
        [
            "layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))",
            "layout_kv = c.get('layout_kv', layout_q)",
            "cu_q = c.get('cu_seqlens_q')",
            "cu_kv = c.get('cu_seqlens_kv')",
            "sq = c.get('seqused_q') or c.get('actual_seq_qlen')",
            "skv = c.get('seqused_kv') or c.get('actual_seq_kvlen')",
            "cu_q_t = _to_int32_tensor(cu_q)",
            "cu_kv_t = _to_int32_tensor(cu_kv)",
            "sq_t = _to_int32_tensor(sq)",
            "skv_t = _to_int32_tensor(skv)",
            "if layout_q == 'TND' and cu_q_t is not None:",
            "    _bs = cu_q_t.shape[0] - 1",
            "else:",
            "    _bs = c.get('B', 1)",
            "_msq = c.get('max_seqlen_q')",
            "if _msq is None:",
            "    _msq = max(sq) if sq else (int((cu_q_t[1:]-cu_q_t[:-1]).max()) if cu_q_t is not None else c.get('S1', 1))",
            "else:",
            "    _msq = int(_msq)",
            "_msk = c.get('max_seqlen_kv')",
            "if _msk is None:",
            "    _msk = max(skv) if skv else (int((cu_kv_t[1:]-cu_kv_t[:-1]).max()) if cu_kv_t is not None else c.get('S2', c.get('S1', 1)))",
            "else:",
            "    _msk = int(_msk)",
            "",
        ]
    )

    if is_hot:
        lines.extend(
            [
                "inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)",
                "bt = c.get('block_table')",
                "if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):",
                "    bt_t = torch.tensor(bt, dtype=torch.int32, device=device)",
                "    if bt_t.dim() == 1:",
                "        bt_t = bt_t.unsqueeze(0)",
                "    inputs['block_table'] = bt_t",
                "q, k, v = inputs['q'], inputs['k'], inputs['v']",
                "bt_dev = inputs.get('block_table')",
                "meta = flash_attn_metadata(",
                "    cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,",
                "    seqused_q=sq_t, seqused_kv=skv_t,",
                "    num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],",
                "    batch_size=_bs, max_seqlen_q=_msq, max_seqlen_kv=_msk,",
                "    mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,",
                "    layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
                ")",
                "if c.get('mask_mode', 0) == 3:",
                "    _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)",
                "else:",
                "    _m = None",
                "torch.npu.synchronize()",
                f"for _ in range({runs}):",
                "    torch.npu.synchronize()",
                "    flash_attn(q, k, v, block_table=bt_dev, sinks=None, attn_mask=_m,",
                "        metadata=meta, cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,",
                "        seqused_q=sq_t, seqused_kv=skv_t,",
                "        softmax_scale=1/(c['D']**0.5), mask_mode=c.get('mask_mode',0),",
                "        win_left=-1, win_right=-1, max_seqlen_q=_msq, max_seqlen_kv=_msk,",
                "        layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
                "        return_softmax_lse=0)",
                "torch.npu.synchronize()",
            ]
        )
    else:
        lines.extend(
            [
                "meta = flash_attn_metadata(",
                "    cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,",
                "    seqused_q=sq_t, seqused_kv=skv_t,",
                "    num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],",
                "    batch_size=_bs, max_seqlen_q=_msq, max_seqlen_kv=_msk,",
                "    mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,",
                "    layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
                ")",
                "if c.get('mask_mode', 0) == 3:",
                "    _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)",
                "else:",
                "    _m = None",
                f"for _ in range({runs}):",
                "    inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)",
                "    bt = c.get('block_table')",
                "    bt_t = None",
                "    if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):",
                "        bt_t = torch.tensor(bt, dtype=torch.int32, device=device)",
                "        if bt_t.dim() == 1:",
                "            bt_t = bt_t.unsqueeze(0)",
                "    elif str(layout_kv).startswith('PA_'):",
                "        print(f'WARN: block_table missing for PA case, bt={type(bt)}')",
                "    inputs['block_table'] = bt_t",
                "    q, k, v = inputs['q'], inputs['k'], inputs['v']",
                "    bt_dev = inputs.get('block_table')",
                "    _f1 = torch.randn(32, 32768, 1, 128, dtype=q.dtype, device=device)",
                "    _f2 = torch.randn(32, 32768, 1, 128, dtype=q.dtype, device=device)",
                "    torch.npu.synchronize()",
                "    flash_attn(q, k, v, block_table=bt_dev, sinks=None, attn_mask=_m,",
                "        metadata=meta, cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,",
                "        seqused_q=sq_t, seqused_kv=skv_t,",
                "        softmax_scale=1/(c['D']**0.5), mask_mode=c.get('mask_mode',0),",
                "        win_left=-1, win_right=-1, max_seqlen_q=_msq, max_seqlen_kv=_msk,",
                "        layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
                "        return_softmax_lse=0)",
                "torch.npu.synchronize()",
                "torch.npu.empty_cache()",
            ]
        )

    lines.extend(
        [
            "print('CASE_DONE')",
        ]
    )
    return "\n".join(lines)


def _serialize_params(c: Dict) -> str:
    s = {}
    for k, v in c.items():
        if hasattr(v, "shape"):
            s[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            s[k] = list(v)
        else:
            s[k] = v
    return repr(s)


def run_all_perf(
    cases: List[Dict],
    runs: int,
    cold_thr: int,
    output_dir: Path,
    case_names: List[str] = None,
    device_id: int = 0,
    one_by_one: bool = False,
) -> Optional[Path]:
    """Run each case as a separate msprof invocation. Returns merged op_summary.csv."""
    import csv as csv_mod

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = case_names or [f"case_{i}" for i in range(len(cases))]
    log_lines = []
    all_csvs = []

    perf_csv = output_dir / "perf.csv"
    csv_f = open(perf_csv, "w", newline="")
    csv_w = csv_mod.DictWriter(csv_f, ["#", "case", "dir", "avg_us", "mode", "error"])
    csv_w.writeheader()

    for i, (c, name) in enumerate(zip(cases, names)):
        tag = f"{'hot' if c.get('S1', 9999) > cold_thr else 'cold'}{i}"
        case_dir_name = name.replace("/", "_")
        case_dir = output_dir / case_dir_name
        case_dir.mkdir(parents=True, exist_ok=True)

        script = build_case_script(c, runs, name, device_id)
        script_path = case_dir / "_perf_sweep.py"
        script_path.write_text(script)

        msprof_dir = case_dir / "msprof_output"
        if msprof_dir.exists():
            shutil.rmtree(msprof_dir)

        cmd = [
            "msprof",
            f"--output={msprof_dir}",
            "--aic-mode=task-based",
            sys.executable,
            str(script_path),
        ]
        logger.info(f"[msprof {i + 1}/{len(cases)}] {name}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        crashed = False
        error_msg = ""
        avg_us = 0.0
        mode = "hot" if c.get("S1", 9999) > cold_thr else "cold"

        if result.returncode != 0:
            crashed = True
            error_msg = f"exit={result.returncode}"
        if "CASE_DONE" not in stdout:
            crashed = True
            error_msg = error_msg or "CASE_DONE not found"
        if stderr:
            error_msg = (error_msg + "; " + stderr)[:500]

        log_lines.append(
            json.dumps(
                {
                    "tag": tag,
                    "ok": not crashed,
                    "error": error_msg,
                    "name": name,
                }
            )
        )

        prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
        if prof_dirs:
            csv_files = sorted(
                prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv")
            )
            if csv_files:
                dst = case_dir / "op_summary.csv"
                shutil.copy(csv_files[0], dst)
                all_csvs.append(dst)
                avg_us = _quick_avg(dst, runs)
                logger.info(f"  [{name}] OK  avg={avg_us:.1f}us")
            else:
                logger.error(f"  [{name}] NO CSV")
                crashed = True
                error_msg = error_msg or "no op_summary.csv"
        else:
            error_msg = error_msg or "no PROF directory"
            logger.error(f"  [{name}] NO PROF_DIR {error_msg}")
            crashed = True

        csv_w.writerow(
            {
                "#": i + 1,
                "case": name,
                "dir": case_dir_name,
                "avg_us": "" if crashed else f"{avg_us:.1f}",
                "mode": "CRASH" if crashed else mode,
                "error": error_msg,
            }
        )
        csv_f.flush()

    csv_f.close()

    # Write log
    log_path = output_dir / "_perf_log.jsonl"
    with open(log_path, "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Merge all op_summary.csv into one
    if not all_csvs:
        return None

    merged = output_dir / "op_summary.csv"
    _merge_csvs(all_csvs, merged)
    logger.info(f"[perf] merged {len(all_csvs)} CSV → {merged}")
    return merged


def _merge_csvs(csv_paths: List[Path], out_path: Path):
    import csv as csv_mod

    rows = []
    header = None
    for cp in csv_paths:
        with open(cp) as f:
            reader = csv_mod.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                rows.append(row)
    if header is None:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _quick_avg(csv_path: Path, runs: int, op_name: str = "FlashAttn") -> float:
    """Quickly compute avg duration from a single op_summary.csv."""
    from utils.perf_parser import parse_op_summary

    entries = parse_op_summary(str(csv_path), op_name)
    durs = [d for _, d in entries]
    if not durs:
        return 0.0
    is_hot = len(durs) >= runs
    if is_hot:
        durs = durs[1:]  # discard 1st
    else:
        durs = sorted(durs)
        if len(durs) >= 3:
            durs = durs[1:-1]
    return sum(durs) / len(durs) if durs else 0.0


# ══════════════════════════════════════════════════════════
# Batch mode: all cases in one msprof script
# ══════════════════════════════════════════════════════════


def build_batch_script(
    cases: List[Dict],
    runs: int,
    cold_thr: int,
    case_names: List[str] = None,
    device_id: int = 0,
    output_dir: str = ".",
) -> str:
    """Build a single script with all cases for batch msprof."""
    hot = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) > cold_thr]
    cold = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) <= cold_thr]
    names = case_names or [f"case_{i}" for i in range(len(cases))]

    case_dicts = _serialize_cases(cases)
    hot_names = repr([names[i] for i, _ in hot])
    cold_names = repr([names[i] for i, _ in cold])

    lines = [
        "import sys, os",
        f"_pytests_dir = {repr(str(Path(__file__).resolve().parent.parent))}",
        "sys.path.insert(0, _pytests_dir)",
        "import torch, torch_npu",
        f"torch.npu.set_device({device_id})",
        "from core.data import flash_attn_inputs",
        "from cann_ops_transformer.ops import flash_attn, flash_attn_metadata",
        "device = torch.device('npu:0')",
        f"_hot_cases = {case_dicts['hot']}",
        f"_cold_cases = {case_dicts['cold']}",
        f"_hot_names = {hot_names}",
        f"_cold_names = {cold_names}",
        f"_perf_log = open({repr(str(Path(output_dir) / '_perf_log.jsonl'))}, 'w')",
        f"print(f'PERF total={len(cases)} hot={len(hot)} cold={len(cold)} runs={runs}')",
        "",
    ]

    for idx, (orig_i, c) in enumerate(hot):
        lines.extend(_gen_batch_hot(idx, runs, len(hot)))

    for idx, (orig_i, c) in enumerate(cold):
        lines.extend(_gen_batch_cold(idx, runs, len(cold)))

    lines.append("print('PERF_DONE')")
    return "\n".join(lines)


def _serialize_cases(cases):
    """For batch mode: serialize only the cases used."""
    hot_items = []
    cold_items = []
    thr = 16
    for c in cases:
        s = _serialize_one(c)
        if c.get("S1", 9999) > thr:
            hot_items.append(s)
        else:
            cold_items.append(s)
    return {"hot": repr(hot_items), "cold": repr(cold_items)}


def _serialize_one(c):
    s = {}
    for k, v in c.items():
        if hasattr(v, "shape"):
            s[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            s[k] = list(v)
        else:
            s[k] = v
    return s


def _gen_batch_hot(idx, runs, total):
    return [
        f"c = _hot_cases[{idx}]",
        f"name = _hot_names[{idx}]",
        f"print(f'  [batch] {idx + 1}/{total}  {{name}}')",
        f"print('[CASE_START] hot{idx}')",
        "try:",
        "    layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))",
        "    layout_kv = c.get('layout_kv', layout_q)",
        "    inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)",
        "    bt = c.get('block_table')",
        "    if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):",
        "        bt_t = torch.tensor(bt, dtype=torch.int32, device=device)",
        "        if bt_t.dim() == 1: bt_t = bt_t.unsqueeze(0)",
        "        inputs['block_table'] = bt_t",
        "    q,k,v = inputs['q'],inputs['k'],inputs['v']; bt_dev = inputs.get('block_table')",
        "    cu_q = c.get('cu_seqlens_q'); cu_kv = c.get('cu_seqlens_kv')",
        "    sq = c.get('seqused_q'); skv = c.get('seqused_kv')",
        "    cu_q_t = torch.tensor(cu_q,dtype=torch.int32,device=device) if cu_q is not None else None",
        "    cu_kv_t = torch.tensor(cu_kv,dtype=torch.int32,device=device) if cu_kv is not None else None",
        "    sq_t = torch.tensor(sq,dtype=torch.int32,device=device) if sq is not None else None",
        "    skv_t = torch.tensor(skv,dtype=torch.int32,device=device) if skv is not None else None",
        "    _bs = cu_q_t.shape[0]-1 if layout_q=='TND' and cu_q_t is not None else c.get('B',1)",
        "    _msq = c.get('max_seqlen_q')",
        "    if _msq is None:",
        "        _msq = max(sq) if sq else (int((cu_q_t[1:]-cu_q_t[:-1]).max()) if cu_q_t is not None else c.get('S1',1))",
        "    else:",
        "        _msq = int(_msq)",
        "    _msk = c.get('max_seqlen_kv')",
        "    if _msk is None:",
        "        _msk = max(skv) if skv else (int((cu_kv_t[1:]-cu_kv_t[:-1]).max()) if cu_kv_t is not None else c.get('S2',c.get('S1',1)))",
        "    else:",
        "        _msk = int(_msk)",
        "    meta = flash_attn_metadata(",
        "        cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,",
        "        seqused_q=sq_t, seqused_kv=skv_t,",
        "        num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],",
        "        batch_size=_bs, max_seqlen_q=_msq, max_seqlen_kv=_msk,",
        "        mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,",
        "        layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
        "    )",
        "    _m = torch.triu(torch.ones(2048,2048),diagonal=1).to(dtype=torch.int8,device=device) if c.get('mask_mode',0)==3 else None",
        "    torch.npu.synchronize()",
        f"    for _ in range({runs}):",
        "        torch.npu.synchronize()",
        "        flash_attn(q,k,v,block_table=bt_dev,sinks=None,attn_mask=_m,",
        "            metadata=meta,cu_seqlens_q=cu_q_t,cu_seqlens_kv=cu_kv_t,",
        "            seqused_q=sq_t,seqused_kv=skv_t,",
        "            softmax_scale=1/(c['D']**0.5),mask_mode=c.get('mask_mode',0),",
        "            win_left=-1,win_right=-1,max_seqlen_q=_msq,max_seqlen_kv=_msk,",
        "            layout_q=layout_q,layout_kv=layout_kv,layout_out=c.get('layout_out',layout_q),",
        "            return_softmax_lse=0)",
        "    torch.npu.synchronize()",
        f'    _perf_log.write(\'{{"tag":"hot{idx}","ok":true}}\\n\'); _perf_log.flush()',
        f"    print(f'[CASE_END] hot{idx}  OK')",
        "except Exception as e:",
        "    import traceback; traceback.print_exc()",
        f'    _perf_log.write(f\'{{"tag":"hot{idx}","ok":false,"error":{{str(e)}}}}\\n\'); _perf_log.flush()',
        f"    print(f'[CASE_END] hot{idx}  CRASH  {{e}}')",
        "finally:",
        "    torch.npu.synchronize()",
        "    torch.npu.empty_cache()",
        "    import gc; gc.collect()",
        f"print(f'  [done] hot {idx + 1}/{total}')",
        "",
    ]


def _gen_batch_cold(idx, runs, total):
    return [
        f"c = _cold_cases[{idx}]",
        f"name = _cold_names[{idx}]",
        f"print(f'  [batch] {idx + 1}/{total}  {{name}}')",
        f"print('[CASE_START] cold{idx}')",
        "try:",
        "    layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))",
        "    layout_kv = c.get('layout_kv', layout_q)",
        "    cu_q = c.get('cu_seqlens_q'); cu_kv = c.get('cu_seqlens_kv')",
        "    sq = c.get('seqused_q'); skv = c.get('seqused_kv')",
        "    cu_q_t = torch.tensor(cu_q,dtype=torch.int32,device=device) if cu_q is not None else None",
        "    cu_kv_t = torch.tensor(cu_kv,dtype=torch.int32,device=device) if cu_kv is not None else None",
        "    sq_t = torch.tensor(sq,dtype=torch.int32,device=device) if sq is not None else None",
        "    skv_t = torch.tensor(skv,dtype=torch.int32,device=device) if skv is not None else None",
        "    _bs = cu_q_t.shape[0]-1 if layout_q=='TND' and cu_q_t is not None else c.get('B',1)",
        "    _msq = c.get('max_seqlen_q')",
        "    if _msq is None:",
        "        _msq = max(sq) if sq else (int((cu_q_t[1:]-cu_q_t[:-1]).max()) if cu_q_t is not None else c.get('S1',1))",
        "    else:",
        "        _msq = int(_msq)",
        "    _msk = c.get('max_seqlen_kv')",
        "    if _msk is None:",
        "        _msk = max(skv) if skv else (int((cu_kv_t[1:]-cu_kv_t[:-1]).max()) if cu_kv_t is not None else c.get('S2',c.get('S1',1)))",
        "    else:",
        "        _msk = int(_msk)",
        "    meta = flash_attn_metadata(",
        "        cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,",
        "        seqused_q=sq_t, seqused_kv=skv_t,",
        "        num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],",
        "        batch_size=_bs, max_seqlen_q=_msq, max_seqlen_kv=_msk,",
        "        mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,",
        "        layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
        "    )",
        "    _m = torch.triu(torch.ones(2048,2048),diagonal=1).to(dtype=torch.int8,device=device) if c.get('mask_mode',0)==3 else None",
        f"    for _ in range({runs}):",
        "        inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)",
        "        bt = c.get('block_table')",
        "        if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):",
        "            bt_t = torch.tensor(bt, dtype=torch.int32, device=device)",
        "            if bt_t.dim() == 1: bt_t = bt_t.unsqueeze(0)",
        "            inputs['block_table'] = bt_t",
        "        q,k,v = inputs['q'],inputs['k'],inputs['v']; bt_dev = inputs.get('block_table')",
        "        _f1 = torch.randn(32,32768,1,128,dtype=q.dtype,device=device)",
        "        _f2 = torch.randn(32,32768,1,128,dtype=q.dtype,device=device)",
        "        torch.npu.synchronize()",
        "        flash_attn(q,k,v,block_table=bt_dev,sinks=None,attn_mask=_m,",
        "            metadata=meta,cu_seqlens_q=cu_q_t,cu_seqlens_kv=cu_kv_t,",
        "            seqused_q=sq_t,seqused_kv=skv_t,",
        "            softmax_scale=1/(c['D']**0.5),mask_mode=c.get('mask_mode',0),",
        "            win_left=-1,win_right=-1,max_seqlen_q=_msq,max_seqlen_kv=_msk,",
        "            layout_q=layout_q,layout_kv=layout_kv,layout_out=c.get('layout_out',layout_q),",
        "            return_softmax_lse=0)",
        "    torch.npu.synchronize(); torch.npu.empty_cache()",
        f'    _perf_log.write(\'{{"tag":"cold{idx}","ok":true}}\\n\'); _perf_log.flush()',
        f"    print(f'[CASE_END] cold{idx}  OK')",
        "except Exception as e:",
        "    import traceback; traceback.print_exc()",
        f'    _perf_log.write(f\'{{"tag":"cold{idx}","ok":false,"error":{{str(e)}}}}\\n\'); _perf_log.flush()',
        f"    print(f'[CASE_END] cold{idx}  CRASH  {{e}}')",
        "finally:",
        "    torch.npu.synchronize()",
        "    torch.npu.empty_cache()",
        "    import gc; gc.collect()",
        f"print(f'  [done] cold {idx + 1}/{total}')",
        "",
    ]


def run_batch_msprof(script: str, output_dir: Path) -> Optional[Path]:
    """Run batch msprof, return path to op_summary.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_perf_sweep.py").write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = [
        "msprof",
        f"--output={msprof_dir}",
        "--aic-mode=task-based",
        sys.executable,
        str(output_dir / "_perf_sweep.py"),
    ]
    logger.info(f"[msprof] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.error(f"[msprof stderr]\n{result.stderr.strip()}")
    if result.returncode != 0:
        logger.error(f"[msprof] exit={result.returncode}")

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        logger.error("[msprof] no PROF_*")
        return None
    csv_files = sorted(prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        logger.error("[msprof] no op_summary CSV")
        return None
    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    logger.info(f"[msprof] op_summary -> {dst}")
    return dst
