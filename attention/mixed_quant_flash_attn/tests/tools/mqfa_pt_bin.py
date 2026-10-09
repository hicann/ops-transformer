# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Compare / convert between pytest input.pt and TTK case_gen input_N_*.bin.

Both frameworks shove the same logical op signature into a positional tensor
list. The index -> tensor name mapping follows tests/assets/spec.py:

    idx  name
    0    q
    1    k
    2    v
    3    k_descale
    4    v_descale
    5    block_table
    6    cu_seqlens_q
    7    seqused_q
    8    seqused_kv
    9    sinks
    10   attn_mask
    11   metadata          (pytest keeps None; generated on the fly)

pytest stores input.pt as a dict; ttk stores one raw-memory .bin per tensor,
named input_<idx>_<dtype>.bin. We compare them byte-for-byte (raw storage).

Usage:
    python mqfa_pt_bin.py compare  <input.pt> <ttk_case_dir>
    python mqfa_pt_bin.py cmpout   <cpu_or_npu.pt> <ttk_case_dir>
    python mqfa_pt_bin.py pt2bin   <input.pt> <out_dir>
    python mqfa_pt_bin.py bin2pt   <ttk_case_dir> <out.pt> [template.pt]
    python mqfa_pt_bin.py merge    <input.pt> <ttk_case_dir> <out.pt>
    python mqfa_pt_bin.py ptout2bin <cpu_or_npu.pt> <out_dir>
    python mqfa_pt_bin.py bin2ptout <ttk_case_dir> <out.pt>

Output layout (pytest cpu/npu .pt is a dict; ttk dumps golden_N_<dtype>__shape_*.bin):
    pytest key      ttk bin
    attn_out        golden_0_<dtype>__shape_*.bin
    softmax_lse     golden_1_<dtype>__shape_*.bin
"""

import os
import sys
import hashlib

import numpy as np
import torch


IDX2NAME = {
    0: "q",
    1: "k",
    2: "v",
    3: "k_descale",
    4: "v_descale",
    5: "block_table",
    6: "cu_seqlens_q",
    7: "seqused_q",
    8: "seqused_kv",
    9: "sinks",
    10: "attn_mask",
    11: "metadata",
}


def tensor_to_bytes(t):
    """Raw memory bytes of a CPU tensor, normalizing torch-specific dtypes
    that numpy does not know (float8_e8m0 -> uint8 view)."""
    if t is None:
        return b""
    t = t.detach().cpu().contiguous()
    if t.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        return t.view(torch.uint8).numpy().tobytes()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def md5(data):
    return hashlib.md5(data).hexdigest()


def load_pt_input(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            tensors[k] = v
    return tensors


def find_bin_files(ttk_case_dir):
    """Return {idx: bin_path} parsed from input_<idx>_*.bin names."""
    out = {}
    for f in sorted(os.listdir(ttk_case_dir)):
        if not f.startswith("input_"):
            continue
        try:
            idx = int(f.split("_")[1])
        except (ValueError, IndexError):
            continue
        out[idx] = os.path.join(ttk_case_dir, f)
    return out


def compare(input_pt, ttk_case_dir):
    tensors = load_pt_input(input_pt)
    bins = find_bin_files(ttk_case_dir)
    print(f"pytest: {input_pt}")
    print(f"ttk   : {ttk_case_dir}\n")
    n_missing = 0
    for idx in sorted(set(IDX2NAME) | set(bins)):
        name = IDX2NAME.get(idx, f"?_{idx}")
        if idx not in bins:
            print(f"[{idx:2d}] {name:14s}  ttK BIN ABSENT")
            n_missing += 1
            continue
        bin_path = bins[idx]
        pt_t = tensors.get(name)
        bin_raw = open(bin_path, "rb").read()
        pt_raw = tensor_to_bytes(pt_t)
        pt_md5 = md5(pt_raw)
        bin_md5 = md5(bin_raw)
        if pt_raw == bin_raw or pt_md5 == bin_md5:
            status = "SAME"
        else:
            if pt_t is not None and pt_raw:
                same_len = (
                    "len-match"
                    if len(pt_raw) == len(bin_raw)
                    else f"len {len(pt_raw)} vs {len(bin_raw)}"
                )
            else:
                same_len = "pt-empty"
            status = "DIFF (" + same_len + ")"
        print(f"[{idx:2d}] {name:14s}  md5(pt)={pt_md5}  md5(bin)={bin_md5}  {status}")
    print(f"\n{len(IDX2NAME) - n_missing} bins checked, {n_missing} ttk bins missing.")
    sys.exit(1 if n_missing else 0)


def find_golden_files(ttk_case_dir):
    """Return {idx: bin_path} parsed from golden_<idx>_<dtype>__shape_*.bin names."""
    out = {}
    for f in sorted(os.listdir(ttk_case_dir)):
        if not f.startswith("golden_"):
            continue
        try:
            idx = int(f.split("_")[1])
        except (ValueError, IndexError):
            continue
        out[idx] = os.path.join(ttk_case_dir, f)
    return out


def load_golden_bin(bin_path):
    """Load a ttk golden .bin into a flat numpy float32 array by dtype hint in name."""
    base = os.path.basename(bin_path)
    dt = base.split("_")[2]  # golden_<idx>_<dtype>__shape_...
    npt = {"float16": np.float16, "float32": np.float32, "bfloat16": bf16_dtype()}.get(
        dt, np.float32
    )
    return np.fromfile(bin_path, dtype=npt).astype(np.float32)


def bf16_dtype():
    try:
        from ml_dtypes import bfloat16

        return bfloat16
    except Exception:
        return None


def cmpout(pt_out_path, ttk_case_dir, rtol=1e-2, atol=1e-3):
    """Compare pytest output .pt (cpu or npu) against ttk golden bins.

    pytest output keys -> ttk golden index:
        attn_out    -> golden_0
        softmax_lse -> golden_1
    Numeric comparison (inputs differ, so never byte-exact). Reports max abs/rel
    error and pass/fail per output. Exit 0 if all pass.
    """
    data = torch.load(pt_out_path, map_location="cpu", weights_only=False)
    goldens = find_golden_files(ttk_case_dir)
    print(f"pytest: {pt_out_path}")
    print(f"ttk   : {ttk_case_dir}\n")
    keys = {0: "attn_out", 1: "softmax_lse"}
    n_missing = 0
    failed = 0

    for k in keys.values():
        if k not in data:
            hint = ""
            if "input" in os.path.basename(pt_out_path):
                hint = "  (这是 _input.pt，不含输出；请传 _cpu.pt 或 _npu.pt)"
            else:
                avail = (
                    ", ".join(k2 for k2 in ("attn_out", "softmax_lse") if k2 in data)
                    or "无 attn_out/softmax_lse"
                )
                hint = f"  (该文件只有: {avail})"
            print(f"    WARN: pytest 输出缺少 key '{k}'{hint}")
            n_missing += 1

    for idx, name in keys.items():
        pt_t = data.get(name)
        if pt_t is None:
            print(f"[{idx}] {name:14s}  ABSENT in pytest output")
            continue
        if idx not in goldens:
            print(f"[{idx}] {name:14s}  ABSENT in ttk goldens")
            n_missing += 1
            continue
        pt = pt_t.detach().cpu().float().numpy().reshape(-1)
        tb = load_golden_bin(goldens[idx])
        if pt.size != tb.size:
            print(f"[{idx}] {name:14s}  size mismatch pt={pt.size} bin={tb.size}  FAIL")
            failed += 1
            continue
        # pytest 用 +inf 表示全 mask 行，ttk 用 -inf，同一语义；任何 inf/nan 视为一致
        pt_inf = ~np.isfinite(pt)
        tb_inf = ~np.isfinite(tb)
        inf_agree = np.array_equal(pt_inf, tb_inf)
        finite = np.isfinite(pt) & np.isfinite(tb)
        abs_err = np.abs(pt[finite] - tb[finite]) if finite.any() else np.array([0.0])
        rel_err = (
            abs_err / np.maximum(np.abs(tb[finite]), atol)
            if finite.any()
            else np.array([0.0])
        )
        passed = bool(inf_agree and (abs_err <= atol).all() and (rel_err <= rtol).all())
        print(
            f"[{idx}] {name:14s}  max_abserr={abs_err.max():.3e} "
            f"max_relerr={rel_err.max():.3e}  {'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            failed += 1
    print(
        f"\n{'ALL PASS' if failed == 0 and n_missing == 0 else 'FAILED'}: "
        f"{failed} failed, {n_missing} missing golden."
    )
    sys.exit(1 if (failed or n_missing) else 0)


# 输出 key -> ttk golden 序号（与 Output layout 一致）
OUT_KEY2IDX = {"attn_out": 0, "softmax_lse": 1}


def _fmt_shape(s):
    return "x".join(str(x) for x in s)


def _tensor_to_bin_file(t, base_dtype):
    """把 torch 张量写成一个 raw-memory bin。返回 (data, dtype_str)。"""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8).numpy(), "float8_e8m0"
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy(), "bfloat16"
    return t.numpy(), base_dtype


def ptout2bin(pt_out_path, out_dir):
    """pytest 输出 .pt (cpu/npu, dict: attn_out/softmax_lse) -> ttk golden bins。"""
    os.makedirs(out_dir, exist_ok=True)
    data = torch.load(pt_out_path, map_location="cpu", weights_only=False)
    for key, idx in OUT_KEY2IDX.items():
        t = data.get(key)
        if t is None:
            print(f"[{idx}] {key:14s}  ABSENT in pytest output, skip")
            continue
        base = (
            "float32"
            if key == "softmax_lse"
            else (
                "float16" if t.dtype == torch.float16 else str(t.dtype).split(".")[-1]
            )
        )
        arr, dstr = _tensor_to_bin_file(t, base)
        shape = "x".join(str(x) for x in t.shape)
        path = os.path.join(out_dir, f"golden_{idx}_{dstr}__shape_{shape}.bin")
        arr.tofile(path)
        print(f"[{idx}] {key:14s} shape={tuple(t.shape)} {dstr}  -> {path}")


def bin2ptout(ttk_case_dir, out_pt):
    """ttk golden bins -> pytest 输出 .pt (dict: attn_out/softmax_lse)。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    goldens = find_golden_files(ttk_case_dir)
    out = {}
    for key, idx in OUT_KEY2IDX.items():
        if idx not in goldens:
            print(f"[{idx}] {key:14s}  ABSENT in ttk goldens, skip")
            continue
        raw = open(goldens[idx], "rb").read()
        if not raw:
            print(f"[{idx}] {key:14s}  empty, skip")
            continue
        arr = load_golden_bin(goldens[idx])
        shape = parse_shape_from_name(os.path.basename(goldens[idx]))
        if key == "softmax_lse":
            t = (
                torch.from_numpy(arr).float().reshape(shape)
                if shape
                else torch.from_numpy(arr).float()
            )
        else:
            t = (
                torch.from_numpy(arr).half().reshape(shape)
                if shape
                else torch.from_numpy(arr).half()
            )
        out[key] = t
        print(
            f"[{idx}] {key:14s} shape={tuple(t.shape)} {t.dtype}  <- {os.path.basename(goldens[idx])}"
        )
    torch.save(out, out_pt)
    print(f"wrote {out_pt}")


def parse_shape_from_name(name):
    """golden_N_<dtype>__shape_128x2x40x128.bin -> (128,2,40,128)"""
    marker = "__shape_"
    if marker not in name:
        return None
    part = name.split(marker, 1)[1].rsplit(".", 1)[0]
    dims = [int(x) for x in part.split("x") if x != ""]
    return tuple(dims) if dims else None


def pt2bin(input_pt, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tensors = load_pt_input(input_pt)
    for idx, name in IDX2NAME.items():
        t = tensors.get(name)
        if t is None:
            open(os.path.join(out_dir, f"input_{idx}_none.bin"), "wb").close()
            print(f"[{idx:2d}] {name:14s}  -> none (empty)")
            continue
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.float8_e8m0fnu:
            npd = t.view(torch.uint8).numpy()
            dtype_str = "float8_e8m0"
        else:
            npd = t.numpy()
            dtype_str = str(t.dtype).split(".")[-1]
        path = os.path.join(out_dir, f"input_{idx}_{dtype_str}.bin")
        npd.tofile(path)
        print(f"[{idx:2d}] {name:14s}  shape={tuple(t.shape)} {dtype_str}  -> {path}")


def merge(target_pt, ttk_case_dir, out_pt):
    """Build a pytest-style input.pt whose TENSORS come from ttk bins but whose
    scalar params/keys are taken from target_pt (the original pytest input.pt).
    This lets pytest run on ttk's exact inputs."""
    orig = torch.load(target_pt, map_location="cpu", weights_only=False)
    new = {}
    for k, v in orig.items():
        if torch.is_tensor(v):
            new[k] = None
        else:
            new[k] = v
    templ_dir = ttk_case_dir if os.path.isdir(ttk_case_dir) else None
    bins = find_bin_files(templ_dir) if templ_dir else {}
    for idx, name in IDX2NAME.items():
        if idx in bins:
            new[name] = restore_bin_tensor(bins[idx], orig.get(name))
        else:
            new[name] = orig.get(name)
    os.makedirs(os.path.dirname(os.path.abspath(out_pt)) or ".", exist_ok=True)
    torch.save(new, out_pt)
    print(
        f"wrote {out_pt} (tensors from ttk, params from {os.path.basename(target_pt)})"
    )
    for k, v in new.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    print("  ... params preserved (scalar keys not printed)")


def restore_bin_tensor(bin_path, tmpl):
    raw = open(bin_path, "rb").read()
    base = os.path.basename(bin_path)
    if not raw:
        return None
    dtype_str = base.rsplit(".", 1)[0].split("_")[-1]
    if torch.is_tensor(tmpl):
        if tmpl.dtype == torch.float8_e8m0fnu:
            return (
                torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
                .view(torch.float8_e8m0fnu)
                .reshape(tmpl.shape)
            )
        if tmpl.dtype == torch.bfloat16:
            return (
                torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy())
                .view(torch.bfloat16)
                .reshape(tmpl.shape)
            )
        npt = np.dtype(str(tmpl.dtype).split(".")[-1])
        return torch.from_numpy(np.frombuffer(raw, dtype=npt).copy()).reshape(
            tmpl.shape
        )
    npt = {
        "float16": np.float16,
        "float32": np.float32,
        "uint8": np.uint8,
        "int8": np.int8,
        "int32": np.int32,
        "int16": np.int16,
    }.get(dtype_str)
    arr = np.frombuffer(raw, dtype=npt if npt is not None else np.uint8).copy()
    if dtype_str == "float8_e8m0":
        return torch.from_numpy(arr).view(torch.float8_e8m0fnu)
    return torch.from_numpy(arr)


def bin2pt(ttk_case_dir, out_pt, template_pt=None):
    """Convert ttk input_N_*.bin back to a pytest-style input.pt.

    Raw bins carry no shape/dtype-header, so pass the original pytest
    input.pt via template_pt to restore exact shapes and scalar params.
    Without a template, tensors come back flattened with dtype from filename.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    template = load_pt_input(template_pt) if template_pt else None
    bins = find_bin_files(ttk_case_dir)
    save = {}
    for idx, name in IDX2NAME.items():
        if idx not in bins:
            save[name] = None
            continue
        path = bins[idx]
        raw = open(path, "rb").read()
        if not raw:
            save[name] = None
            continue
        t = template.get(name) if template else None
        # dtype encoded in filename: <base>_<idx>_<dtype>.bin
        base = os.path.basename(path)
        dtype_str = base.rsplit(".", 1)[0].split("_")[-1]
        if t is not None and torch.is_tensor(t):
            npt = {
                "float16": np.float16,
                "float32": np.float32,
                "uint8": np.uint8,
                "int8": np.int8,
                "int32": np.int32,
                "int16": np.int16,
            }.get(dtype_str)
            if t.dtype == torch.float8_e8m0fnu:
                arr = np.frombuffer(raw, dtype=np.uint8).copy()
                store = torch.from_numpy(arr).view(torch.float8_e8m0fnu)
            elif t.dtype == torch.bfloat16:
                arr = np.frombuffer(raw, dtype=np.int16).copy()
                store = torch.from_numpy(arr).view(torch.bfloat16)
            else:
                npt = npt if npt is not None else np.dtype(str(t.dtype).split(".")[-1])
                arr = np.frombuffer(raw, dtype=npt).copy()
                store = torch.from_numpy(arr)
            save[name] = store.reshape(t.shape)
        else:
            npt = {
                "float16": np.float16,
                "float32": np.float32,
                "uint8": np.uint8,
                "int8": np.int8,
                "int32": np.int32,
                "int16": np.int16,
            }.get(dtype_str)
            arr = np.frombuffer(raw, dtype=npt if npt is not None else np.uint8).copy()
            if dtype_str == "float8_e8m0":
                save[name] = torch.from_numpy(arr).view(torch.float8_e8m0fnu)
            else:
                save[name] = torch.from_numpy(arr)
    torch.save({k: v for k, v in save.items() if v is not None}, out_pt)
    print(f"wrote {out_pt}")
    for k, v in save.items():
        print(f"  {k}: {None if v is None else tuple(v.shape)}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif cmd == "cmpout":
        cmpout(sys.argv[2], sys.argv[3])
    elif cmd == "ptout2bin":
        ptout2bin(sys.argv[2], sys.argv[3])
    elif cmd == "bin2ptout":
        bin2ptout(sys.argv[2], sys.argv[3])
    elif cmd == "pt2bin":
        pt2bin(sys.argv[2], sys.argv[3])
    elif cmd == "bin2pt":
        bin2pt(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "merge":
        merge(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)
