#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unified MC2 DFX dump parser.

Pass one dump file to parse only that file, or pass an operator name and an
optional dump directory to parse a complete dump directory.  The script
discovers and dispatches tiling_data, args_info, workspace segments, CCU
xn/cke address tables, and URMA context files automatically.  Both modes
create one independent readable TXT report for every parsed source dump; no
combined report or NumPy output is generated.
"""

import argparse
import importlib
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

PROFILES = {
    "allgathermatmulv2": {
        "name": "AllGatherMatmulV2",
        "backend": "parse_all_gather_matmul_v2_tiling.py",
        "module": "parse_all_gather_matmul_v2_tiling",
        "path": "CCU",
    },
    "allgathermatmulv3": {
        "name": "AllGatherMatmulV3",
        "backend": "parse_all_gather_matmul_v3_tiling.py",
        "module": "parse_all_gather_matmul_v3_tiling",
        "path": "AIV + URMA",
    },
    "alltoallmatmul": {
        "name": "AlltoAllMatmul",
        "backend": "parse_allto_all_matmul_tiling.py",
        "module": "parse_allto_all_matmul_tiling",
        "path": "mainline CCU / Apace CCU (auto-detect)",
    },
    "alltoallmatmulv2": {
        "name": "AlltoAllMatmulV2",
        "backend": "parse_allto_all_matmul_v2_tiling.py",
        "module": "parse_allto_all_matmul_v2_tiling",
        "path": "AIV + URMA",
    },
    "matmulreducescatterv2": {
        "name": "MatmulReduceScatterV2",
        "backend": "parse_matmul_reduce_scatter_v2_tiling.py",
        "module": "parse_matmul_reduce_scatter_v2_tiling",
        "path": "CCU",
    },
}

ALIASES = {
    "agmmv2": "allgathermatmulv2",
    "agmmv3": "allgathermatmulv3",
    "a2amm": "alltoallmatmul",
    "a2ammv2": "alltoallmatmulv2",
    "alltoallmatmulapace": "alltoallmatmul",
    "alltoallmatmulmainline": "alltoallmatmul",
    "matmulreducescatter": "matmulreducescatterv2",
    "mmrs": "matmulreducescatterv2",
    "mmrsv2": "matmulreducescatterv2",
}

FILE_KINDS = (
    ("tiling", re.compile(r"^tiling_data_", re.IGNORECASE)),
    ("args", re.compile(r"^args_info_", re.IGNORECASE)),
    ("xn", re.compile(r"^xn_addr_info_", re.IGNORECASE)),
    ("cke", re.compile(r"^cke_addr_info_", re.IGNORECASE)),
    ("comm_context", re.compile(r"^comm_context_", re.IGNORECASE)),
    ("workspace", re.compile(r"^workspace_seg\d+_type\d+_", re.IGNORECASE)),
)

KIND_ORDER = {
    "tiling": 0,
    "args": 1,
    "xn": 2,
    "cke": 3,
    "comm_context": 4,
    "workspace": 5,
}

WORKSPACE_SEG_TYPE_NAMES = {
    0: "LIB_API",
    1: "ND2NZ",
    2: "GATHER",
    3: "GATHER_SCALE1",
    4: "GATHER_SCALE",
    5: "COMM_OUT",
    6: "PERMUTE_OUT",
    7: "BIAS",
    8: "MM_RESULT",
    9: "RECV_BUF",
    10: "COMM_INT8",
    11: "DYNAMIC_QUANT",
    12: "STATE_DUMP",
    13: "MM_WORKSPACE",
    255: "RESERVED",
}

WORKSPACE_SEG_DESCRIPTIONS = {
    0: "系统库预留 Workspace，不属于算子用户数据",
    1: "ND 到 NZ 格式转换的中间缓冲区",
    2: "Gather 通信数据缓冲区",
    3: "Gather 的第一路 Scale 缓冲区",
    4: "Gather Scale 缓冲区",
    5: "通信输出缓冲区",
    6: "转置输出缓冲区",
    7: "Bias 缓冲区",
    8: "Matmul 结果；在部分融合通路中也作为通信发送缓冲区",
    9: "通信接收缓冲区",
    10: "INT8 通信临时缓冲区",
    11: "动态量化临时缓冲区",
    12: "每核 StateDump，记录执行位置和通信 commit/wait 情况",
    13: "Matmul 内部临时 Workspace，不是 Matmul 结果，元素类型和逻辑形状不固定",
    255: "对齐或预留空间",
}

WORKSPACE_PREVIEW_SIZE = 64

# Directory mode may contain several calls from the same rank. Cache the
# TilingData layout result by file so RuntimeInfo parsing does not repeatedly
# scan the same binary or emit the same fallback warning.
_AIC_CORE_NUM_CACHE = {}


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_profile(operator_name):
    key = normalize(operator_name)
    key = ALIASES.get(key, key)
    if key not in PROFILES:
        choices = ", ".join(profile["name"] for profile in PROFILES.values())
        raise ValueError(
            f"不支持的算子名 {operator_name!r}。支持: {choices}; 别名: mmrs/a2amm/agmmv2/agmmv3"
        )
    return key, PROFILES[key]


def classify_file(path):
    for kind, pattern in FILE_KINDS:
        if pattern.match(path.name):
            return kind
    return None


def matches_operator(path, profile_key):
    normalized_name = normalize(path.name)
    if profile_key == "alltoallmatmul":
        return (
            "alltoallmatmul" in normalized_name
            and "alltoallmatmulv2" not in normalized_name
        )
    if profile_key == "matmulreducescatterv2":
        return "matmulreducescatter" in normalized_name
    return profile_key in normalized_name


def detect_profile_from_file(path):
    """Infer the operator profile from one dump filename or sibling TilingData."""
    matched_keys = [key for key in PROFILES if matches_operator(path, key)]
    if not matched_keys:
        normalized_name = normalize(path.name)
        for alias in sorted(ALIASES, key=len, reverse=True):
            if alias in normalized_name:
                matched_keys = [ALIASES[alias]]
                break
    if not matched_keys and path.parent.is_dir():
        sibling_keys = set()
        for sibling in path.parent.iterdir():
            if not sibling.is_file() or classify_file(sibling) != "tiling":
                continue
            sibling_keys.update(
                key for key in PROFILES if matches_operator(sibling, key)
            )
        matched_keys = sorted(sibling_keys)
    if len(matched_keys) != 1:
        if matched_keys:
            names = ", ".join(PROFILES[key]["name"] for key in matched_keys)
            raise ValueError(f"无法唯一确定 {path.name} 所属的算子，候选: {names}")
        raise ValueError(
            f"无法从文件名 {path.name!r} 识别算子；文件名中应包含受支持的标准算子名"
        )
    profile_key = matched_keys[0]
    return profile_key, PROFILES[profile_key]


def discover_files(dump_dir, output_dir, profile_key):
    all_files = []
    for path in dump_dir.rglob("*"):
        if not path.is_file() or output_dir in path.parents:
            continue
        # Skip analysis outputs from any previous run, even one rooted at a
        # different dump_dir, so stale TXT reports are never re-parsed as dumps.
        if any(parent.name == output_dir.name for parent in path.parents):
            continue
        all_files.append(path)
    files = []
    for path in all_files:
        kind = classify_file(path)
        if kind is not None and matches_operator(path, profile_key):
            files.append((kind, path))
    return sorted(files, key=lambda item: (KIND_ORDER[item[0]], str(item[1]).lower()))


def read_data(path):
    with path.open("rb") as file_handle:
        return file_handle.read()


def read_preview(path, size=256):
    with path.open("rb") as file_handle:
        return file_handle.read(size)


def emit_decimal_byte_preview(lines, data, echo=True):
    """Print an unambiguous decimal preview when the segment element type is unknown."""
    append = (lambda value: emit(lines, value)) if echo else lines.append
    append("previewType: uint8 raw bytes")
    append(f"previewRange: byte[0:{len(data)}]")
    append(f"previewAllZero: {not any(data)}")
    append("decimalPreview (each value is one raw uint8 byte):")
    for offset in range(0, len(data), 16):
        row = data[offset : offset + 16]
        values = " ".join(f"{value:3d}" for value in row)
        append(f"  byte[{offset:04d}:{offset + len(row):04d}] : {values}")


def segment_identity(path, prefix):
    match = re.match(rf"^{prefix}_seg(\d+)_type(\d+)_", path.name, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def infer_aic_core_num(profile, tiling_file):
    cache_key = (profile["name"], tiling_file.resolve())
    if cache_key in _AIC_CORE_NUM_CACHE:
        return _AIC_CORE_NUM_CACHE[cache_key]

    core_num = None
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = importlib.import_module(profile["module"])
        data = read_data(tiling_file)
        name = profile["name"]
        if name == "AllGatherMatmulV3":
            core_num = (
                module.parse_tiling_data(data).get("mmTile", {}).get("usedCoreNum")
            )
        elif name == "AlltoAllMatmulV2":
            core_num = (
                module.parse_tiling_data(data)
                .get("tileQbmmTilingData", {})
                .get("usedCoreNum")
            )
        elif name == "AlltoAllMatmul":
            tiling_format = module.detect_tiling_format(data)
            if tiling_format == "apace":
                result = module.parse_apace_tiling_data(data)
                core_num = result.get("tileQbmmTilingData", {}).get("usedCoreNum")
            else:
                info = module.parse_allto_all_matmul_tiling_info(
                    data, module.OFFSET_TILING_INFO
                )
                core_num = info.get("aicCoreNum")
        else:
            layout = (
                module.compute_layout()
                if name == "AllGatherMatmulV2"
                else module.compute_mrs_v2_layout()
            )
            is_quant = module.detect_quant_mode(data, layout)
            core_num = (
                module.parse_tiling_data(data, layout, is_quant)
                .get("param", {})
                .get("aicCoreNum")
            )
    except Exception:
        core_num = None
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)
    _AIC_CORE_NUM_CACHE[cache_key] = core_num
    return core_num


def infer_apace_tiling_files(profile, tiling_files):
    """Return the exact AlltoAllMatmul TilingData files using Apace CCU."""
    if profile["name"] != "AlltoAllMatmul":
        return set()
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        module = importlib.import_module(profile["module"])
        apace_files = set()
        for tiling_file in tiling_files:
            try:
                if module.detect_tiling_format(read_data(tiling_file)) == "apace":
                    apace_files.add(tiling_file)
            except (OSError, struct.error, ValueError, KeyError):
                continue
        return apace_files
    except ImportError:
        return set()
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)


def operator_suffix(path, profile):
    """Return the invocation suffix shared by tiling/workspace files."""
    candidates = [profile["name"]]
    if profile["name"] == "MatmulReduceScatterV2":
        candidates.append("MatmulReduceScatter")
    lowered = path.name.lower()
    for candidate in candidates:
        index = lowered.find(candidate.lower())
        if index >= 0:
            return lowered[index + len(candidate) :]
    return lowered


def build_core_index(profile, tiling_files):
    exact = {}
    per_parent = {}
    values = set()
    by_file = {}
    for tiling_file in tiling_files:
        core_num = infer_aic_core_num(profile, tiling_file)
        if core_num is None:
            continue
        by_file[tiling_file] = core_num
        exact[(tiling_file.parent, operator_suffix(tiling_file, profile))] = core_num
        per_parent.setdefault(tiling_file.parent, set()).add(core_num)
        values.add(core_num)
    return exact, per_parent, values, by_file


def core_num_for_file(path, profile, core_index):
    exact, per_parent, values = core_index[:3]
    core_num = exact.get((path.parent, operator_suffix(path, profile)))
    if core_num is not None:
        return core_num
    parent_values = per_parent.get(path.parent, set())
    if len(parent_values) == 1:
        return next(iter(parent_values))
    if len(values) == 1:
        return next(iter(values))
    return None


def trailing_sequence(path):
    """Return the final numeric sequence used by dump filenames as a call timestamp."""
    values = re.findall(r"\d+", path.name)
    return int(values[-1]) if values else None


def nearest_tiling_file(path, profile, tiling_files):
    """Select the same/nearest TilingData file without requiring another CLI argument."""
    if not tiling_files:
        return None
    suffix = operator_suffix(path, profile)
    for tiling_file in tiling_files:
        if operator_suffix(tiling_file, profile) == suffix:
            return tiling_file
    target_sequence = trailing_sequence(path)
    if target_sequence is not None:
        candidates = []
        for tiling_file in tiling_files:
            sequence = trailing_sequence(tiling_file)
            if sequence is not None:
                candidates.append(
                    (abs(sequence - target_sequence), str(tiling_file), tiling_file)
                )
        if candidates:
            return min(candidates)[2]
    return tiling_files[0] if len(tiling_files) == 1 else None


def related_tiling_files(path, profile_key):
    """Find TilingData beside a single input file for automatic StateDump context."""
    if not path.parent.is_dir():
        return []
    return sorted(
        sibling
        for sibling in path.parent.iterdir()
        if sibling.is_file()
        and classify_file(sibling) == "tiling"
        and matches_operator(sibling, profile_key)
    )


def group_tiling_files_by_parent(tiling_files):
    """Index TilingData by rank directory for per-invocation matching."""
    grouped = {}
    for tiling_file in tiling_files:
        grouped.setdefault(tiling_file.parent, []).append(tiling_file)
    for values in grouped.values():
        values.sort()
    return grouped


def workspace_runtime_context(
    path, profile, tiling_files, core_index, apace_tiling_files
):
    """Match one StateDump segment to its nearest sibling TilingData."""
    selected_tiling = nearest_tiling_file(path, profile, tiling_files)
    if selected_tiling is not None:
        by_file = core_index[3] if len(core_index) > 3 else {}
        core_num = by_file.get(selected_tiling)
        if core_num is None:
            core_num = infer_aic_core_num(profile, selected_tiling)
        return core_num, selected_tiling in apace_tiling_files, selected_tiling
    return core_num_for_file(path, profile, core_index), False, None


def parse_address_table(input_path):
    """Parse a little-endian uint64 XN/CKE table into readable text lines."""
    byte_size = input_path.stat().st_size
    row_size = 64 * 8
    if byte_size % row_size != 0:
        raise ValueError(f"文件大小 {byte_size} 不是完整 512 字节行的整数倍")
    data = read_data(input_path)
    table = [
        struct.unpack_from("<64Q", data, offset)
        for offset in range(0, len(data), row_size)
    ]
    nonzero = sum(1 for row in table for value in row if value != 0)
    lines = [
        f"source: {input_path.name}",
        "dtype: uint64 (little-endian)",
        f"shape: ({len(table)}, 64)",
        f"nonzero: {nonzero}",
    ]
    for index, row in enumerate(table):
        cells = " ".join(f"0x{value:016x}" for value in row)
        lines.append(f"row[{index:04d}] {cells}")
    return len(table), len(table) * 64, nonzero, lines


def backend_command(profile, kind, path, aic_core_num, use_apace=False):
    command = [sys.executable, str(SCRIPT_DIR / profile["backend"]), str(path)]
    if kind == "args":
        command.append("--args")
    elif kind == "workspace":
        command.append("--workspace")
        if aic_core_num is not None:
            command.extend(("--aic-core-num", str(aic_core_num)))
        if use_apace:
            command.append("--apace")
    elif kind == "comm_context":
        command.append("--comm-ctx")
    return command


def run_backend(profile, kind, path, aic_core_num, use_apace=False):
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    if kind != "tiling":
        # Args/Workspace parsing does not use TilingData offsets. Avoid an
        # unrelated CANN-layout fallback warning in those per-file reports.
        child_env["MC2_LAYOUT_QUIET_FALLBACK"] = "1"
    process = subprocess.run(
        backend_command(profile, kind, path, aic_core_num, use_apace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        check=False,
    )
    return process.returncode, process.stdout.rstrip()


def emit(lines, value=""):
    print(value)
    lines.append(value)


def text_output_path(output_dir, relative_path):
    """Return the independent TXT path corresponding to one source dump."""
    output_path = output_dir / relative_path.parent / f"{relative_path.name}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_text_result(output_path, profile, kind, relative_path, lines):
    """Save one source file's readable parsing result."""
    header = [
        f"operator: {profile['name']}",
        f"kind: {kind}",
        f"source: {relative_path}",
        "=" * 88,
    ]
    output_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def parse_single_file(input_path, parser):
    """Parse one dump file and save one corresponding TXT report.

    Single-file mode deliberately accepts only one positional file argument.
    Operator/type detection and optional sibling TilingData lookup are handled
    internally.  The TXT is placed below the input file's directory so the raw
    Dump is never overwritten:
      <input_dir>/mc2_dump_analysis/<Operator>/<input_name>.txt
    """
    kind = classify_file(input_path)
    if kind is None:
        parser.error(
            "不支持的文件名；支持 tiling_data_*、args_info_*、workspace_segN_typeM_*、"
            "xn_addr_info_*、cke_addr_info_* 和 comm_context_*"
        )
    try:
        profile_key, profile = detect_profile_from_file(input_path)
    except ValueError as error:
        parser.error(str(error))

    # Keep single-file output naming identical to directory mode: append
    # ".txt" to the complete source name, including its original suffix.
    output_dir = input_path.parent / "mc2_dump_analysis" / profile["name"]
    output_path = text_output_path(output_dir, Path(input_path.name))
    result_lines = []

    print("=" * 88)
    print(f"MC2 single-file analysis: {profile['name']} | kind: {kind}")
    print(f"source: {input_path}")
    print("=" * 88)

    if kind in ("xn", "cke"):
        try:
            _, _, _, parsed_lines = parse_address_table(input_path)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        for line in parsed_lines:
            emit(result_lines, line)
        save_text_result(
            output_path, profile, kind, Path(input_path.name), result_lines
        )
        print(f"saved TXT: {output_path}")
        return 0

    if kind == "workspace":
        identity = segment_identity(input_path, "workspace")
        if identity is not None and identity[1] != 12:
            seg_index, seg_type = identity
            parsed_lines = [
                f"segIdx: {seg_index}",
                f"segType: {seg_type} ({WORKSPACE_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})",
                f"dataMeaning: {WORKSPACE_SEG_DESCRIPTIONS.get(seg_type, '未知 Workspace 段')}",
                f"fileSize: {input_path.stat().st_size} bytes",
            ]
            emit_decimal_byte_preview(
                parsed_lines,
                read_preview(input_path, WORKSPACE_PREVIEW_SIZE),
                echo=False,
            )
            parsed_lines.append(
                "typedViewHint: 使用 parse_matrix.py 并指定 dtype/rows/cols 查看真实数值"
            )
            for line in parsed_lines:
                emit(result_lines, line)
            save_text_result(
                output_path, profile, kind, Path(input_path.name), result_lines
            )
            print(f"saved TXT: {output_path}")
            return 0

    aic_core_num = None
    use_apace = False
    if kind == "workspace":
        tiling_files = related_tiling_files(input_path, profile_key)
        core_index = build_core_index(profile, tiling_files)
        apace_tiling_files = infer_apace_tiling_files(profile, tiling_files)
        aic_core_num, use_apace, selected_tiling = workspace_runtime_context(
            input_path, profile, tiling_files, core_index, apace_tiling_files
        )
        if selected_tiling is not None:
            emit(result_lines, f"relatedTilingData: {selected_tiling.name}")

    return_code, output = run_backend(
        profile, kind, input_path, aic_core_num, use_apace
    )
    if output:
        for line in output.splitlines():
            emit(result_lines, line)
    if kind == "comm_context":
        file_size = input_path.stat().st_size
        emit(result_lines, "")
        emit(
            result_lines,
            "rawDataHint: 如需按十进制查看 comm_context 原始字节，可使用 parse_matrix.py：",
        )
        emit(
            result_lines,
            f'python3 "{SCRIPT_DIR / "parse_matrix.py"}" "{input_path}" '
            f"--dtype uint8 --rows 1 --cols {file_size} --row 0 --col-range 0 64",
        )
    if return_code != 0:
        emit(result_lines, f"ERROR: backend exited with code {return_code}")
    save_text_result(output_path, profile, kind, Path(input_path.name), result_lines)
    print(f"saved TXT: {output_path}")
    return return_code


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="自动解析单个 MC2 Dump 文件，或按算子名批量解析 Dump 目录",
        epilog=(
            "单文件: python3 parse_mc2_dump.py /path/to/workspace_seg2_type12_MatmulReduceScatterV2.xxx\n"
            "批量:   python3 parse_mc2_dump.py AllGatherMatmulV3 /path/to/dump"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="单个 Dump 文件路径；批量解析时填写算子名或别名，例如 AlltoAllMatmul、mmrs",
    )
    parser.add_argument(
        "dump_dir", nargs="?", help="批量解析的 Dump 根目录，省略时为当前目录"
    )
    args = parser.parse_args(argv)

    target_path = Path(args.target).expanduser()
    looks_like_dump = classify_file(target_path) is not None
    if args.dump_dir is None and (target_path.exists() or looks_like_dump):
        input_path = target_path.resolve()
        if not input_path.is_file():
            parser.error(f"Dump 文件不存在: {input_path}")
        return parse_single_file(input_path, parser)

    try:
        profile_key, profile = resolve_profile(args.target)
    except ValueError as error:
        parser.error(str(error))

    dump_dir = Path(args.dump_dir or ".").expanduser().resolve()
    if not dump_dir.is_dir():
        parser.error(f"dump 目录不存在: {dump_dir}")
    output_dir = dump_dir / "mc2_dump_analysis" / profile["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Old versions wrote one combined report.  Remove it so users do not read
    # stale results after switching to per-source TXT output.
    (output_dir / "analysis.txt").unlink(missing_ok=True)

    files = discover_files(dump_dir, output_dir, profile_key)
    if not files:
        parser.error(f"在 {dump_dir} 中没有找到 {profile['name']} 的受支持 dump 文件")

    tiling_files = [path for kind, path in files if kind == "tiling"]
    core_index = build_core_index(profile, tiling_files)
    tiling_files_by_parent = group_tiling_files_by_parent(tiling_files)
    apace_tiling_files = infer_apace_tiling_files(profile, tiling_files)
    core_values = sorted(core_index[2])
    core_summary = (
        core_values[0] if len(core_values) == 1 else (core_values or "unknown")
    )
    console_lines = []
    emit(console_lines, "=" * 88)
    emit(
        console_lines, f"MC2 dump analysis: {profile['name']} | path: {profile['path']}"
    )
    emit(console_lines, f"dump dir: {dump_dir}")
    emit(
        console_lines,
        f"matched files: {len(files)} | inferred AIC cores: {core_summary}",
    )
    emit(console_lines, f"TXT output: {output_dir}")
    emit(console_lines, "=" * 88)

    failures = 0
    for kind, path in files:
        relative_path = path.relative_to(dump_dir)
        output_path = text_output_path(output_dir, relative_path)
        result_lines = []
        emit(console_lines)
        emit(console_lines, f"[{kind}] {relative_path}")
        emit(console_lines, "-" * 88)

        if kind in ("xn", "cke"):
            try:
                rows, values, nonzero, result_lines = parse_address_table(path)
                emit(
                    console_lines,
                    f"parsed: dtype=uint64 shape=({rows}, 64) values={values} nonzero={nonzero}",
                )
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                result_lines.append(f"ERROR: {error}")
                emit(console_lines, result_lines[-1])
            save_text_result(output_path, profile, kind, relative_path, result_lines)
            emit(console_lines, f"saved TXT: {output_path}")
            continue

        # Large data segments only need metadata and a short preview in their
        # own report. RuntimeInfo remains structurally parsed by the backend.
        if kind == "workspace":
            identity = segment_identity(path, "workspace")
            if identity is not None and identity[1] != 12:
                seg_index, seg_type = identity
                emit(result_lines, f"segIdx: {seg_index}")
                emit(
                    result_lines,
                    f"segType: {seg_type} ({WORKSPACE_SEG_TYPE_NAMES.get(seg_type, 'UNKNOWN')})",
                )
                emit(
                    result_lines,
                    f"dataMeaning: {WORKSPACE_SEG_DESCRIPTIONS.get(seg_type, '未知 Workspace 段')}",
                )
                emit(result_lines, f"fileSize: {path.stat().st_size} bytes")
                emit_decimal_byte_preview(
                    result_lines, read_preview(path, WORKSPACE_PREVIEW_SIZE)
                )
                emit(
                    result_lines,
                    "typedViewHint: 使用 parse_matrix.py 并指定 dtype/rows/cols 查看真实数值",
                )
                save_text_result(
                    output_path, profile, kind, relative_path, result_lines
                )
                emit(console_lines, f"saved TXT: {output_path}")
                continue

        aic_core_num = None
        use_apace = False
        if kind == "workspace":
            sibling_tilings = tiling_files_by_parent.get(path.parent, [])
            aic_core_num, use_apace, selected_tiling = workspace_runtime_context(
                path, profile, sibling_tilings, core_index, apace_tiling_files
            )
            if selected_tiling is not None:
                emit(result_lines, f"relatedTilingData: {selected_tiling.name}")
        return_code, output = run_backend(profile, kind, path, aic_core_num, use_apace)
        if output:
            for line in output.splitlines():
                emit(result_lines, line)
        if kind == "comm_context":
            file_size = path.stat().st_size
            emit(result_lines, "")
            emit(
                result_lines,
                "rawDataHint: 如需按十进制查看 comm_context 原始字节，可使用 parse_matrix.py：",
            )
            emit(
                result_lines,
                f'python3 "{SCRIPT_DIR / "parse_matrix.py"}" "{path}" '
                f"--dtype uint8 --rows 1 --cols {file_size} --row 0 --col-range 0 64",
            )
        if return_code != 0:
            failures += 1
            emit(result_lines, f"ERROR: backend exited with code {return_code}")
        save_text_result(output_path, profile, kind, relative_path, result_lines)
        emit(console_lines, f"saved TXT: {output_path}")

    print()
    print(f"解析完成: {len(files) - failures}/{len(files)} 个文件成功")
    print(f"逐文件 TXT 位于: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
