#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import glob
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import yaml

OP_DOMAINS = (
    "attention",
    "ffn",
    "gmm",
    "mamba",
    "mc2",
    "mhc",
    "moe",
    "posembedding",
)
SKIP_EXTS = {".md", ".json", ".ini"}
EXCLUDE_DIRS = {"common", "3rd", "tools"}
REPO_ROOT = Path(__file__).resolve().parents[2]
WHITELIST_PATH = REPO_ROOT / "tests" / "compile_scope_whitelist.yaml"
DEPEND_RE = re.compile(
    r"set\(\s*[A-Za-z0-9_]*depends\s+(.+?)\s+(?:PARENT_SCOPE|CACHE)\b",
    re.I,
)
LONG_SOC = {
    "ascend310p": "Ascend310P3",
    "ascend310b": "Ascend310B1",
    "ascend910": "Ascend910A",
    "ascend910b": "Ascend910B1",
    "ascend910_93": "Ascend910_9391",
    "ascend950": "Ascend950PR_9599",
    "ascend960dt": "Ascend960DT_968B7",
    "ascend350": "Ascend350_355e",
    "kirinx90": "KirinX90",
    "kirin9030": "Kirin9030",
    "mc62": "",
    "ascend5162a": "Ascend5162A",
    "ascend610lite": "Ascend610Lite",
}
_WHITELIST_CACHE = None


def read_changed_files(changed_file):
    with open(changed_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            path = raw.strip().replace(os.sep, "/")
            if not path or path.startswith("/"):
                continue
            if Path(path).suffix.lower() in SKIP_EXTS or "docs/" in path:
                continue
            yield path


def iter_ops(experimental):
    for domain in OP_DOMAINS:
        bases = [REPO_ROOT / domain]
        if experimental:
            bases.append(REPO_ROOT / "experimental" / domain)
        for base in bases:
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if child.is_dir() and child.name not in EXCLUDE_DIRS:
                    yield domain, child.name, child


def load_whitelist():
    global _WHITELIST_CACHE
    if _WHITELIST_CACHE is not None:
        return _WHITELIST_CACHE
    if not WHITELIST_PATH.is_file():
        _WHITELIST_CACHE = {}
        return _WHITELIST_CACHE
    with open(WHITELIST_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    raw_ops = cfg.get("ops") or {}
    ops = {}
    for key, entry in raw_ops.items():
        if not isinstance(entry, dict):
            continue
        if "impact" in entry or "soc" in entry:
            ops[key] = entry
            continue
        for op_name, sub_entry in entry.items():
            if isinstance(sub_entry, dict):
                ops[f"{key}/{op_name}"] = sub_entry
    for key, entry in ops.items():
        soc_value = entry.get("soc")
        if isinstance(soc_value, list):
            valid = any(str(item).strip() for item in soc_value)
        else:
            valid = bool(str(soc_value).strip())
        if not valid:
            raise ValueError(f"{WHITELIST_PATH}: {key} missing required soc")
    _WHITELIST_CACHE = ops
    return _WHITELIST_CACHE


def dep_key_from_path(path):
    parts = Path(str(path)).parts
    if len(parts) >= 2 and parts[0] in OP_DOMAINS:
        return f"{parts[0]}/{parts[1]}"
    return None


def add_edge(edges, dep, op_key):
    if dep and dep != op_key:
        edges[dep].add(op_key)


def parse_cmake_depends(edges, domain, op_name, op_path, op_key):
    cmake = op_path / "op_host" / "CMakeLists.txt"
    if not cmake.is_file():
        return
    text = cmake.read_text(encoding="utf-8", errors="ignore")
    for match in DEPEND_RE.finditer(text):
        for token in re.split(r"[\s]+", match.group(1).strip()):
            if not token or "$" in token:
                continue
            dep = dep_key_from_path(token)
            if dep:
                add_edge(edges, dep, op_key)


def build_edges(ops):
    edges = defaultdict(set)
    for domain, op_name, op_path in ops:
        op_key = f"{domain}/{op_name}"
        parse_cmake_depends(edges, domain, op_name, op_path, op_key)
    return edges


def changed_keys(changed_file, experimental):
    keys = set()
    for path in read_changed_files(changed_file):
        parts = path.split("/")
        domain = op_name = None
        if experimental and len(parts) >= 3 and parts[0] == "experimental":
            domain, op_name = parts[1], parts[2]
        elif len(parts) >= 2:
            domain, op_name = parts[0], parts[1]
        if domain not in OP_DOMAINS or op_name is None:
            continue
        if op_name in ("common", "3rd"):
            keys.add(f"{domain}/{op_name}")
            continue
        base = (
            REPO_ROOT / "experimental" / domain / op_name
            if experimental
            else REPO_ROOT / domain / op_name
        )
        if base.is_dir():
            keys.add(f"{domain}/{op_name}")
    return keys


def reverse_closure(edges, starts):
    seen = set(starts)
    queue = deque(starts)
    while queue:
        dep = queue.popleft()
        for nxt in edges.get(dep, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def split_socs(text):
    if not text:
        return []
    if isinstance(text, (list, tuple)):
        return [str(x).strip().lower() for x in text if str(x).strip()]
    return [x.strip().lower() for x in re.split(r"[,;]", str(text)) if x.strip()]


def def_supports(op_path, soc):
    long_soc = LONG_SOC.get(soc, "")
    soc_pattern = re.escape(soc)
    if long_soc:
        soc_pattern = f"(?:{soc_pattern}|{re.escape(long_soc)})"
    add_config_re = re.compile(r"AddConfig\s*\(\s*[\"']?(?:" + soc_pattern + r")[\"']?")
    for def_file in glob.glob(str(op_path / "**" / "*def.cpp"), recursive=True):
        try:
            text = Path(def_file).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if add_config_re.search(text):
            return True
    return False


def filter_by_soc(ops, socs):
    if not socs:
        return ops
    return [item for item in ops if any(def_supports(item[2], soc) for soc in socs)]


def whitelist_selection(key, entry, by_key, target_socs):
    domain = key.split("/", 1)[0]
    impact = entry.get("impact")
    if impact is None:
        impact_keys = [key]
    else:
        impact_keys = []
        for item in impact:
            item = str(item)
            impact_keys.append(item if "/" in item else f"{domain}/{item}")
    entry_socs = split_socs(entry.get("soc"))
    selected = set()
    for impact_key in impact_keys:
        if impact_key not in by_key:
            continue
        domain, name, path = by_key[impact_key]
        if target_socs and not set(entry_socs).intersection(target_socs):
            continue
        selected.add(name)
    return selected


def check_whitelist():
    try:
        whitelist = load_whitelist()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    by_key = {f"{domain}/{name}": path for domain, name, path in iter_ops(False)}
    valid_socs = set(LONG_SOC)
    errors = []
    for key, entry in sorted(whitelist.items()):
        if key not in by_key:
            errors.append(f"{key}: operator not found")
            continue
        domain = key.split("/", 1)[0]
        impact = entry.get("impact")
        if impact is not None and not isinstance(impact, list):
            errors.append(f"{key}: impact must be a list")
            continue
        for item in impact or []:
            item = str(item)
            impact_key = item if "/" in item else f"{domain}/{item}"
            if impact_key not in by_key:
                errors.append(f"{key}: impact target not found: {item}")
        for soc in split_socs(entry.get("soc")):
            if soc not in valid_socs:
                errors.append(f"{key}: invalid soc: {soc}")
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        return 1
    print(f"whitelist check passed: {len(whitelist)} entries")
    return 0


def collect_compile_ops(changed_file, soc_text, experimental):
    ops = list(iter_ops(experimental))
    by_key = {f"{domain}/{name}": (domain, name, path) for domain, name, path in ops}
    starts = changed_keys(changed_file, experimental)
    whitelist = load_whitelist()
    target_socs = split_socs(soc_text)

    chosen_names = set()
    fallback_starts = set()
    for key in sorted(starts):
        entry = whitelist.get(key)
        if entry is None:
            fallback_starts.add(key)
        else:
            chosen_names.update(whitelist_selection(key, entry, by_key, target_socs))

    if fallback_starts:
        edges = build_edges(ops)
        selected = reverse_closure(edges, fallback_starts)
        chosen = [by_key[key] for key in selected if key in by_key]
        chosen = filter_by_soc(chosen, target_socs)
        chosen_names.update(name for _, name, _ in chosen)
    return sorted(chosen_names)


def main():
    parser = argparse.ArgumentParser(
        description="Select compile scope from CMake and def.cpp"
    )
    parser.add_argument("-f", "--file", help="changed files list")
    parser.add_argument(
        "--soc", default="", help="target soc list, comma/semicolon separated"
    )
    parser.add_argument(
        "--experimental", action="store_true", help="include experimental operators"
    )
    parser.add_argument(
        "--check", action="store_true", help="validate whitelist and exit"
    )
    args = parser.parse_args()
    if args.check:
        sys.exit(check_whitelist())
    if not args.file:
        parser.error("-f/--file is required unless --check")
    ops = collect_compile_ops(args.file, args.soc, args.experimental)
    print((";".join(ops) + ";") if ops else "")


if __name__ == "__main__":
    main()
