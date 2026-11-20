# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import re
import logging
from pathlib import Path

NEW_OPS_PATH = [
    "mc2",
    "attention",
    "ffn",
    "gmm",
    "moe",
]


class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map


BlackList = {
    "fused_infer_attention_score",
    "moe_distribute_combine_shmem",
    "moe_distribute_dispatch_shmem",
    "rope_matrix",
    "quant_sals_indexer",
    "quant_sals_indexer_metadata",
    "sparse_flash_attention_antiquant",
    "sparse_flash_attention_antiquant_metadata",
}


def extract_operator_name(file_path, is_experimental):
    path_parts = file_path.lstrip("/").split("/")
    domain, operator_name = _get_domain_and_op(path_parts, is_experimental)
    if domain is None:
        return ""

    # 判断是否直接返回默认名称（黑名单、common、实验路径不存在、op_host缺失）
    if _should_return_default(domain, operator_name, path_parts, is_experimental):
        return _get_default_name(domain)

    # 非实验分支且 domain 不在 NEW_OPS_PATH 时，返回默认名称（空或 attention 特殊值）
    if is_experimental != "TRUE" and domain not in NEW_OPS_PATH:
        return _get_default_name(domain)

    # 其他情况返回 operator_name
    return operator_name


def _get_domain_and_op(path_parts, is_experimental):
    """从路径部分提取域和算子名"""
    if is_experimental == "TRUE":
        if len(path_parts) >= 3:
            return path_parts[1], path_parts[2]
    else:
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
    return None, None


def _should_return_default(domain, operator_name, path_parts, is_experimental):
    """检查是否应使用默认名称（而不是 operator_name）"""
    if operator_name in BlackList:
        return True

    # 检查是否是 common 或 experimental 路径不存在
    exp_path = f"experimental/{domain}/{operator_name}"
    if operator_name == "common" or not os.path.exists(exp_path):
        return True

    # 实验分支额外检查 op_host 目录
    if is_experimental == "TRUE":
        # 构造 parent 路径（原代码用 Path(*path_parts[:3])）
        if len(path_parts) >= 3:
            parent = Path(*path_parts[:3])
            target = parent / "op_host"
            if not (target.exists() and target.is_dir()):
                return True
    return False


def _get_default_name(domain):
    """根据域返回默认名称（目前只有 attention 特殊处理）"""
    if domain == "attention":
        return "nsa_compress_attention_infer"
    return ""


def get_operator_info_from_ci(changed_file_info_from_ci, is_experimental):
    """
    get operator change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or OperatorChangeInfo
    """

    def is_skippable_file(line):
        ext = os.path.splitext(line)[-1].lower()
        return ext in (".md",)

    def process_line(line, operators_set, files_map):
        """处理单行：提取算子名并更新集合和映射"""
        line = line.strip()
        if is_skippable_file(line):
            return
        operator_name = extract_operator_name(line, is_experimental)
        if operator_name:
            operators_set.add(operator_name)
            if operator_name not in files_map:
                files_map[operator_name] = []
            files_map[operator_name].append(line)

    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error(
            "[ERROR] change file is not exist, can not get file change info in this pull request."
        )
        return None

    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        changed_operators = set()
        operator_file_map = {}

        for line in lines:
            process_line(line, changed_operators, operator_file_map)

    return OperatorChangeInfo(
        changed_operators=list(changed_operators), operator_file_map=operator_file_map
    )


def find_def_cpp_files(operators, operator_file_map, is_experimental):
    """
    Find def.cpp files for each operator
    :param operators: list of operator names
    :param operator_file_map: map of operator name to file paths
    :param is_experimental: whether in experimental branch
    :return: dict mapping operator name to list of def.cpp file paths
    """
    op_to_def_files = {}
    for op_name in operators:
        if op_name not in operator_file_map:
            continue
        for file_path in operator_file_map[op_name]:
            path_parts = file_path.lstrip("/").split("/")
            domain, _ = _get_domain_and_op(path_parts, is_experimental)
            if domain is None:
                continue

            if is_experimental == "TRUE":
                search_dir = f"experimental/{domain}/{op_name}"
            else:
                search_dir = f"{domain}/{op_name}"

            if not os.path.exists(search_dir):
                continue

            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith("def.cpp"):
                        full_path = os.path.join(root, f)
                        if op_name not in op_to_def_files:
                            op_to_def_files[op_name] = []
                        if full_path not in op_to_def_files[op_name]:
                            op_to_def_files[op_name].append(full_path)
    return op_to_def_files


def check_soc_registered(def_cpp_file, soc):
    """
    Check if a SOC is registered in the def.cpp file
    :param def_cpp_file: path to def.cpp file
    :param soc: SOC name to check (e.g., 'ascend950')
    :return: True if SOC is registered, False otherwise
    """
    try:
        with open(def_cpp_file, "r", encoding="utf-8") as f:
            content = f.read()
            pattern = rf'this->AICore\(\)\.AddConfig\(["\']?{soc}["\']?'
            if re.search(pattern, content):
                return True
    except Exception as e:
        logging.warning(f"[WARN] Failed to read {def_cpp_file}: {e}")
    return False


def filter_operators_by_def_and_soc(operators, op_to_def_files, soc):
    """
    Filter operators that have def.cpp files with the specified SOC registered
    :param operators: list of operator names
    :param op_to_def_files: dict mapping operator name to list of def.cpp file paths
    :param soc: SOC name to check
    :return: list of filtered operator names, list of valid def.cpp files
    """
    filtered_operators = []
    valid_def_files = []

    for op_name in operators:
        if op_name not in op_to_def_files or not op_to_def_files[op_name]:
            logging.info(
                f"[INFO] Operator '{op_name}' has no def.cpp file, filtered out."
            )
            continue

        has_valid_def = False
        for def_file in op_to_def_files[op_name]:
            if check_soc_registered(def_file, soc):
                has_valid_def = True
                if def_file not in valid_def_files:
                    valid_def_files.append(def_file)
                break

        if has_valid_def:
            filtered_operators.append(op_name)
        else:
            logging.info(
                f"[INFO] Operator '{op_name}' has no def.cpp with SOC '{soc}' registered, filtered out."
            )

    return filtered_operators, valid_def_files


def add_experimental_metadata_companions(operators, ops_change_info, is_experimental):
    """Add sibling <op>_metadata operators for experimental packages.

    The experimental package build (build.sh --experimental --PR_PKG) uses this
    parser instead of parse_compile_scope.py. Metadata operators are AICPU-only
    and have no def.cpp, so they must be added after the def.cpp/SOC filter.
    """
    if is_experimental != "TRUE":
        return operators

    result = list(operators)
    seen = set(result)
    for op_name in operators:
        file_list = ops_change_info.operator_file_map.get(op_name, [])
        if not file_list:
            continue
        path_parts = file_list[0].lstrip("/").split("/")
        domain, _ = _get_domain_and_op(path_parts, is_experimental)
        if domain is None:
            continue
        metadata_name = f"{op_name}_metadata"
        if metadata_name in seen or metadata_name in BlackList:
            continue
        if os.path.isdir(f"experimental/{domain}/{metadata_name}"):
            result.append(metadata_name)
            seen.add(metadata_name)
            logging.info(
                f"[INFO] Add experimental metadata companion '{metadata_name}' for '{op_name}'."
            )
    return result


def get_change_ops_list(changed_file_info_from_ci, is_experimental, soc):
    ops_change_info = get_operator_info_from_ci(
        changed_file_info_from_ci, is_experimental
    )
    if not ops_change_info:
        logging.info("[INFO] not found ops change info, run all c++.")
        return None

    op_to_def_files = find_def_cpp_files(
        ops_change_info.changed_operators,
        ops_change_info.operator_file_map,
        is_experimental,
    )

    filtered_operators, valid_def_files = filter_operators_by_def_and_soc(
        ops_change_info.changed_operators, op_to_def_files, soc
    )

    if not filtered_operators:
        if soc == "ascend950":
            filtered_operators = ["all_gather_matmul_v2"]
            logging.info(
                "[INFO] No operators found for ascend950, using default: all_gather_matmul_v2"
            )
        elif soc == "ascend910b":
            filtered_operators = ["nsa_compress_attention_infer"]
            logging.info(
                "[INFO] No operators found for ascend910b, using default: nsa_compress_attention_infer"
            )

    filtered_operators = add_experimental_metadata_companions(
        filtered_operators, ops_change_info, is_experimental
    )

    return ";".join(filtered_operators)


if __name__ == "__main__":
    soc = sys.argv[3] if len(sys.argv) > 3 else ""
    ops_str = get_change_ops_list(sys.argv[1], sys.argv[2], soc)
    print(ops_str)
