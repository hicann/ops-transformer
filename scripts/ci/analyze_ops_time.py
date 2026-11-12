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

import os
import re
import csv
import logging
from collections import defaultdict


script_dir = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.join(script_dir, "../../build/")
build_dir = os.path.abspath(build_dir)
workdir = os.path.join(build_dir, "binary")
workdir = os.path.abspath(workdir)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_txt_content(content):
    """Parse txt file content and extract required fields"""
    pattern = (
        r"Build \[([^\]]+)\] op_name \[([^\]]+)\] index \[([^\]]+)\] "
        r"bin_file \[([^\]]+)\] duration \[([^\]]+)\] size \[([^\]]+)\] "
        r"start_time \[([^\]]+)\] end_time \[([^\]]+)\]"
    )
    match = re.search(pattern, content)

    if not match:
        return None

    return {
        "soc": match.group(1),
        "op_name": match.group(2),
        "index": match.group(3),
        "bin_file": match.group(4),
        "duration": match.group(5),
        "size": match.group(6),
        "start_time": match.group(7),
        "end_time": match.group(8),
    }


def remove_apt_suffix(op_name):
    """If op_name ends with generation/arch suffix (_a2/_a3/_apt/_archNN) remove it"""
    op_name = re.sub(r"(?:_arch\d+(?:_\d+)?)$", "", op_name)
    for suffix in ("_a2", "_a3", "_apt"):
        if op_name.endswith(suffix):
            return op_name[: -len(suffix)]
    return op_name


def find_txt_files():
    """Find all txt files in specified directory structure"""
    txt_files = []

    # Get build/binary directory under current directory
    base_dir = workdir

    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist")
        return []

    # Iterate through all soc directories
    for soc_dir in os.listdir(base_dir):
        gen_dir = os.path.join(base_dir, soc_dir, "gen")
        # Find all txt files in gen directory
        for file in os.listdir(gen_dir):
            match_txt = re.match(r"([^-]+)-(\d+)\.txt", file)
            if match_txt:
                txt_files.append(os.path.join(gen_dir, file))

    return txt_files


def process_txt_file(txt_files):
    merge_dict = defaultdict(
        lambda: {"duration": 0, "max_duration": 0.0, "size": 0, "intervals": []}
    )
    for txt_file in txt_files:
        try:
            # Extract op_name and index from filename
            filename = os.path.basename(txt_file)
            # Use regex to match {op_name}-{index}.txt format
            filename_match = re.match(r"([^-]+)-(\d+)\.txt", filename)
            if not filename_match:
                logger.warning(f"Filename format mismatch: {filename}")
                continue

            # Read file content
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Parse content
            parsed_data = parse_txt_content(content)
            if not parsed_data:
                logger.warning(f"File content format mismatch: {txt_file}")
                continue
            # Build data row
            key = (parsed_data["soc"], parsed_data["op_name"])
            duration = float(parsed_data["duration"])
            merge_dict[key]["duration"] += duration
            merge_dict[key]["max_duration"] = max(
                merge_dict[key]["max_duration"], duration
            )
            merge_dict[key]["size"] += float(parsed_data["size"])
            merge_dict[key]["intervals"].append(
                (float(parsed_data["start_time"]), float(parsed_data["end_time"]))
            )
            merge_dict[key]["soc"] = parsed_data["soc"]

        except Exception as e:
            logger.error(f"Error processing file {txt_file}: {e}")
            continue
    return merge_dict


def union_duration(intervals):
    total = 0.0
    cur_start = None
    cur_end = None
    for start, end in sorted(intervals):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def parse_cmake_cache(filepath, key):
    """
    纯 Python 解析 CMakeCache.txt
    格式通常是: KEY:TYPE=VALUE
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith("#"):
                    continue

                # 检查是否以目标 key 开头
                if line.startswith(f"{key}="):
                    # 分割一次，防止 VALUE 中包含等号
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts
    except FileNotFoundError:
        logger.error(f"Error find CMakeCache.txt file {filepath}")
    return None


def main():
    group_trans_op_name = {}
    # Get all txt files in current directory
    txt_files = find_txt_files()

    if not txt_files:
        logger.info("No txt files found")
        return

    merge_dict = process_txt_file(txt_files)

    value = parse_cmake_cache(
        os.path.join(build_dir, "CMakeCache.txt"), "ASCEND_OP_NAME:STRING"
    )
    if value:
        parts = value[1].split(",")
        last_part = ""
        for part in parts:
            if part and part[0].isdigit():
                group_trans_op_name[last_part] = last_part + "," + part
            else:
                group_trans_op_name[part] = part
                last_part = part

    data_list = []
    # Append merged rows
    for (soc, op_name), values in merge_dict.items():
        real_op_name = remove_apt_suffix(op_name)
        real_op_name = group_trans_op_name.get(real_op_name, real_op_name)
        merged_row = {
            "op_name": f"{real_op_name}",
            "duration": int(union_duration(values["intervals"])),
            "sum_duration": int(values["duration"]),
            "max_duration": int(values["max_duration"]),
            "size": int(values["size"]),
            "soc": soc,
        }
        data_list.append(merged_row)
    # Sort by op_name
    data_list.sort(key=lambda x: x["op_name"])

    # Write to CSV file
    if data_list:
        fieldnames = [
            "op_name",
            "duration",
            "sum_duration",
            "max_duration",
            "size",
            "soc",
        ]
        by_soc = defaultdict(list)
        for row in data_list:
            by_soc[row["soc"]].append(row)
        for soc_version, soc_rows in by_soc.items():
            csv_filename = os.path.join(build_dir, f"ops_cost_{soc_version}.csv")
            with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(soc_rows)
            logger.info(f"CSV file generated: {csv_filename}")

        # Print statistics
        logger.info("Statistics:")
        logger.info(f"Total records: {len(data_list)}")
    else:
        logger.info("No valid data files found")


if __name__ == "__main__":
    main()
