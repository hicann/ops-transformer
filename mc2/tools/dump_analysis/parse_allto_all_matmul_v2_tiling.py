#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""AlltoAllMatmulV2 (AIV + URMA) dump parser.

This module deliberately reuses the Apace primitive parsers from
parse_all_gather_matmul_v3_tiling.py, while keeping the operator-specific
tiling offsets, kernel arguments and RuntimeInfo position tags here.
"""

import argparse
import logging
import os
import re
import struct
import sys

import parse_all_gather_matmul_v3_tiling as common


logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s] %(message)s")

# 当前源码布局:
# [DfxDumpInfo 424B][CommTilingData 40B][scale CommTilingData 40B]
# [QuantMatmulTilingData 64B][localMatmul 4B][4B padding]
TILING_DATA_SIZE = 576
OFFSET_WORKSPACE_LAYOUT = 0
OFFSET_PEERMEM_DATA_SIZE = common.WORKSPACE_LAYOUT_INFO_SIZE  # 208
OFFSET_PEERMEM_LAYOUT = OFFSET_PEERMEM_DATA_SIZE + 8  # 216
OFFSET_COMM_TILING = 424
OFFSET_SCALE_COMM_TILING = OFFSET_COMM_TILING + 40  # 464
OFFSET_MM_TILING = OFFSET_SCALE_COMM_TILING + 40  # 504
OFFSET_LOCAL_MATMUL = OFFSET_MM_TILING + 64  # 568

ARGS_INDEX_NAMES = [
    "context",
    "x1",
    "x2",
    "bias",
    "x1_scale",
    "x2_scale",
    "y",
    "all2all_out",
    "workspaceGM",
    "tilingGM",
]

POS_TAG_MAP = {
    0: "COMM_BEFORE",
    1: "COMP_CUBE_MATMUL_BEFORE",
    2: "COMP_LOCAL_BEFORE",
}

PEERMEM_SEG_TYPE_MAP = {
    0: "DATA",
    1: "SCALE",
    2: "BIAS",
    3: "FLAG",
    255: "RESERVED",
}


def parse_segment_layout(data, offset, type_map):
    if offset + common.WORKSPACE_LAYOUT_INFO_SIZE > len(data):
        return {"error": f"data too short for layout at offset {offset}"}
    total_size, seg_count, _ = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for index in range(min(seg_count, common.MAX_WORKSPACE_SEGMENTS)):
        item_offset = offset + 16 + index * common.WORKSPACE_SEG_INFO_SIZE
        seg_offset, seg_size = struct.unpack_from("<2Q", data, item_offset)
        seg_type = struct.unpack_from("<B", data, item_offset + 16)[0]
        segments.append(
            {
                "offset": seg_offset,
                "size": seg_size,
                "type": seg_type,
                "typeName": type_map.get(seg_type, f"UNKNOWN({seg_type})"),
            }
        )
    return {"totalSize": total_size, "segCount": seg_count, "segments": segments}


def parse_tiling_data(data):
    if len(data) < TILING_DATA_SIZE:
        logging.warning(
            "文件大小 %d < 结构体大小 %d，解析可能不完整", len(data), TILING_DATA_SIZE
        )
    return {
        "commTilingData": common.parse_comm_tiling_data(data, OFFSET_COMM_TILING),
        "scaleCommTilingData": common.parse_comm_tiling_data(
            data, OFFSET_SCALE_COMM_TILING
        ),
        "tileQbmmTilingData": common.parse_quant_matmul_tiling_data(
            data, OFFSET_MM_TILING
        ),
        "localMatmul": struct.unpack_from("<I", data, OFFSET_LOCAL_MATMUL)[0]
        if len(data) >= OFFSET_LOCAL_MATMUL + 4
        else None,
        "workspaceLayout": common.parse_workspace_layout_info(
            data, OFFSET_WORKSPACE_LAYOUT
        ),
        "peermemDataSize": struct.unpack_from("<Q", data, OFFSET_PEERMEM_DATA_SIZE)[0]
        if len(data) >= OFFSET_PEERMEM_DATA_SIZE + 8
        else None,
        "peermemLayout": parse_segment_layout(
            data, OFFSET_PEERMEM_LAYOUT, PEERMEM_SEG_TYPE_MAP
        ),
    }


def parse_args_info(data):
    count = min(len(ARGS_INDEX_NAMES), len(data) // 8)
    values = struct.unpack_from(f"<{count}Q", data, 0)
    result = {
        f"[{i}] {ARGS_INDEX_NAMES[i]}": f"0x{values[i]:016x}" for i in range(count)
    }
    result["_dumpFileSize"] = len(data)
    result["_validArgCount"] = count
    return result


def parse_runtime_info(data, aic_core_num=None):
    slots = []
    active_count = 0
    for slot_index in range(len(data) // common.RUNTIME_INFO_PER_CORE_SIZE):
        offset = slot_index * common.RUNTIME_INFO_PER_CORE_SIZE
        magic_num, core_id = struct.unpack_from("<IH", data, offset)
        turn, position, phase, commit_count, wait_count = struct.unpack_from(
            "<5B", data, offset + 6
        )
        active = magic_num == common.RUNTIME_INFO_MAGIC
        if active:
            active_count += 1
        slots.append(
            {
                "slotIdx": slot_index,
                "magicNum": magic_num,
                "magicValid": active,
                "coreId": core_id,
                "coreType": ("C核" if core_id < aic_core_num else "V核")
                if active and aic_core_num is not None
                else "未知",
                "execTurn": turn,
                "execPosition": position,
                "execPositionName": POS_TAG_MAP.get(position, f"UNKNOWN({position})"),
                "commPhase": phase,
                "commPhaseName": common.RUNTIME_PHASE_MAP.get(
                    phase, f"UNKNOWN({phase})"
                ),
                "commCommitCount": commit_count,
                "commWaitCount": wait_count,
            }
        )
    return {
        "segmentSize": len(data),
        "maxSlots": len(slots),
        "activeSlots": active_count,
        "aicCoreNum": aic_core_num,
        "slots": slots,
    }


def workspace_segment_info(filename):
    match = re.match(
        r"workspace_seg(\d+)_type(\d+)_", os.path.basename(filename), re.IGNORECASE
    )
    return (int(match.group(1)), int(match.group(2))) if match else None


def print_result(title, result):
    common.print_section(title)
    common.print_kv(title, result)


def main():
    parser = argparse.ArgumentParser(
        description="AlltoAllMatmulV2 (AIV + URMA) dump 解析脚本"
    )
    parser.add_argument("bin_file")
    parser.add_argument("--args", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument("--aic-core-num", type=int)
    parser.add_argument("--comm-ctx", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        return 1
    with open(args.bin_file, "rb") as file_handle:
        data = file_handle.read()
    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    title = "AlltoAllMatmulV2"
    if args.args:
        result = parse_args_info(data)
        title += " args_info"
    elif args.workspace:
        seg_info = workspace_segment_info(args.bin_file)
        if seg_info is None:
            logging.error("文件名不匹配 workspace_seg{idx}_type{type}_* 模式")
            return 1
        seg_index, seg_type = seg_info
        result = {
            "segIdx": seg_index,
            "segmentType": seg_type,
            "segmentTypeName": common.WS_SEG_TYPE_MAP.get(
                seg_type, f"UNKNOWN({seg_type})"
            ),
            "fileSize": len(data),
        }
        if seg_type == 12:
            result["runtimeInfo"] = parse_runtime_info(data, args.aic_core_num)
        else:
            result["hexPreview"] = data[:256].hex()
        title += f" workspace seg{seg_index}"
    elif args.comm_ctx:
        result = common.parse_urma_comm_context(data)
        title += " CommContext"
    else:
        result = parse_tiling_data(data)
        title += " tiling_data"

    if args.comm_ctx:
        common.print_urma_comm_context(result)
    elif args.workspace and result.get("segmentType") == 12:
        common.print_section(title)
        common.print_kv(
            "segment",
            {key: value for key, value in result.items() if key != "runtimeInfo"},
        )
        common.print_runtime_info_segment(result["runtimeInfo"])
    else:
        print_result(title, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
