#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
AllGatherMatmulV2 TilingData / Workspace / RuntimeInfo dump 解析脚本

结构体布局采用混合模式 (tiling_layout.py): TCubeTiling 等随 CANN 版本漂移的
尺寸从 set_env.sh 对应环境的 kernel_tiling.h 实时解析; DFX 自有结构
(RCSTiling/WorkspaceLayoutInfo 等) 使用内置常量表。无 CANN 环境时告警并用
默认尺寸继续。

用法:
  # 解析 tiling data (自动检测量化/非量化, 并用 WorkspaceLayoutInfo 合法性校验纠错)
  python parse_all_gather_matmul_v2_tiling.py tiling_data_*.bin

  # 解析 args_info
  python parse_all_gather_matmul_v2_tiling.py args_info_*.bin --args

  # 解析 workspace 段文件
  python parse_all_gather_matmul_v2_tiling.py workspace_seg*_type12_*.bin --workspace --aic-core-num N

"""

import argparse
import logging
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiling_layout import LayoutError, compute_layout

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ====== 与布局无关的常量 ======
TILING_DUMP_SIZE = 2048
RUNTIME_INFO_MAGIC = 0x5A5A5A5A

# args_info_*.bin 中 kernel 参数索引
ARGS_INDEX_NAMES = [
    "hccl_context",  # [0] runtime 注入的隐藏 CCU context
    "aGM",  # [1]
    "bGM",  # [2]
    "biasGM",  # [3]
    "scaleInv1",  # [4]
    "scaleInv2",  # [5]
    "scale",  # [6]
    "cGM",  # [7]
    "gatherOut",  # [8]
    "amax",  # [9]
    "workspaceGM",  # [10]
    "tilingGM",  # [11]
]

# PosTag 映射 (AllGatherMatmulV2 算子)
POS_TAG_MAP_AGMMV2 = {
    0: "COMM_BEFORE",
    1: "COMP_LOCAL_BEFORE",
    2: "COMP_GATHER_BEFORE",
    3: "FINALIZE_BEFORE",
}

# RuntimePhase 映射
RUNTIME_PHASE_MAP = {
    0: "COMM_INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "COMM_FINALIZE",
}

# Workspace 段类型映射 (WorkspaceSegType, DFX 自定义, 随算子源码仓演进)
WS_SEG_TYPE_MAP = {
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


def get_layout():
    """实时计算布局 (无 CANN 环境时 tiling_layout 内部告警并使用默认尺寸)。"""
    try:
        layout = compute_layout()
    except LayoutError as e:
        logging.error("布局计算失败: %s", e)
        sys.exit(1)
    for h in layout["headers"]:
        logging.info("布局头文件: %s", h)
    for key, name in (
        ("v2", "AllGatherMatmulTilingDataV2"),
        ("fp8", "AllGatherMatmulTilingDataFp8"),
    ):
        o = layout[key]["offsets"]
        logging.info(
            "布局计算: %s: %dB, dumpInfo@%d, param@%d, workspaceLayout@%d",
            name,
            layout[key]["size"],
            o["dumpInfo"],
            o["param"],
            o["workspaceLayout"],
        )
    return layout


def safe_str(raw: bytes) -> str:
    idx = raw.find(b"\x00")
    if idx >= 0:
        raw = raw[:idx]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def _u32(data, off):
    if off + 4 > len(data):
        return None
    return struct.unpack_from("<I", data, off)[0]


def _u64(data, off):
    if off + 8 > len(data):
        return None
    return struct.unpack_from("<Q", data, off)[0]


def parse_mc2_init_tiling(data: bytes, offset: int, size: int) -> dict:
    if offset + size > len(data):
        return {"error": "data too short"}
    return {"size": size}


def parse_mc2_cc_tiling(data: bytes, offset: int) -> dict:
    # Mc2CcTiling 为 512B HCCL 不透明块, 内部布局由 HCCL 决定, 不随本头文件变化
    if offset + 512 > len(data):
        return {"error": "data too short"}
    skip_local, skip_buffer, step_size, version = struct.unpack_from(
        "<4B", data, offset
    )
    comm_engine = struct.unpack_from("<B", data, offset + 13)[0]
    src_data_type = struct.unpack_from("<B", data, offset + 14)[0]
    dst_data_type = struct.unpack_from("<B", data, offset + 15)[0]
    group_name = safe_str(data[offset + 16 : offset + 16 + 128])
    alg_config = safe_str(data[offset + 144 : offset + 144 + 128])
    op_type, reduce_type = struct.unpack_from("<2I", data, offset + 272)
    return {
        "skipLocalRankCopy": skip_local,
        "skipBufferWindowCopy": skip_buffer,
        "stepSize": step_size,
        "version": version,
        "commEngine": comm_engine,
        "srcDataType": src_data_type,
        "dstDataType": dst_data_type,
        "groupName": group_name,
        "algConfig": alg_config,
        "opType": op_type,
        "reduceType": reduce_type,
    }


def parse_rcs_tiling(data: bytes, offset: int, rcs: dict) -> dict:
    if offset + rcs["size"] > len(data):
        return {"error": "data too short for RCSTiling"}
    out = {}
    for name, (off, size) in rcs["offsets"].items():
        fmt = "<Q" if size == 8 else "<I"
        out[name] = struct.unpack_from(fmt, data, offset + off)[0]
    return out


def detect_quant_mode(data: bytes, layout: dict) -> bool:
    """识别当前 DFX 量化通路。

    新版 TilingData 已移除 quantMode 字段。当前源码只有量化 Kernel 构建
    非空 WorkspaceLayout/StateDump，因此用 dumpInfo 起点的合法段表识别该通路。
    """
    ws = parse_workspace_layout_info(
        data, layout["workspace"]["layoutOffset"], layout["workspace"]
    )
    return validate_ws_layout(ws, layout["workspace"])


def parse_tiling_data(data: bytes, layout: dict, is_quant: bool) -> dict:
    key = "fp8" if is_quant else "v2"
    o = layout[key]["offsets"]
    init_size = o["mc2CcTiling"] - o["mc2InitTiling"]
    return {
        "mc2InitTiling": parse_mc2_init_tiling(data, o["mc2InitTiling"], init_size),
        "mc2CcTiling": parse_mc2_cc_tiling(data, o["mc2CcTiling"]),
        "param": parse_rcs_tiling(data, o["param"], layout["rcs"]),
        "dataType": _u32(data, o["dataType"]),
        "debugMode": _u32(data, o["debugMode"]),
    }


def parse_workspace_seg_info(data: bytes, offset: int) -> dict:
    if offset + 24 > len(data):
        return {"error": "data too short"}
    seg_offset, seg_size, seg_type = struct.unpack_from("<QQB", data, offset)
    return {
        "offset": seg_offset,
        "size": seg_size,
        "type": seg_type,
        "typeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})"),
    }


def parse_workspace_layout_info(data: bytes, offset: int, ws: dict) -> dict:
    if offset + ws["layoutInfoSize"] > len(data):
        return {"error": "data too short for WorkspaceLayoutInfo"}
    total_size, seg_count, reserved = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for i in range(min(seg_count, ws["maxSegments"])):
        seg_offset = offset + 16 + i * ws["segInfoSize"]
        segments.append(parse_workspace_seg_info(data, seg_offset))
    return {
        "totalSize": total_size,
        "segCount": seg_count,
        "segments": segments,
    }


def validate_ws_layout(ws: dict, wsinfo: dict) -> bool:
    """校验 WorkspaceLayoutInfo 是否像一份被正确写入的布局 (用于分支纠错)。"""
    if not ws or "error" in ws:
        return False
    segs = ws.get("segments") or []
    if ws.get("segCount", -1) != len(segs):
        return False
    if not (1 <= len(segs) <= wsinfo["maxSegments"]):
        return False
    total = ws.get("totalSize", 0)
    if not (0 < total < (1 << 44)):
        return False
    end = 0
    for s in segs:
        if "error" in s:
            return False
        if not (0 <= s["type"] <= 255):
            return False
        if s["offset"] < end or s["size"] < 0 or s["size"] > (1 << 40):
            return False
        end = s["offset"] + s["size"]
    return total >= end


def select_branch(data: bytes, layout: dict):
    """按 Mc2InitTiling/Mc2CcTiling 后的 DfxDumpInfo 返回 WorkspaceLayout。"""
    is_quant = detect_quant_mode(data, layout)
    wsinfo = layout["workspace"]
    off = wsinfo["layoutOffset"]
    ws = parse_workspace_layout_info(data, off, wsinfo)
    if validate_ws_layout(ws, wsinfo):
        return is_quant, ws
    logging.warning(
        "offset %d 的 WorkspaceLayoutInfo 未通过合法性校验，"
        "该 dump 可能不是当前 mc2Init/mc2Cc/dumpInfo 布局，或段表尚未写入",
        off,
    )
    return is_quant, None


def parse_runtime_info_per_core(data: bytes, offset: int) -> dict:
    magic_num, core_id = struct.unpack_from("<IH", data, offset)
    exec_turn, exec_position, comm_phase, commit_count, wait_count = struct.unpack_from(
        "<5B", data, offset + 6
    )
    return {
        "magicNum": magic_num,
        "magicValid": magic_num == RUNTIME_INFO_MAGIC,
        "coreId": core_id,
        "execTurn": exec_turn,
        "execPosition": exec_position,
        "execPositionName": POS_TAG_MAP_AGMMV2.get(
            exec_position, f"UNKNOWN({exec_position})"
        ),
        "commPhase": comm_phase,
        "commPhaseName": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "commCommitCount": commit_count,
        "commWaitCount": wait_count,
    }


def parse_runtime_info_segment(
    data: bytes, aic_core_num=None, per_core_size=512
) -> dict:
    max_slots = len(data) // per_core_size
    slots = []
    active_count = 0
    for i in range(max_slots):
        offset = i * per_core_size
        slot = parse_runtime_info_per_core(data, offset)
        slot["slotIdx"] = i
        if slot.get("magicValid", False):
            active_count += 1
            if aic_core_num is not None:
                slot["coreType"] = "C核" if slot["coreId"] < aic_core_num else "V核"
            else:
                slot["coreType"] = "未知"
            slots.append(slot)
        else:
            slots.append(slot)
    return {
        "segmentSize": len(data),
        "maxSlots": max_slots,
        "activeSlots": active_count,
        "aicCoreNum": aic_core_num,
        "slots": slots,
    }


def parse_workspace_segment_file(
    data: bytes, seg_type: int, aic_core_num=None, per_core_size=512
) -> dict:
    type_name = WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})")
    result = {
        "segmentType": seg_type,
        "segmentTypeName": type_name,
        "fileSize": len(data),
    }
    if seg_type == 12:
        rt_info = parse_runtime_info_segment(data, aic_core_num, per_core_size)
        result["runtimeInfo"] = rt_info
    else:
        result["hexPreview"] = data[:256].hex()
        result["note"] = f"段类型 {type_name}，原始二进制数据。"
    return result


def detect_workspace_seg_from_filename(filename: str):
    import re

    basename = os.path.basename(filename)
    m = re.match(r"workspace_seg(\d+)_type(\d+)_", basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_args_info(data: bytes) -> dict:
    num_valid_args = len(ARGS_INDEX_NAMES)
    num_ptrs = min(num_valid_args, len(data) // 8)
    ptrs = struct.unpack_from(f"<{num_ptrs}Q", data, 0)
    args = {}
    for i in range(num_ptrs):
        name = ARGS_INDEX_NAMES[i]
        args[f"[{i}] {name}"] = f"0x{ptrs[i]:016x}"
    args["_dumpFileSize"] = len(data)
    args["_validArgCount"] = num_ptrs
    return args


def print_section(title: str):
    logging.info("=" * 70)
    logging.info("  %s", title)
    logging.info("=" * 70)


def print_kv(title: str, d: dict, indent: int = 0):
    prefix = "  " * indent
    logging.info("%s--- %s ---", prefix, title)
    for k, v in d.items():
        if isinstance(v, dict):
            print_kv(k, v, indent + 1)
        elif isinstance(v, list):
            logging.info("%s  %-30s: %s", prefix, k, v)
        else:
            logging.info("%s  %-30s: %s", prefix, k, v)


def print_tiling_data(result: dict, is_quant: bool, layout: dict):
    key = "fp8" if is_quant else "v2"
    o = layout[key]["offsets"]
    print_section(
        "Mc2InitTiling (HCCL Init, %dB)" % (o["mc2CcTiling"] - o["mc2InitTiling"])
    )
    print_kv("mc2InitTiling", result["mc2InitTiling"])

    print_section("Mc2CcTiling (HCCL CC, 512B)")
    print_kv("mc2CcTiling", result["mc2CcTiling"])

    print_section(
        f"RCSTiling ({layout['rcs']['size']}B)  [{'QUANT' if is_quant else 'NON-QUANT'}]"
    )
    print_kv("param", result["param"])

    print_section(f"Extra Fields ({key}: {layout[key]['size']}B)")
    print_kv(
        "extra",
        {
            "dataType": result["dataType"],
            "debugMode": result["debugMode"],
        },
    )


def print_args_info(result: dict):
    print_section("Args Info (kernel GM pointer)")
    dump_file_size = result.pop("_dumpFileSize", 0)
    valid_arg_count = result.pop("_validArgCount", 0)
    for name, addr in result.items():
        if name.startswith("_"):
            continue
        logging.info("  %-30s: %s", name, addr)
    logging.info("")
    logging.info("  有效 args 数量: %d", valid_arg_count)
    logging.info("  dump 文件大小: %d bytes", dump_file_size)


def print_workspace_layout_info(ws_layout: dict, layout_info_size: int):
    if "error" in ws_layout:
        logging.error("  %s", ws_layout["error"])
        return
    print_section(f"WorkspaceLayoutInfo ({layout_info_size}B)")
    logging.info("  %-30s: %s bytes", "totalSize", ws_layout["totalSize"])
    logging.info("  %-30s: %s", "segCount", ws_layout["segCount"])
    logging.info("")
    logging.info(
        "  %-4s  %-12s  %-12s  %-6s  %s", "Idx", "Offset", "Size", "Type", "TypeName"
    )
    logging.info("  " + "-" * 66)
    for i, seg in enumerate(ws_layout["segments"]):
        if "error" in seg:
            logging.error("  [%d]  %s", i, seg["error"])
            continue
        logging.info(
            "  [%d]  0x%010x  %10d  %-6d  %s",
            i,
            seg["offset"],
            seg["size"],
            seg["type"],
            seg["typeName"],
        )


def print_runtime_info_segment(rt_info: dict):
    print_section("StateDump Segment (type=12, 64KB)")
    logging.info("  %-30s: %d bytes", "segmentSize", rt_info["segmentSize"])
    logging.info(
        "  %-30s: %d / %d",
        "activeSlots (magic matched)",
        rt_info["activeSlots"],
        rt_info["maxSlots"],
    )
    if rt_info.get("aicCoreNum") is None:
        logging.warning("  未提供 --aic-core-num，C/V 核分类不可靠（显示为'未知'）。")
    else:
        logging.info(
            "  %-30s: %d (coreId < %d 为 C核, 否则为 V核)",
            "aicCoreNum",
            rt_info["aicCoreNum"],
            rt_info["aicCoreNum"],
        )
    logging.info("")
    logging.info(
        "  --- 仅显示 magicNum 校验通过的 slot (0x%08X) ---", RUNTIME_INFO_MAGIC
    )
    active_slots = [s for s in rt_info["slots"] if s.get("magicValid", False)]
    if not active_slots:
        logging.warning("  无有效 slot，可能 runtime info 未被写入。")
    else:
        for slot in active_slots:
            core_type = slot.get("coreType", "未知")
            logging.info(
                "  Core %-4d (%s):  turn=%d, pos=%d (%s), phase=%s, commit=%d, wait=%d",
                slot["coreId"],
                core_type,
                slot["execTurn"],
                slot["execPosition"],
                slot["execPositionName"],
                slot["commPhaseName"],
                slot["commCommitCount"],
                slot["commWaitCount"],
            )


def print_workspace_segment(result: dict):
    seg_type = result["segmentType"]
    type_name = result["segmentTypeName"]
    print_section(
        f"Workspace Segment (type={seg_type}/{type_name}, {result['fileSize']}B)"
    )
    if seg_type == 12:
        print_runtime_info_segment(result["runtimeInfo"])
    else:
        logging.info("  段类型: %d (%s)", seg_type, type_name)
        logging.info("  文件大小: %d bytes", result["fileSize"])
        logging.info("  Hex 预览 (前 256 字节):")
        hex_str = result.get("hexPreview", "")
        for i in range(0, min(len(hex_str), 512), 64):
            offset_hex = f"{i // 2:04x}"
            chunk = hex_str[i : i + 64]
            formatted = " ".join(chunk[j : j + 2] for j in range(0, len(chunk), 2))
            logging.info("    %s: %s", offset_hex, formatted)


def main():
    parser = argparse.ArgumentParser(
        description="AllGatherMatmulV2 TilingData / Workspace / RuntimeInfo dump 解析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bin_file", help="dump 产生的 bin 文件路径")
    parser.add_argument("--args", action="store_true", help="解析 args_info_*.bin")
    parser.add_argument(
        "--workspace", action="store_true", help="解析 workspace 段文件"
    )
    parser.add_argument(
        "--aic-core-num",
        type=int,
        metavar="N",
        default=None,
        help="AIC 核数，用于区分 C/V 核",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()

    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    if args.args:
        result = parse_args_info(data)
        print_args_info(result)
        return

    layout = get_layout()

    if args.workspace:
        seg_info = detect_workspace_seg_from_filename(args.bin_file)
        if seg_info is None:
            logging.error("文件名不匹配 workspace_seg{idx}_type{type}_* 模式")
            sys.exit(1)
        seg_idx, seg_type = seg_info
        logging.info(
            "检测到 workspace 段文件: segIdx=%d, segType=%d (%s)",
            seg_idx,
            seg_type,
            WS_SEG_TYPE_MAP.get(seg_type, "UNKNOWN"),
        )
        if args.aic_core_num is not None:
            logging.info("使用 --aic-core-num=%d 区分 C/V 核", args.aic_core_num)
        result = parse_workspace_segment_file(
            data,
            seg_type,
            args.aic_core_num,
            layout["workspace"]["runtimeInfoPerCoreSize"],
        )
        result["segIdx"] = seg_idx
        print_workspace_segment(result)
        return

    # 默认: tiling_data 解析
    is_quant, ws_layout = select_branch(data, layout)
    mode_str = "量化" if is_quant else "非量化"
    key = "fp8" if is_quant else "v2"
    struct_name = (
        "AllGatherMatmulTilingDataFp8" if is_quant else "AllGatherMatmulTilingDataV2"
    )
    logging.info("解析模式: %s（新版 TilingData 不再保存 quantMode 字段）", mode_str)
    logging.info("结构体 %s: %dB", struct_name, layout[key]["size"])

    result = parse_tiling_data(data, layout, is_quant)

    if ws_layout is not None:
        result["workspaceLayout"] = ws_layout
        logging.info(
            "检测到 WorkspaceLayoutInfo (offset=%d, segCount=%d)",
            layout[key]["offsets"]["workspaceLayout"],
            ws_layout["segCount"],
        )

    print_tiling_data(result, is_quant, layout)
    if "workspaceLayout" in result:
        print_workspace_layout_info(
            result["workspaceLayout"], layout["workspace"]["layoutInfoSize"]
        )


if __name__ == "__main__":
    main()
