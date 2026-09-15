#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
MatmulReduceScatterV2 TilingData / Workspace / RuntimeInfo dump 解析脚本

结构体布局采用混合模式 (tiling_layout.py): TCubeTiling 等随 CANN 版本漂移的
尺寸从 set_env.sh 对应环境的 kernel_tiling.h 实时解析; DFX 自有结构使用内置
常量表。无 CANN 环境时告警并用默认尺寸继续。

用法:
  python parse_matmul_reduce_scatter_v2_tiling.py tiling_data_*.bin
  python parse_matmul_reduce_scatter_v2_tiling.py args_info_*.bin --args
  python parse_matmul_reduce_scatter_v2_tiling.py workspace_seg*_type12_*.bin --workspace --aic-core-num N
"""

import argparse
import logging
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiling_layout import LayoutError, compute_mrs_v2_layout

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

TILING_DUMP_SIZE = 2048
RUNTIME_INFO_MAGIC = 0x5A5A5A5A
MMRS_AIV_PER_AIC = 2
WORKSPACE_PREVIEW_SIZE = 64

# RCSTiling 输出字段 (布局偏移由 tiling_layout 计算)
RCS_FIELDS_SHOWN = [
    "rankDim",
    "rankID",
    "tileCnt",
    "tailM",
    "tailCnt",
    "rankM",
    "rankN",
    "rankK",
    "aicCoreNum",
    "gatherLen",
]

ARGS_INDEX_NAMES = [
    "hccl_context",  # [0] runtime 注入的隐藏 CCU context
    "aGM",  # [1]
    "bGM",  # [2]
    "biasGM",  # [3]
    "x1ScaleGM",  # [4]
    "x2ScaleGM",  # [5]
    "quantScaleGM",  # [6]
    "cGM",  # [7]
    "amaxOutGM",  # [8]
    "workspaceGM",  # [9]
    "tilingGM",  # [10]
]

# 与 op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h / quant_bmm_reduce_scatter_fp8_hif8.h
# 中定义的 POS_* 常量保持一致
POS_TAG_MAP_MRSV2 = {
    0: "COMM_BEFORE",  # AlltoAll/ReduceScatter 提交或 hccl.Wait 前
    1: "SCALE_COMM_BEFORE",  # scale 通信前（当前 kernel 未使用）
    2: "COMP_CUBE_MATMUL_BEFORE",  # C 核 Matmul 前
    3: "COMP_VEC_REDUCE_BEFORE",  # V 核 ReduceSum 前
}

RUNTIME_PHASE_MAP = {
    0: "COMM_INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "COMM_FINALIZE",
    255: "RESERVED",
}

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

WS_SEG_DESCRIPTION_MAP = {
    0: "系统库预留 Workspace，不属于算子用户数据",
    1: "ND 到 NZ 格式转换的中间缓冲区",
    2: "Gather 通信数据缓冲区",
    3: "Gather 的第一路 Scale 缓冲区",
    4: "Gather Scale 缓冲区",
    5: "通信输出缓冲区",
    6: "转置输出缓冲区",
    7: "Bias 缓冲区",
    8: "Matmul 结果；MMRS 中也可作为通信发送缓冲区",
    9: "通信接收缓冲区；A2A 路径中供后续 ReduceSum 使用",
    10: "INT8 通信临时缓冲区",
    11: "动态量化临时缓冲区",
    12: "每核 StateDump，记录执行位置和通信 commit/wait 情况",
    13: "Matmul 内部临时 Workspace，不是 Matmul 结果，元素类型和逻辑形状不固定",
    255: "对齐或预留空间",
}


def get_layout():
    """实时计算布局 (无 CANN 环境时 tiling_layout 内部告警并使用默认尺寸)。"""
    try:
        layout = compute_mrs_v2_layout()
    except LayoutError as e:
        logging.error("布局计算失败: %s", e)
        sys.exit(1)
    for h in layout["headers"]:
        logging.info("布局头文件: %s", h)
    for key, name in (
        ("nonQuant", "MatmulReduceScatterV2TilingData"),
        ("quant", "QuantBatchMatmulV3ReduceScatterTilingData"),
    ):
        b = layout[key]
        logging.info(
            "布局计算: %s: %dB, dumpInfo@%d, param@%d, workspaceLayout@%d",
            name,
            b["size"],
            layout["offsets"]["dumpInfo"],
            layout["offsets"]["param"],
            b["wsLayoutOffset"],
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


def parse_mc2_init_tiling(data, offset, size):
    return {"size": size} if offset + size <= len(data) else {"error": "too short"}


def parse_mc2_cc_tiling(data, offset):
    if offset + 512 > len(data):
        return {"error": "too short"}
    skip_local, skip_buffer, step_size, version = struct.unpack_from(
        "<4B", data, offset
    )
    comm_engine = struct.unpack_from("<B", data, offset + 13)[0]
    group_name = safe_str(data[offset + 16 : offset + 16 + 128])
    alg_config = safe_str(data[offset + 144 : offset + 144 + 128])
    op_type, reduce_type = struct.unpack_from("<2I", data, offset + 272)
    return {
        "skipLocalRankCopy": skip_local,
        "skipBufferWindowCopy": skip_buffer,
        "stepSize": step_size,
        "version": version,
        "commEngine": comm_engine,
        "groupName": group_name,
        "algConfig": alg_config,
        "opType": op_type,
        "reduceType": reduce_type,
    }


def parse_rcs_tiling(data, offset, layout):
    rcs = layout["rcs"]
    if offset + rcs["size"] > len(data):
        return {"error": "too short"}
    out = {}
    for name in RCS_FIELDS_SHOWN:
        off, size = rcs["offsets"][name]
        out[name] = struct.unpack_from("<Q" if size == 8 else "<I", data, offset + off)[
            0
        ]
    return out


def detect_quant_mode(data, layout):
    """新版结构已移除 quantMode；当前 DFX Workspace 段表由量化实现写入。"""
    ws = parse_workspace_layout_info(
        data, layout["workspace"]["layoutOffset"], layout["workspace"]
    )
    return workspace_layout_is_valid(ws, layout["workspace"])


def parse_tiling_data(data, layout, is_quant):
    o = layout["offsets"]
    return {
        "mc2InitTiling": parse_mc2_init_tiling(
            data, o["mc2InitTiling"], o["mc2CcTiling"] - o["mc2InitTiling"]
        ),
        "mc2CcTiling": parse_mc2_cc_tiling(data, o["mc2CcTiling"]),
        "param": parse_rcs_tiling(data, o["param"], layout),
        "dataType": struct.unpack_from("<I", data, o["dataType"])[0],
        "debugMode": struct.unpack_from("<I", data, o["debugMode"])[0],
    }


def parse_workspace_seg_info(data, offset):
    if offset + 24 > len(data):
        return {"error": "too short"}
    seg_offset, seg_size, seg_type = struct.unpack_from("<QQB", data, offset)
    return {
        "offset": seg_offset,
        "size": seg_size,
        "type": seg_type,
        "typeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})"),
    }


def parse_workspace_layout_info(data, offset, ws):
    if offset + ws["layoutInfoSize"] > len(data):
        return {"error": "too short"}
    total_size, seg_count, _ = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for i in range(min(seg_count, ws["maxSegments"])):
        segments.append(
            parse_workspace_seg_info(data, offset + 16 + i * ws["segInfoSize"])
        )
    return {"totalSize": total_size, "segCount": seg_count, "segments": segments}


def workspace_layout_is_valid(layout, ws):
    """段表合理性校验: 偏移漂移时 segCount/totalSize 会解出垃圾值"""
    if not isinstance(layout, dict) or "error" in layout:
        return False
    seg_count = layout.get("segCount", 0)
    total_size = layout.get("totalSize", 0)
    if not 1 <= seg_count <= ws["maxSegments"]:
        return False
    if total_size <= 0:
        return False
    return all(seg["offset"] + seg["size"] <= total_size for seg in layout["segments"])


def parse_runtime_info_per_core(data, offset):
    if offset + 512 > len(data):
        return {"error": "too short"}
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
        "execPositionName": POS_TAG_MAP_MRSV2.get(
            exec_position, f"UNKNOWN({exec_position})"
        ),
        "commPhase": comm_phase,
        "commPhaseName": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "commCommitCount": commit_count,
        "commWaitCount": wait_count,
    }


def parse_runtime_info_segment(data, aic_core_num=None, per_core_size=512):
    max_slots = len(data) // per_core_size
    slots = []
    active_count = 0
    for i in range(max_slots):
        slot = parse_runtime_info_per_core(data, i * per_core_size)
        slot["slotIdx"] = i
        if aic_core_num is None:
            slot["coreType"] = "未知"
        elif i < aic_core_num:
            slot["coreType"] = "C核"
        elif i < aic_core_num * (1 + MMRS_AIV_PER_AIC):
            slot["coreType"] = "V核"
        else:
            slot["coreType"] = "未使用"
        if slot.get("magicValid", False):
            active_count += 1
        slots.append(slot)

    if aic_core_num is None:
        expected_slots = max_slots
        aiv_core_num = None
    else:
        aiv_core_num = aic_core_num * MMRS_AIV_PER_AIC
        expected_slots = min(max_slots, aic_core_num + aiv_core_num)

    magic_bytes = struct.pack("<I", RUNTIME_INFO_MAGIC)
    magic_offsets = []
    search_from = 0
    while True:
        found = data.find(magic_bytes, search_from)
        if found < 0:
            break
        magic_offsets.append(found)
        search_from = found + 1
    unaligned_magic_offsets = [
        offset for offset in magic_offsets if offset % per_core_size != 0
    ]

    expected = slots[:expected_slots]
    valid_expected = [slot for slot in expected if slot.get("magicValid", False)]
    return {
        "segmentSize": len(data),
        "maxSlots": max_slots,
        "activeSlots": active_count,
        "expectedSlots": expected_slots,
        "validExpectedSlots": len(valid_expected),
        "aicCoreNum": aic_core_num,
        "aivCoreNum": aiv_core_num,
        "commitTotal": sum(slot["commCommitCount"] for slot in valid_expected),
        "waitTotal": sum(slot["commWaitCount"] for slot in valid_expected),
        "unalignedMagicOffsets": unaligned_magic_offsets[:16],
        "slots": slots,
    }


def parse_workspace_segment_file(data, seg_type, aic_core_num=None, per_core_size=512):
    result = {
        "segmentType": seg_type,
        "segmentTypeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})"),
        "segmentDescription": WS_SEG_DESCRIPTION_MAP.get(seg_type, "未知 Workspace 段"),
        "fileSize": len(data),
    }
    if seg_type == 12:
        result["runtimeInfo"] = parse_runtime_info_segment(
            data, aic_core_num, per_core_size
        )
    else:
        preview = data[:WORKSPACE_PREVIEW_SIZE]
        result["previewType"] = "uint8 raw bytes"
        result["previewOffset"] = 0
        result["decimalPreview"] = list(preview)
        result["previewAllZero"] = not any(preview)
    return result


def detect_workspace_seg_from_filename(filename):
    import re

    m = re.match(r"workspace_seg(\d+)_type(\d+)_", os.path.basename(filename))
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_args_info(data):
    num_ptrs = min(len(ARGS_INDEX_NAMES), len(data) // 8)
    ptrs = struct.unpack_from(f"<{num_ptrs}Q", data, 0)
    args = {
        f"[{i}] {ARGS_INDEX_NAMES[i]}": f"0x{ptrs[i]:016x}" for i in range(num_ptrs)
    }
    args["_dumpFileSize"] = len(data)
    args["_validArgCount"] = num_ptrs
    return args


def print_section(title):
    logging.info("=" * 70)
    logging.info("  %s", title)
    logging.info("=" * 70)


def print_kv(title, d, indent=0):
    prefix = "  " * indent
    logging.info("%s--- %s ---", prefix, title)
    for k, v in d.items():
        if isinstance(v, dict):
            print_kv(k, v, indent + 1)
        else:
            logging.info("%s  %-30s: %s", prefix, k, v)


def print_tiling_data(result, is_quant, layout):
    o = layout["offsets"]
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
    print_section("Extra Fields")
    print_kv(
        "extra", {"dataType": result["dataType"], "debugMode": result["debugMode"]}
    )


def print_runtime_info_segment(rt_info):
    print_section("StateDump Segment (type=12, 64KB)")
    logging.info("  %-30s: %d bytes", "segmentSize", rt_info["segmentSize"])
    logging.info("  %-30s: %d", "maxSlots", rt_info["maxSlots"])
    logging.info("  %-30s: %d", "expectedSlots", rt_info["expectedSlots"])
    logging.info(
        "  %-30s: %d / %d",
        "validExpectedSlots",
        rt_info["validExpectedSlots"],
        rt_info["expectedSlots"],
    )
    if rt_info.get("aicCoreNum") is None:
        logging.warning(
            "  未提供 --aic-core-num，将显示文件中的全部 slot，C/V 核分类未知。"
        )
    else:
        logging.info("  %-30s: %d", "aicCoreNum", rt_info["aicCoreNum"])
        logging.info("  %-30s: %d", "aivCoreNum", rt_info["aivCoreNum"])
    logging.info("  %-30s: %d", "commitTotal(valid slots)", rt_info["commitTotal"])
    logging.info("  %-30s: %d", "waitTotal(valid slots)", rt_info["waitTotal"])
    logging.info("")
    if rt_info["validExpectedSlots"] == 0:
        logging.warning(
            "  预期核对应的 slot 均未发现 magic 0x%08X。", RUNTIME_INFO_MAGIC
        )
        logging.warning(
            "  这表示 StateDump 未被 Kernel 写入，或 Dump 的 StateDump 起始地址/时机不正确；"
            "下表仍显示每个 slot 的原始 commit/wait 字节。"
        )
    if rt_info["unalignedMagicOffsets"]:
        logging.warning(
            "  在非 512B 对齐位置发现 magic，可能存在段起点错位: %s",
            rt_info["unalignedMagicOffsets"],
        )

    logging.info("  --- 每核通信计数（valid=no 时数值只是未初始化的原始字节） ---")
    logging.info(
        "  %-5s %-4s %-5s %-10s %-5s %-22s %-14s %-6s %-6s",
        "slot",
        "核型",
        "valid",
        "storedCore",
        "turn",
        "position",
        "phase",
        "commit",
        "wait",
    )
    for slot in rt_info["slots"][: rt_info["expectedSlots"]]:
        is_valid = slot["magicValid"]
        stored_core = str(slot["coreId"]) if is_valid else "-"
        turn = str(slot["execTurn"]) if is_valid else "-"
        position = (
            f"{slot['execPosition']}/{slot['execPositionName']}" if is_valid else "-"
        )
        phase = slot["commPhaseName"] if is_valid else "-"
        logging.info(
            "  %-5d %-4s %-5s %-10s %-5s %-22s %-14s %-6d %-6d",
            slot["slotIdx"],
            slot["coreType"],
            "yes" if is_valid else "no",
            stored_core,
            turn,
            position,
            phase,
            slot["commCommitCount"],
            slot["commWaitCount"],
        )


def print_workspace_segment(result):
    seg_type = result["segmentType"]
    print_section(
        f"Workspace Segment (type={seg_type}/{result['segmentTypeName']}, {result['fileSize']}B)"
    )
    logging.info("  %-30s: %s", "dataMeaning", result["segmentDescription"])
    if seg_type == 12:
        print_runtime_info_segment(result["runtimeInfo"])
    else:
        values = result.get("decimalPreview", [])
        logging.info("  %-30s: %s", "previewType", result.get("previewType"))
        logging.info("  %-30s: %s", "previewRange", f"byte[0:{len(values)}]")
        logging.info("  %-30s: %s", "previewAllZero", result.get("previewAllZero"))
        logging.info("  --- 十进制预览（每个值对应一个原始 uint8 字节） ---")
        for offset in range(0, len(values), 16):
            row = values[offset : offset + 16]
            logging.info(
                "  byte[%04d:%04d] : %s",
                offset,
                offset + len(row),
                " ".join(f"{value:3d}" for value in row),
            )
        logging.info(
            "  如需按 BF16/FP16/FP8 等真实类型查看矩阵，请使用 parse_matrix.py 并指定 dtype/rows/cols。"
        )


def select_branch(data, layout):
    """按 Mc2InitTiling/Mc2CcTiling 后的 DfxDumpInfo 返回 WorkspaceLayout。"""
    is_quant = detect_quant_mode(data, layout)
    ws = layout["workspace"]
    off = ws["layoutOffset"]
    ws_layout = parse_workspace_layout_info(data, off, ws)
    if workspace_layout_is_valid(ws_layout, ws):
        return is_quant, ws_layout
    logging.warning(
        "offset %d 的 WorkspaceLayoutInfo 未通过合法性校验，"
        "该 dump 可能不是当前 mc2Init/mc2Cc/dumpInfo 布局，或段表尚未写入",
        off,
    )
    return is_quant, None


def main():
    parser = argparse.ArgumentParser(description="MatmulReduceScatterV2 dump 解析脚本")
    parser.add_argument("bin_file", help="bin 文件路径")
    parser.add_argument("--args", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument("--aic-core-num", type=int, metavar="N", default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()
    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    if args.workspace:
        layout = get_layout()
        seg_info = detect_workspace_seg_from_filename(args.bin_file)
        if seg_info is None:
            logging.error("文件名不匹配 workspace_seg{idx}_type{type}_* 模式")
            sys.exit(1)
        seg_idx, seg_type = seg_info
        result = parse_workspace_segment_file(
            data,
            seg_type,
            args.aic_core_num,
            layout["workspace"]["runtimeInfoPerCoreSize"],
        )
        result["segIdx"] = seg_idx
        print_workspace_segment(result)
        return

    if args.args:
        result = parse_args_info(data)
        print_section("Args Info")
        for k, v in result.items():
            if not k.startswith("_"):
                logging.info("  %-30s: %s", k, v)
        return

    layout = get_layout()

    is_quant, ws_layout = select_branch(data, layout)
    logging.info(
        "解析模式: %s（新版 TilingData 不再保存 quantMode 字段）",
        "量化" if is_quant else "非量化",
    )
    result = parse_tiling_data(data, layout, is_quant)
    if ws_layout is not None:
        result["workspaceLayout"] = ws_layout
    print_tiling_data(result, is_quant, layout)
    if "workspaceLayout" in result:
        print_section("WorkspaceLayoutInfo")
        print_kv("layout", result["workspaceLayout"])


if __name__ == "__main__":
    main()
