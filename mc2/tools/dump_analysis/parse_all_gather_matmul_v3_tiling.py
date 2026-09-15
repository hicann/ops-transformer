#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.

# !
#  \file parse_all_gather_matmul_v3_tiling.py
#  \brief AllGatherMatmulV3 (AllGatherMxMatmulUrma) TilingData / Workspace / RuntimeInfo /
#         CommContext dump 解析脚本。
#         解析异常 dump 产生的 tiling_data_*.bin、args_info_*.bin、
#         workspace_seg*_type*_*.bin 或 comm_context_*.bin 文件。
#
# 用法:
#   python parse_all_gather_matmul_v3_tiling.py <bin_file> [--args]
#                                                   [--workspace] [--comm-ctx]
#                                                   [--aic-core-num N]
#
# 模式:
#   默认         解析 tiling_data_*.bin（AllGatherMxMatmulUrmaTilingData, 536B）
#   --args       解析 args_info_*.bin（GM 地址指针）
#   --workspace  解析 workspace_seg*_type*_*.bin（type=12 解析 StateDump）
#   --comm-ctx   解析 comm_context_*.bin（CommContext = CommUdmaContext + CommUbmemContext, 1552B）

import sys
import os
import struct
import argparse
import logging

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ====== 结构体大小常量（与 apace C++ 定义保持一致）======
# AllGatherMxMatmulUrmaTilingData 内存布局（536B）:
#   DfxDumpInfo           dumpInfo         offset 0,   424B
#     WorkspaceLayoutInfo workspaceLayout  offset 0,   208B
#     uint64_t            peermemDataSize  offset 208, 8B
#     PeermemLayoutInfo   peermemLayout    offset 216, 208B
#   QuantMatmulTilingData mmTile           offset 424, 64B
#   CommTilingData        commTile         offset 488, 40B
#   uint8_t               isBias           offset 528, 1B  (+7B padding)
ALL_GATHER_MX_MATMUL_URMA_TILING_DATA_SIZE = 536
TILING_DUMP_SIZE = 2048

# QuantMatmulTilingData (64B = 14*u32 + 3*u8 + 5B padding)
QUANT_MATMUL_TILING_DATA_SIZE = 64

# CommTilingData (40B = 5*u64)
COMM_TILING_DATA_SIZE = 40

# ====== Workspace 结构体大小常量（与 mc2_tiling_struct.h 保持一致）======
WORKSPACE_SEG_INFO_SIZE = 24  # 8+8+1+7
MAX_WORKSPACE_SEGMENTS = 8
WORKSPACE_LAYOUT_INFO_SIZE = (
    16 + MAX_WORKSPACE_SEGMENTS * WORKSPACE_SEG_INFO_SIZE
)  # 208
RUNTIME_INFO_PER_CORE_SIZE = 512
RUNTIME_INFO_CORE_MAX = 128
RUNTIME_INFO_TOTAL_SIZE = RUNTIME_INFO_CORE_MAX * RUNTIME_INFO_PER_CORE_SIZE  # 65536
RUNTIME_INFO_MAGIC = 0x5A5A5A5A

# ====== 偏移量常量 ======
DFX_DUMP_INFO_SIZE = 424
OFFSET_WS_LAYOUT = 0
OFFSET_PEERMEM_DATA_SIZE = WORKSPACE_LAYOUT_INFO_SIZE  # 208
OFFSET_PEERMEM_LAYOUT = OFFSET_PEERMEM_DATA_SIZE + 8  # 216
OFFSET_MM_TILE = DFX_DUMP_INFO_SIZE  # 424
OFFSET_COMM_TILE = OFFSET_MM_TILE + QUANT_MATMUL_TILING_DATA_SIZE  # 488
OFFSET_IS_BIAS = OFFSET_COMM_TILE + COMM_TILING_DATA_SIZE  # 528

# ====== CommContext 常量（与 Apace::AivComm::CommContext / moe_distribute_comm_ctx.h 镜像保持一致）======
URMA_MAX_RANK_NUM = 64
# CommUdmaContext (数据面): rankId + rankSize + channelHandles[64] + commBufferAddrs[64]
COMM_UDMA_CONTEXT_SIZE = 4 + 4 + URMA_MAX_RANK_NUM * 8 * 2  # 1032B
# CommUbmemContext (同步面): rankId + rankSize + commBufferAddrs[64]，无 channelHandles
COMM_UBMEM_CONTEXT_SIZE = 4 + 4 + URMA_MAX_RANK_NUM * 8  # 520B
# CommContext: udmaCtx 在前，ubmemCtx 紧随其后（offset 1032）
COMM_CONTEXT_SIZE = COMM_UDMA_CONTEXT_SIZE + COMM_UBMEM_CONTEXT_SIZE  # 1552B

# args_info_*.bin 中 kernel 参数索引（注册算子 all_gather_matmul_v3）
ARGS_INDEX_NAMES = [
    "context",  # [0] CommContext*
    "x1",  # [1] aGM (input A)
    "x2",  # [2] bGM (input B)
    "bias",  # [3] biasGM
    "x1_scale",  # [4] aScaleGM
    "x2_scale",  # [5] bScaleGM
    "y",  # [6] cGM (output)
    "gather_out",  # [7] (unused)
    "amax_out",  # [8] (unused)
    "workspaceGM",  # [9] workspace
    "tilingGM",  # [10] tiling data pointer
]

# Workspace 段类型枚举（与 mc2_tiling_struct.h WorkspaceSegType 保持一致）
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

PEERMEM_SEG_TYPE_MAP = {
    0: "DATA",
    1: "SCALE",
    2: "BIAS",
    3: "FLAG",
    255: "RESERVED",
}

# RuntimePhase 枚举（与 mc2_tiling_struct.h RuntimePhase 保持一致）
RUNTIME_PHASE_MAP = {
    0: "COMM_INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "COMM_FINALIZE",
    255: "RESERVED",
}

# PosTag 映射 (AllGatherMatmulV3 算子, 值由 all_gather_mx_matmul_urma_impl.h 就地定义)
# 来源: all_gather_mx_matmul_urma_impl.h
POS_TAG_MAP_AGMMV3 = {
    0: "COMM_BEFORE",
    1: "COMP_CUBE_MATMUL_BEFORE",
}


def safe_str(raw: bytes) -> str:
    """将 C 风格 char[] 转为 Python 字符串，截断于第一个 '\\0'。"""
    idx = raw.find(b"\x00")
    if idx >= 0:
        raw = raw[:idx]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def parse_quant_matmul_tiling_data(data: bytes, offset: int) -> dict:
    """解析 QuantMatmulTilingData（64B）。

    布局: 14*u32(56B) + 3*u8(3B) + 5B padding
    字段顺序与 quant_matmul_tiling_data.h 保持一致。
    """
    if offset + QUANT_MATMUL_TILING_DATA_SIZE > len(data):
        return {"error": "data too short for QuantMatmulTilingData"}
    u32_fields = struct.unpack_from("<14I", data, offset)
    step_k, n_buffer_num, db_l0c = struct.unpack_from("<3B", data, offset + 56)
    return {
        "m": u32_fields[0],
        "n": u32_fields[1],
        "k": u32_fields[2],
        "baseM": u32_fields[3],
        "baseN": u32_fields[4],
        "baseK": u32_fields[5],
        "scaleKL1": u32_fields[6],
        "mTailTile": u32_fields[7],
        "nTailTile": u32_fields[8],
        "mBaseTailSplitCnt": u32_fields[9],
        "nBaseTailSplitCnt": u32_fields[10],
        "mTailMain": u32_fields[11],
        "nTailMain": u32_fields[12],
        "usedCoreNum": u32_fields[13],
        "stepK": step_k,
        "nBufferNum": n_buffer_num,
        "dbL0c": db_l0c,
    }


def parse_comm_tiling_data(data: bytes, offset: int) -> dict:
    """解析 CommTilingData（40B = 5 * uint64_t）。"""
    if offset + COMM_TILING_DATA_SIZE > len(data):
        return {"error": "data too short for CommTilingData"}
    fields = struct.unpack_from("<5Q", data, offset)
    return {
        "splitAxisTileSize": fields[0],
        "splitAxisTileCnt": fields[1],
        "splitAxisTailSize": fields[2],
        "splitAxisTailCnt": fields[3],
        "nonSplitAxisSize": fields[4],
    }


def parse_workspace_seg_info(data: bytes, offset: int) -> dict:
    """解析单个 WorkspaceSegInfo（24B）。"""
    if offset + WORKSPACE_SEG_INFO_SIZE > len(data):
        return {"error": "data too short for WorkspaceSegInfo"}
    seg_offset, seg_size = struct.unpack_from("<2Q", data, offset)
    seg_type = struct.unpack_from("<B", data, offset + 16)[0]
    return {
        "offset": seg_offset,
        "size": seg_size,
        "type": seg_type,
        "typeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})"),
    }


def parse_workspace_layout_info(data: bytes, offset: int) -> dict:
    """解析 WorkspaceLayoutInfo（208B）。"""
    if offset + WORKSPACE_LAYOUT_INFO_SIZE > len(data):
        return {
            "error": f"data too short for WorkspaceLayoutInfo (need offset {offset} + {WORKSPACE_LAYOUT_INFO_SIZE})"
        }
    total_size, seg_count, reserved = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for i in range(min(seg_count, MAX_WORKSPACE_SEGMENTS)):
        seg_offset = offset + 16 + i * WORKSPACE_SEG_INFO_SIZE
        segments.append(parse_workspace_seg_info(data, seg_offset))
    return {
        "totalSize": total_size,
        "segCount": seg_count,
        "segments": segments,
    }


def parse_peermem_layout_info(data: bytes, offset: int) -> dict:
    """解析 PeermemLayoutInfo（208B），结构与 workspace layout 相同但类型枚举不同。"""
    if offset + WORKSPACE_LAYOUT_INFO_SIZE > len(data):
        return {
            "error": f"data too short for PeermemLayoutInfo (need offset {offset} + {WORKSPACE_LAYOUT_INFO_SIZE})"
        }
    total_size, seg_count, _ = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for index in range(min(seg_count, MAX_WORKSPACE_SEGMENTS)):
        seg_offset = offset + 16 + index * WORKSPACE_SEG_INFO_SIZE
        item_offset, item_size = struct.unpack_from("<2Q", data, seg_offset)
        item_type = struct.unpack_from("<B", data, seg_offset + 16)[0]
        segments.append(
            {
                "offset": item_offset,
                "size": item_size,
                "type": item_type,
                "typeName": PEERMEM_SEG_TYPE_MAP.get(
                    item_type, f"UNKNOWN({item_type})"
                ),
            }
        )
    return {"totalSize": total_size, "segCount": seg_count, "segments": segments}


def parse_tiling_data(data: bytes) -> dict:
    """解析 AllGatherMxMatmulUrmaTilingData（536B）。"""
    if len(data) < ALL_GATHER_MX_MATMUL_URMA_TILING_DATA_SIZE:
        logging.warning(
            "文件大小 %d < 结构体大小 %d，解析可能不完整",
            len(data),
            ALL_GATHER_MX_MATMUL_URMA_TILING_DATA_SIZE,
        )

    result = {
        "mmTile": parse_quant_matmul_tiling_data(data, OFFSET_MM_TILE),
        "commTile": parse_comm_tiling_data(data, OFFSET_COMM_TILE),
        "isBias": struct.unpack_from("<B", data, OFFSET_IS_BIAS)[0]
        if len(data) > OFFSET_IS_BIAS
        else None,
        "workspaceLayout": parse_workspace_layout_info(data, OFFSET_WS_LAYOUT),
        "peermemDataSize": struct.unpack_from("<Q", data, OFFSET_PEERMEM_DATA_SIZE)[0]
        if len(data) >= OFFSET_PEERMEM_DATA_SIZE + 8
        else None,
        "peermemLayout": parse_peermem_layout_info(data, OFFSET_PEERMEM_LAYOUT),
    }
    return result


def parse_args_info(data: bytes) -> dict:
    """解析 args_info_*.bin（AllGatherMatmulV3 有 11 个有效 args，index 0-10）。"""
    num_valid_args = len(ARGS_INDEX_NAMES)
    num_ptrs = min(num_valid_args, len(data) // 8)
    ptrs = struct.unpack_from(f"<{num_ptrs}Q", data, 0)
    args = {}
    for i in range(num_ptrs):
        name = ARGS_INDEX_NAMES[i]
        args[f"[{i}] {name}"] = f"0x{ptrs[i]:016x}"
    args["_dumpFileSize"] = len(data)
    args["_validArgCount"] = num_ptrs
    args["_note"] = (
        f"dump 代码固定拷贝 min(devArgsLen, 256) 字节，AllGatherMatmulV3 有效 args 为 {num_ptrs} 个。"
        f" 实际 devArgsLen 请查看日志中 'Tiling dump start' 行的 devArgsLen 字段。"
    )
    return args


def parse_runtime_info_per_core(data: bytes, offset: int) -> dict:
    """解析单个 StateDumpPerCore（512B）。"""
    if offset + RUNTIME_INFO_PER_CORE_SIZE > len(data):
        return {"error": "data too short for StateDumpPerCore"}
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
        "execPositionName": POS_TAG_MAP_AGMMV3.get(
            exec_position, f"UNKNOWN({exec_position})"
        ),
        "commPhase": comm_phase,
        "commPhaseName": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "commCommitCount": commit_count,
        "commWaitCount": wait_count,
    }


def parse_runtime_info_segment(data: bytes, aic_core_num=None) -> dict:
    """解析 workspace_segN_type12_*.bin（RuntimeInfo 段，64KB = 128 * 512B）。

    aic_core_num: tiling data 中的 mmTile.usedCoreNum，用于区分 C/V 核。
                  AIC coreId 范围 0 ~ aicCoreNum-1; AIV coreId 范围 aicCoreNum ~ 2*aicCoreNum-1
                  (MIX_AIC_1_1 模式下 AIV slotIdx = aicCoreNum + GetBlockIdx())。
                  为 None 时 coreType 标记为 "未知"。
    """
    max_slots = len(data) // RUNTIME_INFO_PER_CORE_SIZE
    slots = []
    active_count = 0
    for i in range(max_slots):
        offset = i * RUNTIME_INFO_PER_CORE_SIZE
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


def detect_workspace_seg_from_filename(filename: str) -> tuple:
    """从文件名检测是否为 workspace_seg 文件，返回 (seg_idx, seg_type) 或 None。"""
    import re

    basename = os.path.basename(filename)
    m = re.match(r"workspace_seg(\d+)_type(\d+)_", basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_workspace_segment_file(data: bytes, seg_type: int, aic_core_num=None) -> dict:
    """解析 workspace_seg*_type*_*.bin 文件，根据段类型选择解析方式。"""
    type_name = WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})")
    result = {
        "segmentType": seg_type,
        "segmentTypeName": type_name,
        "fileSize": len(data),
    }
    if seg_type == 12:  # WS_SEG_RUNTIME_INFO
        rt_info = parse_runtime_info_segment(data, aic_core_num)
        result["runtimeInfo"] = rt_info
    else:
        result["hexPreview"] = data[:256].hex()
        result["note"] = (
            f"段类型 {type_name}，原始二进制数据，请结合 tiling 参数人工分析。"
        )
    return result


def parse_urma_comm_context(data: bytes) -> dict:
    """解析 comm_context_*.bin（CommContext = CommUdmaContext + CommUbmemContext, 1552B）。

    udmaCtx 布局 (offset 0, 1032B):
      uint32_t rankId
      uint32_t rankSize
      uint64_t channelHandles[64]   (512B)
      uint64_t commBufferAddrs[64]  (512B)
    ubmemCtx 布局 (offset 1032, 520B):
      uint32_t rankId
      uint32_t rankSize
      uint64_t commBufferAddrs[64]  (512B)  ← [rankId] 为本 rank barrier sync buffer 地址
    """
    if len(data) < COMM_CONTEXT_SIZE:
        return {
            "error": f"data too short for CommContext (need {COMM_CONTEXT_SIZE}, got {len(data)})"
        }
    rank_id, rank_size = struct.unpack_from("<2I", data, 0)
    channel_handles = list(struct.unpack_from(f"<{URMA_MAX_RANK_NUM}Q", data, 8))
    comm_buffer_addrs = list(
        struct.unpack_from(f"<{URMA_MAX_RANK_NUM}Q", data, 8 + URMA_MAX_RANK_NUM * 8)
    )

    ub_rank_id, ub_rank_size = struct.unpack_from("<2I", data, COMM_UDMA_CONTEXT_SIZE)
    ub_comm_buffer_addrs = list(
        struct.unpack_from(f"<{URMA_MAX_RANK_NUM}Q", data, COMM_UDMA_CONTEXT_SIZE + 8)
    )

    # 仅保留非零的 channel/buffer 地址
    channels = {}
    buffers = {}
    for i in range(URMA_MAX_RANK_NUM):
        if channel_handles[i] != 0:
            channels[i] = f"0x{channel_handles[i]:016x}"
        if comm_buffer_addrs[i] != 0:
            buffers[i] = f"0x{comm_buffer_addrs[i]:016x}"

    ub_buffers = {}
    for i in range(URMA_MAX_RANK_NUM):
        if ub_comm_buffer_addrs[i] != 0:
            ub_buffers[i] = f"0x{ub_comm_buffer_addrs[i]:016x}"

    self_win_addr = comm_buffer_addrs[rank_id] if rank_id < URMA_MAX_RANK_NUM else 0
    self_sync_addr = (
        ub_comm_buffer_addrs[ub_rank_id] if ub_rank_id < URMA_MAX_RANK_NUM else 0
    )
    return {
        "rankId": rank_id,
        "rankSize": rank_size,
        "selfWinAddr": f"0x{self_win_addr:016x}" if self_win_addr else "0x0",
        "channelHandles": channels,
        "commBufferAddrs": buffers,
        "ubmemCtx": {
            "rankId": ub_rank_id,
            "rankSize": ub_rank_size,
            "selfSyncBufAddr": f"0x{self_sync_addr:016x}" if self_sync_addr else "0x0",
            "commBufferAddrs": ub_buffers,
        },
        "fileSize": len(data),
    }


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


def print_tiling_data(result: dict):
    """人类可读格式打印 tiling data。"""
    print_section("QuantMatmulTilingData (mmTile, 64B)")
    print_kv("mmTile", result["mmTile"])

    print_section("CommTilingData (commTile, 40B)")
    print_kv("commTile", result["commTile"])

    print_section("Extra Fields")
    print_kv(
        "extra",
        {
            "isBias": result.get("isBias"),
            "peermemDataSize": result.get("peermemDataSize"),
        },
    )

    ws_layout = result.get("workspaceLayout")
    if ws_layout and "error" not in ws_layout:
        print_workspace_layout_info(ws_layout)

    peermem_layout = result.get("peermemLayout")
    if peermem_layout and "error" not in peermem_layout:
        print_section("PeermemLayoutInfo (208B)")
        print_kv("peermemLayout", peermem_layout)


def print_args_info(result: dict):
    """人类可读格式打印 args info。"""
    print_section("Args Info (kernel GM pointer)")
    dump_file_size = result.pop("_dumpFileSize", 0)
    valid_arg_count = result.pop("_validArgCount", 0)
    note = result.pop("_note", "")
    for name, addr in result.items():
        if name.startswith("_"):
            continue
        logging.info("  %-30s: %s", name, addr)
    logging.info("")
    logging.info(
        "  有效 args 数量: %d (index 0-%d)", valid_arg_count, valid_arg_count - 1
    )
    logging.info("  dump 文件大小: %d bytes", dump_file_size)
    logging.info("  提示: tilingGM 应在 [10] 位置 (offset=80)")
    tiling_gm = result.get("[10] tilingGM", "N/A")
    logging.info("  [10] tilingGM = %s", tiling_gm)
    if note:
        logging.info("  %s", note)


def print_workspace_layout_info(ws_layout: dict):
    """人类可读格式打印 WorkspaceLayoutInfo。"""
    if "error" in ws_layout:
        logging.error("  %s", ws_layout["error"])
        return
    print_section("WorkspaceLayoutInfo (208B)")
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
    """人类可读格式打印 RuntimeInfo 段。"""
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
        logging.warning(
            "  请先解析 tiling_data_*.bin 获取 mmTile.usedCoreNum，再通过 --aic-core-num N 传入。"
        )
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
        logging.warning(
            "  无有效 slot，可能 runtime info 未被写入（异常发生在 Init 之前）。"
        )
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
    """人类可读格式打印 workspace 段文件。"""
    seg_type = result["segmentType"]
    type_name = result["segmentTypeName"]
    print_section(
        f"Workspace Segment (type={seg_type}/{type_name}, {result['fileSize']}B)"
    )
    if seg_type == 12:  # RUNTIME_INFO
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
        logging.info("")
        logging.info("  %s", result.get("note", ""))


def print_urma_comm_context(result: dict):
    """人类可读格式打印 CommContext。"""
    if "error" in result:
        print_section("CommContext (CommUdmaContext + CommUbmemContext)")
        logging.error("  %s", result["error"])
        return
    print_section(
        f"CommContext (CommUdmaContext + CommUbmemContext, {result['fileSize']}B)"
    )
    logging.info("  --- udmaCtx (数据面, offset 0, %dB) ---", COMM_UDMA_CONTEXT_SIZE)
    logging.info("  %-30s: %d", "rankId", result["rankId"])
    logging.info("  %-30s: %d", "rankSize", result["rankSize"])
    logging.info("  %-30s: %s", "selfWinAddr", result["selfWinAddr"])
    logging.info("")
    logging.info("  --- channelHandles (非零) ---")
    for idx, addr in result["channelHandles"].items():
        logging.info("    %-28s: %s", f"channel[{idx}]", addr)
    logging.info("")
    logging.info("  --- commBufferAddrs (非零) ---")
    for idx, addr in result["commBufferAddrs"].items():
        logging.info("    %-28s: %s", f"buffer[{idx}]", addr)

    ubmem = result.get("ubmemCtx", {})
    logging.info("")
    logging.info(
        "  --- ubmemCtx (同步面, offset %d, %dB) ---",
        COMM_UDMA_CONTEXT_SIZE,
        COMM_UBMEM_CONTEXT_SIZE,
    )
    logging.info("  %-30s: %s", "rankId", ubmem.get("rankId"))
    logging.info("  %-30s: %s", "rankSize", ubmem.get("rankSize"))
    logging.info("  %-30s: %s", "selfSyncBufAddr", ubmem.get("selfSyncBufAddr"))
    logging.info("")
    logging.info("  --- ubmemCtx commBufferAddrs (非零) ---")
    for idx, addr in ubmem.get("commBufferAddrs", {}).items():
        logging.info("    %-28s: %s", f"syncBuf[{idx}]", addr)


def main():
    parser = argparse.ArgumentParser(
        description="AllGatherMatmulV3 (AllGatherMxMatmulUrma) TilingData / Workspace / RuntimeInfo / "
        "CommContext dump 解析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 解析 tiling data
  python parse_all_gather_matmul_v3_tiling.py tiling_data_AllGatherMatmulV3.0.1.20260819.bin

  # 解析 args_info（验证 tilingGM offset）
  python parse_all_gather_matmul_v3_tiling.py args_info_AllGatherMatmulV3.0.1.20260819.bin --args

  # 解析 workspace 段文件（自动从文件名识别段类型）
  python parse_all_gather_matmul_v3_tiling.py workspace_seg1_type12_AllGatherMatmulV3.0.1.20260819.bin --workspace

  # 解析 workspace 段文件并区分 C/V 核
  python parse_all_gather_matmul_v3_tiling.py workspace_seg1_type12_*.bin --workspace --aic-core-num 24

  # 解析 comm_context（通信拓扑）
  python parse_all_gather_matmul_v3_tiling.py comm_context_AllGatherMatmulV3.0.1.20260819.bin --comm-ctx

        """,
    )
    parser.add_argument("bin_file", help="dump 产生的 bin 文件路径")
    parser.add_argument(
        "--args", action="store_true", help="解析 args_info_*.bin（GM 地址指针）"
    )
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="解析 workspace_seg*_type*_*.bin（段文件，type=12 自动解析 StateDump）",
    )
    parser.add_argument(
        "--comm-ctx",
        action="store_true",
        help="解析 comm_context_*.bin（CommContext = CommUdmaContext + CommUbmemContext, 1552B, 通信拓扑）",
    )
    parser.add_argument(
        "--aic-core-num",
        type=int,
        metavar="N",
        default=None,
        help="AIC 核数，用于 workspace 段解析时区分 C/V 核。"
        " 从 tiling_data 解析结果的 mmTile.usedCoreNum 字段获取。"
        " 未提供时 C/V 核分类不可靠（显示为'未知'）。",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()

    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    # --- comm_context 解析 ---
    if args.comm_ctx:
        result = parse_urma_comm_context(data)
        print_urma_comm_context(result)
        return

    # --- workspace 段文件解析 ---
    if args.workspace:
        seg_info = detect_workspace_seg_from_filename(args.bin_file)
        if seg_info is None:
            logging.error(
                "文件名不匹配 workspace_seg{idx}_type{type}_* 模式: %s", args.bin_file
            )
            logging.error("请确保文件名格式为 workspace_segN_typeM_*。")
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
        result = parse_workspace_segment_file(data, seg_type, args.aic_core_num)
        result["segIdx"] = seg_idx
        print_workspace_segment(result)
        return

    # --- args_info 解析 ---
    if args.args:
        result = parse_args_info(data)
        print_args_info(result)
        return

    # --- 默认: tiling_data 解析 ---
    logging.info("解析模式: AllGatherMxMatmulUrmaTilingData (MX量化, 536B)")

    struct_size = ALL_GATHER_MX_MATMUL_URMA_TILING_DATA_SIZE
    if len(data) < struct_size:
        logging.warning(
            "文件大小 %d < 结构体大小 %d，解析可能不完整", len(data), struct_size
        )

    result = parse_tiling_data(data)
    logging.info(
        "结构体 AllGatherMxMatmulUrmaTilingData: %dB, dump 文件 %dB, 尾部 %dB 为 padding",
        struct_size,
        len(data),
        max(0, len(data) - struct_size),
    )

    # 如果提供了 --aic-core-num，记录用于 runtime info 解析提示
    if args.aic_core_num is not None:
        used_core_num = result.get("mmTile", {}).get("usedCoreNum", "N/A")
        logging.info(
            "mmTile.usedCoreNum=%s, --aic-core-num=%d", used_core_num, args.aic_core_num
        )

    print_tiling_data(result)


if __name__ == "__main__":
    main()
