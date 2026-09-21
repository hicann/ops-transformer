#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.

# !
#  \file parse_allto_all_matmul_tiling.py
#  \brief AlltoAllMatmul TilingData / Workspace / RuntimeInfo 解析脚本。
#         解析异常 dump 产生的 tiling_data_*.bin、args_info_*.bin 或 workspace_seg*.bin 文件。
#
# 用法:
#   python parse_allto_all_matmul_tiling.py <bin_file> [--args]
#
# 模式:
#   默认         自动识别主线非量化、主线量化或 Apace CCU tiling_data_*.bin
#   --args       解析 args_info_*.bin（GM 地址指针）
#   --workspace  解析 workspace_seg*_type*_*.bin（自动识别段类型，type=12 解析 RuntimeInfo）

import sys
import os
import struct
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiling_layout import LayoutError, compute_allto_all_layout

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ====== 固定结构体大小常量（DFX 自有, 不随 CANN 版本漂移）======
MC2_INIT_TILING_SIZE = 64
MC2_CC_TILING_SIZE = 512
ALLTO_ALL_MATMUL_TILING_INFO_SIZE = 88  # 10*u32(40) + 6*u64(48)

# ====== 以下常量依赖 TCubeTiling (随 CANN 版本漂移), 由 tiling_layout 实时计算 ======
_LAYOUT = compute_allto_all_layout()

TCUBE_TILING_SIZE = _LAYOUT["tcubeTilingSize"]
MC2_MM_V3_TILING_DATA_SIZE = _LAYOUT["nonQuant"]["blockSize"]  # TCubeTiling + 9*u32
MC2_QUANT_BMM_TILING_PARAMS_SIZE = _LAYOUT["quant"][
    "blockSize"
]  # 128 + TCubeTiling + 24 + 12

ALLTO_ALL_MATMUL_TILING_PREFIX_SIZE = _LAYOUT["nonQuant"]["tileOffset"]
ALLTO_ALL_QUANT_MATMUL_TILING_PREFIX_SIZE = _LAYOUT["quant"]["tileOffset"]
ALLTO_ALL_MATMUL_TILING_DATA_SIZE = _LAYOUT["nonQuant"]["size"]
ALLTO_ALL_QUANT_MATMUL_TILING_DATA_SIZE = _LAYOUT["quant"]["size"]
APACE_ALLTO_ALL_MATMUL_TILING_DATA_SIZE = _LAYOUT["apace"]["size"]
TILING_DUMP_SIZE = 2048  # dump 固定大小

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

# WorkspaceLayoutInfo 是 DfxDumpInfo 的首成员；当前三种格式的 dumpInfo
# 均位于 Mc2InitTiling + Mc2CcTiling 之后（默认 offset 576）。
OFFSET_WS_LAYOUT_NON_QUANT = _LAYOUT["nonQuant"]["wsLayoutOffset"]
OFFSET_WS_LAYOUT_QUANT = _LAYOUT["quant"]["wsLayoutOffset"]
OFFSET_APACE_COMM_TILING = _LAYOUT["apace"]["commOffset"]
OFFSET_APACE_QUANT_MATMUL = _LAYOUT["apace"]["quantMatmulOffset"]
OFFSET_APACE_LOCAL_MATMUL = _LAYOUT["apace"]["localMatmulOffset"]
OFFSET_WS_LAYOUT_APACE = _LAYOUT["apace"]["wsLayoutOffset"]


def init_layout(cann=None):
    """按当前 CANN 环境重新计算布局常量并打印布局摘要。"""
    global _LAYOUT
    try:
        layout = compute_allto_all_layout(cann=cann)
    except LayoutError as e:
        logging.error("布局计算失败: %s", e)
        sys.exit(1)
    _LAYOUT = layout
    g = globals()
    g["TCUBE_TILING_SIZE"] = layout["tcubeTilingSize"]
    g["MC2_MM_V3_TILING_DATA_SIZE"] = layout["nonQuant"]["blockSize"]
    g["MC2_QUANT_BMM_TILING_PARAMS_SIZE"] = layout["quant"]["blockSize"]
    g["ALLTO_ALL_MATMUL_TILING_PREFIX_SIZE"] = layout["nonQuant"]["tileOffset"]
    g["ALLTO_ALL_QUANT_MATMUL_TILING_PREFIX_SIZE"] = layout["quant"]["tileOffset"]
    g["ALLTO_ALL_MATMUL_TILING_DATA_SIZE"] = layout["nonQuant"]["size"]
    g["ALLTO_ALL_QUANT_MATMUL_TILING_DATA_SIZE"] = layout["quant"]["size"]
    g["APACE_ALLTO_ALL_MATMUL_TILING_DATA_SIZE"] = layout["apace"]["size"]
    g["OFFSET_WS_LAYOUT_NON_QUANT"] = layout["nonQuant"]["wsLayoutOffset"]
    g["OFFSET_WS_LAYOUT_QUANT"] = layout["quant"]["wsLayoutOffset"]
    g["OFFSET_WS_LAYOUT_APACE"] = layout["apace"]["wsLayoutOffset"]
    g["OFFSET_APACE_COMM_TILING"] = layout["apace"]["commOffset"]
    g["OFFSET_APACE_QUANT_MATMUL"] = layout["apace"]["quantMatmulOffset"]
    g["OFFSET_APACE_LOCAL_MATMUL"] = layout["apace"]["localMatmulOffset"]
    g["OFFSET_DFX_DUMP_INFO"] = layout["offsets"]["dumpInfo"]
    g["OFFSET_MC2_INIT_TILING"] = layout["offsets"]["mc2InitTiling"]
    g["OFFSET_MC2_CC_TILING"] = layout["offsets"]["mc2CcTiling"]
    g["OFFSET_TILING_INFO"] = layout["offsets"]["tilingInfo"]
    g["OFFSET_TILE_MM_TILING"] = layout["nonQuant"]["tileOffset"]
    g["OFFSET_TAIL_MM_TILING_NON_QUANT"] = (
        g["OFFSET_TILE_MM_TILING"] + layout["nonQuant"]["blockSize"]
    )
    g["OFFSET_TILE_QUANT_MM_TILING"] = layout["quant"]["tileOffset"]
    g["OFFSET_TAIL_QUANT_MM_TILING"] = layout["quant"]["tailOffset"]
    for h in layout["headers"]:
        logging.info("布局头文件: %s", h)
    for key, name in (
        ("nonQuant", "AlltoAllMatmulTilingData"),
        ("quant", "AlltoAllQuantMatmulTilingData"),
        ("apace", "Apace hcommAllToAllMatmulTilingData"),
    ):
        b = layout[key]
        logging.info(
            "布局计算: %s: %dB, workspaceLayout@%d",
            name,
            b["size"],
            b["wsLayoutOffset"],
        )
    return layout


# 量化模式枚举（x1QuantMode / x2QuantMode，与 op_proto.h 保持一致）
# x1QuantMode: 0=非量化, 7=动态pertoken量化(KC), 6=MX量化
# x2QuantMode: 0=非量化, 2=perchannel量化(KC), 6=MX量化
X1_QUANT_MODE_MAP = {
    0: "NON_QUANT",
    7: "KC_DYNAMIC_PERTOKEN",
    6: "MX_QUANT",
}
X2_QUANT_MODE_MAP = {
    0: "NON_QUANT",
    2: "KC_PERCHANNEL",
    6: "MX_QUANT",
}

# ====== 偏移量常量 ======
OFFSET_MC2_INIT_TILING = _LAYOUT["offsets"]["mc2InitTiling"]
OFFSET_MC2_CC_TILING = _LAYOUT["offsets"]["mc2CcTiling"]
OFFSET_DFX_DUMP_INFO = _LAYOUT["offsets"]["dumpInfo"]
OFFSET_TILING_INFO = _LAYOUT["offsets"]["tilingInfo"]
OFFSET_TILE_MM_TILING = _LAYOUT["nonQuant"]["tileOffset"]
OFFSET_TAIL_MM_TILING_NON_QUANT = _LAYOUT["nonQuant"]["tailOffset"]
OFFSET_TILE_QUANT_MM_TILING = _LAYOUT["quant"]["tileOffset"]
OFFSET_TAIL_QUANT_MM_TILING = _LAYOUT["quant"]["tailOffset"]

# args_info_*.bin 中 kernel 参数索引
ARGS_INDEX_NAMES = [
    "hccl_context",  # [0] runtime 注入
    "x1",  # [1]
    "x2",  # [2]
    "bias",  # [3]
    "x1_scale",  # [4]
    "x2_scale",  # [5]
    "comm_scale",  # [6]
    "x1_offset",  # [7]
    "x2_offset",  # [8]
    "y",  # [9]
    "all2all_out",  # [10]
    "workspaceGM",  # [11]
    "tilingGM",  # [12]
]

# HCCL 数据类型枚举
HCCL_DATA_TYPE_MAP = {
    0: "HCCL_DATA_TYPE_INT8",
    1: "HCCL_DATA_TYPE_INT16",
    2: "HCCL_DATA_TYPE_INT32",
    3: "HCCL_DATA_TYPE_INT64",
    4: "HCCL_DATA_TYPE_FLOAT16",
    5: "HCCL_DATA_TYPE_FLOAT32",
    6: "HCCL_DATA_TYPE_INT128",
    7: "HCCL_DATA_TYPE_FLOAT64",
    8: "HCCL_DATA_TYPE_BF16",
    9: "HCCL_DATA_TYPE_INT20",
    10: "HCCL_DATA_TYPE_UINT8",
    11: "HCCL_DATA_TYPE_UINT16",
    12: "HCCL_DATA_TYPE_UINT32",
    13: "HCCL_DATA_TYPE_UINT64",
    14: "HCCL_DATA_TYPE_FLOAT8_E5M2",
    15: "HCCL_DATA_TYPE_FLOAT8_E4M3FN",
    16: "HCCL_DATA_TYPE_BFLOAT16",
    17: "HCCL_DATA_TYPE_INT4",
    18: "HCCL_DATA_TYPE_UINT4",
    19: "HCCL_DATA_TYPE_FLOAT4_E2M1",
}

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

# RuntimePhase 枚举（与 mc2_tiling_struct.h RuntimePhase 保持一致, 仅 HCCL 五态）
RUNTIME_PHASE_MAP = {
    0: "COMM_INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "COMM_FINALIZE",
    255: "RESERVED",
}

# PosTag 映射 (allto_all_matmul 算子, 值由各 kernel 就地定义)
# 值与各 kernel 文件内 constexpr 一一对应，新增类别时同步更新此处
#
# 来源: arch35/allto_all_mx_quant_matmul_pipeline.h
POS_TAG_MAP_ALLTO_ALL = {
    0: "COMM_BEFORE",
    1: "SCALE_COMM_BEFORE",
    2: "COMP_CUBE_MATMUL_BEFORE",
    3: "COMP_VEC_PERMUTE_BEFORE",
}

# Apace 路径 (usingApaceImpl_) 的 position 枚举与主线不同:
# 见 common/op_kernel/apace/kernel/fusions/all_to_all_quant_matmul/
#     all_to_all_mx_quant_matmul_hcomm_impl.h 中的 POS_* 常量
POS_TAG_MAP_ALLTO_ALL_APACE = {
    0: "COMM_BEFORE",  # 通信下发前
    1: "COMP_LOCAL_BEFORE",  # local 块 matmul 前
    2: "COMP_REMOTE_BEFORE",  # 远端(通信结果) matmul 前
    3: "FINALIZE_BEFORE",  # hccl Finalize 前
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


def parse_mc2_init_tiling_inner(data: bytes, offset: int) -> dict:
    """解析 Mc2InitTilingInner（64B），实际布局见 CANN hccl_tiling_msg.h。"""
    if offset + MC2_INIT_TILING_SIZE > len(data):
        return {"error": "data too short for Mc2InitTiling"}
    version, mc2_hcomm_cnt = struct.unpack_from("<2I", data, offset)
    offsets = struct.unpack_from("<8I", data, offset + 8)
    debug_mode, prepare_position = struct.unpack_from("<2B", data, offset + 40)
    queue_num, comm_block_num = struct.unpack_from("<2H", data, offset + 42)
    dev_type = struct.unpack_from("<B", data, offset + 46)[0]
    return {
        "version": version,
        "mc2HcommCnt": mc2_hcomm_cnt,
        "offset": list(offsets),
        "debugMode": debug_mode,
        "preparePosition": prepare_position,
        "queueNum": queue_num,
        "commBlockNum": comm_block_num,
        "devType": dev_type,
    }


def parse_mc2_cc_tiling_inner(data: bytes, offset: int) -> dict:
    """解析 Mc2CcTilingInner（512B），实际布局见 CANN hccl_tiling_msg.h。"""
    if offset + MC2_CC_TILING_SIZE > len(data):
        return {"error": "data too short for Mc2CcTiling"}
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


def parse_allto_all_matmul_tiling_info(data: bytes, offset: int) -> dict:
    """解析当前源码的 AlltoAllMatmulTilingInfo（88B）。"""
    if offset + ALLTO_ALL_MATMUL_TILING_INFO_SIZE > len(data):
        return {"error": "data too short for AlltoAllMatmulTilingInfo"}
    fields = struct.unpack_from("<10I6Q", data, offset)
    hccl_dt = fields[12]
    return {
        "rankDim": fields[0],
        "tileM": fields[1],
        "tileCnt": fields[2],
        "tailM": fields[3],
        "tailCnt": fields[4],
        "biasLen": fields[5],
        "rankM": fields[6],
        "rankN": fields[7],
        "rankK": fields[8],
        "aicCoreNum": fields[9],
        "commLen": fields[10],
        "permuteLen": fields[11],
        "hcclDataType": hccl_dt,
        "hcclDataTypeName": HCCL_DATA_TYPE_MAP.get(hccl_dt, f"UNKNOWN({hccl_dt})"),
        "x1ScaleOptionalLen": fields[13],
        "dynamicExtraSpace": fields[14],
        "x1QuantDtype": fields[15],
    }


TCUBE_TILING_FIELDS = [
    "usedCoreNum",
    "M",
    "N",
    "Ka",
    "Kb",
    "singleCoreM",
    "singleCoreN",
    "singleCoreK",
    "baseM",
    "baseN",
    "baseK",
    "depthA1",
    "depthB1",
    "stepM",
    "stepN",
    "isBias",
    "transLength",
    "iterateOrder",
    "shareMode",
    "shareL1Size",
    "shareL0CSize",
    "shareUbSize",
    "batchM",
    "batchN",
    "singleBatchM",
    "singleBatchN",
    "stepKa",
    "stepKb",
    "depthAL1CacheUB",
    "depthBL1CacheUB",
    "dbL0A",
    "dbL0B",
    "dbL0C",
    "ALayoutInfoB",
    "ALayoutInfoS",
    "ALayoutInfoN",
    "ALayoutInfoG",
    "ALayoutInfoD",
    "BLayoutInfoB",
    "BLayoutInfoS",
    "BLayoutInfoN",
    "BLayoutInfoG",
    "BLayoutInfoD",
    "CLayoutInfoB",
    "CLayoutInfoS1",
    "CLayoutInfoN",
    "CLayoutInfoG",
    "CLayoutInfoS2",
    "BatchNum",
    "mxTypePara",
]

MC2_MM_V3_EXT_FIELDS = [
    "mTailCnt",
    "nTailCnt",
    "kTailCnt",
    "mBaseTailSplitCnt",
    "nBaseTailSplitCnt",
    "mTailMain",
    "nTailMain",
    "isHf32",
    "aswWindowLen",
]


def parse_tcube_tiling(data: bytes, offset: int) -> dict:
    """解析 TCubeTiling（大小 TCUBE_TILING_SIZE 随 CANN 版本由 tiling_layout 计算）。

    已知字段名覆盖当前 50 个 (200B); CANN 版本变化导致字段更多时,
    超出部分以 reservedFieldN 占位, 少于时仅解析已知字段。"""
    if offset + TCUBE_TILING_SIZE > len(data):
        return {"error": "data too short for TCubeTiling"}
    count = TCUBE_TILING_SIZE // 4
    values = struct.unpack_from(f"<{count}i", data, offset)
    names = TCUBE_TILING_FIELDS[:count] + [
        f"reservedField{i}" for i in range(len(TCUBE_TILING_FIELDS), count)
    ]
    return dict(zip(names, list(values)))


def parse_mc2_mm_v3_tiling_data(data: bytes, offset: int) -> dict:
    """解析 Mc2MatMulV3TilingData（236B = TCubeTiling + 9 × uint32）。"""
    if offset + MC2_MM_V3_TILING_DATA_SIZE > len(data):
        return {"error": "data too short for Mc2MatMulV3TilingData"}
    tcube = parse_tcube_tiling(data, offset)
    ext_values = struct.unpack_from(
        f"<{len(MC2_MM_V3_EXT_FIELDS)}I", data, offset + TCUBE_TILING_SIZE
    )
    ext = dict(zip(MC2_MM_V3_EXT_FIELDS, list(ext_values)))
    return {"tCubeTiling": tcube, **ext}


def parse_mc2_quant_bmm_data_params(data: bytes, offset: int) -> dict:
    """解析 Mc2QuantBatchMatmulV3DataParams（128B = 32 × uint32）。"""
    quant_fields = [
        "batchA",
        "batchB",
        "batchC",
        "batchA1",
        "batchA2",
        "batchA3",
        "batchA4",
        "batchB1",
        "batchB2",
        "batchB3",
        "batchB4",
        "batchC1",
        "batchC2",
        "batchC3",
        "batchC4",
        "singleCoreBatch",
        "isPerTensor",
        "isPertoken",
        "isDoubleScale",
        "biasThreeDim",
        "ubCalcM",
        "ubCalcN",
        "needUbBuffer",
        "realSingleCoreM",
        "realSingleCoreN",
        "biasDtype",
        "ubSize",
        "isMClash",
        "isNClash",
        "groupSizeM",
        "groupSizeN",
        "groupSizeK",
    ]
    if offset + 128 > len(data):
        return {"error": "data too short for Mc2QuantBatchMatmulV3DataParams"}
    values = struct.unpack_from(f"<{len(quant_fields)}I", data, offset)
    return dict(zip(quant_fields, list(values)))


def parse_mc2_l2cache_tile_params(data: bytes, offset: int) -> dict:
    """解析 Mc2L2cacheTileParams（24B = 6 × uint32）。"""
    l2_fields = [
        "mTileCntL2",
        "nTileCntL2",
        "mTileBlock",
        "nTileBlock",
        "calOrder",
        "isBasicTiling",
    ]
    if offset + 24 > len(data):
        return {"error": "data too short for Mc2L2cacheTileParams"}
    values = struct.unpack_from(f"<{len(l2_fields)}I", data, offset)
    return dict(zip(l2_fields, list(values)))


def parse_mc2_sliding_window_params(data: bytes, offset: int) -> dict:
    """解析 Mc2SlidingWindowParams（12B = 3 × uint32: mTailTile, nTailTile, alignNTailSplit）。"""
    if offset + 12 > len(data):
        return {"error": "data too short for Mc2SlidingWindowParams"}
    m_tail_tile, n_tail_tile, align_n_tail_split = struct.unpack_from(
        "<3I", data, offset
    )
    return {
        "mTailTile": m_tail_tile,
        "nTailTile": n_tail_tile,
        "alignNTailSplit": align_n_tail_split,
    }


def parse_mc2_quant_bmm_tiling_data_params(data: bytes, offset: int) -> dict:
    """解析 Mc2QuantBatchMatmulV3TilingDataParams（364B）。"""
    if offset + MC2_QUANT_BMM_TILING_PARAMS_SIZE > len(data):
        return {"error": "data too short for Mc2QuantBatchMatmulV3TilingDataParams"}
    params = parse_mc2_quant_bmm_data_params(data, offset)
    tcube = parse_tcube_tiling(data, offset + 128)
    l2cache = parse_mc2_l2cache_tile_params(data, offset + 128 + TCUBE_TILING_SIZE)
    sliding = parse_mc2_sliding_window_params(
        data, offset + 128 + TCUBE_TILING_SIZE + 24
    )
    return {
        "params": params,
        "matmulTiling": tcube,
        "tileL2cacheTiling": l2cache,
        "adaptiveSlidingWin": sliding,
    }


def parse_non_quant_tiling_data(data: bytes) -> dict:
    """解析非量化 AlltoAllMatmulTilingData（结构大小由 TCubeTiling 动态计算）。"""
    return {
        "mc2InitTiling": parse_mc2_init_tiling_inner(data, OFFSET_MC2_INIT_TILING),
        "mc2CcTiling": parse_mc2_cc_tiling_inner(data, OFFSET_MC2_CC_TILING),
        "alltoAllMatmulTilingInfo": parse_allto_all_matmul_tiling_info(
            data, OFFSET_TILING_INFO
        ),
        "mc2MmV3TileTilingData": parse_mc2_mm_v3_tiling_data(
            data, OFFSET_TILE_MM_TILING
        ),
        "mc2MmV3TailTilingData": parse_mc2_mm_v3_tiling_data(
            data, OFFSET_TAIL_MM_TILING_NON_QUANT
        ),
    }


def parse_quant_tiling_data(data: bytes) -> dict:
    """解析量化 AlltoAllQuantMatmulTilingData（结构大小由 TCubeTiling 动态计算）。"""
    return {
        "mc2InitTiling": parse_mc2_init_tiling_inner(data, OFFSET_MC2_INIT_TILING),
        "mc2CcTiling": parse_mc2_cc_tiling_inner(data, OFFSET_MC2_CC_TILING),
        "alltoAllQuantMatmulTilingInfo": parse_allto_all_matmul_tiling_info(
            data, OFFSET_TILING_INFO
        ),
        "mc2QuantMmTileTilingData": parse_mc2_quant_bmm_tiling_data_params(
            data, OFFSET_TILE_QUANT_MM_TILING
        ),
        "mc2QuantMmTailTilingData": parse_mc2_quant_bmm_tiling_data_params(
            data, OFFSET_TAIL_QUANT_MM_TILING
        ),
    }


def parse_apace_quant_matmul_tiling(data: bytes, offset: int) -> dict:
    """解析 Apace QuantMatmulTilingData（64B）。"""
    if offset + 64 > len(data):
        return {"error": "data too short for Apace QuantMatmulTilingData"}
    values = struct.unpack_from("<14I", data, offset)
    step_k, n_buffer_num, db_l0c = struct.unpack_from("<3B", data, offset + 56)
    names = (
        "m",
        "n",
        "k",
        "baseM",
        "baseN",
        "baseK",
        "scaleKL1",
        "mTailTile",
        "nTailTile",
        "mBaseTailSplitCnt",
        "nBaseTailSplitCnt",
        "mTailMain",
        "nTailMain",
        "usedCoreNum",
    )
    result = dict(zip(names, values))
    result.update({"stepK": step_k, "nBufferNum": n_buffer_num, "dbL0c": db_l0c})
    return result


def parse_apace_comm_tiling(data: bytes, offset: int) -> dict:
    """解析 Apace CommTilingData（48B = 6 * uint64_t）。"""
    if offset + 48 > len(data):
        return {"error": "data too short for Apace CommTilingData"}
    values = struct.unpack_from("<6Q", data, offset)
    names = (
        "splitAxisTileSize",
        "splitAxisTileCnt",
        "splitAxisTailSize",
        "splitAxisTailCnt",
        "nonSplitAxisSize",
        "slotNum",
    )
    return dict(zip(names, values))


def parse_apace_tiling_data(data: bytes) -> dict:
    """解析主算子 usingApaceImpl_ 分支的 hcommAllToAllMatmulTilingData。"""
    if len(data) < APACE_ALLTO_ALL_MATMUL_TILING_DATA_SIZE:
        return {
            "error": f"data too short for Apace tiling data: "
            f"{len(data)} < {APACE_ALLTO_ALL_MATMUL_TILING_DATA_SIZE}"
        }
    return {
        "mc2InitTiling": parse_mc2_init_tiling_inner(data, OFFSET_MC2_INIT_TILING),
        "mc2CcTiling": parse_mc2_cc_tiling_inner(data, OFFSET_MC2_CC_TILING),
        "commTilingData": parse_apace_comm_tiling(data, OFFSET_APACE_COMM_TILING),
        "tileQbmmTilingData": parse_apace_quant_matmul_tiling(
            data, OFFSET_APACE_QUANT_MATMUL
        ),
        "localMatmul": struct.unpack_from("<I", data, OFFSET_APACE_LOCAL_MATMUL)[0],
        "workspaceLayout": parse_workspace_layout_info(data, OFFSET_WS_LAYOUT_APACE),
    }


def is_valid_workspace_layout(layout: dict) -> bool:
    """按 dump 侧相同约束检查布局，减少不同 tiling 结构误判。"""
    if "error" in layout:
        return False
    total_size = layout.get("totalSize", 0)
    seg_count = layout.get("segCount", 0)
    segments = layout.get("segments", [])
    if (
        total_size <= 0
        or not 0 < seg_count <= MAX_WORKSPACE_SEGMENTS
        or len(segments) != seg_count
    ):
        return False
    last_end = 0
    for segment in segments:
        offset = segment.get("offset", 0)
        size = segment.get("size", 0)
        seg_type = segment.get("type")
        if (
            size <= 0
            or offset != last_end
            or offset + size > total_size
            or seg_type not in WS_SEG_TYPE_MAP
        ):
            return False
        last_end = offset + size
    return last_end == total_size


def detect_tiling_format(data: bytes) -> str:
    """自动区分主线非量化、主线量化和 usingApaceImpl_ CCU tiling。

    新版结构不再携带 quantMode。Apace 的 CommTilingData 为 6 个小于 2^32
    的 uint64，主线同位置则是两个相邻 uint32 字段；结合 QMM 尺寸和
    localMatmul(0/1) 可稳定区分。当前非空 DFX WorkspaceLayout 对应 MX 量化通路。
    """
    workspace_layout = parse_workspace_layout_info(data, OFFSET_WS_LAYOUT_APACE)
    comm = parse_apace_comm_tiling(data, OFFSET_APACE_COMM_TILING)
    qmm = parse_apace_quant_matmul_tiling(data, OFFSET_APACE_QUANT_MATMUL)
    local_matmul = (
        struct.unpack_from("<I", data, OFFSET_APACE_LOCAL_MATMUL)[0]
        if len(data) >= OFFSET_APACE_LOCAL_MATMUL + 4
        else None
    )
    comm_values = list(comm.values()) if "error" not in comm else []
    apace_plausible = (
        len(comm_values) == 6
        and all(0 <= value < (1 << 32) for value in comm_values)
        and (comm_values[0] > 0 or comm_values[2] > 0)
        and (comm_values[1] > 0 or comm_values[3] > 0)
        and comm_values[4] > 0
        and "error" not in qmm
        and all(qmm.get(name, 0) > 0 for name in ("m", "n", "k"))
        and local_matmul in (0, 1)
    )
    if apace_plausible and is_valid_workspace_layout(workspace_layout):
        return "apace"
    return "quant" if detect_quant_mode(data) else "non_quant"


def detect_quant_mode(data: bytes) -> bool:
    """新版 TilingData 已移除量化模式字段；非空合法段表对应当前 MX DFX 通路。"""
    return is_valid_workspace_layout(
        parse_workspace_layout_info(data, OFFSET_WS_LAYOUT_QUANT)
    )


def parse_args_info(data: bytes) -> dict:
    """解析 args_info_*.bin（AlltoAllMatmul 有 13 个有效 args，index 0-12）。"""
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
        f"dump 代码固定拷贝 min(devArgsLen, 256) 字节，AlltoAllMatmul 有效 args 为 {num_ptrs} 个。"
        f" 实际 devArgsLen 请查看日志中 'Tiling dump start' 行的 devArgsLen 字段。"
    )
    return args


# ====== Workspace 解析函数 ======


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


def parse_runtime_info_per_core(
    data: bytes, offset: int, pos_tag_map=POS_TAG_MAP_ALLTO_ALL
) -> dict:
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
        "execPositionName": pos_tag_map.get(exec_position, f"UNKNOWN({exec_position})"),
        "commPhase": comm_phase,
        "commPhaseName": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "commCommitCount": commit_count,
        "commWaitCount": wait_count,
    }


def parse_runtime_info_segment(
    data: bytes, aic_core_num=None, pos_tag_map=POS_TAG_MAP_ALLTO_ALL
) -> dict:
    """解析 workspace_segN_type12_*.bin（RuntimeInfo 段，64KB = 128 × 512B）。

    aic_core_num: tiling data 中的 aicCoreNum，用于区分 C/V 核。
                  AIC coreId 范围 0 ~ aicCoreNum-1; AIV coreId 范围 aicCoreNum ~ 3*aicCoreNum-1
                  (MIX_AIC_1_2 模式下 AIV slotIdx = aicCoreNum + GetBlockIdx())。
                  为 None 时 coreType 标记为 "未知"，需通过 --aic-core-num 传入。
    """
    max_slots = len(data) // RUNTIME_INFO_PER_CORE_SIZE
    slots = []
    active_count = 0
    for i in range(max_slots):
        offset = i * RUNTIME_INFO_PER_CORE_SIZE
        slot = parse_runtime_info_per_core(data, offset, pos_tag_map)
        slot["slotIdx"] = i
        if slot.get("magicValid", False):
            active_count += 1
            if aic_core_num is not None:
                slot["coreType"] = "C核" if slot["coreId"] < aic_core_num else "V核"
            else:
                slot["coreType"] = "未知"
            slots.append(slot)
        else:
            # 未写入的 slot 也保留，但标记为 inactive
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


def parse_workspace_segment_file(
    data: bytes, seg_type: int, aic_core_num=None, pos_tag_map=POS_TAG_MAP_ALLTO_ALL
) -> dict:
    """解析 workspace_seg*_type*_*.bin 文件，根据段类型选择解析方式。"""
    type_name = WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})")
    result = {
        "segmentType": seg_type,
        "segmentTypeName": type_name,
        "fileSize": len(data),
    }
    if seg_type == 12:  # WS_SEG_RUNTIME_INFO
        rt_info = parse_runtime_info_segment(data, aic_core_num, pos_tag_map)
        result["runtimeInfo"] = rt_info
    else:
        # 非 RuntimeInfo 段：做基础 hex 统计
        result["hexPreview"] = data[:256].hex()
        result["note"] = (
            f"段类型 {type_name}，原始二进制数据，请结合 tiling 参数人工分析。"
        )
    return result


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


def print_tiling_data(result: dict, is_quant: bool):
    """人类可读格式打印 tiling data。"""
    info_key = (
        "alltoAllQuantMatmulTilingInfo" if is_quant else "alltoAllMatmulTilingInfo"
    )
    mm_tile_key = "mc2QuantMmTileTilingData" if is_quant else "mc2MmV3TileTilingData"
    mm_tail_key = "mc2QuantMmTailTilingData" if is_quant else "mc2MmV3TailTilingData"

    print_section("Mc2InitTiling (HCCL Init, 64B)")
    print_kv("mc2InitTilingInner", result["mc2InitTiling"])

    print_section("Mc2CcTiling (HCCL CC, 512B)")
    print_kv("mc2CcTilingInner", result["mc2CcTiling"])

    print_section(
        f"AlltoAllMatmulTilingInfo ({ALLTO_ALL_MATMUL_TILING_INFO_SIZE}B)  [{'QUANT' if is_quant else 'NON-QUANT'}]"
    )
    print_kv(info_key, result[info_key])

    print_section("Matmul Tile Tiling (头块)")
    print_kv(mm_tile_key, result[mm_tile_key])

    print_section("Matmul Tail Tiling (尾块)")
    print_kv(mm_tail_key, result[mm_tail_key])


def print_args_info(result: dict):
    """人类可读格式打印 args info。"""
    print_section("Args Info (kernel GM pointer)")
    dump_file_size = result.pop("_dumpFileSize", 0)
    valid_arg_count = result.pop("_validArgCount", 0)
    for name, addr in result.items():
        if name.startswith("_"):
            continue
        logging.info("  %-30s: %s", name, addr)
    logging.info("")
    logging.info(
        "  有效 args 数量: %d (index 0-%d)", valid_arg_count, valid_arg_count - 1
    )
    logging.info("  dump 文件大小: %d bytes", dump_file_size)
    logging.info("  提示: tilingGM 应在 [12] 位置 (offset=96)")
    tiling_gm = result.get("[12] tilingGM", "N/A")
    logging.info("  [12] tilingGM = %s", tiling_gm)


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
            "  请先解析 tiling_data_*.bin 获取 aicCoreNum，再通过 --aic-core-num N 传入。"
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
            # 格式化为每行 32 字节（64 hex chars）
            formatted = " ".join(chunk[j : j + 2] for j in range(0, len(chunk), 2))
            logging.info("    %s: %s", offset_hex, formatted)
        logging.info("")
        logging.info("  %s", result.get("note", ""))


def main():
    parser = argparse.ArgumentParser(
        description="AlltoAllMatmul TilingData / Workspace / RuntimeInfo dump 解析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 解析 tiling data（自动识别主线非量化、主线量化或 Apace CCU）
  python parse_allto_all_matmul_tiling.py tiling_data_AlltoAllMatmul.0.1.20260819.bin

  # 解析 args_info（验证 tilingGM offset）
  python parse_allto_all_matmul_tiling.py args_info_AlltoAllMatmul.0.1.20260819.bin --args

  # 解析 workspace 段文件（自动从文件名识别段类型）
  python parse_allto_all_matmul_tiling.py workspace_seg4_type12_AlltoAllMatmul.0.1.20260819.bin --workspace

        """,
    )
    parser.add_argument("bin_file", help="dump 产生的 bin 文件路径")
    parser.add_argument(
        "--args", action="store_true", help="解析 args_info_*.bin（GM 地址指针）"
    )
    parser.add_argument(
        "--workspace",
        action="store_true",
        help="解析 workspace_seg*_type*_*.bin（段文件，type=12 自动解析 RuntimeInfo）",
    )
    parser.add_argument(
        "--aic-core-num",
        type=int,
        metavar="N",
        default=None,
        help="AIC 核数，用于 workspace 段解析时区分 C/V 核。"
        " 从 tiling_data 解析结果的 aicCoreNum 字段获取。"
        " 未提供时 C/V 核分类不可靠（显示为'未知'）。",
    )
    parser.add_argument(
        "--apace",
        action="store_true",
        help="workspace 段解析时使用 Apace 路径(usingApaceImpl_)的 position 枚举标签"
        "(1=COMP_LOCAL_BEFORE, 2=COMP_REMOTE_BEFORE, 3=FINALIZE_BEFORE)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()

    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

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
        pos_tag_map = (
            POS_TAG_MAP_ALLTO_ALL_APACE if args.apace else POS_TAG_MAP_ALLTO_ALL
        )
        if args.apace:
            logging.info("使用 Apace 路径 position 枚举标签")
        result = parse_workspace_segment_file(
            data, seg_type, args.aic_core_num, pos_tag_map
        )
        result["segIdx"] = seg_idx
        print_workspace_segment(result)
        return

    # --- args_info 解析 ---
    if args.args:
        result = parse_args_info(data)
        print_args_info(result)
        return

    # --- 默认: tiling_data 解析 ---
    tiling_format = detect_tiling_format(data)
    if tiling_format == "apace":
        logging.info(
            "解析模式: Apace CCU (usingApaceImpl_=true, workspaceLayout offset=%d)",
            OFFSET_WS_LAYOUT_APACE,
        )
        result = parse_apace_tiling_data(data)
        if "error" in result:
            logging.error("解析失败: %s", result["error"])
            sys.exit(1)
        print_section("Apace hcommAllToAllMatmulTilingData")
        print_kv("tilingData", result)
        print_workspace_layout_info(result["workspaceLayout"])
        return

    is_quant = tiling_format == "quant"
    if is_quant:
        logging.info("解析模式: 主线量化（新版 TilingData 不再保存 x1/x2QuantMode）")
    else:
        logging.info("解析模式: 主线非量化")

    struct_size = (
        ALLTO_ALL_QUANT_MATMUL_TILING_DATA_SIZE
        if is_quant
        else ALLTO_ALL_MATMUL_TILING_DATA_SIZE
    )
    if len(data) < struct_size:
        logging.warning(
            "文件大小 %d < 结构体大小 %d，解析可能不完整", len(data), struct_size
        )

    if is_quant:
        result = parse_quant_tiling_data(data)
        logging.info(
            "结构体 AlltoAllQuantMatmulTilingData: %dB, dump 文件 %dB, 尾部 %dB 为 padding",
            struct_size,
            len(data),
            len(data) - struct_size,
        )
    else:
        result = parse_non_quant_tiling_data(data)
        logging.info(
            "结构体 AlltoAllMatmulTilingData: %dB, dump 文件 %dB, 尾部 %dB 为 padding",
            struct_size,
            len(data),
            len(data) - struct_size,
        )

    # WorkspaceLayoutInfo 位于 Mc2InitTiling + Mc2CcTiling 之后的 DfxDumpInfo 起点。
    ws_offset = OFFSET_WS_LAYOUT_QUANT if is_quant else OFFSET_WS_LAYOUT_NON_QUANT
    if len(data) >= ws_offset + WORKSPACE_LAYOUT_INFO_SIZE:
        ws_layout = parse_workspace_layout_info(data, ws_offset)
        if "error" not in ws_layout and ws_layout.get("segCount", 0) > 0:
            result["workspaceLayout"] = ws_layout
            logging.info(
                "检测到 WorkspaceLayoutInfo (offset=%d, segCount=%d, totalSize=%d)",
                ws_offset,
                ws_layout["segCount"],
                ws_layout["totalSize"],
            )

    print_tiling_data(result, is_quant)
    if "workspaceLayout" in result:
        print_workspace_layout_info(result["workspaceLayout"])


if __name__ == "__main__":
    main()
