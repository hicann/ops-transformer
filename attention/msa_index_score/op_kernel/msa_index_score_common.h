/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_index_score_common.h
 * \brief host / device 共用的编译期常量与 TilingKey 定义。
 *
 * 该头文件不依赖 AscendC，可被 op_host 的 tiling 直接 include。
 */

#ifndef MSA_INDEX_SCORE_COMMON_H
#define MSA_INDEX_SCORE_COMMON_H

#include <cstdint>

namespace MsaIndexScoreNs {

// 一个 sparse block（同时也是一个 paged-cache page）的 token 数。
// 该值同时是 BlockMmad 的 L1TileShape::N，必须是编译期常量。
constexpr uint32_t MSA_BLOCK_SIZE = 128;

// 一个 M-tile 的行数。行 = 请求内 (token, head) 扁平索引，head 为低位。
constexpr uint32_t MSA_ROW_TILE_M = 128;

// Cube L0A fractal / 向量归约对齐行数。末尾短 tile 不足该值时向前重叠。
constexpr uint32_t MSA_M_ALIGN = 16;

// BlockMmad 的 L1TileShape::K。headDim <= 该值时 kLoop = 1，
// headDim 更大时 BlockMmad 内部自动多轮 K 迭代。
constexpr uint32_t MSA_K_TILE = 128;

// 一个 S workspace tile 覆盖的 sparse block 数（AIC 每产出这么多 block 同步一次 AIV）。
// 取 8 使得每行的归约结果恰为 8 个 fp32 = 32B，输出可以按 32B 对齐整块写回。
constexpr uint32_t MSA_BLOCKS_PER_STILE = 8;

// S workspace 轮转级数。AIC 领先 AIV 的未消费 tile 数 <= MSA_WORKSPACE_STAGES - 1。
// 5 -> 8：AIV 乒乓化后速度提升，加深缓冲让 AIC 几乎不因 AIV 停顿（wait_id2→0）。
// 非量化路径 S 为 fp16，8 级 × 256KB × 20 核 = 40MB，仍可整体驻留 L2。
constexpr uint32_t MSA_WORKSPACE_STAGES = 8;

// 一个 S workspace tile 的元素数：[MSA_ROW_TILE_M, MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE]
// 非量化路径元素为 fp16（fixpipe F322F16 直接写），int8 路径为 fp32。
constexpr uint32_t MSA_STILE_ELEM_NUM = MSA_ROW_TILE_M * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
// 非量化 S tile 的字节数（fp16）。
constexpr uint32_t MSA_STILE_BYTES_FP16 = MSA_STILE_ELEM_NUM * sizeof(uint16_t);
// int8 路径 S tile 的字节数（fp32）。
constexpr uint32_t MSA_STILE_BYTES_FP32 = MSA_STILE_ELEM_NUM * sizeof(float);

// int8 路径：每个 AIC 私有一块 K cast 暂存，容纳一个 S-tile 内全部 page
// （AIV 先 cast，AIC 再 Mmad，避免逐 page 细粒度同步）。
// 单页布局 [blockSize, headDim] 行主序，按 j*P*D 排列。
constexpr uint32_t MSA_K_SCRATCH_ELEM_NUM = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * MSA_K_TILE;
// A2/A3 int8：四槽 scratch。WaitS(st) 后 slot[st%4] 已空，gather(st+4) 叠在
// AIC 的 cube(st+1)+cube(st+2)+cube(st+3) 上；epilogue(st) 再叠在 cube(st+4) 上。950 不用。
constexpr uint32_t MSA_K_SCRATCH_STAGES_A2 = 4;
// 950 一页握手：每 AIC 只需一页 K scratch。
constexpr uint32_t MSA_A5_K_SCRATCH_ELEM_NUM = MSA_BLOCK_SIZE * MSA_K_TILE;
constexpr uint32_t MSA_A5_S_STAGES = 2;
constexpr uint32_t MSA_A5_FP8_C0 = 32;
constexpr uint32_t MSA_A5_S_PAGE_ELEM_NUM = MSA_ROW_TILE_M * MSA_BLOCK_SIZE;

#ifndef MSA_A5_USE_C2UB
#define MSA_A5_USE_C2UB 1
#endif

// AIV 单次归约处理的行数。受 level-0 向量 API repeatTimes(uint8_t) 上限约束：
// MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE <= 255，且乘积需为 8 的倍数。
constexpr uint32_t MSA_ROWS_PER_PASS = 16;

// 归约展开后的逻辑行数：把 [rows, BLOCKS, BLOCK_SIZE] 视作 [rows*BLOCKS, BLOCK_SIZE]。
constexpr uint32_t MSA_REDUCE_ROWS = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE;

// 向量单元一个 repeat 处理的 fp32 数 / 一个 32B datablock 的 fp32 数。
constexpr uint32_t MSA_FP32_PER_REPEAT = 64;
constexpr uint32_t MSA_FP32_PER_BLOCK = 8;
// 向量 DataCopy 的 32B datablock。
constexpr uint32_t MSA_DATABLOCK_BYTES = 32;
// AscendC Duplicate 比较掩码字数（2×uint64）。
constexpr uint32_t MSA_VEC_MASK_WORDS = 2;
// MIX 核比：1 AIC : 2 AIV。
constexpr uint32_t MSA_AIV_PER_AIC = 2;
// 950 C_to_UB：dual-dst 后每 AIV 半个 M-tile × N=128 fp32 pingpong。
constexpr uint32_t MSA_A5_S_PING_ROWS = MSA_ROW_TILE_M / MSA_AIV_PER_AIC;
constexpr uint32_t MSA_A5_S_PING_ELEM_NUM = MSA_A5_S_PING_ROWS * MSA_BLOCK_SIZE;
constexpr uint32_t MSA_A5_S_PING_BYTES = MSA_A5_S_PING_ELEM_NUM * static_cast<uint32_t>(sizeof(float));
// block16 一次 MMAD 最多并几个 128 列窗。nWin=4 时 L0C 单级放下 128×512。
constexpr uint32_t MSA_BLOCK16_MAX_WIN_GROUP = 4;
constexpr uint32_t MSA_A5_S_PING_BYTES_B16 = MSA_A5_S_PING_ROWS * MSA_BLOCK16_MAX_WIN_GROUP * MSA_BLOCK_SIZE * 2U;
// UB 手工布局 512B 对齐。
constexpr uint32_t MSA_UB_ALIGN_BYTES = 512;

// BlockMmad L1B pingpong 级数：950 int8 仍为 2（单页 scratch 竞态）；A2/A3 int8 在
// arch22 kernel 里改用 MSA_L1B_STAGES_FP16（双槽后安全）。
constexpr uint32_t MSA_L1B_STAGES_INT8 = 2;
constexpr uint32_t MSA_L1B_STAGES_FP16 = 3;

// Shared byte offset for AIC/AIV handoff of an already-NZ MXFP4 Q-scale tile.
template <uint32_t STAGES, bool WIDE>
struct MsaMxL1Layout {
    static constexpr uint32_t COLS = MSA_BLOCK_SIZE * (WIDE ? 4U : 2U);
    static constexpr uint32_t SCALE_A_OFFSET = MSA_ROW_TILE_M * MSA_K_TILE + STAGES * COLS * MSA_K_TILE;
};

// score 输出末维的对齐粒度。
constexpr uint32_t MSA_SCORE_STRIDE_ALIGN = 16;

// 不可见 block 的填充值（-inf），使其在下游 topk 中必然落选。
constexpr float MSA_FILL_VALUE = -3.4028234663852886e+38F;

// local_mask 强制高分（对齐 Triton decode：init=1e30，local=1e29；local 覆盖 init）。
constexpr float MSA_LOCAL_SCORE_INIT = 1.0e30F;
constexpr float MSA_LOCAL_SCORE_LOCAL = 1.0e29F;

// sparse_mode：0=defaultMask（无因果截断）；3=rightDownCausal（与 LightningIndexer 一致）。
constexpr uint32_t MSA_SPARSE_MODE_DEFAULT = 0;
constexpr uint32_t MSA_SPARSE_MODE_RIGHT_DOWN = 3;

// MiniMax 默认强制选块数（接口文档未暴露 attr；P0 以常量对齐 HF/Triton 常见配置）。
constexpr uint32_t MSA_DEFAULT_INIT_BLOCKS = 0;
constexpr uint32_t MSA_DEFAULT_LOCAL_BLOCKS = 1;

// key 数据排布。BBND / BNBD 为 PageAttention。
// TND / BSND / BNSD 为非分页序列，无 block_table。
// 块长 16 的 BSND/BNSD 必须与 query 布局相同；块长 128 的 BSND/BNSD 配 TND query。
constexpr uint32_t MSA_KEY_LAYOUT_BBND = 0; // [block_num, block_size, N2, D]
constexpr uint32_t MSA_KEY_LAYOUT_BNBD = 1; // [block_num, N2, block_size, D]
constexpr uint32_t MSA_KEY_LAYOUT_TND = 2;  // [T2, N2, D]
constexpr uint32_t MSA_KEY_LAYOUT_BSND = 3; // [B, S, N2, D]
constexpr uint32_t MSA_KEY_LAYOUT_BNSD = 4; // [B, N2, S, D]

// query 布局。TND 为打包变长；BSND / BNSD 为定长 batch，实际长度仍由 actual_seq_qlen 给出。
constexpr uint32_t MSA_QUERY_LAYOUT_TND = 0;  // [T1, N1, D]
constexpr uint32_t MSA_QUERY_LAYOUT_BSND = 1; // [B, S, N1, D]
constexpr uint32_t MSA_QUERY_LAYOUT_BNSD = 2; // [B, N1, S, D]

#ifdef __CCE_AICORE__
#define MSA_FN __aicore__ inline
#else
#define MSA_FN inline
#endif

MSA_FN bool MsaIsPaKeyLayout(uint32_t keyLayout)
{
    return keyLayout == MSA_KEY_LAYOUT_BBND || keyLayout == MSA_KEY_LAYOUT_BNBD;
}

MSA_FN bool MsaIsSeqKeyLayout(uint32_t keyLayout)
{
    return keyLayout == MSA_KEY_LAYOUT_TND || keyLayout == MSA_KEY_LAYOUT_BSND || keyLayout == MSA_KEY_LAYOUT_BNSD;
}

// BBND/BNBD 页长 16。MSA_BLOCK_SIZE 仍是 Cube 的 N：一次 GEMM 的 128 列
// 恰好是 MSA_BLOCKS_PER_STILE 个逻辑 page。
constexpr uint32_t MSA_BLOCK16_SIZE = 16;
static_assert(MSA_BLOCK16_SIZE * MSA_BLOCKS_PER_STILE == MSA_BLOCK_SIZE,
              "16-token pages must pack into one Cube N tile");

// atten_mask 压缩下三角模板边长（sparse_mode=3 时校验）。
constexpr uint32_t MSA_ATTEN_MASK_SIZE = 2048;

// FFTS cross-core flag id，合法范围 0~7（8/9/10 为保留 barrier）。
constexpr uint16_t MSA_FLAG_S_READY = 1;
constexpr uint16_t MSA_FLAG_S_READY_REVERSE = 2;
// int8 路径：AIV cast K→scratch 完成后通知 AIC。A2 四槽用 3/4/5/6。
constexpr uint16_t MSA_FLAG_K_READY = 3;
constexpr uint16_t MSA_FLAG_K_READY_REVERSE = 4;
constexpr uint16_t MSA_FLAG_K_READY_2 = 5;
constexpr uint16_t MSA_FLAG_K_READY_3 = 6;

// Ascend 950 MIX：MODE4 下 AIC 分别给两个 AIV 置位，flagId+16。
constexpr uint32_t MSA_A5_SYNC_MODE4 = 4;
constexpr uint16_t MSA_A5_AIV_FLAG_OFFSET = 16;
constexpr uint16_t MSA_A5_FLAG_CV = 0;       // Cube→Vector：S 已在 GM/UB
constexpr uint16_t MSA_A5_FLAG_VC = 2;       // Vector→Cube：UB slot 可覆盖
constexpr uint16_t MSA_A5_FLAG_K = 4;        // Vector→Cube：K scratch 就绪
constexpr uint16_t MSA_A5_FLAG_L1B_FREE = 5; // Cube→Vector0：L1B 单槽可覆盖

// TND 保持打包行：globalRow * D。BSND 在 batch 内同样是 (token, head) 连续行。
MSA_FN uint64_t MsaQueryTileElem(uint32_t queryLayout, uint32_t batchIdx, uint32_t cuQStart, uint32_t mStart,
                                 uint32_t numQHeads, uint32_t headDim, uint32_t qMaxSeq, uint32_t strideQn)
{
    if (queryLayout == MSA_QUERY_LAYOUT_BSND) {
        return static_cast<uint64_t>(batchIdx) * qMaxSeq * numQHeads * headDim +
               static_cast<uint64_t>(mStart) * headDim;
    }
    return static_cast<uint64_t>(cuQStart * numQHeads + mStart) * strideQn;
}

// 非分页 key 上一个逻辑 block 的起始元素。N2=1 时 BSND 与 BNSD 的 head0 都是连续 token。
MSA_FN uint64_t MsaSeqKeyBlockElem(uint32_t keyLayout, uint32_t batchIdx, uint32_t cuKStart, uint32_t localTok,
                                   uint32_t numKvHeads, uint32_t headDim, uint32_t kMaxSeq)
{
    if (keyLayout == MSA_KEY_LAYOUT_BSND) {
        return (static_cast<uint64_t>(batchIdx) * kMaxSeq * numKvHeads + static_cast<uint64_t>(localTok) * numKvHeads) *
               headDim;
    }
    if (keyLayout == MSA_KEY_LAYOUT_BNSD) {
        return (static_cast<uint64_t>(batchIdx) * numKvHeads * kMaxSeq + localTok) * headDim;
    }
    return static_cast<uint64_t>(cuKStart + localTok) * numKvHeads * headDim;
}

} // namespace MsaIndexScoreNs

// TilingKey：按 query dtype × key 是否 int8 分支。
// 模式编号同时用于 Host 选路与 schMode 模板注册；保持库上已有的数值编码。
#define MSA_TILING_KEY_BF16 0
#define MSA_TILING_KEY_FP16 1
#define MSA_TILING_KEY_BF16_INT8 2
#define MSA_TILING_KEY_FP16_INT8 3
#define MSA_TILING_KEY_HIFLOAT8 4
#define MSA_TILING_KEY_FP8_E5M2 5
#define MSA_TILING_KEY_FP8_E4M3FN 6
// 块长 16（BSND/BNSD，或 BBND/BNBD 页长 16）：独立 TilingKey。
#define MSA_TILING_KEY_BF16_BLK16 7
#define MSA_TILING_KEY_FP16_BLK16 8
// 950：max 之后再做 topk。A2/A3 的 13/14 只预留，本阶段 tiling 不发出。
#define MSA_TILING_KEY_BF16_TOPK 9
#define MSA_TILING_KEY_FP16_TOPK 10
#define MSA_TILING_KEY_FP8_E5M2_TOPK 11
#define MSA_TILING_KEY_FP8_E4M3FN_TOPK 12
#define MSA_TILING_KEY_BF16_TOPK_A2 13
#define MSA_TILING_KEY_FP16_TOPK_A2 14
#define MSA_TILING_KEY_MXFP4 15
#define MSA_TILING_KEY_MXFP4_BLK16 16
#define MSA_TILING_KEY_MXFP4_TOPK 18
#define MSA_TILING_KEY_MXFP4_PREFILL 19
#define MSA_TILING_KEY_MXFP4_LATENCY 23

// 块长 16 的 FP8、压缩页和直读模式。
#define MSA_TILING_KEY_FP8_E4M3FN_BLK16 44
#define MSA_TILING_KEY_FP8_E5M2_BLK16 45
#define MSA_TILING_KEY_FP8_E4M3FN_COMPRESSED_PAGE 46
#define MSA_TILING_KEY_FP8_E5M2_COMPRESSED_PAGE 47
#define MSA_TILING_KEY_BF16_COMPRESSED_PAGE 48
#define MSA_TILING_KEY_FP16_COMPRESSED_PAGE 49
#define MSA_TILING_KEY_FP8_E4M3FN_PIPELINED 50
#define MSA_TILING_KEY_FP8_E5M2_PIPELINED 51
#define MSA_TILING_KEY_FP8_E4M3FN_PIPELINED_TOPK 52
#define MSA_TILING_KEY_FP8_E5M2_PIPELINED_TOPK 53
#define MSA_TILING_KEY_FP8_E4M3FN_BALANCED 54
#define MSA_TILING_KEY_FP8_E5M2_BALANCED 55
#define MSA_TILING_KEY_FP8_E4M3FN_BALANCED_TOPK 56
#define MSA_TILING_KEY_FP8_E5M2_BALANCED_TOPK 57
#define MSA_TILING_KEY_FP8_E4M3FN_PIPELINED_BYPASS 58
#define MSA_TILING_KEY_FP8_E5M2_PIPELINED_BYPASS 59
#define MSA_TILING_KEY_FP8_E4M3FN_PIPELINED_TOPK_BYPASS 60
#define MSA_TILING_KEY_FP8_E5M2_PIPELINED_TOPK_BYPASS 61
#define MSA_TILING_KEY_FP8_E4M3FN_BALANCED_BYPASS 62
#define MSA_TILING_KEY_FP8_E5M2_BALANCED_BYPASS 63
#define MSA_TILING_KEY_FP8_E4M3FN_BALANCED_TOPK_BYPASS 64
#define MSA_TILING_KEY_FP8_E5M2_BALANCED_TOPK_BYPASS 65
#define MSA_TILING_KEY_BF16_DIRECT_FLOAT 66
#define MSA_TILING_KEY_FP16_DIRECT_FLOAT 67

constexpr uint32_t MSA_MX_GROUP = 32;
constexpr uint32_t MSA_BLOCK16_MX_SCALE_SCRATCH_BYTES = 64U * 1024U;

constexpr uint32_t MSA_RETURN_SCORE = 0;
constexpr uint32_t MSA_RETURN_INDICES = 1;
constexpr uint32_t MSA_RETURN_INDICES_VALUES = 2;
constexpr uint32_t MSA_BLOCK16_TOPK = 128;
constexpr uint32_t MSA_TOPK_DEFAULT = 16;
constexpr uint32_t MSA_TOPK_MAX = 128;
constexpr uint32_t MSA_TOPK_TRUNK = 1024;
constexpr int32_t MSA_TOPK_INVALID_INDEX = -1;

// 一块 Sort32 最多 2048 列：6 段缓冲约 48KB，落在 256KB UB 里。
constexpr uint32_t MSA_TOPK_SORT_CHUNK = 2048;
// Sort32 一次处理 32 个元素；(score, index) 一对占 2 个 float。
constexpr uint32_t MSA_TOPK_SORT32_LEN = 32;
constexpr uint32_t MSA_TOPK_SORT_PAIR = 2;
// Gather 的 byte offset = 元素下标 << 2。sizeof(float) 与 sizeof(int32) 都是 4。
constexpr uint32_t MSA_TOPK_GATHER_BYTE_SHIFT = 2;
// 长度表放在 trunk 的 128KB 处，q / k 各最多 2048 个 int32。
constexpr uint32_t MSA_TOPK_LEN_FLOAT_OFF = 32768;
constexpr uint32_t MSA_TOPK_LEN_CAP = 2048;
// 行元数据放在 160KB 处。q、kv、token 偏移各 4096 个 int32，依次排列。
constexpr uint32_t MSA_TOPK_META_FLOAT_OFF = 40960;
constexpr uint32_t MSA_TOPK_META_CAP = 4096;
// 256 列、k=128 的稠密路径，一次最多拼 8 行。
constexpr uint32_t MSA_TOPK_DENSE_COLS = 256;
constexpr uint32_t MSA_TOPK_DENSE_K = MSA_TOPK_MAX;
constexpr uint32_t MSA_TOPK_DENSE_BATCH = 8;
constexpr uint32_t MSA_TOPK_DENSE_PROP_FLOATS = MSA_TOPK_DENSE_COLS * MSA_TOPK_SORT_PAIR;
constexpr uint32_t MSA_TOPK_SORT32_PAIR_FLOATS = MSA_TOPK_SORT32_LEN * MSA_TOPK_SORT_PAIR;
// MrgSort 四路源。0b1111 四路都参与，0b0011 只合并前两路。
constexpr uint32_t MSA_TOPK_MRG_SRC_NUM = 4;
constexpr uint32_t MSA_TOPK_MRG_MASK_ALL = 0b1111U;
constexpr uint32_t MSA_TOPK_MRG_MASK_LO2 = 0b0011U;
// 尾块列号加上 2^20，排在任何前缀列号之后。填充列的索引键落到最后。
constexpr float MSA_TOPK_TAIL_KEY_BIAS = 1048576.0F;
constexpr int32_t MSA_TOPK_PAD_INDEX = 0x7FFFFFFF;

MSA_FN constexpr uint32_t MsaAlignUpPow2(uint32_t value, uint32_t align)
{
    return (value + align - 1U) & ~(align - 1U);
}

#endif // MSA_INDEX_SCORE_COMMON_H
