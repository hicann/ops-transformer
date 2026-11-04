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
 * \file quant_lightning_indexer_v2_fast_service_cube_arch35.h
 * \brief Cube侧数据预取、矩阵乘及Fixpipe结果交接
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_CUBE_ARCH35_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_CUBE_ARCH35_H

#include "kernel_operator_fixpipe_intf.h"
#include "kernel_operator_mm_intf.h"
#include "quant_lightning_indexer_v2_fast_common_arch35.h"

namespace QLIV2FastCubeCommon {
using namespace AscendC;
using namespace QLIV2Fast;
// L0按4个Q组、2个K槽及2个结果槽复用；地址单位为字节，视图长度单位为元素
constexpr int64_t L0_QUERY_SLOT_BYTES = 16U * 1024U;
constexpr int64_t L0_KEY_SLOT_BYTES = 16U * 1024U;
constexpr int64_t L0_RESULT_SLOT_BYTES = 128U * 1024U;
constexpr uint32_t L0_QUERY_VIEW_ELEMENTS = 16U * 1024U;
constexpr uint32_t L0_KEY_VIEW_ELEMENTS = 16U * 1024U;
constexpr uint32_t L0_RESULT_VIEW_ELEMENTS = 16U * 1024U;
constexpr uint32_t L1_QUERY_VIEW_ELEMENTS = 64U * 1024U;
constexpr int64_t L1_KEY_SLOT_BYTES = KEY_TILE_COLUMNS * HEAD_DIM;
constexpr uint32_t QUERY_ROWS_PER_GROUP = 4U;
// LoadData2D的M/N方向以16行为单位，FP8的K方向NZ内层块宽为32个元素
// NZ块宽与MX_SCALE_GROUP_SIZE是不同概念，即使当前值相同也分别命名
constexpr uint32_t CUBE_BLOCK_ROWS = 16U;
constexpr uint32_t FP8_NZ_BLOCK_COLUMNS = 32U;
constexpr uint32_t QUERY_HEADS_PER_M_BLOCK = CUBE_BLOCK_ROWS / QUERY_ROWS_PER_GROUP;
constexpr uint32_t QUERY_K_BLOCKS = HEAD_DIM / FP8_NZ_BLOCK_COLUMNS;
constexpr uint32_t MMAD_RESULT_COLUMNS = 128U;
constexpr uint32_t KEY_N_BLOCKS = MMAD_RESULT_COLUMNS / CUBE_BLOCK_ROWS;
constexpr uint32_t KEY_L1_N_BLOCK_STRIDE = KEY_TILE_COLUMNS / CUBE_BLOCK_ROWS;
constexpr uint32_t RESULT_GROUP_COLUMNS = 2U * MMAD_RESULT_COLUMNS;
// Key tile按256列分为8组，以下常量表示各组索引
constexpr int32_t KEY_GROUP_0 = 0;
constexpr int32_t KEY_GROUP_1 = 1;
constexpr int32_t KEY_GROUP_2 = 2;
constexpr int32_t KEY_GROUP_3 = 3;
constexpr int32_t KEY_GROUP_4 = 4;
constexpr int32_t KEY_GROUP_5 = 5;
constexpr int32_t KEY_GROUP_6 = 6;
constexpr int32_t KEY_GROUP_7 = 7;
constexpr int64_t L0_RESULT_HALF_BYTES_PER_HEAD = QUERY_ROWS_PER_GROUP * MMAD_RESULT_COLUMNS * sizeof(float);
inline constexpr FixpipeConfig kHandoff{CO2Layout::ROW_MAJOR, true};

__aicore__ inline void LoadNd(int64_t dst, GlobalTensor<fp8_e4m3fn_t> src, int32_t rows, int32_t width,
                              int32_t c0Stride)
{
    LocalTensor<fp8_e4m3fn_t> local(TPosition::A1, dst, L1_BUFFER_BYTES - dst);
    // 单矩阵ND转NZ：源行跨度为128个FP8元素，C0跨度由调用方指定，N跨度为1
    Nd2NzParams copy{
        1, static_cast<uint16_t>(rows), static_cast<uint32_t>(width), 0, HEAD_DIM, static_cast<uint16_t>(c0Stride), 1,
        0};
    DataCopy(local, src, copy);
}
template <int32_t Half>
__aicore__ inline void StoreHalf(int64_t dst, int64_t src, uint32_t paddedHeadNum)
{
    LocalTensor<float> output(TPosition::VECCALC, dst, CUBE_RESULT_VIEW_ELEMENTS);
    LocalTensor<float> input(TPosition::CO1, src, CUBE_RESULT_VIEW_ELEMENTS);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> copy;
    // 每个ND子矩阵64列，共4个；双目标交接后每个AIV按128列行跨度读取
    copy.nSize = 64;
    copy.mSize = 4 * paddedHeadNum;
    copy.srcStride = 4 * paddedHeadNum;
    copy.dstStride = 128;
    copy.dualDstCtl = 1;
    copy.unitFlag = 0;
    copy.params.ndNum = 4;
    // 4Q乘补齐头数；ND子矩阵在源/目标中的布局步长分别为16/256乘补齐头数
    copy.params.srcNdStride = 16 * paddedHeadNum;
    copy.params.dstNdStride = 256 * paddedHeadNum;
    Fixpipe<float, float, kHandoff>(output, input, copy);
}
// 数据预取完成后共用矩阵乘及令牌归还，量化差异由输入类型固化
// q/c为L0A/L0C字节偏移，b为L0B视图，BSlot选择Key的L0B缓存槽
template <int32_t BSlot, class QueryType>
__aicore__ inline void Multiply(int64_t q, int64_t c, uint32_t paddedHeadNum, LocalTensor<QueryType> b)
{
    // L0A的4个Q槽占用事件0至3，L0B的两个K槽使用事件4、5
    SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(BSlot + 4));
    WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(BSlot + 4));
    LocalTensor<QueryType> ql0(TPosition::A2, q, L0_QUERY_VIEW_ELEMENTS);
    LocalTensor<float> cl0(TPosition::CO1, c, L0_RESULT_VIEW_ELEMENTS);
    MmadParams multiply;
    multiply.m = QUERY_ROWS_PER_GROUP * paddedHeadNum;
    multiply.n = MMAD_RESULT_COLUMNS;
    multiply.k = HEAD_DIM;
    multiply.unitFlag = 0;
    multiply.disableGemv = true;
    multiply.cmatrixSource = false;
    multiply.cmatrixInitVal = true;
    Mmad(cl0, ql0, b, multiply);
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(BSlot + 4));
}
// MMAD完成后向两个AIV发布同一布局的结果；模式4中AIV1的事件号比AIV0增加16
template <int32_t Slot>
__aicore__ inline void PublishResult(int64_t c, uint32_t paddedHeadNum)
{
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(Slot));
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(Slot + 16));
    const int64_t out = static_cast<int64_t>(Slot) * CUBE_RESULT_SLOT_OFFSET_BYTES;
    // 每次Fixpipe API调用内部包含一次FIX屏障；末尾M事件覆盖两个半块，保持原有缓冲区可用通知与释放顺序
    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(Slot));
    StoreHalf<0>(out, c, paddedHeadNum);
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(Slot));
    CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(Slot));
    CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(Slot + 16));
}
// 两种量化共用令牌恢复，FP8额外回收两个预取令牌
template <bool FP8>
__aicore__ inline void FinishAic()
{
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(0));
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(16));
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(1));
    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(static_cast<uint16_t>(17));
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
    if constexpr (FP8) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }
    SetMMColumnMajor();
}
// 初始化共用矩阵乘令牌，随后分别加载各量化的输入
__aicore__ inline void BeginAic()
{
    SetMMRowMajor();
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
}
} // namespace QLIV2FastCubeCommon
namespace QLIV2FastCubeFp8 {
using namespace AscendC;
using namespace QLIV2FastCubeCommon;
using namespace QLIV2Fast;

// Q进入L0A后，K使用两个256 KiB的L1分区
template <int32_t Slot, bool PAGE>
__aicore__ inline void LoadK(const QLIV2Fast::KeyView<PAGE>& k, int64_t start, int32_t valid)
{
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(Slot + 1));
    k.CopyData(Slot * L1_KEY_SLOT_BYTES, start, valid, 128, KEY_TILE_COLUMNS);
    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(Slot + 1));
}

template <int32_t BSlot>
__aicore__ inline void IssueMmad(int64_t q, int64_t k, int64_t c, int32_t column, uint32_t paddedHeadNum)
{
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(BSlot + 4));
    LocalTensor<fp8_e4m3fn_t> b(TPosition::B2, BSlot * L0_KEY_SLOT_BYTES, L0_KEY_VIEW_ELEMENTS);
    LocalTensor<fp8_e4m3fn_t> kl1(TPosition::A1, k + column * FP8_NZ_BLOCK_COLUMNS, L1_KEY_SLOT_BYTES);
    LoadData2DParamsV2 copy;
    copy.mStartPosition = 0;
    copy.kStartPosition = 0;
    copy.mStep = KEY_N_BLOCKS;
    copy.kStep = QUERY_K_BLOCKS;
    copy.srcStride = KEY_L1_N_BLOCK_STRIDE;
    copy.dstStride = KEY_N_BLOCKS;
    copy.ifTranspose = false;
    LoadData(b, kl1, copy);
    Multiply<BSlot, fp8_e4m3fn_t>(q, c, paddedHeadNum, b);
}

// 每组覆盖256列Key；q/k为L0A/L1字节偏移，Slot选择结果缓存槽
template <int32_t Slot>
__aicore__ inline void ComputeKeyGroup(int64_t q, int64_t k, int32_t keyGroupIndex, uint32_t paddedHeadNum)
{
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(Slot));
    const int64_t c = static_cast<int64_t>(Slot) * L0_RESULT_SLOT_BYTES;
    // 每次加载128列Key到L0B并计算，两个16 KiB槽交替使用
    IssueMmad<0>(q, k, c, keyGroupIndex * RESULT_GROUP_COLUMNS, paddedHeadNum);
    IssueMmad<1>(q, k, c + paddedHeadNum * L0_RESULT_HALF_BYTES_PER_HEAD, keyGroupIndex * RESULT_GROUP_COLUMNS + 128,
                 paddedHeadNum);
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(Slot));
    PublishResult<Slot>(c, paddedHeadNum);
}

__aicore__ inline void ComputeQueryGroup(int64_t q, int64_t k, int32_t valid, uint32_t paddedHeadNum)
{
    // 计算同一4Q在当前Key tile上的全部group
    // 按Key列范围依次计算8组，交替复用两个结果槽
    if (valid == KEY_TILE_COLUMNS) {
        ComputeKeyGroup<0>(q, k, KEY_GROUP_0, paddedHeadNum);
        ComputeKeyGroup<1>(q, k, KEY_GROUP_1, paddedHeadNum);
        ComputeKeyGroup<0>(q, k, KEY_GROUP_2, paddedHeadNum);
        ComputeKeyGroup<1>(q, k, KEY_GROUP_3, paddedHeadNum);
        ComputeKeyGroup<0>(q, k, KEY_GROUP_4, paddedHeadNum);
        ComputeKeyGroup<1>(q, k, KEY_GROUP_5, paddedHeadNum);
        ComputeKeyGroup<0>(q, k, KEY_GROUP_6, paddedHeadNum);
        ComputeKeyGroup<1>(q, k, KEY_GROUP_7, paddedHeadNum);
    } else { // 尾分支按照256列group处理，正式调用pad到2048，无效尾列由vector屏蔽
        for (int32_t keyGroupIndex = 0; keyGroupIndex < valid / RESULT_GROUP_COLUMNS; keyGroupIndex += 2) {
            ComputeKeyGroup<0>(q, k, keyGroupIndex, paddedHeadNum);
            ComputeKeyGroup<1>(q, k, keyGroupIndex + 1, paddedHeadNum);
        }
    }
}
template <int32_t Slot>
__aicore__ inline void ProcessKeyTile(int32_t valid, uint32_t paddedHeadNum)
{
    // 等待一个key tile到L1，用4个Q块依次计算并释放L1
    WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(Slot + 1));
    const int64_t k = static_cast<int64_t>(Slot) * L1_KEY_SLOT_BYTES;
    ComputeQueryGroup(0U, k, valid, paddedHeadNum);
    ComputeQueryGroup(L0_QUERY_SLOT_BYTES, k, valid, paddedHeadNum);
    ComputeQueryGroup(2U * L0_QUERY_SLOT_BYTES, k, valid, paddedHeadNum);
    ComputeQueryGroup(3U * L0_QUERY_SLOT_BYTES, k, valid, paddedHeadNum);
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(Slot + 1));
}

template <int32_t Block>
__aicore__ inline void LoadQBlock(uint32_t paddedHeadNum)
{
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(Block));
    LocalTensor<fp8_e4m3fn_t> ql0(TPosition::A2, static_cast<int64_t>(Block) * L0_QUERY_SLOT_BYTES,
                                  L0_QUERY_VIEW_ELEMENTS);
    LocalTensor<fp8_e4m3fn_t> ql1(
        TPosition::A1, static_cast<int64_t>(Block) * QUERY_ROWS_PER_GROUP * paddedHeadNum * FP8_NZ_BLOCK_COLUMNS,
        L1_QUERY_VIEW_ELEMENTS);
    LoadData2DParamsV2 copy;
    copy.mStartPosition = 0;
    copy.kStartPosition = 0;
    copy.mStep = paddedHeadNum / QUERY_HEADS_PER_M_BLOCK;
    copy.kStep = QUERY_K_BLOCKS;
    copy.srcStride = paddedHeadNum;
    copy.dstStride = paddedHeadNum / QUERY_HEADS_PER_M_BLOCK;
    copy.ifTranspose = false;
    LoadData(ql0, ql1, copy);
    SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(Block));
}

template <bool PAGE>
__aicore__ inline void RunAic(GM_ADDR qAddress, const QLIV2Fast::KeyView<PAGE>& k, uint32_t s2, uint32_t validQueries,
                              uint32_t paddedHeadNum)
{
    GlobalTensor<fp8_e4m3fn_t> q;
    q.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(qAddress));

    BeginAic();
    // Query每行128维，NZ目标C0跨度为16行乘补齐头数；只从GM搬运有效Query行
    LoadNd(0, q, validQueries * paddedHeadNum, 128, 16 * paddedHeadNum);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    // L1到L0A按4行Query分组，每组4*补齐头数*128个FP8元素，四组覆盖16行
    LoadQBlock<0>(paddedHeadNum);
    LoadQBlock<1>(paddedHeadNum);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    LoadQBlock<2>(paddedHeadNum);
    LoadQBlock<3>(paddedHeadNum);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    LoadK<0>(k, 0, KEY_TILE_COLUMNS); // Load Key GM->L1: 等待L1可复用
    LoadK<1>(k, KEY_TILE_COLUMNS, KEY_TILE_COLUMNS);

    // 已预取列数保留原生计数位宽，搬运地址参数单独提升为64位
    uint32_t nextKeyTileStart = 2 * KEY_TILE_COLUMNS;
    int32_t slot0ValidKeyCount = KEY_TILE_COLUMNS, slot1ValidKeyCount = KEY_TILE_COLUMNS;
    while (true) {
        // half0: 计算slot0, 随后复用slot0预取下一块
        ProcessKeyTile<0>(slot0ValidKeyCount, paddedHeadNum);
        if (nextKeyTileStart == s2) {
            ProcessKeyTile<1>(slot1ValidKeyCount, paddedHeadNum);
            break;
        }
        slot0ValidKeyCount = (s2 - nextKeyTileStart) < KEY_TILE_COLUMNS ? s2 - nextKeyTileStart : KEY_TILE_COLUMNS;
        LoadK<0>(k, static_cast<int64_t>(nextKeyTileStart), slot0ValidKeyCount);
        nextKeyTileStart += slot0ValidKeyCount;
        // half1: 计算slot1, 随后复用slot1预取下一块
        ProcessKeyTile<1>(slot1ValidKeyCount, paddedHeadNum);
        if (nextKeyTileStart == s2) {
            ProcessKeyTile<0>(slot0ValidKeyCount, paddedHeadNum);
            break;
        }
        slot1ValidKeyCount = (s2 - nextKeyTileStart) < KEY_TILE_COLUMNS ? s2 - nextKeyTileStart : KEY_TILE_COLUMNS;
        LoadK<1>(k, static_cast<int64_t>(nextKeyTileStart), slot1ValidKeyCount);
        nextKeyTileStart += slot1ValidKeyCount;
    }
    FinishAic<true>();
}
} // namespace QLIV2FastCubeFp8

namespace QLIV2FastCubeMx {
using namespace AscendC;
using namespace QLIV2FastCubeCommon;
using namespace QLIV2Fast;
constexpr uint32_t KEY_TILE_COLUMN_COUNT = KEY_TILE_COLUMNS;
constexpr int64_t KEY_L1_SLOT_BYTES = HEAD_DIM * KEY_TILE_COLUMN_COUNT;
constexpr int64_t QUERY_L1_BYTES = QUERY_STAGE_DATA_BYTES;
// 启动分两次各预取半个K tile；稳态先搬128行的首个NZ块，再补齐前256行及其余行
constexpr int32_t K_PREFETCH_HALF_ROWS = KEY_TILE_COLUMNS / 2;
constexpr int32_t K_PREFETCH_STEP_ROWS = MMAD_RESULT_COLUMNS;
constexpr int32_t K_PREFETCH_PREFIX_ROWS = RESULT_GROUP_COLUMNS;
constexpr int32_t K_MIN_TAIL_ROWS = 2 * RESULT_GROUP_COLUMNS;
constexpr int64_t K_PREFETCH_STEP_BYTES = K_PREFETCH_STEP_ROWS * FP8_NZ_BLOCK_COLUMNS;
// 首个256行、32维K片段占8 KiB，之后为本轮Q及其MX scale暂存区
constexpr int64_t K_PREFETCH_PREFIX_BYTES = K_PREFETCH_PREFIX_ROWS * FP8_NZ_BLOCK_COLUMNS;
// 每个D=128行有4个E8M0 scale，按uint16_t成对搬运，对应基线scalePerToken/FP8_TWO
constexpr uint32_t MX_SCALE_PACKED_ROW_ELEMENTS = (HEAD_DIM / MX_SCALE_GROUP_SIZE) / sizeof(uint16_t);
constexpr uint32_t KEY_SCALE_VIEW_ELEMENTS = KEY_TILE_COLUMNS * (HEAD_DIM / MX_SCALE_GROUP_SIZE);
constexpr uint32_t QUERY_SCALE_VIEW_ELEMENTS = QUERY_TILE_ROWS * MAX_HEAD_NUM * (HEAD_DIM / MX_SCALE_GROUP_SIZE);
constexpr int64_t QUERY_L1_OFFSET = KEY_L1_SLOT_BYTES + K_PREFETCH_PREFIX_BYTES;
constexpr int64_t QUERY_SCALE_L1_OFFSET = QUERY_L1_OFFSET + QUERY_L1_BYTES;
static_assert(2 * KEY_L1_SLOT_BYTES == L1_BUFFER_BYTES);
static_assert(QUERY_SCALE_L1_OFFSET + QUERY_SCALE_VIEW_ELEMENTS <= 2 * KEY_L1_SLOT_BYTES);

template <bool PAGE>
__aicore__ inline void DataCopy(int64_t dst, const QLIV2Fast::KeyView<PAGE>& src, int64_t start, int32_t rows,
                                int32_t d, int32_t column = 0)
{
    src.CopyData(dst, start, rows, d, KEY_TILE_COLUMN_COUNT, column);
}

__aicore__ inline void LoadScale(int64_t dst, GlobalTensor<uint16_t> src, int32_t rows)
{
    LocalTensor<uint16_t> local(TPosition::A1, dst, (L1_BUFFER_BYTES - dst) / sizeof(uint16_t));
    // 每行4个E8M0字节按2个uint16搬运，源D跨度及目标C0跨度均为2，N跨度为1
    Dn2NzParams copy{1,
                     MX_SCALE_PACKED_ROW_ELEMENTS,
                     static_cast<uint32_t>(rows),
                     0,
                     MX_SCALE_PACKED_ROW_ELEMENTS,
                     MX_SCALE_PACKED_ROW_ELEMENTS,
                     1,
                     0};
    DataCopy(local, src, copy);
}

template <bool PAGE>
__aicore__ inline void PreloadTail(int32_t slot, const QLIV2Fast::KeyView<PAGE>& src, int64_t start, int32_t valid)
{
    DataCopy(slot * KEY_L1_SLOT_BYTES + K_PREFETCH_PREFIX_BYTES, src, start + K_PREFETCH_PREFIX_ROWS,
             valid - K_PREFETCH_PREFIX_ROWS, HEAD_DIM);
    DataCopy(slot * KEY_L1_SLOT_BYTES + KEY_TILE_COLUMN_COUNT * FP8_NZ_BLOCK_COLUMNS, src, start,
             K_PREFETCH_PREFIX_ROWS, HEAD_DIM - FP8_NZ_BLOCK_COLUMNS, FP8_NZ_BLOCK_COLUMNS);
}

template <int32_t Block>
__aicore__ inline void LoadQBlock(uint32_t paddedHeadNum)
{
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(Block));
    LocalTensor<mx_fp8_e4m3_t> ql0(TPosition::A2, static_cast<int64_t>(Block) * L0_QUERY_SLOT_BYTES,
                                   L0_QUERY_VIEW_ELEMENTS);
    LocalTensor<fp8_e4m3fn_t> ql1(
        TPosition::A1,
        QUERY_L1_OFFSET + static_cast<int64_t>(Block) * QUERY_ROWS_PER_GROUP * paddedHeadNum * FP8_NZ_BLOCK_COLUMNS,
        L1_QUERY_VIEW_ELEMENTS);
    LoadData2DParamsV2 copy;
    copy.mStartPosition = 0;
    copy.kStartPosition = 0;
    copy.mStep = paddedHeadNum / QUERY_HEADS_PER_M_BLOCK;
    copy.kStep = QUERY_K_BLOCKS;
    copy.srcStride = paddedHeadNum;
    copy.dstStride = paddedHeadNum / QUERY_HEADS_PER_M_BLOCK;
    copy.ifTranspose = false;
    LocalTensor<fp8_e8m0_t> scales(TPosition::A1, QUERY_SCALE_L1_OFFSET, QUERY_SCALE_VIEW_ELEMENTS);
    LoadData2DMxParams mxCopy;
    mxCopy.xStartPosition = Block * (paddedHeadNum / QUERY_HEADS_PER_M_BLOCK);
    mxCopy.yStartPosition = 0;
    mxCopy.xStep = paddedHeadNum / QUERY_HEADS_PER_M_BLOCK;
    mxCopy.yStep = MX_SCALE_PACKED_ROW_ELEMENTS;
    mxCopy.srcStride = MX_SCALE_PACKED_ROW_ELEMENTS;
    mxCopy.dstStride = MX_SCALE_PACKED_ROW_ELEMENTS;
    LoadData(ql0, ql1, scales, copy, mxCopy);
    SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(Block));
    WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(Block));
}

template <int32_t BSlot, bool PAGE = false>
__aicore__ inline void Issue(int64_t q, int64_t k, int64_t c, int32_t column, uint32_t paddedHeadNum,
                             const QLIV2Fast::KeyView<PAGE>& nextK = {}, int64_t nextKeyTileStart = 0,
                             int64_t remaining = 0)
{
    // 从当前L1 key和另一半的scale load到L0B；可穿插下一个key的部分搬运，再MMAD
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(BSlot + 4));
    LocalTensor<mx_fp8_e4m3_t> b(TPosition::B2, BSlot * L0_KEY_SLOT_BYTES, L0_KEY_VIEW_ELEMENTS);
    LocalTensor<fp8_e4m3fn_t> kl1(TPosition::A1, k + column * FP8_NZ_BLOCK_COLUMNS, L1_KEY_SLOT_BYTES);
    LoadData2DParamsV2 copy;
    copy.mStartPosition = 0;
    copy.kStartPosition = 0;
    copy.mStep = KEY_N_BLOCKS;
    copy.kStep = QUERY_K_BLOCKS;
    copy.srcStride = KEY_L1_N_BLOCK_STRIDE;
    copy.dstStride = KEY_N_BLOCKS;
    copy.ifTranspose = false;
    const int64_t keyL1Offset = k;
    LocalTensor<fp8_e8m0_t> scales(TPosition::A1, keyL1Offset ^ KEY_L1_SLOT_BYTES, KEY_SCALE_VIEW_ELEMENTS);
    LoadData2DMxParams mxCopy;
    mxCopy.xStartPosition = column / CUBE_BLOCK_ROWS;
    mxCopy.yStartPosition = 0;
    mxCopy.xStep = KEY_N_BLOCKS;
    mxCopy.yStep = MX_SCALE_PACKED_ROW_ELEMENTS;
    mxCopy.srcStride = MX_SCALE_PACKED_ROW_ELEMENTS;
    mxCopy.dstStride = MX_SCALE_PACKED_ROW_ELEMENTS;
    LoadData(b, kl1, scales, copy, mxCopy);
    if (remaining > 0) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        DataCopy((keyL1Offset ^ KEY_L1_SLOT_BYTES) + K_PREFETCH_STEP_BYTES, nextK,
                 nextKeyTileStart + K_PREFETCH_STEP_ROWS, K_PREFETCH_STEP_ROWS, FP8_NZ_BLOCK_COLUMNS);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
        if (remaining > static_cast<int32_t>(KEY_TILE_COLUMN_COUNT)) {
            PreloadTail(
                static_cast<int32_t>(keyL1Offset / KEY_L1_SLOT_BYTES), nextK, nextKeyTileStart + KEY_TILE_COLUMN_COUNT,
                remaining - static_cast<int32_t>(KEY_TILE_COLUMN_COUNT) < static_cast<int32_t>(KEY_TILE_COLUMN_COUNT) ?
                    remaining - static_cast<int32_t>(KEY_TILE_COLUMN_COUNT) :
                    static_cast<int32_t>(KEY_TILE_COLUMN_COUNT));
        }
    }
    Multiply<BSlot, mx_fp8_e4m3_t>(q, c, paddedHeadNum, b);
}

// WaitFull控制第二次MMAD前是否等待Key预取完成；q/k为L0A/L1字节偏移
template <int32_t Slot, bool WaitFull = false, bool PAGE = false>
__aicore__ inline void ComputeKeyGroup(int64_t q, int64_t k, int32_t keyGroupIndex, uint32_t paddedHeadNum,
                                       const QLIV2Fast::KeyView<PAGE>& nextK = {}, int64_t nextKeyTileStart = 0,
                                       int64_t remaining = 0)
{
    // 两次mmad及交接
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(Slot));
    const int64_t c = static_cast<int64_t>(Slot) * L0_RESULT_SLOT_BYTES;
    Issue<0>(q, k, c, keyGroupIndex * RESULT_GROUP_COLUMNS, paddedHeadNum);
    if constexpr (WaitFull) {
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    }
    Issue<1>(q, k, c + paddedHeadNum * L0_RESULT_HALF_BYTES_PER_HEAD,
             keyGroupIndex * RESULT_GROUP_COLUMNS + MMAD_RESULT_COLUMNS, paddedHeadNum, nextK, nextKeyTileStart,
             remaining);
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(Slot));
    PublishResult<Slot>(c, paddedHeadNum);
}

// Startup仅用于首个Key tile，额外等待启动阶段的后半块搬运完成
template <bool WaitFull = false, bool Startup = false>
__aicore__ inline void ComputeQueryGroup(int64_t q, int64_t k, int32_t valid, uint32_t paddedHeadNum)
{
    // 组织一个4Q子块的8个group；实参0~7为256列组索引，模板参数0/1为交替结果槽
    ComputeKeyGroup<0, WaitFull>(q, k, KEY_GROUP_0, paddedHeadNum);
    ComputeKeyGroup<1, false>(q, k, KEY_GROUP_1, paddedHeadNum);
    if (valid == static_cast<int32_t>(KEY_TILE_COLUMN_COUNT)) {
        ComputeKeyGroup<0, false>(q, k, KEY_GROUP_2, paddedHeadNum);
        ComputeKeyGroup<1, false>(q, k, KEY_GROUP_3, paddedHeadNum);
        if constexpr (Startup) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
        }
        ComputeKeyGroup<0, false>(q, k, KEY_GROUP_4, paddedHeadNum);
        ComputeKeyGroup<1, false>(q, k, KEY_GROUP_5, paddedHeadNum);
        ComputeKeyGroup<0, false>(q, k, KEY_GROUP_6, paddedHeadNum);
        ComputeKeyGroup<1, false>(q, k, KEY_GROUP_7, paddedHeadNum);
    } else {
        for (int32_t keyGroupIndex = 2; keyGroupIndex < valid / RESULT_GROUP_COLUMNS; keyGroupIndex += 2) {
            ComputeKeyGroup<0, false>(q, k, keyGroupIndex, paddedHeadNum);
            ComputeKeyGroup<1, false>(q, k, keyGroupIndex + 1, paddedHeadNum);
        }
    }
}

template <int32_t Slot, bool Startup = false, bool PAGE = false>
__aicore__ inline void ProcessKeyTile(const QLIV2Fast::KeyView<PAGE>& k, int64_t start, int64_t s2,
                                      uint32_t paddedHeadNum)
{
    // 消费当前K tile；在最后4Q处理中交错准备下一个K的scale及被延迟的数据头部
    const int32_t valid = (s2 - start) < static_cast<int32_t>(KEY_TILE_COLUMN_COUNT) ?
                              s2 - start :
                              static_cast<int32_t>(KEY_TILE_COLUMN_COUNT);
    const int64_t nextKeyTileStart = start + KEY_TILE_COLUMN_COUNT;
    const int32_t nextValidKeyCount = (s2 - nextKeyTileStart) < static_cast<int32_t>(KEY_TILE_COLUMN_COUNT) ?
                                          s2 - nextKeyTileStart :
                                          static_cast<int32_t>(KEY_TILE_COLUMN_COUNT);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
    const int64_t tile = static_cast<int64_t>(Slot) * KEY_L1_SLOT_BYTES;
    ComputeQueryGroup<true, Startup>(0U, tile, valid, paddedHeadNum);
    ComputeQueryGroup<false, false>(L0_QUERY_SLOT_BYTES, tile, valid, paddedHeadNum);
    ComputeQueryGroup<false, false>(2U * L0_QUERY_SLOT_BYTES, tile, valid, paddedHeadNum);
    // 最后4Q沿用组索引0~7及交替结果槽，将下一Key tile的预取穿插到各组计算之间
    ComputeKeyGroup<0, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_0, paddedHeadNum);
    if (nextKeyTileStart < s2) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        k.CopyMxScale(Slot * KEY_L1_SLOT_BYTES, nextKeyTileStart, nextValidKeyCount);
    }
    ComputeKeyGroup<1, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_1, paddedHeadNum, k, nextKeyTileStart,
                              valid == K_MIN_TAIL_ROWS ? s2 - nextKeyTileStart : 0);
    if (valid == static_cast<int32_t>(KEY_TILE_COLUMN_COUNT)) {
        ComputeKeyGroup<0, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_2, paddedHeadNum);
        ComputeKeyGroup<1, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_3, paddedHeadNum);
        if (nextKeyTileStart < s2) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            DataCopy((Slot ^ 1) * KEY_L1_SLOT_BYTES, k, nextKeyTileStart, K_PREFETCH_STEP_ROWS, FP8_NZ_BLOCK_COLUMNS);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
        }
        ComputeKeyGroup<0, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_4, paddedHeadNum);
        ComputeKeyGroup<1, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_5, paddedHeadNum);
        ComputeKeyGroup<0, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_6, paddedHeadNum);
        ComputeKeyGroup<1, false>(3U * L0_QUERY_SLOT_BYTES, tile, KEY_GROUP_7, paddedHeadNum, k, nextKeyTileStart,
                                  s2 - nextKeyTileStart);
    } else {
        for (int32_t keyGroupIndex = 2; keyGroupIndex < valid / RESULT_GROUP_COLUMNS; keyGroupIndex += 2) {
            ComputeKeyGroup<0, false>(3U * L0_QUERY_SLOT_BYTES, tile, keyGroupIndex, paddedHeadNum);
            ComputeKeyGroup<1, false>(3U * L0_QUERY_SLOT_BYTES, tile, keyGroupIndex + 1, paddedHeadNum, k,
                                      nextKeyTileStart,
                                      (keyGroupIndex + 2) * RESULT_GROUP_COLUMNS == valid ? s2 - nextKeyTileStart : 0);
        }
    }
}

template <bool PAGE>
__aicore__ inline void RunAic(GM_ADDR qAddress, const QLIV2Fast::KeyView<PAGE>& k, GM_ADDR qsAddress, int64_t s2,
                              uint32_t validQueries, uint32_t paddedHeadNum)
{
    GlobalTensor<fp8_e4m3fn_t> q;
    q.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(qAddress));
    GlobalTensor<uint16_t> qs;
    qs.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(qsAddress));

    BeginAic();
    k.CopyMxScale(KEY_L1_SLOT_BYTES, 0, KEY_TILE_COLUMN_COUNT);
    LoadScale(QUERY_SCALE_L1_OFFSET, qs, validQueries * paddedHeadNum); // 将Query scale从GM搬运到L1
    LoadNd(QUERY_L1_OFFSET, q, validQueries * paddedHeadNum, HEAD_DIM, QUERY_TILE_ROWS * paddedHeadNum);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    DataCopy(0, k, 0, K_PREFETCH_HALF_ROWS, HEAD_DIM);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    DataCopy(K_PREFETCH_HALF_ROWS * FP8_NZ_BLOCK_COLUMNS, k, K_PREFETCH_HALF_ROWS, K_PREFETCH_HALF_ROWS, HEAD_DIM);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadQBlock<0>(paddedHeadNum);
    LoadQBlock<1>(paddedHeadNum);
    LoadQBlock<2>(paddedHeadNum);
    LoadQBlock<3>(paddedHeadNum);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    PreloadTail(1, k, KEY_TILE_COLUMN_COUNT, KEY_TILE_COLUMN_COUNT); // 预取下一key tile可提前覆盖部分
    ProcessKeyTile<0, true>(k, 0, s2, paddedHeadNum);
    for (int64_t start = KEY_TILE_COLUMN_COUNT; start < s2; start += 2 * KEY_TILE_COLUMN_COUNT) {
        ProcessKeyTile<1, false>(k, start, s2, paddedHeadNum);
        if (start + static_cast<int32_t>(KEY_TILE_COLUMN_COUNT) < s2) {
            ProcessKeyTile<0, false>(k, start + KEY_TILE_COLUMN_COUNT, s2, paddedHeadNum);
        }
    }
    FinishAic<false>();
}
} // namespace QLIV2FastCubeMx

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_SERVICE_CUBE_ARCH35_H
