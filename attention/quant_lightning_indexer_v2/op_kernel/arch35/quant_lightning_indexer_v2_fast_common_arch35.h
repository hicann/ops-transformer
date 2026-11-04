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
 * \file quant_lightning_indexer_v2_fast_common_arch35.h
 * \brief 公共常量、工作区布局及TND、分页Key访问
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_COMMON_ARCH35_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_COMMON_ARCH35_H

#include <cstdint>
#include "kernel_operator_data_copy_intf.h"
#include "kernel_operator_block_sync_intf.h"
#include "quant_lightning_indexer_v2_common_arch35.h"
#include "../quant_lightning_indexer_v2_metadata.h"

namespace QLIV2Fast {
using namespace AscendC;

// 流水基本块：Q按16行、K按2048列推进；TopK状态容量与K块宽含义独立
constexpr uint32_t CROSS_CORE_SYNC_MODE = QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4;
// AIV0完成Query及其scale暂存后通知Cube，供非对齐头数路径读取
constexpr uint16_t QUERY_STAGE_READY_EVENT_ID = 7;
constexpr uint32_t QUERY_TILE_ROWS = 16U;
constexpr int32_t KEY_TILE_COLUMNS = 2048;
constexpr int32_t TOPK_STATE_CAPACITY = optiling::QLI_V2_A5_FAST_SPARSE_CAPACITY;
constexpr int32_t TOPK_CANDIDATE_CAPACITY = 8192;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t MAX_HEAD_NUM = 32U;
constexpr uint32_t MX_SCALE_GROUP_SIZE = 32U;
constexpr uint32_t DATA_BLOCK_BYTES = 32U;       // GM/UB搬运的基本对齐块大小
constexpr uint32_t VECTOR_REGISTER_BYTES = 256U; // 单个向量寄存器的字节数
constexpr uint32_t HEAD_NUM_ALIGNMENT = 8U;      // 每头一个float权重，8头使每行按32字节对齐
constexpr uint32_t AIVS_PER_CUBE = 2U;           // Arch35每个Cube组对应两个AIV
constexpr uint16_t AIV1_EVENT_OFFSET = 16U;      // mode4中同组AIV1的事件号比AIV0增加16
// 以下容量以字节计；LocalTensor的长度另按元素数计算
constexpr int64_t L1_BUFFER_BYTES = 512U * 1024U;
constexpr int64_t QUERY_STAGE_DATA_BYTES = QUERY_TILE_ROWS * MAX_HEAD_NUM * HEAD_DIM;
constexpr uint32_t CUBE_RESULT_VIEW_ELEMENTS = 32U * 1024U; // float元素数，Cube/Vector共用交接视图
constexpr int64_t CUBE_RESULT_SLOT_OFFSET_BYTES = 256U;     // 两个交接槽的起始偏移，不是独立分配容量
constexpr int64_t EFFECTIVE_WEIGHT_UB_OFFSET = CUBE_RESULT_VIEW_ELEMENTS * sizeof(float);
constexpr uint16_t BF16_NEGATIVE_INF_BITS = 0xff80U;
constexpr uint32_t MX_SCALE_ONE_PACKED = 0x7f7f7f7fU; // 四个E8M0编码1，按uint32_t搬运

// Kernel侧工作区布局，所有大小和偏移均以字节计；Host侧独立预留同等容量
constexpr int64_t STATE_KEY_BYTES =
    static_cast<int64_t>(QUERY_TILE_ROWS) * TOPK_STATE_CAPACITY * static_cast<int64_t>(sizeof(uint16_t));
constexpr int64_t CANDIDATE_BYTES =
    static_cast<int64_t>(QUERY_TILE_ROWS) * TOPK_CANDIDATE_CAPACITY * static_cast<int64_t>(sizeof(uint16_t));
constexpr int64_t CANDIDATE_INDEX_OFFSET = STATE_KEY_BYTES + CANDIDATE_BYTES;
constexpr int64_t STATE_INDEX_OFFSET = CANDIDATE_INDEX_OFFSET + CANDIDATE_BYTES;
constexpr int64_t PIPELINE_WORKSPACE_BYTES = STATE_INDEX_OFFSET + static_cast<int64_t>(QUERY_TILE_ROWS) *
                                                                      TOPK_STATE_CAPACITY *
                                                                      static_cast<int64_t>(sizeof(int32_t));
constexpr int64_t SCALE_FLAG_BYTES = 64U; // 两个AIV各占一个32字节通信槽
// 全局依次排列全部核流水区、全部核scale通信区和全部核Query暂存区，非单核交错排列
constexpr int64_t QUERY_STAGE_POOL_OFFSET_PER_CORE = PIPELINE_WORKSPACE_BYTES + SCALE_FLAG_BYTES;
constexpr int64_t QUERY_STAGE_BYTES =
    static_cast<int64_t>(QUERY_TILE_ROWS) * MAX_HEAD_NUM * (HEAD_DIM + static_cast<int64_t>(sizeof(float)));
constexpr int64_t CORE_WORKSPACE_BYTES = QUERY_STAGE_POOL_OFFSET_PER_CORE + QUERY_STAGE_BYTES;

// 将混合任务的物理核编号转换为Cube组编号，正常任务比例为1或2
__aicore__ inline uint32_t GetCubeGroupIdx()
{
    const int64_t taskRatio = GetTaskRation();
    // 除零保护：API返回0时将除数置为1，保证下方除法的除数非零；正常任务映射不变
    const int64_t safeTaskRatio = (taskRatio == 0) ? 1 : taskRatio;
    return static_cast<uint32_t>(GetBlockIdx() / safeTaskRatio);
}

__aicore__ inline void LoadHeadRow(const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint32_t heads,
                                   uint32_t paddedHeads)
{
    // 单行float搬运：每头4字节；源/目的无额外步长，右侧补零至paddedHeads个float
    DataCopyExtParams copy{1U, heads * 4U, 0U, 0U, 0U};
    DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(paddedHeads - heads), 0.0f};
    DataCopyPad(dst, src, copy, pad);
}

template <class T>
__aicore__ inline void Load(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)
{
    if (count == 0U) {
        return;
    }
    // 单段连续GM到UB搬运，blockLen以字节计；不设置额外步长或左右补齐
    DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
    DataCopyPadExtParams<T> padding{false, 0U, 0U, static_cast<T>(0)};
    DataCopyPad(dst, src, copy, padding);
}
template <class T>
__aicore__ inline void Store(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
{
    if (count == 0U) {
        return;
    }
    // 单段连续UB到GM搬运，blockLen为count个T的字节数，两个stride均为0
    DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
    DataCopyPad(dst, src, copy);
}

// Cube、FP8加权归约和scale检查共用逻辑token寻址
// PAGE=false时在编译期消除分页分支，保留TND连续搬运路径
template <bool PAGE>
struct KeyView {
    GlobalTensor<fp8_e4m3fn_t> data;
    GlobalTensor<uint16_t> mxScale;
    GlobalTensor<float> fp8Scale;
    GlobalTensor<int32_t> table;
    uint32_t pageSize = 0;
    int64_t validRows = INT64_MAX;
    int64_t dataPageStride = 0;
    int64_t scalePageStride = 0;

    __aicore__ inline void Init(GM_ADDR key, GM_ADDR scales, GM_ADDR blockTable, uint32_t blockSize, int64_t keyStride,
                                int64_t scaleStride, uint32_t scaleRowElements)
    {
        data.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(key));
        mxScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(scales));
        fp8Scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        if constexpr (PAGE) {
            table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
            pageSize = blockSize;
            // FP8的D=128，每个元素1字节；MX每行4个E8M0字节，FP8 scale每行1个float
            dataPageStride = keyStride != 0 ? keyStride : static_cast<int64_t>(blockSize) * HEAD_DIM;
            scalePageStride = scaleStride != 0 ? scaleStride : static_cast<int64_t>(blockSize) * scaleRowElements;
        }
    }

    __aicore__ inline int64_t Offset(int64_t start, int64_t pageStride, uint32_t rowElements) const
    {
        if constexpr (PAGE) {
            // batch内token索引来自int32序列信息；物理页字节地址使用64位乘加
            const uint32_t row = static_cast<uint32_t>(start);
            const uint32_t physical = static_cast<uint32_t>(table.GetValue(row / pageSize));
            return static_cast<int64_t>(physical) * pageStride + static_cast<int64_t>(row % pageSize) * rowElements;
        } else {
            return start * rowElements;
        }
    }

    __aicore__ inline uint32_t Segment(int64_t start, uint32_t remaining, int64_t pageStride,
                                       uint32_t rowElements) const
    {
        if constexpr (!PAGE) {
            return remaining;
        } else {
            const uint32_t row = static_cast<uint32_t>(start);
            uint32_t rows = pageSize - row % pageSize;
            if (rows >= remaining) {
                return remaining;
            }
            if (pageStride != static_cast<int64_t>(pageSize) * rowElements) {
                return rows;
            }
            // 连续物理页可合并为一次数据搬运
            const uint32_t logical = row / pageSize;
            const uint32_t first = static_cast<uint32_t>(table.GetValue(logical));
            for (uint32_t next = 1; rows < remaining; ++next) {
                if (static_cast<uint32_t>(table.GetValue(logical + next)) != first + next) {
                    break;
                }
                const uint32_t add = (remaining - rows) < pageSize ? remaining - rows : pageSize;
                rows += add;
            }
            return rows;
        }
    }

    // 搬运范围不超过validRows，调用方在TopK前屏蔽Cube补齐位置的输出
    __aicore__ inline void CopyData(int64_t dst, int64_t start, uint32_t rows, uint32_t width, uint32_t c0Stride,
                                    uint32_t column = 0) const
    {
        if (start >= validRows) {
            return;
        }
        rows = rows < (validRows - start) ? rows : validRows - start;
        for (uint32_t copied = 0; copied < rows;) {
            const uint32_t count = Segment(start + copied, rows - copied, dataPageStride, 128U);
            LocalTensor<fp8_e4m3fn_t> target(TPosition::A1, dst + static_cast<int64_t>(copied) * 32U,
                                             L1_BUFFER_BYTES - dst - static_cast<int64_t>(copied) * 32U);
            // 单矩阵ND到NZ：源每行128个FP8元素；NZ的C0为32字节，copied按此推进目标地址
            // 末尾的1为NZ的N方向步长，其余两个矩阵间距为0；c0Stride由目标tile布局指定
            Nd2NzParams params{1,        static_cast<uint16_t>(count),    width, 0,
                               HEAD_DIM, static_cast<uint16_t>(c0Stride), 1,     0};
            DataCopy(target, data[Offset(start + copied, dataPageStride, 128U) + column], params);
            copied += count;
        }
    }

    __aicore__ inline void CopyMxScale(int64_t dst, int64_t start, uint32_t rows) const
    {
        if (start >= validRows) {
            return;
        }
        rows = rows < (validRows - start) ? rows : validRows - start;
        for (uint32_t copied = 0; copied < rows;) {
            const uint32_t count = Segment(start + copied, rows - copied, scalePageStride, 4U);
            LocalTensor<uint16_t> target(TPosition::A1, dst + static_cast<int64_t>(copied) * 4U,
                                         (L1_BUFFER_BYTES - dst - static_cast<int64_t>(copied) * 4U) / 2U);
            // 每行4个E8M0字节按2个uint16搬运；4字节行偏移除2后用于uint16视图寻址
            // 单矩阵DN到NZ，源D步长和目标C0步长均为2，N步长为1，无矩阵间额外间距
            Dn2NzParams params{1, 2, count, 0, 2, 2, 1, 0};
            DataCopy(target, mxScale[Offset(start + copied, scalePageStride, 4U) / 2U], params);
            copied += count;
        }
    }

    __aicore__ inline void CopyFp8Scale(const LocalTensor<float>& dst, int64_t start, uint32_t rows) const
    {
        if (start >= validRows) {
            return;
        }
        rows = rows < (validRows - start) ? rows : validRows - start;
        for (uint32_t copied = 0; copied < rows;) {
            const uint32_t count = Segment(start + copied, rows - copied, scalePageStride, 1U);
            // 每行一个float scale（4字节）；单段搬运，源/目的无额外步长且不补齐有效数据
            DataCopyExtParams params{1, count * 4U, 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(dst[static_cast<int64_t>(copied)], fp8Scale[Offset(start + copied, scalePageStride, 1U)],
                        params, pad);
            copied += count;
        }
    }
};
} // namespace QLIV2Fast

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_COMMON_ARCH35_H
