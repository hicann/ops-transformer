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
 * \file quant_lightning_indexer_v2_fast_kernel_arch35.h
 * \brief 16Q任务调度及Cube/Vector计算分发
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH35_H
#define QUANT_LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH35_H

#include "kernel_operator_cache_intf.h"
#include "kernel_operator_vec_duplicate_intf.h"
#include "quant_lightning_indexer_v2_fast_common_arch35.h"
#include "../quant_lightning_indexer_v2_template_tiling_key.h"
#include "quant_lightning_indexer_v2_fast_service_cube_arch35.h"
#include "quant_lightning_indexer_v2_fast_service_vector_arch35.h"

namespace QLIV2FastScaleGuard {
using namespace AscendC;
// 小分页的逐页搬运成本较高，分摊检查才足以抵消全核同步开销；大分页保留组内检查
constexpr uint32_t CROSS_CORE_SCALE_CHECK_MAX_PAGE_ROWS = 32U;
constexpr uint32_t SCALE_SCAN_ELEMENTS = 8192U;
constexpr int64_t SCALE_SCAN_RESULT_UB_OFFSET = SCALE_SCAN_ELEMENTS * sizeof(float);
constexpr int64_t SCALE_SYNC_RESULT_UB_OFFSET = SCALE_SCAN_RESULT_UB_OFFSET + 32U;
constexpr uint32_t SCALE_RESULT_VIEW_ELEMENTS = 8U; // 搬运块对齐的结果视图
// 每个AIV占32字节（8个uint32），两个AIV共用64字节通信区
// 槽内第0个字存汇总结果，第1个字存跨核归并前的分片结果，防止汇总覆盖仍待读取的分片
constexpr uint32_t SCALE_FLAG_SLOT_WORDS = QLIV2Fast::DATA_BLOCK_BYTES / sizeof(uint32_t);
constexpr uint32_t FINAL_SCALE_FLAG_WORD = 0U;
constexpr uint32_t PARTIAL_SCALE_FLAG_WORD = 1U;
constexpr uint16_t SCALE_RESULT_READY_EVENT = 5U; // AIV写回结果后通知组内其他核
constexpr uint16_t SCALE_RESULT_READ_EVENT = 6U;  // 组内各核已读完结果后释放通信区

// 数据来源决定搬运方式，检查与同步顺序保持一致
__aicore__ inline void CopyScale(LocalTensor<float> dst, const GlobalTensor<float>& src, int64_t begin, uint32_t size)
{
    QLIV2Fast::Load(dst, src[begin], size);
}

template <bool PAGE>
__aicore__ inline void CopyScale(LocalTensor<float> dst, const QLIV2Fast::KeyView<PAGE>& src, int64_t begin,
                                 uint32_t size)
{
    src.CopyFp8Scale(dst, begin, size);
}

// src/count为scale来源及有效元素数，sub为当前Cube组内的AIV编号（0或1）
// PAGE_ATTENTION表示Key分页布局；useCrossCoreScaleCheck启用全AIV分摊扫描，否则由组内两个AIV平分
template <bool PAGE_ATTENTION, class ScaleSource>
__aicore__ inline bool HasInvalidPart(const ScaleSource& src, int64_t count, uint32_t sub, bool useCrossCoreScaleCheck)
{
    LocalTensor<float> input(TPosition::VECCALC, 0, SCALE_SCAN_ELEMENTS);
    LocalTensor<uint32_t> resultFlag(TPosition::VECCALC, SCALE_SCAN_RESULT_UB_OFFSET, SCALE_RESULT_VIEW_ELEMENTS);
    int64_t begin, half;
    if (PAGE_ATTENTION && useCrossCoreScaleCheck) {
        // 全部AIV均分256元素对齐的片段，尾核允许为空；每个有效scale只扫描一次
        const uint32_t workers = GetBlockNum() * QLIV2Fast::AIVS_PER_CUBE;
        const uint32_t worker = QLIV2Fast::GetCubeGroupIdx() * QLIV2Fast::AIVS_PER_CUBE + sub;
        // 分片按256个float对齐；255是向上取整的偏置，不是额外扫描元素
        const int64_t chunk = (((count + workers - 1U) / workers + 255U) / 256U) * 256U;
        begin = (static_cast<int64_t>(worker) * chunk) < count ? worker * chunk : count;
        half = (count - begin) < chunk ? count - begin : chunk;
    } else {
        // 两个AIV各承担一半，分界仍按256个float对齐，尾段只检查有效元素
        const int64_t alignedHalf = (((count + 1U) / 2U + 255U) / 256U) * 256U;
        const int64_t split = alignedHalf < count ? alignedHalf : count;
        half = sub == 0U ? split : count - split;
        begin = sub == 0U ? 0U : split;
    }
    for (int64_t offset = 0; offset < half; offset += SCALE_SCAN_ELEMENTS) {
        const uint32_t size =
            static_cast<uint32_t>((half - offset) < SCALE_SCAN_ELEMENTS ? half - offset : SCALE_SCAN_ELEMENTS);
        CopyScale(input, src, begin + offset, size);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        QLIV2FastVector1::InvalidScaleVF(reinterpret_cast<__ubuf__ uint32_t*>(resultFlag.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()), size);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const bool invalid = resultFlag.GetValue(0) != 0U;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (invalid) {
            return true;
        }
    }
    return false;
}
// 各AIV检查所有batch中分配给自己的片段，再汇总整次调用的回退结果
template <bool PAGE>
__aicore__ inline bool HasInvalidScalesLocal(GM_ADDR qs, const QLIV2Fast::KeyView<PAGE>& key, int64_t q, int64_t k,
                                             uint32_t heads)
{
    GlobalTensor<float> query;
    query.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qs));
    const uint32_t sub = GetSubBlockIdx();
    const bool useCrossCoreScaleCheck = PAGE && (key.pageSize <= CROSS_CORE_SCALE_CHECK_MAX_PAGE_ROWS);
    return HasInvalidPart<PAGE>(query, q * heads, sub, useCrossCoreScaleCheck) ||
           HasInvalidPart<PAGE>(key, k, sub, useCrossCoreScaleCheck);
}

// invalid为当前AIV的检查结果，workspace提供结果通信区
// PAGE_ATTENTION表示Key分页布局；启用useCrossCoreScaleCheck时先全AIV汇总，否则仅组内归并
// 最终向所属Cube组发布一致的无效scale判定，返回值用于决定是否回退旧模板
template <bool PAGE_ATTENTION>
__aicore__ inline bool SynchronizeInvalidScales(bool invalid, GM_ADDR workspace, bool useCrossCoreScaleCheck)
{
    const uint32_t core = QLIV2Fast::GetCubeGroupIdx();
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace));
    flags = flags[static_cast<int64_t>(GetBlockNum()) *
                      (QLIV2Fast::PIPELINE_WORKSPACE_BYTES / static_cast<int64_t>(sizeof(uint32_t))) +
                  static_cast<int64_t>(core) * (QLIV2Fast::SCALE_FLAG_BYTES / static_cast<int64_t>(sizeof(uint32_t)))];
    if ASCEND_IS_AIV {
        const uint32_t sub = GetSubBlockIdx();
        LocalTensor<uint32_t> result(TPosition::VECCALC, SCALE_SYNC_RESULT_UB_OFFSET, SCALE_RESULT_VIEW_ELEMENTS);
        result.SetValue(0, invalid ? 1U : 0U);
        // 只写回一个uint32判定值（4字节），无源/目的额外步长，最后一项为保留字段
        DataCopyExtParams copy{1, sizeof(uint32_t), 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        if (PAGE_ATTENTION && useCrossCoreScaleCheck) {
            DataCopyPad(flags[sub * SCALE_FLAG_SLOT_WORDS + PARTIAL_SCALE_FLAG_WORD], result, copy);
            // 确认本核结果已写入GM后进行AIV全核同步，避免读取未发布的分片
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            SyncAll();
            GlobalTensor<uint32_t> allFlags;
            allFlags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
                (workspace + static_cast<int64_t>(GetBlockNum()) * QLIV2Fast::PIPELINE_WORKSPACE_BYTES)));
            invalid = false;
            for (uint32_t group = 0U; group < GetBlockNum(); ++group) {
                auto groupFlags =
                    allFlags[group * (QLIV2Fast::SCALE_FLAG_BYTES / static_cast<int64_t>(sizeof(uint32_t)))];
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(groupFlags);
                // 两个AIV的分片结果分别位于第1和第9个uint32，即各自32字节槽内的第1个字
                invalid = invalid || (groupFlags.GetValue(PARTIAL_SCALE_FLAG_WORD) != 0U) ||
                          (groupFlags.GetValue(SCALE_FLAG_SLOT_WORDS + PARTIAL_SCALE_FLAG_WORD) != 0U);
            }
            // 分片结果与汇总结果分开存放，其他核归并时不会读到被覆盖的值
            result.SetValue(0, invalid ? 1U : 0U);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(flags[sub * SCALE_FLAG_SLOT_WORDS + FINAL_SCALE_FLAG_WORD], result, copy);
        } else {
            DataCopyPad(flags[sub * SCALE_FLAG_SLOT_WORDS + FINAL_SCALE_FLAG_WORD], result, copy);
        }
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SCALE_RESULT_READY_EVENT);
        CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READY_EVENT);
        // MTE3写入不更新Scalar缓存；通信区会跨调用复用，读取前须刷新缓存
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(flags);
        // 汇总结果位于两个32字节槽的首字，原偏移为0和8个uint32
        invalid = (flags.GetValue(FINAL_SCALE_FLAG_WORD) != 0U) ||
                  (flags.GetValue(SCALE_FLAG_SLOT_WORDS + FINAL_SCALE_FLAG_WORD) != 0U);
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READ_EVENT);
        CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READ_EVENT);
    } else if ASCEND_IS_AIC {
        // AIV1使用AIV0事件号加16，因此就绪/确认事件原始编号分别为5/21和6/22
        CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READY_EVENT);
        CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READY_EVENT +
                                                                   QLIV2Fast::AIV1_EVENT_OFFSET);
        // MTE3写入不更新Scalar缓存；通信区会跨调用复用，读取前须刷新缓存
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(flags);
        invalid = (flags.GetValue(FINAL_SCALE_FLAG_WORD) != 0U) ||
                  (flags.GetValue(SCALE_FLAG_SLOT_WORDS + FINAL_SCALE_FLAG_WORD) != 0U);
        // 等待生产方MTE3写入完成后再确认，确保两个Vector核进入流水前都能读取两条判定缓存行
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(SCALE_RESULT_READY_EVENT);
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(SCALE_RESULT_READY_EVENT +
                                                                    QLIV2Fast::AIV1_EVENT_OFFSET);
        CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READ_EVENT);
        CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_S>(SCALE_RESULT_READ_EVENT +
                                                                   QLIV2Fast::AIV1_EVENT_OFFSET);
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(SCALE_RESULT_READ_EVENT);
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_FIX>(SCALE_RESULT_READ_EVENT +
                                                                    QLIV2Fast::AIV1_EVENT_OFFSET);
    }
    return invalid;
}
} // namespace QLIV2FastScaleGuard

namespace QLIV2Kernel {
using namespace AscendC;
template <bool IS_MX, bool PAGE>
// 量化模式和Key布局在编译期区分；头数统一在运行时处理
class QLIV2FastKernel {
public:
    __aicore__ static inline void Run(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR qs, GM_ADDR ks, GM_ADDR indices,
                                      GM_ADDR values, GM_ADDR workspace, uint32_t s1, uint32_t s2, bool returnValues,
                                      int64_t keyLengthOffset, const QLIV2Fast::KeyView<PAGE>& keyView,
                                      int64_t taskOffset, uint32_t requested, uint32_t heads)
    {
        // 头数非8对齐原因：每行权重和q scale在UB中起点保持32字节对齐；4个Query头展开为矩阵行，8对齐更好与当前流水配套
        const uint32_t paddedHeadNum =
            ((heads + (QLIV2Fast::HEAD_NUM_ALIGNMENT - 1U)) / QLIV2Fast::HEAD_NUM_ALIGNMENT) *
            QLIV2Fast::HEAD_NUM_ALIGNMENT;
        // block/blocks分别为当前Cube组编号和Cube组总数，每组包含一个AIC及两个AIV
        const uint32_t block = QLIV2Fast::GetCubeGroupIdx();
        const uint32_t blocks = GetBlockNum();
        // 使用TPipe初始化的矩阵乘同步令牌，结束时恢复，供析构流程使用
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        }
        // 按全局任务号(taskOffset + tile)分配到当前Cube核及其配对的两个AIV核
        for (uint32_t tile = static_cast<uint32_t>((block + blocks - taskOffset % blocks) % blocks);
             tile < (s1 + (QLIV2Fast::QUERY_TILE_ROWS - 1U)) / QLIV2Fast::QUERY_TILE_ROWS; tile += blocks) {
            const int64_t base = static_cast<int64_t>(tile) * QLIV2Fast::QUERY_TILE_ROWS;
            const uint32_t validQueries = static_cast<uint32_t>(
                (s1 - base) < QLIV2Fast::QUERY_TILE_ROWS ? s1 - base : QLIV2Fast::QUERY_TILE_ROWS);
            // 因果模式的keyLengthOffset为K-Q，无mask模式为K，每行有效长度不超过实际K，无mask任务尾块同样按实际K截断
            const int64_t lastRowK = keyLengthOffset + base + validQueries;
            const int64_t alignedLastRowK = (lastRowK + (QLIV2Fast::KEY_TILE_COLUMNS - 1U)) /
                                            QLIV2Fast::KEY_TILE_COLUMNS * QLIV2Fast::KEY_TILE_COLUMNS;
            const uint32_t taskK = static_cast<uint32_t>(s2 < alignedLastRowK ? s2 : alignedLastRowK); // vector实际范围
            // cube规整计算
            const uint32_t paddedK = (taskK + (QLIV2Fast::KEY_TILE_COLUMNS - 1U)) / QLIV2Fast::KEY_TILE_COLUMNS *
                                     QLIV2Fast::KEY_TILE_COLUMNS;
            GM_ADDR cubeQuery = q + static_cast<int64_t>(base) * heads * 128U;
            GM_ADDR cubeScale = qs + static_cast<int64_t>(base) * heads * 4U;
            if (heads != paddedHeadNum) {
                GM_ADDR staged = workspace +
                                 static_cast<int64_t>(blocks) * QLIV2Fast::QUERY_STAGE_POOL_OFFSET_PER_CORE +
                                 static_cast<int64_t>(block) * QLIV2Fast::QUERY_STAGE_BYTES;
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0U) {
                        StageQuery(cubeQuery, cubeScale, staged, validQueries, heads, paddedHeadNum);
                    }
                } else if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
                        QLIV2Fast::QUERY_STAGE_READY_EVENT_ID);
                }
                cubeQuery = staged;
                cubeScale = staged + QLIV2Fast::QUERY_STAGE_DATA_BYTES;
            }
            if ASCEND_IS_AIC {
                if constexpr (IS_MX) {
                    QLIV2FastCubeMx::RunAic(cubeQuery, keyView, cubeScale, paddedK, validQueries, paddedHeadNum);
                } else {
                    QLIV2FastCubeFp8::RunAic(cubeQuery, keyView, paddedK, validQueries, paddedHeadNum);
                }
            } else if ASCEND_IS_AIV {
                QLIV2FastVector::RunVector<IS_MX>(
                    w, qs, ks, workspace + static_cast<int64_t>(block) * QLIV2Fast::PIPELINE_WORKSPACE_BYTES, indices,
                    values, s1, keyLengthOffset, base, returnValues, requested, taskK, keyView, paddedHeadNum, heads);
            }
        }
        if ASCEND_IS_AIC {
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        }
    }

private:
    // 由第一个AIV完成当前任务补齐，待GM数据写入完成后通知Cube读取
    __aicore__ static inline void StageQuery(GM_ADDR q, GM_ADDR qs, GM_ADDR staged, uint32_t rows, uint32_t heads,
                                             uint32_t paddedHeadNum)
    {
        LocalTensor<uint32_t> queryUb(TPosition::VECCALC, 0U, QLIV2Fast::QUERY_STAGE_DATA_BYTES / sizeof(uint32_t));
        GlobalTensor<uint32_t> queryGm, queryOut;
        queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(q));
        queryOut.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(staged));
        Duplicate(queryUb, static_cast<uint32_t>(0), 16U * paddedHeadNum * 32U);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        // 每行Query为heads*128字节；UB头数补齐间隔按32字节块计，每个补齐头占4块
        DataCopyExtParams queryCopy{static_cast<uint16_t>(rows), heads * 128U, 0U, (paddedHeadNum - heads) * 4U, 0U};
        DataCopyPadExtParams<uint32_t> noPad{false, 0U, 0U, 0U};
        DataCopyPad(queryUb, queryGm, queryCopy, noPad);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID7);
        // 16行Query，每头128字节即32个uint32；此DataCopy重载的长度单位为元素
        DataCopy(queryOut, queryUb, 16U * paddedHeadNum * 32U);
        if constexpr (IS_MX) {
            LocalTensor<uint32_t> scaleUb(TPosition::VECCALC, QLIV2Fast::QUERY_STAGE_DATA_BYTES,
                                          QLIV2Fast::QUERY_TILE_ROWS * QLIV2Fast::MAX_HEAD_NUM);
            GlobalTensor<uint32_t> scaleGm, scaleOut;
            scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(qs));
            scaleOut.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>((staged + QLIV2Fast::QUERY_STAGE_DATA_BYTES)));
            // E8M0编码0x7f表示1；补齐的Query行和头的数据均为0
            Duplicate(scaleUb, static_cast<uint32_t>(QLIV2Fast::MX_SCALE_ONE_PACKED), 16U * paddedHeadNum);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
            // 每头4个E8M0字节打包为一个uint32；每行拷贝heads*4字节，右侧补齐编码1
            DataCopyExtParams scaleCopy{static_cast<uint16_t>(rows), heads * 4U, 0U, 0U, 0U};
            DataCopyPadExtParams<uint32_t> onePad{true, 0U, static_cast<uint8_t>(paddedHeadNum - heads),
                                                  QLIV2Fast::MX_SCALE_ONE_PACKED};
            DataCopyPad(scaleUb, scaleGm, scaleCopy, onePad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID7);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID7);
            DataCopy(scaleOut, scaleUb, 16U * paddedHeadNum);
        }
        CrossCoreSetFlag<QLIV2Fast::CROSS_CORE_SYNC_MODE, PIPE_MTE3>(QLIV2Fast::QUERY_STAGE_READY_EVENT_ID);
        // RunVector会立即复用UB，包括queryUb使用的暂存区，须先等待搬运完成
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    }
};

// 分发层负责整次调用的回退检查和batch调度，各计算服务负责单个任务
template <int32_t DT_Q, int32_t DT_K, int32_t PAGE_ATTENTION, int32_t Q_LAYOUT_T, int32_t K_LAYOUT_T>
class QLIV2FastDispatch {
    using TilingData = QLIV2TilingData;
    static constexpr uint32_t QUANT_MODE_FP8 = 1U;
    static constexpr uint32_t QUANT_MODE_MXFP8 = 3U;

    struct Context {
        GM_ADDR query;
        GM_ADDR key;
        GM_ADDR weights;
        GM_ADDR queryScale;
        GM_ADDR keyScale;
        GM_ADDR cuSeqlensQ;
        GM_ADDR cuSeqlensK;
        GM_ADDR sequsedK;
        GM_ADDR blockTable;
        GM_ADDR outputIdxOffset;
        GM_ADDR metadata;
        GM_ADDR sparseIndices;
        GM_ADDR sparseValues;
        GM_ADDR userWorkspace;
        const TilingData* tilingData;

        struct BatchInfo {
            int64_t qPrefix;
            // 长度读取自int32序列信息；计数与64位地址偏移分开保存
            uint32_t qCount;
            uint32_t kCount;
            QLIV2Fast::KeyView<PAGE_ATTENTION> keyView;
        };

        // scale检查与正式调度共用同一batch描述，保证长度、分页寻址及非连续stride解释一致
        template <bool IS_MX>
        __aicore__ inline BatchInfo ReadBatch(uint32_t batch) const
        {
            GlobalTensor<int32_t> qLengths, kLengths;
            qLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensQ));
            if constexpr (PAGE_ATTENTION) {
                kLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedK));
            } else {
                kLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensK));
            }
            BatchInfo info;
            info.qPrefix = qLengths.GetValue(batch);
            info.qCount = static_cast<uint32_t>(static_cast<int64_t>(qLengths.GetValue(batch + 1U)) - info.qPrefix);
            const int64_t kPrefix = PAGE_ATTENTION ? 0U : kLengths.GetValue(batch);
            info.kCount =
                static_cast<uint32_t>(PAGE_ATTENTION ? kLengths.GetValue(batch) :
                                                       static_cast<int64_t>(kLengths.GetValue(batch + 1U)) - kPrefix);
            if (info.qCount == 0U) {
                return info;
            }
            info.keyView.validRows = info.kCount;
            info.keyView.Init(
                key + static_cast<int64_t>(kPrefix) * 128U, keyScale + static_cast<int64_t>(kPrefix) * 4U,
                blockTable == nullptr ? nullptr :
                                        blockTable + static_cast<int64_t>(batch) * tilingData->maxBlockNumPerBatch * 4U,
                tilingData->blockSize, tilingData->keyStride0, tilingData->keyDequantScaleStride0, IS_MX ? 4U : 1U);
            return info;
        }

        template <bool IS_MX>
        __aicore__ inline void RunBatches()
        {
            int64_t taskOffset = 0U;
            for (uint32_t batch = 0U; batch < tilingData->bSize; ++batch) {
                const auto info = ReadBatch<IS_MX>(batch);
                const int64_t qPrefix = info.qPrefix;
                const uint32_t qCount = info.qCount, kCount = info.kCount;
                if (qCount == 0U) {
                    continue;
                }
                const auto& keyView = info.keyView;
                const int64_t qOffset = static_cast<int64_t>(qPrefix) * tilingData->gSize;
                // sparseMode=3为右下因果mask，行偏移取K-Q；无mask模式取K，每行可访问全部Key
                const int64_t keyLengthOffset =
                    tilingData->sparseMode == 3U ? static_cast<int64_t>(kCount) - qCount : static_cast<int64_t>(kCount);
                // Query每头128个FP8字节；weight/scale/int32索引为4字节，BF16 value为2字节
                QLIV2FastKernel<IS_MX, PAGE_ATTENTION>::Run(
                    query + qOffset * 128U, key, weights + qOffset * 4U, queryScale + qOffset * 4U, keyScale,
                    sparseIndices + static_cast<int64_t>(qPrefix) * tilingData->sparseCount * 4U,
                    sparseValues + static_cast<int64_t>(qPrefix) * tilingData->sparseCount * 2U, userWorkspace, qCount,
                    kCount, tilingData->returnValue != 0U, keyLengthOffset, keyView, taskOffset,
                    tilingData->sparseCount, tilingData->gSize);
                // 跨batch延续全局任务编号，与metadata中的轮转分核规则保持一致
                taskOffset += (qCount + (QLIV2Fast::QUERY_TILE_ROWS - 1U)) / QLIV2Fast::QUERY_TILE_ROWS;
            }
        }

        __aicore__ inline bool TryRun()
        {
            if (!IsA5FastAuthorized(metadata) || outputIdxOffset != nullptr) {
                return false;
            }
            bool invalidScale = false;
            // 分发前检查所有batch，确保Cube和Vector统一选择执行路径
            if (tilingData->quantMode == QUANT_MODE_FP8) {
                if ASCEND_IS_AIV {
                    for (uint32_t batch = 0U; batch < tilingData->bSize; ++batch) {
                        const auto info = ReadBatch<false>(batch);
                        if (info.qCount == 0U) {
                            continue;
                        }
                        const bool batchInvalid = QLIV2FastScaleGuard::HasInvalidScalesLocal(
                            queryScale + static_cast<int64_t>(info.qPrefix) * tilingData->gSize * 4U, info.keyView,
                            info.qCount, info.kCount, tilingData->gSize);
                        invalidScale = invalidScale || batchInvalid;
                    }
                }
                invalidScale = QLIV2FastScaleGuard::SynchronizeInvalidScales<PAGE_ATTENTION>(
                    invalidScale, userWorkspace,
                    tilingData->blockSize <= QLIV2FastScaleGuard::CROSS_CORE_SCALE_CHECK_MAX_PAGE_ROWS);
            }
            if (invalidScale) {
                return false;
            }
            if (tilingData->quantMode == QUANT_MODE_FP8) {
                RunBatches<false>();
            } else {
                RunBatches<true>();
            }
            return true;
        }
    };

    __aicore__ static inline bool IsA5FastAuthorized(GM_ADDR metadata)
    {
        if (metadata == nullptr) {
            return false;
        }
        GlobalTensor<uint32_t> metadataGm;
        metadataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
        const uint32_t metadataAicNum = metadataGm.GetValue(QLI_V2_METADATA_AIC_NUM_INDEX);
        const uint32_t metadataAivNum = metadataGm.GetValue(QLI_V2_METADATA_AIV_NUM_INDEX);
        const int64_t fastFlagOffset =
            static_cast<int64_t>(GetQliV2MetadataExtensionOffset(metadataAicNum, metadataAivNum)) +
            QLI_V2_A5_FAST_FLAG_OFFSET;
        return metadataGm.GetValue(fastFlagOffset) == QLI_V2_A5_FAST_ENABLE;
    }

public:
    __aicore__ static inline bool TryRun(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale,
                                         GM_ADDR keyScale, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedK,
                                         GM_ADDR blockTable, GM_ADDR outputIdxOffset, GM_ADDR metadata,
                                         GM_ADDR sparseIndices, GM_ADDR sparseValues, GM_ADDR userWorkspace,
                                         const TilingData* tilingData)
    {
        if constexpr (DT_Q == QLIV2_TPL_FLOAT8_E4M3FN && DT_K == QLIV2_TPL_FLOAT8_E4M3FN &&
                      Q_LAYOUT_T == QLIV2_LAYOUT_TND &&
                      ((PAGE_ATTENTION == 0 && K_LAYOUT_T == QLIV2_LAYOUT_TND) ||
                       (PAGE_ATTENTION == 1 && K_LAYOUT_T == QLIV2_LAYOUT_PA_BBND))) {
            Context context{query,      key,           weights,      queryScale,    keyScale,
                            cuSeqlensQ, cuSeqlensK,    sequsedK,     blockTable,    outputIdxOffset,
                            metadata,   sparseIndices, sparseValues, userWorkspace, tilingData};
            return context.TryRun();
        }
        return false;
    }
};
} // namespace QLIV2Kernel

#endif // QUANT_LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH35_H
