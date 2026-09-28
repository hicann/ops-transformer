/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GMM_ADD_C_16_H_
#define _GMM_ADD_C_16_H_

#include "grouped_matmul_utils.h"

using namespace AscendC;

namespace ops_gmm_add_c_16 {

/*@brief store variables for core split configuration
 */
struct MNConfig {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t mIdx;
    uint32_t nIdx;
    uint32_t blockDimM;
    uint32_t blockDimN;
    uint32_t singleM;
    uint32_t singleN;
    // offset params
    uint32_t totalK = 0;
    uint32_t cuK = 0;
    uint32_t groupIdx = 0;
};

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType>
class GMMProcess {
private:
    __aicore__ inline void Process_();

    ComputeType &computeOp; // inernal computation operator
    const GmmAddTilingData &tilingData;
    uint32_t coreIdx;
    MNConfig mnConfig;

public:
    /** @brief constructor */
    __aicore__ inline GMMProcess(ComputeType &computeOp_, const GmmAddTilingData &tilingData)
        : computeOp(computeOp_),
          tilingData(tilingData)
    {}

    __aicore__ inline void Init();

    __aicore__ inline void Process();
};

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Init()
{
    coreIdx = GetBlockIdx();
    mnConfig.m = tilingData.gmmBaseParams.M;
    mnConfig.n = tilingData.gmmBaseParams.N;
    mnConfig.totalK = tilingData.gmmBaseParams.TotalK;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process()
{
    Process_();
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process_()
{
    uint32_t count = 0;

    uint32_t dimM = Ceil(mnConfig.m, tilingData.gmmBaseParams.singleM);
    uint32_t dimN = Ceil(mnConfig.n, tilingData.gmmBaseParams.singleN);
    mnConfig.blockDimM = dimM;
    mnConfig.blockDimN = dimN;
    mnConfig.singleM = tilingData.gmmBaseParams.singleM;
    mnConfig.singleN = tilingData.gmmBaseParams.singleN;

    for (uint32_t groupIdx(0); groupIdx < tilingData.gmmBaseParams.groupNum; ++groupIdx) {
        mnConfig.k = computeOp.CalcKForThisGroup(groupIdx);
        if (mnConfig.k == 0) {
            continue;
        }
        mnConfig.groupIdx = groupIdx;
        uint32_t curCount = count + dimM * dimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + tilingData.gmmBaseParams.usedCoreNum;
        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - count) / dimN;
            mnConfig.nIdx = (curBlock - count) % dimN;
            computeOp.MMCompute(groupIdx, mnConfig);
            curBlock += tilingData.gmmBaseParams.usedCoreNum;
        }
        count = curCount % tilingData.gmmBaseParams.usedCoreNum;

        // update cumulative k rows
        mnConfig.cuK += mnConfig.k;
    }
}

/** @brief intenal computation class
 */
template <typename mmType, typename AT, typename CT, typename GT = int64_t, bool trans_a = false, bool sync = false>
class GMMCompute {
public:
    using BT = AT;

    /** @brief constructor */
    __aicore__ inline GMMCompute(mmType &mm_, TPipe &pipe_, const GmmAddTilingData &tilingData_)
        : mm(mm_),
          pipe(pipe_),
          tilingData(tilingData_)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group, GM_ADDR weightGrad, GM_ADDR y,
                                GM_ADDR workspace);

    __aicore__ inline void MMCompute(uint32_t groupIdx, const MNConfig &mnConfig);

    __aicore__ inline int32_t CalcKForThisGroup(int group_id)
    {
        return static_cast<int32_t>(groupGm.GetValue(group_id));
    }

    __aicore__ inline void CopyInMatmulRes()
    {
        // 从GM搬运real_rows * curSingleN到ub，inner轴对齐到curSingleN,右侧补零
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = real_rows;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = curSingleN * sizeof(CT);
        // 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes),
        // 如果源操作数的逻辑位置为GM,则单位为Byte
        copyParams.srcStride = (tilingData.gmmBaseParams.singleN - curSingleN) * sizeof(CT);
        // 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，如果目的操作数的逻辑位置为GM，则单位为Byte
        copyParams.dstStride = 0;

        DataCopyPadExtParams<CT> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 0;
        // 连续搬运数据块右侧需要补充的数据范围，单位为元素个数。leftPadding+rightPadding的字节数之和不能超过32Bytes。
        padParams.rightPadding = curSingleNAligned - curSingleN;

        DataCopyPad(matmulResTensor, workspaceGm[static_cast<uint64_t>(offsetM) * tilingData.gmmBaseParams.singleN],
                    copyParams, padParams);
    }

    __aicore__ inline void CopyInWeightGrad(const MNConfig &mnConfig)
    {
        // 从GM搬运real_rows * curSingleN到ub，inner轴对齐到curSingleN,右侧补零
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = real_rows;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = curSingleN * sizeof(CT);
        // 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes),
        // 如果源操作数的逻辑位置为GM,则单位为Byte
        copyParams.srcStride = (mnConfig.n - curSingleN) * sizeof(CT);
        // 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，如果目的操作数的逻辑位置为GM，则单位为Byte
        copyParams.dstStride = 0;

        DataCopyPadExtParams<CT> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 0;
        // 连续搬运数据块右侧需要补充的数据范围，单位为元素个数。leftPadding+rightPadding的字节数之和不能超过32Bytes。
        padParams.rightPadding = curSingleNAligned - curSingleN;
        DataCopyPad(weightGradTensor,
                    weightGradGm[static_cast<uint64_t>(yOffset) + static_cast<uint64_t>(offsetM) * mnConfig.n + cuN],
                    copyParams, padParams);
    }

    __aicore__ inline void CopyOut(const MNConfig &mnConfig)
    {
        // 从ub搬运real_rows * curSingleN到GM
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = real_rows;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = curSingleN * sizeof(CT);
        // 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes),
        // 如果源操作数的逻辑位置为GM,则单位为Byte
        copyParams.srcStride = 0;
        // 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，如果目的操作数的逻辑位置为GM，则单位为Byte
        copyParams.dstStride = (mnConfig.n - curSingleN) * sizeof(CT);
        // 写回到输出。注意：y可以和weightGradGm使用相同地址
        DataCopyPad(yGm[static_cast<uint64_t>(yOffset) + static_cast<uint64_t>(offsetM) * mnConfig.n + cuN],
                    weightGradTensor, copyParams);
    }

private:
    // matmul operator
    mmType &mm;
    // memory manager
    TPipe &pipe;
    // tiling
    const GmmAddTilingData &tilingData;

    // inputs & workspace & outputs
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<GT> groupGm;
    GlobalTensor<CT> weightGradGm;
    GlobalTensor<CT> yGm;
    GlobalTensor<CT> workspaceGm;

    // ub tmp buffer
    TBuf<> weightGradBuf;
    TBuf<> matmulResBuf;
    TBuf<> weightGradBufFp32;
    TBuf<> matmulResBufFp32;

    // total elements per ADD
    const int NUMS_PER_ADD = 120 * 128;

    // const values
    const int BLOCK_BYTE = 32;
    const int FLOAT_BYTE = 4;
    const int BLOCK_FLOAT_NUM = BLOCK_BYTE / FLOAT_BYTE;

    // lhs and rhs tensor for Add
    LocalTensor<CT> weightGradTensor;
    LocalTensor<CT> matmulResTensor;
    LocalTensor<float> weightGradTensorFp32;
    LocalTensor<float> matmulResTensorFp32;

    // offset for matmul
    uint32_t curSingleM;
    uint32_t curSingleN;
    uint32_t curSingleNAligned;
    uint32_t cuM;
    uint32_t cuN;

    uint64_t xOffset;
    uint64_t wOffset;
    uint64_t yOffset;

    // offset for Add loop
    uint32_t offsetM;
    uint32_t rows_per_loop;
    uint32_t real_rows;
};

template <typename mmType, typename AT, typename CT, typename GT, bool trans_a, bool sync>
__aicore__ inline void GMMCompute<mmType, AT, CT, GT, trans_a, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group,
                                                                           GM_ADDR weightGrad, GM_ADDR y,
                                                                           GM_ADDR workspace)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(weight));
    groupGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(group));
    weightGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(weightGrad));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(y));

    // workspace offset: elements count
    const auto workspaceOffset = GetBlockIdx() * tilingData.gmmBaseParams.singleM * tilingData.gmmBaseParams.singleN;
    workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(workspace) + workspaceOffset);

    pipe.InitBuffer(weightGradBuf, NUMS_PER_ADD * sizeof(CT));
    pipe.InitBuffer(matmulResBuf, NUMS_PER_ADD * sizeof(CT));
    pipe.InitBuffer(weightGradBufFp32, NUMS_PER_ADD * sizeof(float));
    pipe.InitBuffer(matmulResBufFp32, NUMS_PER_ADD * sizeof(float));
}

template <typename mmType, typename AT, typename CT, typename GT, bool trans_a, bool sync>
__aicore__ inline void GMMCompute<mmType, AT, CT, GT, trans_a, sync>::MMCompute(uint32_t groupIdx,
                                                                                const MNConfig &mnConfig)
{
    curSingleN = mnConfig.singleN;
    cuN = mnConfig.nIdx * mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curSingleN = mnConfig.n - cuN;
    }
    curSingleM = mnConfig.singleM;
    cuM = mnConfig.mIdx * curSingleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = mnConfig.m - cuM;
    }
    // calc offset for this basic block
    if (trans_a) {
        xOffset = static_cast<uint64_t>(mnConfig.cuK) * mnConfig.m + cuM;
    } else {
        xOffset = static_cast<uint64_t>(cuM) * mnConfig.totalK + mnConfig.cuK;
    }
    wOffset = static_cast<uint64_t>(mnConfig.cuK) * mnConfig.n + cuN;
    yOffset =
        static_cast<uint64_t>(mnConfig.groupIdx) * mnConfig.m * mnConfig.n + static_cast<uint64_t>(cuM) * mnConfig.n;
    // 注意，这里的gemm结果存放到workspace里，输出C矩阵的N需要设置成singeN，这样的话，向gm里写入tensorc时，每一行跳singeN这么多个元素
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.totalK, mnConfig.totalK, tilingData.gmmBaseParams.singleN);
    // @lg: 这里的single k，是每个专家对应的token数量
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], trans_a);
    mm.SetTensorB(weightGm[wOffset]);
    // 这里使用同步模式，等待结果ready
    // 结果写入到该core对应的workspace位置
    mm.template IterateAll<sync>(workspaceGm[0], 0);

    ////////////////////// 进行add计算： load lhs、load rhs-> add -> 写回gm
    curSingleNAligned = (curSingleN + BLOCK_FLOAT_NUM - 1) / BLOCK_FLOAT_NUM * BLOCK_FLOAT_NUM;
    rows_per_loop = NUMS_PER_ADD / curSingleNAligned;

    const auto loopCount = (curSingleM + rows_per_loop - 1) / rows_per_loop;
    const auto tailRows = curSingleM - (loopCount - 1) * rows_per_loop;

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());

    for (int idx = 0; idx < loopCount; ++idx) {
        if (idx > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        // 执行本次循环前，M方向的offset
        offsetM = idx * rows_per_loop;
        // 本次循环实际有效的rows
        real_rows = (idx == loopCount - 1 ? tailRows : rows_per_loop);
        // 获取对齐到32B的ub tensor
        const auto element_num = real_rows * curSingleNAligned;
        weightGradTensor = weightGradBuf.Get<CT>(element_num);
        matmulResTensor = matmulResBuf.Get<CT>(element_num);
        weightGradTensorFp32 = weightGradBufFp32.Get<float>(element_num);
        matmulResTensorFp32 = matmulResBufFp32.Get<float>(element_num);
        CopyInMatmulRes();
        CopyInWeightGrad(mnConfig);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        // set发送到mte2队列
        // wait发送到vector队列
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(weightGradTensorFp32, weightGradTensor, RoundMode::CAST_NONE, element_num);
        pipe_barrier(PIPE_V);
        Cast(matmulResTensorFp32, matmulResTensor, RoundMode::CAST_NONE, element_num);
        pipe_barrier(PIPE_V);
        // 直接使用二级接口
        Add(weightGradTensorFp32, weightGradTensorFp32, matmulResTensorFp32, element_num);
        pipe_barrier(PIPE_V);
        Cast(weightGradTensor, weightGradTensorFp32, RoundMode::CAST_ROUND, element_num);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 结果写回GM
        CopyOut(mnConfig);
        if (idx < loopCount - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
}

} // namespace ops_gmm_add_c_16

#endif
