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
 * \file block_epliogue_fag_pre.h
 * \brief Block Epliogue Fag Pre Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
namespace NpuArch::Epilogue::Block {

template <class OutputType_, class UpdateType_, class InputType_>
class BlockEpilogue<EpilogueAtlasA2FAGPre, OutputType_, UpdateType_, InputType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        GM_ADDR dqWrk;
        GM_ADDR dkWrk;
        GM_ADDR dvWrk;
        GM_ADDR tilingData;

        // Methods
        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR dqWrk_, GM_ADDR dkWrk_, GM_ADDR dvWrk_, GM_ADDR tilingData_)
            : dqWrk(dqWrk_),
              dkWrk(dkWrk_),
              dvWrk(dvWrk_),
              tilingData(tilingData_)
        {}
    };

    NpuArch::Arch::Resource<ArchTag> resource;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    LocalTensor<float> zeroTensor;
    uint64_t cBlockIdx;
    uint64_t BLOCK_BYTE = 32;

    uint64_t qPreBlockFactor = 0;
    uint64_t qPreBlockTotal = 0;
    uint64_t qPreTailNumTmp = 0;
    uint64_t qPreTailNum = 0;
    uint64_t qSizeAlign = 0;
    uint64_t initdqSize = 0;
    uint64_t dqOffset = 0;

    uint64_t kvPreBlockFactor = 0;
    uint64_t kvPreBlockTotal = 0;
    uint64_t kvPreTailNumTmp = 0;
    uint64_t kvPreTailNum = 0;
    uint64_t kvSizeAlign = 0;
    uint64_t initdkSize = 0;
    uint64_t dkvOffset = 0;

    uint64_t usedCoreNum = 0;
    uint64_t ubsize = 0;

    __aicore__ inline BlockEpilogue(Params const &params)
    {
        cBlockIdx = GetBlockIdx();
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedCoreNum = tilingData->usedVecCoreNum; // 先按这个把，得适配

        qSizeAlign = tilingData->dqSize;
        kvSizeAlign = tilingData->dkvSize;
        ubsize = tilingData->ubSize / BLOCK_BYTE * BLOCK_BYTE; // 32字节对齐

        qPreBlockFactor = (qSizeAlign + usedCoreNum - 1) / usedCoreNum; // 每个vec 处理的元素数量 向上取整
        qPreBlockTotal =
            (qSizeAlign + qPreBlockFactor - 1) / qPreBlockFactor; // 一共需要处理次数 向上取整， 也理解需要的核数
        qPreTailNumTmp = qSizeAlign % qPreBlockFactor; // remain量， 尾核数量
        qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

        kvPreBlockFactor = (kvSizeAlign + usedCoreNum - 1) / usedCoreNum;
        kvPreBlockTotal = (kvSizeAlign + kvPreBlockFactor - 1) / kvPreBlockFactor;
        kvPreTailNumTmp = kvSizeAlign % kvPreBlockFactor;
        kvPreTailNum = kvPreTailNumTmp == 0 ? kvPreBlockFactor : kvPreTailNumTmp;

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dqWrk);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dkWrk);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dvWrk);

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreTailNum : qPreBlockFactor;
        dqOffset = ((uint64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreTailNum : kvPreBlockFactor;
        dkvOffset = ((uint64_t)cBlockIdx) * kvPreBlockFactor;

        zeroTensor = resource.ubBuf.template GetBufferByByte<float>(0);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()();

    template <>
    __aicore__ inline void operator()<AscendC::AIC>()
    {}

    template <>
    __aicore__ inline void operator()<AscendC::AIV>()
    {
        if (cBlockIdx >= usedCoreNum) {
            return;
        }

        uint64_t maxDataCount = (ubsize) / sizeof(float);
        Duplicate(zeroTensor, (float)0.0, maxDataCount);

        // dq
        uint64_t cOutElement = initdqSize; // currenr out ele
        uint64_t totalLoop = cOutElement / maxDataCount;
        uint64_t remainOutNum = cOutElement % maxDataCount;

        auto eventID1 = EVENT_ID2;
        set_flag(PIPE_V, PIPE_MTE3, eventID1);
        wait_flag(PIPE_V, PIPE_MTE3, eventID1);

        if (cBlockIdx < qPreBlockTotal) {
            processZero(dqWorkSpaceGm[dqOffset], initdqSize, maxDataCount);
        }

        if (cBlockIdx < kvPreBlockTotal) {
            processZero(dkWorkSpaceGm[dkvOffset], initdkSize, maxDataCount);
            processZero(dvWorkSpaceGm[dkvOffset], initdkSize, maxDataCount);
        }
    }

    __aicore__ inline void processZero(GlobalTensor<float> outGm, uint64_t outCount, uint64_t maxDataCount)
    {
        uint64_t cOutElement = outCount; // currenr out ele
        uint64_t totalLoop = cOutElement / maxDataCount;
        uint64_t remainOutNum = cOutElement % maxDataCount;

        for (uint32_t i = 0; i < totalLoop; i++) {
            DataCopy(outGm[i * maxDataCount], zeroTensor, maxDataCount);
        }

        if (remainOutNum > 0) {
            if (remainOutNum * sizeof(float) % BLOCK_BYTE == 0) {
                DataCopy(outGm[totalLoop * maxDataCount], zeroTensor, remainOutNum);
            } else {
                AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainOutNum * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(outGm[totalLoop * maxDataCount], zeroTensor, copyParams);
            }
        }
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
