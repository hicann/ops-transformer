/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_quant_bmm_perblock.h
 * \brief
 */
#ifndef ALL_GATHER_QUANT_BMM_PERBLOCK_H
#define ALL_GATHER_QUANT_BMM_PERBLOCK_H

#include "all_gather_matmul_base_v2.h"
#include "../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h"
#include "../../../common/op_kernel/qbmm_mix_perblock_noncontiguous.h"

/**
 * 1、依赖tiling结构QuantBatchMatmulV3TilingData
 * 2、依赖新matmul接口
 * 3、低精度类型定义 (json)
 * 4、AllgatherMatmulTilingData结构体, tilingdata + taildata (QuantBatchMatmulV3TilingData)
 */
namespace AllGatherMatmulImpl {
using namespace AscendC;

template <typename AType, typename BType, typename CType, class MmType, Mc2CoreType CoreType,
          uint8_t ServerMode = HCCL_COMM_MODE_CCU>
class AllGatherQuantBmmPerBlock : public AllGatherMatmulBase<AType, CType, CoreType, ServerMode> {
public:
    __aicore__ inline AllGatherQuantBmmPerBlock(MC2GmAddrs *addrs, QuantGmAddrs *quantAddrs,
                                                Mc2Tiling::AllGatherMatmulTilingDataFp8 *tilingData, GM_ADDR contextGM,
                                                __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling, TPipe *tPipe)
        : AllGatherMatmulBase<AType, CType, CoreType, ServerMode>(addrs, quantAddrs, tilingData, contextGM, tPipe)
    {
        tilingData_ = tilingData;
        this->localInfo_.mmTiling = &tilingData_->quantBmmv3LocalTiling;
        this->tileInfo_.mmTiling = &tilingData_->quantBmmv3TileTiling;
        this->tailInfo_.mmTiling = &tilingData_->quantBmmv3TailTiling;
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void QuantMatmulCompute();      /* 计算入口 */
    __aicore__ inline void QuantMatmulLocalCompute(); /* 本地块计算 */
    __aicore__ inline void QuantMatmulGatherCompute(MmType &mmOp, uint32_t count,
                                                    const Mc2Tiling::MC2TileInfo &tileInfo, bool isTail); /* 远端计算 */
private:
    Mc2Tiling::AllGatherMatmulTilingDataFp8 *tilingData_;
};

/**
 * @brief allgather quant bmm perblock 主流程：通信+计算整块+计算尾块+后处理
 * @return void
 */
template <typename AType, typename BType, typename CType, class MmType, Mc2CoreType CoreType, uint8_t ServerMode>
__aicore__ inline void AllGatherQuantBmmPerBlock<AType, BType, CType, MmType, CoreType, ServerMode>::Process()
{
    // 开启通信
    this->HcclStart();
    // 计算 + 远端 Gather 数据计算
    QuantMatmulCompute();
    // 后处理：AIC全核同步+终止hcclserver
    this->HcclFinalize();
}

/**
 * @brief allgather quant bmm 计算流程
 * @return void
 */
template <typename AType, typename BType, typename CType, class MmType, Mc2CoreType CoreType, uint8_t ServerMode>
__aicore__ inline void
AllGatherQuantBmmPerBlock<AType, BType, CType, MmType, CoreType, ServerMode>::QuantMatmulCompute()
{
    // 计算本片的块
    // 当N=0时，跳过本地计算
    if (this->paramInTiling_->rankN != 0) {
        QuantMatmulLocalCompute();
    }

    // 只做本卡计算时，aGM_从本地获取；计算远端卡数据时，aGM_为gather来的数据
    this->addrs_->aGM = this->gatherX1Addr_;

    this->gatherScaleAddr_ = this->gatherScale1Addr_;

    // 先完成scale1通信的wait工作
    this->HcclWaitScale();
    // 仅用AIC核完成通信，添加全核同步确保AIV核可以拿到scale1数据后再参与运算
    SyncAll<false>();

    // 计算其他片的整块
    MmType opTile;
    QuantMatmulGatherCompute(opTile, this->paramInTiling_->tileCnt, this->tileInfo_, false);

    // 计算尾块
    if (this->paramInTiling_->tailCnt > 0) {
        MmType opTail;
        QuantMatmulGatherCompute(opTail, this->paramInTiling_->tailCnt, this->tailInfo_, true);
    }
}

/**
 * @brief 本地块计算流程
 * @return void
 */
template <typename AType, typename BType, typename CType, class MmType, Mc2CoreType CoreType, uint8_t ServerMode>
__aicore__ inline void
AllGatherQuantBmmPerBlock<AType, BType, CType, MmType, CoreType, ServerMode>::QuantMatmulLocalCompute()
{
    // 本卡位置与rankId一致，C矩阵本卡矩阵首地址为rankid*cSize
    GM_ADDR cGM =
        this->addrs_->cGM + static_cast<uint64_t>(this->rankId_) * static_cast<uint64_t>(this->paramInTiling_->rankM) *
                                static_cast<uint64_t>(this->localInfo_.mmTiling->matmulTiling.N) * sizeof(CType);
    this->tPipe_->Destroy();
    this->tPipe_->Init();
    MmType op;
    // Local compute do not need 'strideCount', set to 0.
    uint32_t strideCount = 0;
    op.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->scale1GM,
            this->quantAddrs_->scale2GM, cGM, this->addrs_->workspaceGM, this->localInfo_.mmTiling, this->tPipe_,
            this->batchWeight_, strideCount, false);
    op.Process();
}

/**
 * @brief 其他片计算流程
 * @param count: 切分块数，isTail为true时，应传入tail尾块数，isTail为false时，应传入tile整块数
 * @param qMmv3Tiling: 核切分信息，isTail为true时，应传入tail切分信息，isTail为false时，应传入tile切分信息
 * @param isTail: 当前块是否为尾块
 * @return void
 */
template <typename AType, typename BType, typename CType, class MmType, Mc2CoreType CoreType, uint8_t ServerMode>
__aicore__ inline void
AllGatherQuantBmmPerBlock<AType, BType, CType, MmType, CoreType, ServerMode>::QuantMatmulGatherCompute(
    MmType &mmOp, uint32_t count, const Mc2Tiling::MC2TileInfo &tileInfo, bool isTail)
{
    uint32_t strideCount = this->paramInTiling_->rankM;
    // 当N=0时，跳过计算，只wait通信
    if (this->paramInTiling_->rankN == 0) {
        for (uint32_t i = 0; i < count; i++) {
            this->HcclWait(i, isTail);
        }
        return;
    }
    // 按照切分的片数进行逐片处理
    for (uint32_t i = 0; i < count; i++) {
        // 开始计算前确认是否搬运完成，debug模式只做本片内计算，则不需要判断搬运状态
        this->HcclWait(i, isTail);
        this->tPipe_->Destroy();
        this->tPipe_->Init();
        mmOp.Init(this->addrs_->aGM, this->addrs_->bGM, this->addrs_->biasGM, this->quantAddrs_->scale1GM,
                  this->gatherScaleAddr_, this->addrs_->cGM, this->addrs_->workspaceGM, tileInfo.mmTiling, this->tPipe_,
                  this->batchWeight_, strideCount, true);
        // batchmatmul计算
        mmOp.Process();
        SyncAll<false>();
        // 刷新下一个tile片的A、C矩阵地址
        // 到计算尾块时，因为已经加上了所有整块的地址大小，尾块不需要重新计算起始地址
        this->addrs_->aGM += tileInfo.aAddrOffset;
        this->addrs_->cGM += tileInfo.cAddrOffset;
        this->gatherScaleAddr_ += this->oneCommBlockSizeOffset_;
    }
}

#define INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_PERBLOCK_OP_IMPL(templateClass, CoreType, hcclCommType, isTransA, \
                                                             isTransB, ...) \
    do { \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataFp8); \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataFp8 *)tilingGM; \
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling)); \
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling)); \
        GET_TILING_DATA(tilingData, tilingGM); \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, cGM, gatherOut, workspaceGM}; \
        QuantGmAddrs quantAddrs = {scaleInv2, scaleInv1, scale}; \
        using opType = templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND, CubeFormat::ND, \
                                     CubeFormat::ND, isTransA, isTransB>; \
        AllGatherQuantBmmPerBlock<DTYPE_X1, DTYPE_X2, DTYPE_Y, opType, CoreType, hcclCommType> op( \
            &addrs, &quantAddrs, &tilingData, (__gm__ uint8_t *)context, mc2InitTiling, mc2CcTiling, &pipe); \
        op.Init(mc2InitTiling, mc2CcTiling); \
        op.Process(); \
    } while (0)
} // namespace AllGatherMatmulImpl

#endif // ALL_GATHER_QUANT_BMM_PERBLOCK_H
