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
 * \file all_to_all_matmul_aic.h
 * \brief AIC 侧融合计算——CCU 事件同步原语 + WaitPolicy 逐 tile 等待 + Blaze MXFP8 matmul 编排
 *
 * matmul kernel 为 apace 仓内实现，原位引用不拷入。
 */

#pragma once

#include "ccu_all_to_all.asc"           // CCU 通信服务（同步原语的依赖底座）
#include "ccu_all_to_all_tiling_data.h" // ccuAllToAllMatmulTilingData

// ══════════ §1 AIC 侧同步原语（__aicore__）══════════

// 融合 kernel block 0 调 AllToAllProcess 发起（清扫 → commit），计算核逐轮 WaitCcuCommDone

// commit：写 wake event 通知 CCU 启动通信（block 0 调用）
__aicore__ inline void CommitCcuComm(const CcuMappedAddrs& mappedAddrs)
{
    __gm__ uint64_t* wakeAddr = reinterpret_cast<__gm__ uint64_t*>(mappedAddrs.eventVa[0]);
    AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}

// 帧首 CKE 事件残值清扫 → commit（清零必须在 commit 之前，否则吞 round-done）；block 0 调用
__aicore__ inline void AllToAllProcess(const CcuCommTilingData& comm)
{
    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) != 0) {
        return;
    }

    // events: [0]=wake, [t+1]=data 轮 t done（轮 0 含其搭载的 scale 整块）
    const uint32_t totalEventNum = comm.dataCommTurn + 1;
    for (uint32_t i = 0; i < totalEventNum; i++) {
        __gm__ uint64_t* evAddr = reinterpret_cast<__gm__ uint64_t*>(comm.mappedAddrs.eventVa[i]);
        AscendC::WriteGmByPassDCache<uint64_t>(evAddr, 0);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    CommitCcuComm(comm.mappedAddrs);
}

// 等待 CCU 某轮通信完成：绕 DCache 直读 CKE 寄存器槽位（MMIO uncached 语义）
__aicore__ inline void WaitCcuCommDone(uint64_t eventVa)
{
    __gm__ uint64_t* doneAddr = reinterpret_cast<__gm__ uint64_t*>(eventVa);
    while (AscendC::ReadGmByPassDCache(doneAddr) != 1) {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        AscendC::Nop<800>();
    }
}

// ══════════ §2 WaitPolicy + 融合 matmul 编排 ══════════

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

// CANN>=20261028 bisheng -xasc 的 host pass 不再定义 __CCE_AICORE__，而 apace
// op_state_dump.h 的 Mc2Kernel 命名空间仅在设备 pass 展开；共享 qbmm 内核头的
// Mc2Kernel::OpStateDump 形参/按值成员在 host pass 仍需完整类型可解析。此处提供
// 仅 host pass 生效的空壳（host 代码不触碰其成员，布局无需与设备侧一致）；
// 设备 pass 走 apace 真类，DFX 功能本体不受影响。apace 侧零改动。
#ifndef __CCE_AICORE__
namespace Mc2Kernel {
class OpStateDump {
public:
    OpStateDump() = default;
    void Init(void*, const void*, uint32_t) {}
    void DoDump(uint8_t, uint8_t = 0, uint8_t = 0, uint8_t = 0, uint8_t = 1, uint8_t = 1) {}
};
} // namespace Mc2Kernel
#endif

#include "apace/kernel/matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h"
#include "apace/utils/op_state_dump.h"
#include "ccu_all_to_all_tiling_data.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"

namespace Apace {

// 等待 data 轮 tileIdx done（eventVa[tileIdx+1]；轮 0 已含 tile0 所需的 scale 全量）
struct CcuCommWaitPolicy {
    const CcuMappedAddrs* mappedAddrs_{nullptr};

    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        WaitCcuCommDone(mappedAddrs_->eventVa[tileIdx + 1]);
    }
};

// CCU 直调版 AllToAll + MXFP8 MatMul 融合 impl（Init/Run 去 HCCL 化，通信由
// streamCcu 上的 CCU kernel 独立执行）
template <typename X1Type, typename X2Type, typename YType>
class AllToAllMxQuantMatmulCcuImpl {
public:
    explicit __aicore__ inline AllToAllMxQuantMatmulCcuImpl(QuantMatmulTilingData* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x1Scale, GM_ADDR x2, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR commOut,
                                GM_ADDR commScale, const CcuMappedAddrs& mappedAddrs, uint32_t rankId,
                                uint32_t rankSize, uint32_t tileM);
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::dn_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using BiasType = float;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
                                                    BiasType, LayoutBias>;
    using QuantMatmulKernelImpl =
        Blaze::Gemm::Kernel::AllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler, CcuCommWaitPolicy>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    QuantMatmulTilingData* tilingData_;

    GM_ADDR x1_;
    GM_ADDR x1Scale_;
    GM_ADDR x2_;
    GM_ADDR x2Scale_;
    GM_ADDR y_;
    GM_ADDR commOutGM_;
    GM_ADDR commX1ScaleGM1_;
    CcuMappedAddrs mappedAddrs_;

    uint32_t rankId_{0};
    uint32_t rankSize_{0};
    uint32_t tileM_{0};
    uint64_t originalM_{0};

    Mc2Kernel::OpStateDump opStateDump_{};

    __aicore__ inline void SetupParams(Params& out, MatmulMode matmulMode);
    __aicore__ inline void MatmulProcess(MatmulMode matmulMode);
};

template <typename X1Type, typename X2Type, typename YType>
__aicore__ inline void AllToAllMxQuantMatmulCcuImpl<X1Type, X2Type, YType>::Init(
    GM_ADDR x1, GM_ADDR x1Scale, GM_ADDR x2, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR commOut, GM_ADDR commScale,
    const CcuMappedAddrs& mappedAddrs, uint32_t rankId, uint32_t rankSize, uint32_t tileM)
{
    x1_ = x1;
    x1Scale_ = x1Scale;
    x2_ = x2;
    x2Scale_ = x2Scale;
    y_ = y;
    commOutGM_ = commOut;
    commX1ScaleGM1_ = commScale;
    mappedAddrs_ = mappedAddrs;
    rankId_ = rankId;
    rankSize_ = rankSize;
    tileM_ = tileM;
    originalM_ = static_cast<uint64_t>(tilingData_->m);
    quantMatmulKernelImpl_.GetCommPolicy().mappedAddrs_ = &mappedAddrs_;
    // CCU 版暂不接入 DFX 状态 dump（nullptr Init 为安全空操作）
    opStateDump_.Init(nullptr, nullptr, 0);
}

// 融合 REMOTE 单趟（localMatmul=0：self rank 直读 x1 零拷贝，跨源 L0C 累加）
template <typename X1Type, typename X2Type, typename YType>
__aicore__ inline void AllToAllMxQuantMatmulCcuImpl<X1Type, X2Type, YType>::Run()
{
    MatmulProcess(MatmulMode::REMOTE);
}

template <typename X1Type, typename X2Type, typename YType>
__aicore__ inline void AllToAllMxQuantMatmulCcuImpl<X1Type, X2Type, YType>::SetupParams(Params& out,
                                                                                        MatmulMode matmulMode)
{
    const QuantMatmulTilingData& mmTile = *tilingData_;

    ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = commOutGM_;
    mmadParams.scaleAGmAddr = commX1ScaleGM1_;
    mmadParams.bGmAddr = x2_;
    mmadParams.scaleBGmAddr = x2Scale_;
    mmadParams.cGmAddr = y_;
    mmadParams.biasGmAddr = 0;

    LocalParams localParams{rankId_, rankSize_, originalM_, x1_, x1Scale_, 0UL, rankSize_, matmulMode, tileM_};
    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    BlockSchedulerParams schedulerParams{mmTile.baseM,
                                         mmTile.baseN,
                                         mmTile.mTailTile,
                                         mmTile.nTailTile,
                                         mmTile.mBaseTailSplitCnt,
                                         mmTile.nBaseTailSplitCnt,
                                         mmTile.mTailMain,
                                         mmTile.nTailMain};
    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c, 0U};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template <typename X1Type, typename X2Type, typename YType>
__aicore__ inline void AllToAllMxQuantMatmulCcuImpl<X1Type, X2Type, YType>::MatmulProcess(MatmulMode matmulMode)
{
    Params params;
    SetupParams(params, matmulMode);
    quantMatmulKernelImpl_(params, opStateDump_);
}

} // namespace Apace

// AIC 侧融合编排：workspace 切分 [recvData|recvScale] → AllToAllProcess（block 0 发起）→
// 全核融合 matmul（逐 tile WaitTile 与 CCU 通信 overlap）
__aicore__ inline void AllToAllMatmulProcess(__gm__ void* x1, __gm__ void* x1Scale, __gm__ void* x2,
                                             __gm__ void* x2Scale, __gm__ void* output, __gm__ void* workspace,
                                             ccuAllToAllMatmulTilingData tilingData)
{
    const CcuCommTilingData& comm = tilingData.comm;

    __gm__ uint8_t* wsBytes = reinterpret_cast<__gm__ uint8_t*>(workspace);
    __gm__ void* recvBuf = workspace;
    __gm__ void* recvScaleBuf = reinterpret_cast<__gm__ void*>(wsBytes + comm.workspaceScaleOffset);

    // 通信参数经 GM staging 直达 CCU，AIC 只发令（清扫+commit）
    AllToAllProcess(comm);

    Apace::AllToAllMxQuantMatmulCcuImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> matmulImpl(&tilingData.matmul);
    matmulImpl.Init(reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x1)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x1Scale)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x2)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x2Scale)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(output)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(recvBuf)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(recvScaleBuf)), comm.mappedAddrs, comm.rankId,
                    comm.rankSize, comm.tileM);
    matmulImpl.Run();
}
