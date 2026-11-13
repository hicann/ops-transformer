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
 * \file matmul_all_to_all_hcomm_impl.h
 * \brief 非量化（FP16/BF16）MatmulAlltoAll Apace 融合编排层。
 *
 * 职责分工：本层解释 tiling data、按 tile/tail 双批准备通信上下文，并逐片推进
 * "计算 -> AIC 核间屏障 -> HcclCommunicator 发 AllToAll -> 地址步进"；
 * 计算内核（MatmulAllToAllMmKernel）只负责单片计算。mmad 结果直写 rank-major
 * 发送缓冲，APT 路径的 AIV 转置段在本路径整体消失。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "../../matmul/batch_matmul/matmul_all_to_all_mm_kernel.h"
#include "../../../../mc2_templates/communication/communicator/hccl_communicator.h"
#include "../../../../mc2_templates/communication/primitives/hccl_primitives.h"
#include "matmul_all_to_all_tiling_data.h"

namespace Apace {

/**
 * @brief 非量化 MatmulAlltoAll 的 Apace 融合编排层（先计算后通信）
 *
 * @tparam X1Type         x1 元素类型（fp16/bf16）
 * @tparam X2Type          x2 元素类型（非量化约束 x1/x2/y 同 dtype）
 * @tparam YType           y 输出元素类型
 * @tparam BiasType        bias 元素类型（与 x 同 dtype 或 fp32）
 * @tparam TilingDataType  tiling data 结构（hcommNonQuantMatmulAllToAllTilingData）
 * @tparam ServerType      HCCL 服务器类型（承载 CCU/AICPU 通信引擎）
 * @tparam TransB          x2 是否转置布局（true: (N,K) 存储 DNExt；false: (K,N) 存储 NDExt）
 */
template <typename X1Type, typename X2Type, typename YType, typename BiasType, typename TilingDataType,
          AscendC::HcclServerType ServerType, bool TransB>
class MatmulAllToAllHcommImpl {
public:
    /**
     * @brief 构造：缓存 tiling data 指针并初始化通信器
     *        （HcclCommunicator 同持该指针，从中读 mc2InitTiling/mc2CcTiling 通信配置）
     */
    explicit __aicore__ inline MatmulAllToAllHcommImpl(TilingDataType* tilingData)
        : tilingData_(tilingData),
          communication_(tilingData)
    {}

    /**
     * @brief 缓存 GM 地址并初始化通信器
     *
     * @param x1          本 rank 的 A 输入 [rankM, rankK]（ND 行主序）
     * @param x2           全宽 B 输入 [rankK, fullN]（转置时 [fullN, rankK]）
     * @param bias         全宽 bias [fullN]（可为 nullptr，mmad 内部判空跳过）
     * @param y            输出（AllToAll 的接收端）
     * @param workspaceGM  rank-major 发送缓冲区
     */
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR workspaceGM)
    {
        x1_ = x1;
        x2_ = x2;
        bias_ = bias;
        y_ = y;
        workspaceGM_ = workspaceGM;
        communication_.Init();
    }

    /**
     * @brief 顶层执行：tile/tail 双批推进 + 通信收尾
     *
     * 分两批的原因：host 侧通算切分产出"主块批（tileCnt 片 x tileM 行）+ 尾块批
     * （tailCnt 片 x tailM 行）"，两批的 mm 块形不同（尾块单独出 tiling），需分别
     * ProcessBatch；communication_.End() 等全部通信片完成后收尾。
     */
    __aicore__ inline void Run()
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        if (tilingInfo.tileCnt > 0) {
            ProcessBatch(0, tilingInfo.tileM, tilingInfo.tileCnt, tilingData_->tileMmTilingData);
        }
        if (tilingInfo.tailCnt > 0) {
            const uint64_t mainRows = static_cast<uint64_t>(tilingInfo.tileM) * tilingInfo.tileCnt;
            ProcessBatch(mainRows, tilingInfo.tailM, tilingInfo.tailCnt, tilingData_->tailMmTilingData);
        }
        communication_.End();
    }

private:
    // 帧布局：A/C/bias 恒 NDExt 行主序；B 按逻辑转置选 DNExt（(N,K) 存储）/NDExt（(K,N) 存储）
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    // Blaze 非量化组件：多块调度策略 / mmad 块（bias 走 fixpipe 加入）/ ND 调度块
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;
    // {M: 本rank单片行数(通算切分后的tileM/tailM, 非全量), N: 单rank列宽(=完整N/rankDim, 非全量), K: 完整K, batch: 恒1}
    using ProblemShapeT = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShapeT>;
    using KernelImpl = Blaze::Gemm::Kernel::MatmulAllToAllMmKernel<ProblemShapeT, BlockMmad, BlockScheduler>;
    // 通信器：ServerType 承载 CCU/AICPU 通信引擎维度
    using CommunicationType =
        MC2KernelTemplate::HcclCommunicator<true, ServerType, MC2KernelTemplate::MC2AlltoAllContext, TilingDataType,
                                            MC2KernelTemplate::MC2AlltoAllPrimitives, 1, 0>;

    // AIC 核间屏障 flag id
    static constexpr uint16_t AIC_BARRIER_FLAG_ID = 8U;

    // 等 fixpipe 异步搬出（L0C->GM）全部落盘后再发通信，否则会发出半成品
    __aicore__ inline void SyncAicCores()
    {
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG_ID);
        AscendC::CrossCoreWaitFlag<0, PIPE_S>(AIC_BARRIER_FLAG_ID);
    }

    /**
     * @brief 组装计算内核 Params（GM 地址 + 块形 + 切片步长）
     *
     * @param rowOffset 本批起始行（主块批=0，尾块批=tileM*tileCnt）
     * @param taskM     本批片行数（tileM 或 tailM）
     * @param taskCnt   本批片数（tileCnt 或 tailCnt）
     * @param mmTile    本批的 Blaze mm tiling（host 侧 SWAT 产出）
     */
    __aicore__ inline void SetupParams(typename KernelImpl::Params& params, uint64_t rowOffset, uint32_t taskM,
                                       uint32_t taskCnt, const MatmulTilingData& mmTile)
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNPerPeer =
            static_cast<uint64_t>(tilingInfo.rankN) / static_cast<uint64_t>(tilingInfo.rankDim); // 单 rank 列宽

        params.problemShape = ProblemShapeT{static_cast<int64_t>(mmTile.m), static_cast<int64_t>(mmTile.n),
                                            static_cast<int64_t>(mmTile.k), 1};

        auto& mmParams = params.mmParams;
        mmParams.aGmAddr = x1_ + rowOffset * tilingInfo.rankK * sizeof(X1Type);
        mmParams.bGmAddr = x2_;
        mmParams.cGmAddr = workspaceGM_ + rowOffset * rankNPerPeer * sizeof(YType);
        mmParams.biasGmAddr = bias_; // 可空，mmad 内部判空跳过
        mmParams.mL1 = mmTile.mL1;
        mmParams.nL1 = mmTile.nL1;
        mmParams.kL1 = mmTile.kL1;
        mmParams.mL0 = mmTile.baseM; // L0 块形对应 SWAT 的 baseM/baseN/baseK
        mmParams.nL0 = mmTile.baseN;
        mmParams.kL0 = mmTile.baseK;
        mmParams.l1Stages = mmTile.l1BufferNum;
        // SWAT 块形恰好填满 L0C，l0cDB > 1 才能开双缓冲，否则 L0C 越界
        mmParams.l0cStages = mmTile.l0cDB > 1 ? 2 : 1;
        mmParams.splitKNum = 1; // K 整段单次调用完成

        auto& schParams = params.schedulerParams;
        schParams.mL1 = mmTile.mL1;
        schParams.nL1 = mmTile.nL1;
        schParams.kL1 = mmTile.kL1;
        schParams.baseM = mmTile.baseM;
        schParams.baseN = mmTile.baseN;
        schParams.baseK = mmTile.baseK;
        schParams.mTailCnt = mmTile.mTailCnt;
        schParams.nTailCnt = mmTile.nTailCnt;
        schParams.mBaseTailSplitCnt = mmTile.mBaseTailSplitCnt;
        schParams.nBaseTailSplitCnt = mmTile.nBaseTailSplitCnt;
        // 尾区负载均衡：尾区拆多块时前 N-1 块各取 mTailMain 行、最后一块取剩余，
        // 避免整块尾区独占单核拖慢；SWAT 搜索最优值
        schParams.mTailMain = mmTile.mTailMain;
        schParams.nTailMain = mmTile.nTailMain;

        // rank 区步长（元素数）= rankM*rankNPerPeer；片步长：x1 下一片行首 / 发送缓冲下一片行首（字节）
        params.rankStride = static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        params.rankDim = tilingInfo.rankDim;
        params.sliceParams.aByteStride = static_cast<uint64_t>(taskM) * tilingInfo.rankK * sizeof(X1Type);
        params.sliceParams.cByteStride = static_cast<uint64_t>(taskM) * rankNPerPeer * sizeof(YType);
        params.sliceParams.sliceCount = taskCnt;
    }

    /**
     * @brief 单批通算流水：组装参数后逐片"计算 -> 核间屏障 -> 发通信 -> 地址步进"
     *
     * 逐片循环持有在本层：InitGroup 每批一次；每片 ProcessSlice 算出 [taskM, rankN]
     * 后，SyncAicCores 等全部核把该片写进 GM，communication_.Process 发走该片，地址步进到下一片。
     */
    __aicore__ inline void MatmulProcess(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                         const MatmulTilingData& mmTile)
    {
        typename KernelImpl::Params params;
        SetupParams(params, startRow, taskM, taskCnt, mmTile);
        kernel_.InitGroup(params);
        for (uint32_t sliceIdx = 0; sliceIdx < params.sliceParams.sliceCount; ++sliceIdx) {
            kernel_.ProcessSlice(params);
            SyncAicCores();
            communication_.Process(sliceIdx);
            if (sliceIdx + 1 < params.sliceParams.sliceCount) {
                params.mmParams.aGmAddr += params.sliceParams.aByteStride;
                params.mmParams.cGmAddr += params.sliceParams.cByteStride;
            }
        }
    }

    /**
     * @brief 单批执行：准备通信上下文 -> 预注册全部片 -> 计算逐片触发通信
     *
     * 通信上下文字段语义（MC2AlltoAllContext）：
     *   sendBuffer：本批发送首地址（rank-major 发送缓冲的本批起始行）
     *   recvBuffer：接收首地址（y 的本批起始行）
     *   sendOffset：片步长（taskM*rankNPerPeer 元素），Process(sliceIdx) 按它寻第 idx 片
     *   sendCount：单片发送量（taskM*rankNPerPeer 元素）
     *   strideCount：单 rank 区大小（rankM*rankNPerPeer），AllToAll 按 rank 区交换的步长
     *   hcclDataType：HCCL 数据类型枚举
     * PrepareAll(taskCnt) 一次注册本批全部片，后续 Process(idx) 逐片发出。
     */
    __aicore__ inline void ProcessBatch(uint64_t startRow, uint32_t taskM, uint32_t taskCnt,
                                        const MatmulTilingData& mmTile)
    {
        const auto& tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNPerPeer =
            static_cast<uint64_t>(tilingInfo.rankN) / static_cast<uint64_t>(tilingInfo.rankDim);
        auto* communicationContext = communication_.GetContextPtr();
        communicationContext->sendBuffer = workspaceGM_ + startRow * rankNPerPeer * sizeof(YType);
        communicationContext->recvBuffer = y_ + startRow * rankNPerPeer * sizeof(YType);
        communicationContext->sendOffset = static_cast<uint64_t>(taskM) * rankNPerPeer * sizeof(YType);
        communicationContext->recvOffset = communicationContext->sendOffset;
        communicationContext->sendCount = static_cast<uint64_t>(taskM) * rankNPerPeer;
        communicationContext->strideCount = static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        communicationContext->hcclDataType = tilingInfo.hcclDataType;

        communication_.PrepareAll(taskCnt);
        MatmulProcess(startRow, taskM, taskCnt, mmTile);
    }

    TilingDataType* tilingData_;      // host 下发的 tiling data
    KernelImpl kernel_{};             // 计算内核（无状态，参数每次传入）
    CommunicationType communication_; // HCCL 通信器
    GM_ADDR x1_{nullptr};             // A 输入 [rankM, rankK]
    GM_ADDR x2_{nullptr};             // 全宽 B
    GM_ADDR bias_{nullptr};           // 全宽 bias（可空）
    GM_ADDR y_{nullptr};              // 输出（AllToAll 接收端）
    GM_ADDR workspaceGM_{nullptr};    // rank-major 发送缓冲
};

} // namespace Apace
