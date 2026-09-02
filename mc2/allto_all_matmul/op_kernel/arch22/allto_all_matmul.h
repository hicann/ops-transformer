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
 * \file allto_all_matmul.h
 * \brief
 */

#ifndef ALL_TO_ALL_MATMUL_H
#define ALL_TO_ALL_MATMUL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "allto_all_matmul_tiling.h"
#include "allto_all_matmul_util.h"
#include "allto_all_matmul_block_epilogue_dequant.hpp"
#include "allto_all_matmul_tile_broadcast_add.hpp"

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tla_epilogue_dispatch_policy.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_mul.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_broadcast_one_blk.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_swizzle.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_tile_copy.hpp"

#include "allto_all_matmul.hpp"

using namespace AscendC;
using namespace Catlass;

namespace Mc2Kernel {

// A2AMM : AlltoAllMatmul
#define TemplateA2AMMClass \
    typename AType, typename BType, typename BiasType, typename PerTokenScaleType, typename ScaleType, typename CType, \
        typename AllToAllResultType, bool hasBias, bool transB, int32_t QuantType
#define TemplateA2AMMFunc \
    AType, BType, BiasType, PerTokenScaleType, ScaleType, CType, AllToAllResultType, hasBias, transB, QuantType

using namespace AscendC;
template <TemplateA2AMMClass>
class AlltoAllMatmul : public CommBase {
public:
    __aicore__ inline AlltoAllMatmul(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR scaleGM,
                                GM_ADDR cGM, GM_ADDR allToAllResult, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void AlltoAll();
    __aicore__ inline void Dequant();
    __aicore__ inline void Quant(uint64_t flagIdx, int32_t commIdx);
    __aicore__ inline void QuantPerToken(LocalTensor<float> copyTensor, LocalTensor<float> smoothScaleTensor,
                                         LocalTensor<float> absTensor, LocalTensor<float> reduceMaxTensor,
                                         LocalTensor<float> quantScaleTensor, int32_t actualMoveSize,
                                         int32_t actualMoveToken, int32_t tokenPerMove, int32_t moveIdx,
                                         event_t eventId);
    __aicore__ inline void QuantToken(__gm__ AType *dataSrc, int64_t dataOffset, int32_t coreTokenOffset,
                                      int32_t dataLen, int32_t commIdx);
    __aicore__ inline void QuantTokenSegment(__gm__ AType *dataSrc, int64_t dataOffset, int32_t coreTokenOffset,
                                             int32_t dataLen, int32_t commIdx);
    __aicore__ inline void SmoothQuantProc(event_t eventId, int32_t dataSegmentOffset, int32_t smoothScaleCastOffset,
                                           int32_t actualMoveSize, LocalTensor<float> copyTensor,
                                           LocalTensor<float> smoothScaleTensor);
    __aicore__ inline void CalcTokenMaxValue(LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1,
                                             LocalTensor<float> absTensor0, LocalTensor<float> absTensor1,
                                             LocalTensor<float> smoothScaleTensor0,
                                             LocalTensor<float> smoothScaleTensor1, int32_t castOffset,
                                             __gm__ AType *dataSrc, int64_t dataTokenOffset,
                                             int32_t smoothScaleCastOffset, int32_t sizeScale,
                                             LocalTensor<float> reduceMaxTensor);
    __aicore__ inline void QuantPerSegment(LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1,
                                           LocalTensor<float> absTensor0, LocalTensor<float> absTensor1,
                                           LocalTensor<float> smoothScaleTensor0, LocalTensor<float> smoothScaleTensor1,
                                           int32_t castOffset, __gm__ AType *dataSrc, int64_t dataTokenOffset,
                                           int32_t smoothScaleCastOffset, int32_t sizeScale, float quantScaleReciproal);

private:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR scaleGM_;
    GM_ADDR x1ScaleGM_;
    GM_ADDR allToAllResultGM_;
    GM_ADDR workspaceGM_;

    __gm__ AType *gmPeerMem_;
    __gm__ int8_t *quantAGM_;
    __gm__ int32_t *dequantCGM_;
    GM_ADDR quantScaleGM_;

    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
};

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR x1ScaleGM, GM_ADDR scaleGM, GM_ADDR cGM,
                                                               GM_ADDR allToAllResultGM, GM_ADDR workspaceGM,
                                                               GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    auto tiling = (__gm__ AlltoAllMatmulTilingData *)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclCombineOpParam *)contextGM;
    rank = winContext_->rankId;
    rankSize = tilingData.allToAllMatmulInfo.rankSize;

    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    x1ScaleGM_ = x1ScaleGM;
    scaleGM_ = scaleGM;
    allToAllResultGM_ = allToAllResultGM;
    workspaceGM_ = GetUserWorkspace(workspaceGM);

    CommBase::SetArgs<AType>(rank, rankSize, tilingData);
    this->ub_offset = Catlass::BytesToBits(UB_OFFSET) / Catlass::SizeOfBits<int8_t>::value;

    if constexpr (QuantType == MC2_DYNAMIC_QUANT) {
        quantAGM_ = reinterpret_cast<__gm__ int8_t *>(workspaceGM_);
        dequantCGM_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_ + quantSize);
        quantScaleGM_ = reinterpret_cast<GM_ADDR>(workspaceGM_ + quantSize + dequantSize);
    }
    if constexpr (QuantType == MC2_STATIC_QUANT) {
        x1ScaleGM_ += rank * (m / rankSize) * sizeof(PerTokenScaleType); // 全量化场景，对x1Scale进行偏移
        dequantCGM_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM_);
    }
    kBytes = Catlass::BitsToBytes(k * Catlass::SizeOfBits<AType>::value);
    tokenBytes = Catlass::BitsToBytes(tokenSize * Catlass::SizeOfBits<AType>::value);

    AlltoAllMatmul<TemplateA2AMMFunc>::AICInit();
    AlltoAllMatmul<TemplateA2AMMFunc>::AIVInit();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gmPeerMem_ = reinterpret_cast<__gm__ AType *>(buff[rank]);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNorm();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);
    }
}

// A16W4的tiling
template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        using ArchTag = Arch::AtlasA2;

        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = false;
        constexpr bool aicCalBias =
            (QuantType == MC2_NON_QUANT) && hasBias; // 计算量化后的矩阵乘，bias不由CatlassMatmul负责

        using ElementA = BType; // 非量化场景、量化场景，A、B的入参类型一致；伪量化场景，A需要动态量化成BType
        using ElementB = BType;
        using ElementC = std::conditional_t<QuantType != MC2_NON_QUANT, int32_t,
                                            CType>; // 非量化场景，Btype和CType一致；量化场景计算结果为int32_t
        using ElementBias = BiasType;
        using LayoutA = layout::RowMajor;
        // B转置
        using LayoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
        using LayoutC = layout::RowMajor;
        using LayoutBias = layout::VectorLayout;

        uint32_t realM = m / rankSize;
        uint32_t realK = k * rankSize;
        LayoutA layoutA{static_cast<uint32_t>(realM), static_cast<uint32_t>(realK)};
        LayoutB layoutB{static_cast<uint32_t>(realK), static_cast<uint32_t>(n)};
        LayoutC layoutC{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};
        LayoutBias layoutBias{static_cast<uint32_t>(n)};

        using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                                  Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;

        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using BType_ = Gemm::GemmType<ElementB, LayoutB>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;

        struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
            using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
            using ElementA = typename Base::ElementA;
            using ElementB = typename Base::ElementB;
            using ElementAccumulator = typename Base::ElementAccumulator;

            using CopyGmToL1A = typename Base::CopyGmToL1A;
            using CopyGmToL1B = typename Base::CopyGmToL1B;

            using CopyL1ToL0A = typename Base::CopyL1ToL0A;
            using CopyL1ToL0B = typename Base::CopyL1ToL0B;

            using CopyL0CToGm = typename Base::CopyL0CToGm;
            using BiasTypeSelector = typename Base::BiasTypeSelector;
            using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
            using CopyL1ToBT = typename Base::CopyL1ToBT;
        };

        using TileCopy = TileCopyOpt;
        GemmCoord processSize{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};
        using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        GM_ADDR srcGM = (QuantType == MC2_DYNAMIC_QUANT) ?
                            reinterpret_cast<GM_ADDR>(quantAGM_) :
                            reinterpret_cast<GM_ADDR>(gmPeerMem_); // 动态量化时，需要更改左矩阵读取位置
        GM_ADDR matmulResultGM = (QuantType == MC2_NON_QUANT) ?
                                     cGM_ :
                                     reinterpret_cast<GM_ADDR>(dequantCGM_); // 量化矩阵乘法时，需要修改c矩阵存放地址
        constexpr uint32_t L1TileShapeK =
            std::is_same<BType, int4b_t>::value ? 1024 :
            std::is_same<BType, int8_t>::value  ? 512 :
                                                  256; // 不同的matmul数据类型对应的L1TileShape不同
        constexpr uint32_t L0TileShapeK =
            std::is_same<BType, int4b_t>::value ? 256 :
            std::is_same<BType, int8_t>::value  ? 128 :
                                                  64; // 不同的matmul数据类型对应的L0TileShape不同

        if (m0 == 128) {
            using L1TileShape = GemmShape<128, 256, L1TileShapeK>;
            using L0TileShape = GemmShape<128, 256, L0TileShapeK>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                        CType_, BiasType_, TileCopy>;
            using MatmulKernel =
                Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(srcGM),
                                                 layoutA,
                                                 reinterpret_cast<GM_ADDR>(bGM_),
                                                 layoutB,
                                                 reinterpret_cast<GM_ADDR>(biasGM_),
                                                 reinterpret_cast<GM_ADDR>(matmulResultGM),
                                                 layoutC,
                                                 pValue,
                                                 3,
                                                 0,
                                                 static_cast<int32_t>(rankSize),
                                                 MAX_BLOCK_COUNT};
            matmul_op(params);
        } else {
            using L1TileShape = GemmShape<256, 128, L1TileShapeK>;
            using L0TileShape = GemmShape<256, 128, L0TileShapeK>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                        CType_, BiasType_, TileCopy>;
            using MatmulKernel =
                Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
            MatmulKernel matmul_op;
            typename MatmulKernel::Params params{processSize,
                                                 reinterpret_cast<GM_ADDR>(srcGM),
                                                 layoutA,
                                                 reinterpret_cast<GM_ADDR>(bGM_),
                                                 layoutB,
                                                 reinterpret_cast<GM_ADDR>(biasGM_),
                                                 reinterpret_cast<GM_ADDR>(matmulResultGM),
                                                 layoutC,
                                                 pValue,
                                                 3,
                                                 0,
                                                 static_cast<int32_t>(rankSize),
                                                 MAX_BLOCK_COUNT};
            matmul_op(params);
        }
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantPerToken(
    LocalTensor<float> copyTensor, LocalTensor<float> smoothScaleTensor, LocalTensor<float> absTensor,
    LocalTensor<float> reduceMaxTensor, LocalTensor<float> quantScaleTensor, int32_t actualMoveSize,
    int32_t actualMoveToken, int32_t tokenPerMove, int32_t moveIdx, event_t eventId)
{
    float quantMaxValue = std::is_same_v<BType, int8_t> ? MAX_INT8 : MAX_INT4;
    for (int32_t tokenIdx = 0; tokenIdx < actualMoveToken; tokenIdx++) {
        PipeBarrier<PIPE_V>();
        uint32_t tokenOffset = tokenIdx * tokenSize;
        if (isSmoothQuant) {
            Mul(copyTensor[tokenOffset], copyTensor[tokenOffset], smoothScaleTensor, tokenSize);
            PipeBarrier<PIPE_V>();
        }
        Abs(absTensor, copyTensor[tokenOffset], tokenSize);
        PipeBarrier<PIPE_V>();

        ReduceMax<float>(reduceMaxTensor, absTensor, absTensor, tokenSize);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        float maxValue = reduceMaxTensor.GetValue(0);
        float quantScale = maxValue / quantMaxValue;
        float quantScaleReciproal = quantMaxValue / maxValue;
        quantScaleTensor.SetValue(moveIdx * tokenPerMove + tokenIdx, quantScale);
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        Muls(copyTensor[tokenOffset], copyTensor[tokenOffset], quantScaleReciproal, tokenSize);
        PipeBarrier<PIPE_V>();

        Cast(copyTensor.ReinterpretCast<int32_t>()[tokenOffset], copyTensor[tokenOffset], RoundMode::CAST_RINT,
             tokenSize);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<half>()[tokenOffset],
             copyTensor.ReinterpretCast<int32_t>()[tokenIdx * tokenSize], RoundMode::CAST_ROUND, tokenSize);
    }
    PipeBarrier<PIPE_V>();
    Cast(copyTensor.ReinterpretCast<BType>(), copyTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC,
         actualMoveSize);
    PipeBarrier<PIPE_V>();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantToken(__gm__ AType *dataSrc, int64_t dataOffset,
                                                                     int32_t coreTokenOffset, int32_t dataLen,
                                                                     int32_t commIdx)
{
    int32_t tokenNum = dataLen / tokenSize;
    LocalTensor<float> ubTensor = uBuf_.Get<float>();
    /* 用于存储计算完成的量化系数 */
    LocalTensor<float> quantScaleTensor = ubTensor;

    /* 在smoothQuant场景，用于存储smoothScales系数 */
    int32_t smoothScaleOffset = Block32B<float>::AlignUp(tokenNum);
    LocalTensor<float> smoothScaleTensor = quantScaleTensor[smoothScaleOffset];

    // 用于存储allToAll后的A矩阵
    uint32_t copyTensorOffset = isSmoothQuant ? Block32B<float>::AlignUp(tokenSize) : 0;
    uint32_t smoothScaleCastOffset = Block32B<AType>::AlignUp(copyTensorOffset);
    uint32_t midElementCnt = (USED_UB_SIZE / sizeof(float) - smoothScaleOffset - copyTensorOffset) / BUFFER_NUM;
    uint32_t ub_offset = Block32B<float>::AlignUp(midElementCnt);
    LocalTensor<float> copyTensor0 = smoothScaleTensor[copyTensorOffset];
    LocalTensor<float> copyTensor1 = copyTensor0[ub_offset];

    int32_t copyTensorRemainUbSize =
        midElementCnt - Block32B<float>::AlignUp(tokenSize) - BLOCK_ALIGN_BYTES / sizeof(float);
    int32_t ubTokenAlignedPingPongSize = copyTensorRemainUbSize / tokenSize * tokenSize;
    int32_t pingPongMoveCount = (dataLen + ubTokenAlignedPingPongSize - 1) / ubTokenAlignedPingPongSize;
    int32_t actualMoveSize = ubTokenAlignedPingPongSize;
    int32_t tokenPerMove = tokenNum < actualMoveSize / tokenSize ? tokenNum : actualMoveSize / tokenSize;
    int32_t actualMoveToken = tokenPerMove; /* ub_ping_pong_size已经与tokenSize对齐，因此必然每次搬运整数倍token */
    // 动态量化为INT8场景，每个元素1字节；量化为INT4场景，每两个元素1字节
    uint32_t actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
    uint32_t sizeScale = std::is_same_v<BType, int8_t> ? 1 : 2;

    /* 用于存储计算quantScale的token取abs的结果 */
    int32_t absOffset = Block32B<float>::AlignUp(
        ubTokenAlignedPingPongSize); /* 从GM拷贝的数据用abs_offset_a大小空间，case为float后用abs_offset大小的空间 */
    int32_t castOffset = Block32B<AType>::AlignUp(absOffset); // 换成AType可能不一定32B对齐
    LocalTensor<float> absTensor0 = copyTensor0[absOffset];
    LocalTensor<float> absTensor1 = copyTensor1[absOffset];

    /* 用于存储取abs后，取出token中最大元素的值 */
    int32_t reduceMaxOffset = Block32B<float>::AlignUp(tokenSize);
    LocalTensor<float> reduceMaxTensor0 = absTensor0[reduceMaxOffset];
    LocalTensor<float> reduceMaxTensor1 = absTensor1[reduceMaxOffset];

    if (isSmoothQuant) {
        // smoothQuant场景需要将x1先乘上smoothQuantScale，MUL不支持bf16，因此需要先转换成float类型
        CopyGmToUbufAlignB16(smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset],
                             reinterpret_cast<__gm__ AType *>(x1ScaleGM_), 1, tokenSize * sizeof(AType), 0, 0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(smoothScaleTensor, smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset], RoundMode::CAST_NONE,
             tokenSize);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t moveIdx = 0; moveIdx < pingPongMoveCount; ++moveIdx) {
        if (moveIdx == pingPongMoveCount - 1) {
            actualMoveSize = dataLen - moveIdx * ubTokenAlignedPingPongSize;
            actualMoveToken = actualMoveSize / tokenSize;
            actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
        }
        auto eventId = (moveIdx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (moveIdx & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (moveIdx & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> reduceMaxTensor = (moveIdx & 1) ? reduceMaxTensor0 : reduceMaxTensor1;

        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[castOffset],
                             reinterpret_cast<__gm__ AType *>(dataSrc) + dataOffset, 1, actualMoveSize * sizeof(AType),
                             0, 0);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[castOffset], RoundMode::CAST_NONE, actualMoveSize);
        QuantPerToken(copyTensor, smoothScaleTensor, absTensor, reduceMaxTensor, quantScaleTensor, actualMoveSize,
                      actualMoveToken, tokenPerMove, moveIdx, eventId);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        /* 搬运到GM上时，与peerMem上的相对位置保持不变，后续数据offset可以复用 */
        CopyUbufToGmAlignB16(reinterpret_cast<__gm__ int8_t *>(quantAGM_) + dataOffset / sizeScale,
                             copyTensor.ReinterpretCast<int8_t>(), 1, actualMoveBytes, 0, 0);
        dataOffset += actualMoveSize;
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    CopyUbufToGmAlignB16(reinterpret_cast<__gm__ float *>(quantScaleGM_) + coreTokenOffset + commIdx * mPerLoop,
                         quantScaleTensor, 1, tokenNum * sizeof(float), 0, 0);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::SmoothQuantProc(event_t eventId, int32_t dataSegmentOffset,
                                                                          int32_t smoothScaleCastOffset,
                                                                          int32_t actualMoveSize,
                                                                          LocalTensor<float> copyTensor,
                                                                          LocalTensor<float> smoothScaleTensor)
{
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    CopyGmToUbufAlignB16(smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset],
                         reinterpret_cast<__gm__ AType *>(x1ScaleGM_) + dataSegmentOffset, 1,
                         actualMoveSize * sizeof(AType), 0, 0);
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(smoothScaleTensor, smoothScaleTensor.ReinterpretCast<AType>()[smoothScaleCastOffset], RoundMode::CAST_NONE,
         actualMoveSize);
    PipeBarrier<PIPE_V>();
    Mul(copyTensor, copyTensor, smoothScaleTensor, actualMoveSize);
    PipeBarrier<PIPE_V>();
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CalcTokenMaxValue(
    LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1, LocalTensor<float> absTensor0,
    LocalTensor<float> absTensor1, LocalTensor<float> smoothScaleTensor0, LocalTensor<float> smoothScaleTensor1,
    int32_t castOffset, __gm__ AType *dataSrc, int64_t dataTokenOffset, int32_t smoothScaleCastOffset,
    int32_t sizeScale, LocalTensor<float> reduceMaxTensor)
{
    int32_t actualMoveSize = copyTensorSize;
    int32_t dataSegmentOffset = 0;
    /* 获取当前token的max_abs_value */
    for (int32_t tokenSegmentLoop = 0; tokenSegmentLoop < copyTimes; tokenSegmentLoop++) {
        if (tokenSegmentLoop == copyTimes - 1) {
            actualMoveSize = tokenSize - tokenSegmentLoop * copyTensorSize;
        }
        auto eventId = (tokenSegmentLoop & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (tokenSegmentLoop & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (tokenSegmentLoop & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> smoothScaleTensor = (tokenSegmentLoop & 1) ? smoothScaleTensor0 : smoothScaleTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        /* 下一步需要将copyTensor转换成float类型，目标地址复用copyTensor。为防止踩踏，将AType类型数据内存放在后半段 */
        CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[castOffset],
                             reinterpret_cast<__gm__ AType *>(dataSrc) + dataTokenOffset + dataSegmentOffset, 1,
                             actualMoveSize * sizeof(AType), 0, 0);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[castOffset], RoundMode::CAST_NONE, actualMoveSize);
        PipeBarrier<PIPE_V>();
        if (isSmoothQuant) {
            SmoothQuantProc(eventId, dataSegmentOffset, smoothScaleCastOffset, actualMoveSize, copyTensor,
                            smoothScaleTensor);
        }
        Abs(absTensor, copyTensor, actualMoveSize);
        PipeBarrier<PIPE_V>();
        ReduceMax<float>(copyTensor, absTensor, absTensor, actualMoveSize);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        float currentMaxValue = copyTensor.GetValue(0);
        float lastMaxValue = reduceMaxTensor.GetValue(0);
        float maxValue = currentMaxValue > lastMaxValue ? currentMaxValue : lastMaxValue;
        reduceMaxTensor.SetValue(0, maxValue);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        dataSegmentOffset += actualMoveSize;
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantPerSegment(
    LocalTensor<float> copyTensor0, LocalTensor<float> copyTensor1, LocalTensor<float> absTensor0,
    LocalTensor<float> absTensor1, LocalTensor<float> smoothScaleTensor0, LocalTensor<float> smoothScaleTensor1,
    int32_t castOffset, __gm__ AType *dataSrc, int64_t dataTokenOffset, int32_t smoothScaleCastOffset,
    int32_t sizeScale, float quantScaleReciproal)
{
    int32_t actualMoveSize = copyTensorSize;
    int32_t actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
    int32_t dataSegmentOffset = 0;
    /* 量化当前token */
    for (int32_t tokenSegmentLoop = 0; tokenSegmentLoop < copyTimes; tokenSegmentLoop++) {
        if (tokenSegmentLoop == copyTimes - 1) {
            actualMoveSize = tokenSize - tokenSegmentLoop * copyTensorSize;
            actualMoveBytes = std::is_same_v<BType, int8_t> ? actualMoveSize : actualMoveSize / 2;
        }
        auto eventId = (tokenSegmentLoop & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<float> copyTensor = (tokenSegmentLoop & 1) ? copyTensor0 : copyTensor1;
        LocalTensor<float> absTensor = (tokenSegmentLoop & 1) ? absTensor0 : absTensor1;
        LocalTensor<float> smoothScaleTensor = (tokenSegmentLoop & 1) ? smoothScaleTensor0 : smoothScaleTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        /* 下一步需要将copyTensor转换成float类型，目标地址复用copyTensor。为防止踩踏，将AType类型数据内存放在后半段 */
        CopyGmToUbufAlignB16(copyTensor.ReinterpretCast<AType>()[castOffset],
                             reinterpret_cast<__gm__ AType *>(dataSrc) + dataTokenOffset + dataSegmentOffset, 1,
                             actualMoveSize * sizeof(AType), 0, 0);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(copyTensor, copyTensor.ReinterpretCast<AType>()[castOffset], RoundMode::CAST_NONE, actualMoveSize);
        PipeBarrier<PIPE_V>();
        if (isSmoothQuant) {
            SmoothQuantProc(eventId, dataSegmentOffset, smoothScaleCastOffset, actualMoveSize, copyTensor,
                            smoothScaleTensor);
        }
        Muls(copyTensor, copyTensor, quantScaleReciproal, actualMoveSize);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<int32_t>(), copyTensor, RoundMode::CAST_RINT, actualMoveSize);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<half>(), copyTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
             actualMoveSize);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        PipeBarrier<PIPE_V>();
        Cast(copyTensor.ReinterpretCast<BType>(), copyTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC,
             actualMoveSize);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        CopyUbufToGmAlignB16(
            reinterpret_cast<__gm__ int8_t *>(quantAGM_) + (dataTokenOffset + dataSegmentOffset) / sizeScale,
            copyTensor.ReinterpretCast<int8_t>(), 1, actualMoveBytes, 0, 0);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        dataSegmentOffset += actualMoveSize;
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::QuantTokenSegment(__gm__ AType *dataSrc, int64_t dataOffset,
                                                                            int32_t coreTokenOffset, int32_t dataLen,
                                                                            int32_t commIdx)
{
    int32_t tokenNum = dataLen / tokenSize; /* 当前核实际处理的token数 */
    LocalTensor<float> ubTensor = uBuf_.Get<float>();
    LocalTensor<float> quantScaleTensor = ubTensor;

    int32_t reduceMaxOffset = Block32B<float>::AlignUp(tokenNum);
    LocalTensor<float> reduceMaxTensor = quantScaleTensor[reduceMaxOffset];

    // 用于存储allToAll后的A矩阵
    uint32_t copyTensorOffset = BLOCK_ALIGN_BYTES / sizeof(float);
    uint32_t midElementCnt = (USED_UB_SIZE / sizeof(float) - reduceMaxOffset - copyTensorOffset) / BUFFER_NUM;
    uint32_t ub_offset = Block32B<float>::AlignUp(midElementCnt);
    LocalTensor<float> copyTensor0 = reduceMaxTensor[copyTensorOffset];
    LocalTensor<float> copyTensor1 = copyTensor0[ub_offset];

    int32_t absOffset = Block32B<float>::AlignUp(
        copyTensorSize); /* 从GM拷贝的数据用abs_offset_a大小空间，case为float后用abs_offset大小的空间 */
    int32_t castOffset = Block32B<AType>::AlignUp(absOffset); // 换成AType可能不一定32B对齐
    LocalTensor<float> absTensor0 = copyTensor0[absOffset];
    LocalTensor<float> absTensor1 = copyTensor1[absOffset];

    /* 在smoothQuant场景，用于存储smoothScales系数 */
    int32_t smoothScaleOffset = Block32B<float>::AlignUp(copyTensorSize);
    int32_t smoothScaleCastOffset = Block32B<AType>::AlignDown(smoothScaleOffset);
    LocalTensor<float> smoothScaleTensor0 = absTensor0[smoothScaleOffset];
    LocalTensor<float> smoothScaleTensor1 = absTensor1[smoothScaleOffset];

    uint32_t sizeScale = std::is_same_v<BType, int8_t> ? 1 : 2;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (int32_t tokenLoop = 0; tokenLoop < tokenNum; tokenLoop++) {
        int64_t dataTokenOffset = dataOffset + static_cast<int64_t>(tokenLoop) * tokenSize;
        Duplicate<float>(reduceMaxTensor, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        /* 获取当前token的max_abs_value */
        CalcTokenMaxValue(copyTensor0, copyTensor1, absTensor0, absTensor1, smoothScaleTensor0, smoothScaleTensor1,
                          castOffset, dataSrc, dataTokenOffset, smoothScaleCastOffset, sizeScale, reduceMaxTensor);
        float tokenMaxValue = reduceMaxTensor.GetValue(0);
        float quantMaxValue = std::is_same_v<BType, int8_t> ? MAX_INT8 : MAX_INT4;
        float quantScale = tokenMaxValue / quantMaxValue;
        float quantScaleReciproal = quantMaxValue / tokenMaxValue;
        quantScaleTensor.SetValue(tokenLoop, quantScale);
        /* 量化当前token */
        QuantPerSegment(copyTensor0, copyTensor1, absTensor0, absTensor1, smoothScaleTensor0, smoothScaleTensor1,
                        castOffset, dataSrc, dataTokenOffset, smoothScaleCastOffset, sizeScale, quantScaleReciproal);
    }
    CopyUbufToGmAlignB16(reinterpret_cast<__gm__ float *>(quantScaleGM_) + coreTokenOffset + commIdx * mPerLoop,
                         quantScaleTensor, 1, tokenNum * sizeof(float), 0, 0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Quant(uint64_t flagIdx, int32_t commIdx)
{
    __gm__ AType *dataSrc = (__gm__ AType *)buff[rank];
    int32_t totalDataSize =
        x1DataSize < allToAllSizeAllRanksPerLoop ? x1DataSize : allToAllSizeAllRanksPerLoop; // 实际需要处理的数据量
    int32_t quantSizePerCore = (totalDataSize / quantCoreNum) / tokenSize * tokenSize;       // 每核均分的数据量
    int32_t remainTokenNum = (totalDataSize - quantSizePerCore * quantCoreNum) /
                             tokenSize; // 每个核均分quantSizePerCore之后，剩余的token由前remainTokenNum个核各多分担一个
    uint32_t globalAivIdx = aicIdx * 2 + aivIdx;
    int32_t dataSrcCoreOffset = 0;
    int32_t dataLen = 0;
    int32_t coreTokenOffset = 0;
    int32_t tokenPercore = quantSizePerCore / tokenSize;
    if (globalAivIdx < remainTokenNum) {
        dataSrcCoreOffset = (globalAivIdx % quantCoreNum) * (quantSizePerCore + tokenSize);
        dataLen = dataSrcCoreOffset + (quantSizePerCore + tokenSize) > totalDataSize ?
                      totalDataSize - dataSrcCoreOffset :
                      quantSizePerCore + tokenSize;
        coreTokenOffset = (globalAivIdx % quantCoreNum) * (tokenPercore + 1);
    } else {
        dataSrcCoreOffset = remainTokenNum * (quantSizePerCore + tokenSize) +
                            (globalAivIdx % quantCoreNum - remainTokenNum) * (quantSizePerCore);
        dataLen =
            dataSrcCoreOffset + quantSizePerCore > totalDataSize ? totalDataSize - dataSrcCoreOffset : quantSizePerCore;
        coreTokenOffset =
            remainTokenNum * (tokenPercore + 1) + (globalAivIdx % quantCoreNum - remainTokenNum) * tokenPercore;
    }
    if (dataLen <= 0) {
        return;
    }
    int64_t dataSrcOffset = static_cast<int64_t>(flagIdx) * pingPongBlockSize;
    int64_t dataOffset = dataSrcOffset + dataSrcCoreOffset;
    if (isSegmentK) {
        // token过大，分段量化
        QuantTokenSegment(dataSrc, dataOffset, coreTokenOffset, dataLen, commIdx);
    } else {
        // token较小，一次量化多个
        QuantToken(dataSrc, dataOffset, coreTokenOffset, dataLen, commIdx);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Dequant()
{
    using ArchTag = Arch::AtlasA2;

    using ElementC = int32_t;
    using ElementBias = BiasType;
    using LayoutD = layout::RowMajor;

    uint32_t realM = m / rankSize;
    uint32_t realK = k * rankSize;
    LayoutD layoutD{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};

    using CType_ = Gemm::GemmType<ElementC, layout::RowMajor>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleGType = Gemm::GemmType<ScaleType, layout::VectorLayout>;
    using PerTokenScaleGType = Gemm::GemmType<float, layout::VectorLayout>; // 全量化时，也限定为float
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<CType, layout::RowMajor>;
    layout::VectorLayout layoutScale{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutPerTokenScale{static_cast<uint32_t>(realM)};
    layout::VectorLayout layoutBias{static_cast<uint32_t>(n)};
    using LayoutScale = layout::VectorLayout;
    using LayoutPerTokenScale = layout::VectorLayout;
    using ElementD = CType;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    struct TileCopyDequant
        : public Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType> {
        using Base = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType>;
        using ElementC = typename Base::ElementC;
        using ElementScale = typename Base::ElementX;
        using ElementPerTokenScale = typename Base::ElementY;
        using ElementBias = typename BiasGType::Element;
        using ElementD = typename Base::ElementD;

        using CopyGmToUbC = typename Base::CopyGmToUbC;
        using CopyGmToUbScale = typename Base::CopyGmToUbX;
        using CopyGmToUbPerTokenScale = typename Base::CopyGmToUbY;
        using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasGType>;
        using CopyUbToGmD = typename Base::CopyUbToGmD;
    };

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    GemmCoord problemShape{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    AscendC::GlobalTensor<ElementD> gmD;
    gmD.SetGlobalBuffer((__gm__ ElementD *)cGM_);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)dequantCGM_);

    AscendC::GlobalTensor<ElementBias> gmBias;
    if (biasGM_ != nullptr) {
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)biasGM_);
    }

    uint32_t rowsPerCore = DivCeil(problemShape.m(), blockNum);
    uint32_t rowsThisCore = rowsPerCore;
    uint32_t stRowPerCore = aicIdx * rowsPerCore;
    if (stRowPerCore < problemShape.m()) {
        if (rowsThisCore + stRowPerCore > problemShape.m()) {
            rowsThisCore = problemShape.m() - stRowPerCore;
        }
    } else {
        rowsThisCore = 0;
    }
    MatrixCoord coreOffset(stRowPerCore, 0u);
    auto layoutC = layout::RowMajor{problemShape.m(), n};
    int64_t gmOffsetC = layoutC.GetOffset(coreOffset);
    GemmCoord actualBlockShape{rowsThisCore, n, 1};

    GM_ADDR x1ScaleGM =
        (QuantType == MC2_DYNAMIC_QUANT) ? quantScaleGM_ : x1ScaleGM_; // 根据是否是全量化，决定从哪里获得pertokenScale

    if (m0 == 128) {
        using EpilogueTileShape = MatrixShape<32, 256>;
        using TileRowBroadcastMul =
            Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
        using TileRowBroadcastAdd =
            Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
        using TileBroadcastOneBlk =
            Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul =
            Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
        using QuantBlockEpilogue =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType,
                                           DType, TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                           TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
        QuantBlockEpilogue blockEpilogue(resource);

        using EpilogueParams = typename QuantBlockEpilogue::Params;
        EpilogueParams epilogueParams{
            scaleGM_,  layoutScale,
            x1ScaleGM, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
            biasGM_,   layoutBias};
        blockEpilogue.UpdateParams(epilogueParams);

        blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
    } else {
        using EpilogueTileShape = MatrixShape<64, 128>;
        using TileRowBroadcastMul =
            Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;

        using TileRowBroadcastAdd =
            Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;

        using TileBroadcastOneBlk =
            Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
        using TileOneBlkColumnBroadcastMul =
            Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
        using QuantBlockEpilogue =
            Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType,
                                           DType, TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                           TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
        QuantBlockEpilogue blockEpilogue(resource);

        using EpilogueParams = typename QuantBlockEpilogue::Params;
        EpilogueParams epilogueParams{
            scaleGM_,  layoutScale,
            x1ScaleGM, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
            biasGM_,   layoutBias};
        blockEpilogue.UpdateParams(epilogueParams);

        blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::AlltoAll()
{
    if ASCEND_IS_AIV {
        ResetIpcFlags(BUFFER_NUM);
        PipeBarrier<PIPE_ALL>();
        int64_t src_offset = 0;
        uint32_t elemBytes = sizeof(AType); // 将int4归入该文件时，需要修改
        for (int32_t commIdx = 0; commIdx <= commCount; ++commIdx) {
            uint64_t flagIdx = commIdx % MAX_BLOCK_COUNT;

            if (commIdx == commCount - 1) {
                allToAllSizePerRankPerLoop = allToAllSizePerRank - src_offset;
            }

            if (commIdx >= MAX_BLOCK_COUNT && commIdx < commCount) {
                WaitEvent(flagIdx);
            }

            SetAndWaitAivSync(flagIdx);
            if (commIdx < commCount) {
                CrossRankSyncV1(FLAG_ZERO_IDX, commIdx + 1);
            }
            SetAndWaitAivSync(flagIdx);

            if (aivIdx == 0 && commIdx < commCount && aicIdx < allToAllSendCoreNum) {
                int32_t dstRank = aicIdx / coreNumPerRank;
                int32_t dstLoc = aicIdx % coreNumPerRank;
                int32_t coreOffset = dstLoc * allToAllSizePerCore;
                int32_t dataLen = coreOffset + allToAllSizePerCore > allToAllSizePerRankPerLoop ?
                                      allToAllSizePerRankPerLoop - coreOffset :
                                      allToAllSizePerCore;
                uint64_t dataSrc = dstRank * allToAllSizePerRank + src_offset + coreOffset;
                int64_t dataDst = static_cast<int64_t>(flagIdx) * pingPongBlockSize +
                                  static_cast<int64_t>(coreOffset) * rankSize + static_cast<int64_t>(rank) * k;
                uint32_t copyBytes = Catlass::BitsToBytes(dataLen * Catlass::SizeOfBits<AType>::value);

                if (dataLen > 0) {
                    CopyTokensFromGMToGM(reinterpret_cast<__gm__ int8_t *>(aGM_) + ElemNumToBytes<AType>(dataSrc),
                                         (__gm__ int8_t *)buff[dstRank] + ElemNumToBytes<AType>(dataDst), copyBytes,
                                         kBytes, tokenBytes);
                }
                src_offset += allToAllSizePerRankPerLoop;
            } else if (isAlltoallOut && aivIdx == 1 && commIdx > 0 && aicIdx >= allToAllSendCoreNum &&
                       aicIdx < usedCoreNum) {
                int64_t blockDst = static_cast<int64_t>((commIdx - 1) % MAX_BLOCK_COUNT) * pingPongBlockSize;
                int32_t mThisLoop = commIdx == commCount ? m / rankSize - (commIdx - 1) * mPerLoop : mPerLoop;
                int32_t mThisLoopPerCore = DivCeil(mThisLoop, allToAllRecvCoreNum);
                int32_t mSt = (aicIdx - allToAllSendCoreNum) * mThisLoopPerCore;
                int32_t mThisCoreThisLoop = mSt + mThisLoopPerCore > mThisLoop ? mThisLoop - mSt : mThisLoopPerCore;
                int64_t srcSt = blockDst + static_cast<int64_t>(mSt) * tokenSize;
                int64_t dstSt = (static_cast<int64_t>(commIdx - 1) * mPerLoop + mSt) * tokenSize;
                if (mThisCoreThisLoop > 0) {
                    CopyTokensFromGMToGM(
                        (__gm__ int8_t *)buff[rank] + ElemNumToBytes<AType>(srcSt),
                        reinterpret_cast<__gm__ int8_t *>(allToAllResultGM_) + ElemNumToBytes<AType>(dstSt),
                        mThisCoreThisLoop * tokenBytes, tokenBytes, tokenBytes);
                }
            }

            SetAndWaitAivSync(flagIdx);
            if (commIdx < commCount) {
                CrossRankSyncV1(FLAG_ONE_IDX, commIdx + 1);
            }
            SetAndWaitAivSync(flagIdx);

            if constexpr (QuantType == MC2_DYNAMIC_QUANT) { // 动态量化场景，拷贝完成后，对左矩阵进行quant
                if (commIdx < commCount) {
                    Quant(flagIdx, commIdx);
                    SetAndWaitAivSync(flagIdx);
                }
            }

            if (commIdx < commCount) {
                SetAicSync(flagIdx);
            }
        }

        WaitEvent(FLAG_ZERO_IDX);
        if (commCount % 2 == 0) { // 若AIC计算次数为偶数，则多等一次
            WaitEvent(FLAG_ONE_IDX);
        }

        if constexpr (QuantType != MC2_NON_QUANT) {
            SetAndWaitAivSync(FLAG_ONE_IDX);
            Dequant();
        }

        PipeBarrier<PIPE_ALL>();
        ResetIpcFlags(1);
    }
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Process()
{
    AlltoAll();
    CatlassMatmul();
    SyncAll<false>();
}

} // namespace Mc2Kernel
#endif // ALL_TO_ALL_MATMUL_H
