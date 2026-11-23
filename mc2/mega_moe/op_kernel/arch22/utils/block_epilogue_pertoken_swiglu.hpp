/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_HPP

#include <limits>

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"
#include "gated_activation.hpp"
#include "swiglu_epilogue_common.hpp"
#include "dispatch_policy_custom.hpp"

#include "shared_w8_dequant.hpp"

namespace Catlass::Epilogue::Block {

namespace detail {

// The three policies have the same tiled UB layout and arithmetic. Their
// existing A2/A3 call signatures are provided by the overloads below.
template <class DispatchPolicy_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class PerTokenSwigluImpl {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using Activation = typename DispatchPolicy::Activation;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr uint32_t TILE_LENGTH = DispatchPolicy::TILE_LENGTH;
    static constexpr uint32_t SCALE_BUFFER_COUNT = 2;
    // 输出侧 scale 缓冲批量：结果 scale 在 UB 中按批累积后写回 GM
    static constexpr uint32_t SCALE_BATCH_COUNT = 256;
    static constexpr uint32_t SCALE_BUFFER_BYTES = (SCALE_BATCH_COUNT + BYTE_PER_BLK / sizeof(float)) * sizeof(float);
    static constexpr uint32_t ROW_MAX_BYTES = BYTE_PER_BLK;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // 输入侧 per-token scale 批量预读：批量加载到 UB，避免逐个标量读 GM（读旧值 + 性能差）
    static constexpr uint32_t SCALE_INPUT_COUNT = 256;
    static_assert(SCALE_INPUT_COUNT % (BYTE_PER_BLK / sizeof(ElementPerTokenScale)) == 0,
                  "The scale input batch must be block aligned");
    static constexpr uint32_t SCALE_INPUT_BYTES = SCALE_INPUT_COUNT * sizeof(float);
    static constexpr size_t BUFFER_SIZE =
        UB_STAGES *
            (2 * TILE_LENGTH * sizeof(ElementC) + TILE_LENGTH * sizeof(ElementD) + 3 * TILE_LENGTH * sizeof(float)) +
        SCALE_BUFFER_COUNT * SCALE_BUFFER_BYTES + ROW_MAX_BYTES + SCALE_INPUT_BYTES;
    static_assert(UB_STAGES >= 2, "The pipelined activation epilogue requires double buffering");
    static_assert(SCALE_BATCH_COUNT % (BYTE_PER_BLK / sizeof(ElementPerTokenScale)) == 0,
                  "The scale batch must be block aligned");
    static_assert(BUFFER_SIZE + Activation::MIN_SHARED_TMP_BYTES <= ArchTag::UB_SIZE,
                  "The tiled activation buffers and workspace exceed UB capacity");

    static_assert((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                   std::is_same_v<ElementC, int32_t>) &&
                      (std::is_same_v<ElementD, float> || std::is_same_v<ElementD, int8_t> ||
                       std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>),
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> &&
                      std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::RowMajor>,
                  "The layout template parameters of BlockEpilogue are wrong");

    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
    using CopyUbToGmDequantScale =
        Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>>;

    CATLASS_DEVICE void SetWeightScales(__gm__ uint64_t* scales)
    {
        weightScales_.SetGlobalBuffer(scales);
    }

    struct Params {
        __gm__ ElementPerTokenScale* ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD* ptrD{nullptr};
        LayoutD layoutD{};

        CATLASS_DEVICE Params() {};

        CATLASS_DEVICE
        Params(__gm__ ElementPerTokenScale* ptrPerTokenScale_, LayoutPerTokenScale const& layoutPerTokenScale_,
               __gm__ ElementD* ptrD_, LayoutD const& layoutD_)
            : ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(ptrD_),
              layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    PerTokenSwigluImpl(Arch::Resource<ArchTag> const& resource, int32_t n)
    {
        (void)n;
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += 2 * TILE_LENGTH * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(ElementD);
            ubGateFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);
            ubUpFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);
            ubActivationList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += TILE_LENGTH * sizeof(float);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
        for (uint32_t i = 0; i < SCALE_BUFFER_COUNT; ++i) {
            ubPerTokenScaleOutputList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += SCALE_BUFFER_BYTES;
        }
        ubRowAbsMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += ROW_MAX_BYTES;
        ubPerTokenScaleInput = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += SCALE_INPUT_BYTES;
        sharedTmpBuffer = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
    }

    // Execution receives its tensors directly; retain the explicit Params overload for compatibility.
    CATLASS_DEVICE
    PerTokenSwigluImpl(Arch::Resource<ArchTag> const& resource, int32_t n, Params const& params)
        : PerTokenSwigluImpl(resource, n)
    {
        (void)params;
    }

    CATLASS_DEVICE
    void Finalize()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE ~PerTokenSwigluImpl() {}

    CATLASS_DEVICE
    void UpdateParams(Params const& params_)
    {
        (void)params_;
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC, MatrixCoord const& shapeC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale1,
                    AscendC::GlobalTensor<ElementD> const& gmD,
                    AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale2, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback&& callback = Callback{})
    {
        activationCode_ = activationCode;
        activationParams1_ = activationParams1;
        activationParams2_ = activationParams2;
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t branchLength = shapeC.column() / 2;
        uint32_t loopStartIdx = 0;
        uint32_t tasksForIdx = 0;
        if (!GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)) {
            return;
        }
        uint32_t scaleBatchStart = loopStartIdx;
        uint32_t scaleBatchCount = 0;
        uint32_t scaleBatchIndex = 0;

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            ElementPerTokenScale perTokenScale =
                ::Catlass::Epilogue::Block::detail::ReadSwigluInputScale<SCALE_INPUT_COUNT>(
                    ubPerTokenScaleInput, gmPerTokenScale1, loopIdx, loopStartIdx, tasksForIdx);
            uint32_t tileCount = TileCount(branchLength);
            uint32_t currentStage = 0;
            LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
            ResetRowAbsMax();
            for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint32_t tileOffset = tileIdx * TILE_LENGTH;
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                uint32_t nextStage = NextStage(currentStage);
                uint32_t nextOffset = tileOffset + TILE_LENGTH;
                if (nextOffset < branchLength) {
                    LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                             gmmOutPreRowStride, nextStage);
                }
                ComputeLoadedTile(currentStage, tileLength, tileOffset, branchLength, perTokenScale, true,
                                  activationClamp);
                AccumulateRowAbsMax(currentStage, tileLength);
                currentStage = nextStage;
            }

            ElementPerTokenScale rowAbsMax = ReadRowAbsMax();
            uint32_t scaleStage = scaleBatchIndex % SCALE_BUFFER_COUNT;
            // Reuse waits only after both buffers have been launched, allowing scale MTE3 to overlap the next pass.
            if (scaleBatchCount == 0 && scaleBatchIndex >= SCALE_BUFFER_COUNT) {
                WaitScaleBuffer(scaleStage);
            }
            ubPerTokenScaleOutputList[scaleStage].SetValue(scaleBatchCount, rowAbsMax / 127.0f);
            ElementPerTokenScale quantMultiplier = rowAbsMax > 0.0f ? 127.0f / rowAbsMax : 0.0f;
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            currentStage = 0;
            if (tileCount == 1) {
                // Single-tile fast path: ubActivationList[0] still holds the first-pass activation
                // output, so skip the second LoadTile + reactivation and quantize it directly.
                QuantizeAndCopyTile(0, gmTileD[0], TileLength(branchLength, 0), quantMultiplier);
            } else {
                LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
                for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                    uint32_t tileOffset = tileIdx * TILE_LENGTH;
                    uint32_t tileLength = TileLength(branchLength, tileOffset);
                    uint32_t nextStage = NextStage(currentStage);
                    uint32_t nextOffset = tileOffset + TILE_LENGTH;
                    if (nextOffset < branchLength) {
                        LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                                 gmmOutPreRowStride, nextStage);
                    }
                    ComputeLoadedTile(currentStage, tileLength, tileOffset, branchLength, perTokenScale, true,
                                      activationClamp);
                    QuantizeAndCopyTile(currentStage, gmTileD[tileOffset], tileLength, quantMultiplier);
                    currentStage = nextStage;
                }
            }
            ++scaleBatchCount;
            if (scaleBatchCount == SCALE_BATCH_COUNT) {
                FlushScaleBatch(gmPerTokenScale2, scaleBatchStart, scaleBatchCount, scaleStage);
                scaleBatchStart += scaleBatchCount;
                scaleBatchCount = 0;
                ++scaleBatchIndex;
            }
        }
        if (scaleBatchCount > 0) {
            FlushScaleBatch(gmPerTokenScale2, scaleBatchStart, scaleBatchCount, scaleBatchIndex % SCALE_BUFFER_COUNT);
            ++scaleBatchIndex;
        }
        DrainScaleBuffers(scaleBatchIndex);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC, MatrixCoord const& shapeC,
                    AscendC::GlobalTensor<ElementD> const& gmD, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback&& callback = Callback{},
                    const AscendC::GlobalTensor<ElementPerTokenScale>* inputScale = nullptr)
    {
        activationCode_ = activationCode;
        activationParams1_ = activationParams1;
        activationParams2_ = activationParams2;
        callback();
        uint32_t blockM = shapeC.row();
        uint32_t branchLength = shapeC.column() / 2;
        uint32_t loopStartIdx = 0;
        uint32_t tasksForIdx = 0;
        if (!GetTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx)) {
            return;
        }

        for (uint32_t loopIdx = loopStartIdx; loopIdx < loopStartIdx + tasksForIdx; ++loopIdx) {
            auto gmTileC = gmC[loopIdx * gmmOutPreRowStride];
            auto gmTileD = gmD[loopIdx * branchLength];
            // INT8 GMM1 -> BF16 GMM2: dequantize before activation, then
            // round directly to BF16. There is no second dynamic quantization.
            ElementPerTokenScale perTokenScale = 1.0f;
            if (inputScale != nullptr) {
                perTokenScale = ::Catlass::Epilogue::Block::detail::ReadSwigluInputScale<SCALE_INPUT_COUNT>(
                    ubPerTokenScaleInput, *inputScale, loopIdx, loopStartIdx, tasksForIdx);
            }
            uint32_t tileCount = TileCount(branchLength);
            uint32_t currentStage = 0;
            LoadTile(gmTileC, branchLength, 0, TileLength(branchLength, 0), gmmOutPreRowStride, currentStage);
            for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint32_t tileOffset = tileIdx * TILE_LENGTH;
                uint32_t tileLength = TileLength(branchLength, tileOffset);
                uint32_t nextStage = NextStage(currentStage);
                uint32_t nextOffset = tileOffset + TILE_LENGTH;
                if (nextOffset < branchLength) {
                    LoadTile(gmTileC, branchLength, nextOffset, TileLength(branchLength, nextOffset),
                             gmmOutPreRowStride, nextStage);
                }
                ComputeLoadedTile(currentStage, tileLength, tileOffset, branchLength, perTokenScale,
                                  inputScale != nullptr, activationClamp);
                CopyDirectTile(currentStage, gmTileD[tileOffset], tileLength);
                currentStage = nextStage;
            }
        }
    }

    // The A3 kernels pass Resource; the A2 kernels use the resource-free
    // overloads above. Resource was unused in both original operator bodies.
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC, MatrixCoord const& shapeC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale1,
                    AscendC::GlobalTensor<ElementD> const& gmD,
                    AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale2,
                    Arch::Resource<ArchTag> const& resource, uint32_t epilogueCoreNum = 40,
                    float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback&& callback = Callback{})
    {
        (void)resource;
        operator()(gmC, shapeC, gmPerTokenScale1, gmD, gmPerTokenScale2, epilogueCoreNum, activationClamp,
                   activationCode, activationParams1, activationParams2, gmmOutPreRowStride,
                   static_cast<Callback&&>(callback));
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const& gmC, MatrixCoord const& shapeC,
                    AscendC::GlobalTensor<ElementD> const& gmD, Arch::Resource<ArchTag> const& resource,
                    uint32_t epilogueCoreNum = 40, float activationClamp = 0.0f, uint32_t activationCode = 0,
                    float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                    float activationParams2 = SituActivation::DEFAULT_BETA, uint32_t gmmOutPreRowStride = 1,
                    Callback&& callback = Callback{},
                    const AscendC::GlobalTensor<ElementPerTokenScale>* inputScale = nullptr)
    {
        (void)resource;
        operator()(gmC, shapeC, gmD, epilogueCoreNum, activationClamp, activationCode, activationParams1,
                   activationParams2, gmmOutPreRowStride, static_cast<Callback&&>(callback), inputScale);
    }

private:
    CATLASS_DEVICE
    bool GetTaskRange(uint32_t blockM, uint32_t epilogueCoreNum, uint32_t& loopStartIdx, uint32_t& tasksForIdx)
    {
        return ::Catlass::Epilogue::Block::detail::SwigluTaskRange(blockM, epilogueCoreNum, loopStartIdx, tasksForIdx);
    }

    CATLASS_DEVICE
    void FlushScaleBatch(AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale, uint32_t gmOffset,
                         uint32_t scaleCount, uint32_t scaleStage)
    {
        ::Catlass::Epilogue::Block::detail::FlushSwigluScaleBatch<LayoutPerTokenScale>(
            gmPerTokenScale, ubPerTokenScaleOutputList[scaleStage], copyUbToGmDequantScale, gmOffset, scaleCount,
            scaleStage);
    }

    CATLASS_DEVICE
    void WaitScaleBuffer(uint32_t scaleStage)
    {
        ::Catlass::Epilogue::Block::detail::WaitSwigluScaleBuffer(scaleStage);
    }

    CATLASS_DEVICE
    void DrainScaleBuffers(uint32_t scaleBatchCount)
    {
        ::Catlass::Epilogue::Block::detail::DrainSwigluScaleBuffers(scaleBatchCount, SCALE_BUFFER_COUNT);
    }

    CATLASS_DEVICE
    static uint32_t TileCount(uint32_t branchLength)
    {
        return ::Catlass::Epilogue::Block::detail::SwigluTileCount<TILE_LENGTH>(branchLength);
    }

    CATLASS_DEVICE
    static uint32_t TileLength(uint32_t branchLength, uint32_t tileOffset)
    {
        return ::Catlass::Epilogue::Block::detail::SwigluTileLength<TILE_LENGTH>(branchLength, tileOffset);
    }

    CATLASS_DEVICE
    static uint32_t NextStage(uint32_t stageId)
    {
        return ::Catlass::Epilogue::Block::detail::SwigluNextStage<UB_STAGES>(stageId);
    }

    CATLASS_DEVICE
    void LoadTile(AscendC::GlobalTensor<ElementC> const& gmTileC, uint32_t branchLength, uint32_t tileOffset,
                  uint32_t tileLength, uint32_t gmRowStride, uint32_t stageId)
    {
        auto& ubC = ubCList[stageId];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[stageId]);
        LayoutC tileLayout{1, tileLength};
        LayoutC gmTileLayout{1, tileLength, gmRowStride};
        copyGmToUbC(ubC, gmTileC[tileOffset], tileLayout, gmTileLayout);
        copyGmToUbC(ubC[TILE_LENGTH], gmTileC[branchLength + tileOffset], tileLayout, gmTileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[stageId]);
    }

    CATLASS_DEVICE
    void ComputeLoadedTile(uint32_t stageId, uint32_t tileLength, uint32_t tileOffset, uint32_t branchLength,
                           ElementPerTokenScale perTokenScale, bool applyPerTokenScale, float activationClamp)
    {
        auto& ubC = ubCList[stageId];
        auto& ubGate = ubGateFp32List[stageId];
        auto& ubUp = ubUpFp32List[stageId];
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[stageId]);
        Activation::PrepareBranches(ubGate, ubUp, ubC, TILE_LENGTH, tileLength);
        if constexpr (std::is_same_v<ElementC, int32_t>) {
            ApplySharedW8Scale(ubGate, ubC, weightScales_, tileOffset, tileLength, 2 * TILE_LENGTH,
                               eventUbCVMTE2List[stageId], eventUbCMTE2VList[stageId]);
            ApplySharedW8Scale(ubUp, ubC, weightScales_, branchLength + tileOffset, tileLength, 2 * TILE_LENGTH,
                               eventUbCVMTE2List[stageId], eventUbCMTE2VList[stageId]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[stageId]);
        if (applyPerTokenScale) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubGate, ubGate, perTokenScale, tileLength);
            AscendC::Muls(ubUp, ubUp, perTokenScale, tileLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Activation::Compute(ubActivationList[stageId], ubGate, ubUp, sharedTmpBuffer, tileLength, activationClamp,
                            activationCode_, activationParams1_, activationParams2_);
    }

    CATLASS_DEVICE
    void QuantizeAndCopyTile(uint32_t stageId, AscendC::GlobalTensor<ElementD> const& gmTileD, uint32_t tileLength,
                             ElementPerTokenScale quantMultiplier)
    {
        auto& ubQuantTmp = ubGateFp32List[stageId];
        auto ubQuantS32 = ubQuantTmp.template ReinterpretCast<int32_t>();
        auto ubQuantF16 = ubQuantTmp.template ReinterpretCast<half>();
        AscendC::Muls(ubQuantTmp, ubActivationList[stageId], quantMultiplier, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ubQuantS32, ubQuantTmp, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetDeqScale(static_cast<half>(1.0));
        AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
        AscendC::Cast(ubDList[stageId], ubQuantF16, AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        LayoutD tileLayout{1, tileLength};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        copyUbToGmD(gmTileD, ubDList[stageId], tileLayout, tileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
    }

    CATLASS_DEVICE
    void ResetRowAbsMax()
    {
        ::Catlass::Epilogue::Block::detail::ResetSwigluRowAbsMax(ubRowAbsMax);
    }

    CATLASS_DEVICE
    void AccumulateRowAbsMax(uint32_t stageId, uint32_t tileLength)
    {
        auto& ubAbs = ubGateFp32List[stageId];
        auto& ubTileMax = ubUpFp32List[stageId];
        AscendC::Abs(ubAbs, ubActivationList[stageId], tileLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax<float>(ubTileMax, ubAbs, ubTileMax, tileLength, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(ubRowAbsMax, ubRowAbsMax, ubTileMax, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    ElementPerTokenScale ReadRowAbsMax()
    {
        return ::Catlass::Epilogue::Block::detail::ReadSwigluRowAbsMax(ubRowAbsMax);
    }

    CATLASS_DEVICE
    void CopyDirectTile(uint32_t stageId, AscendC::GlobalTensor<ElementD> const& gmTileD, uint32_t tileLength)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
        AscendC::Cast(ubDList[stageId], ubActivationList[stageId], AscendC::RoundMode::CAST_RINT, tileLength);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        LayoutD tileLayout{1, tileLength};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[stageId]);
        copyUbToGmD(gmTileD, ubDList[stageId], tileLayout, tileLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[stageId]);
    }

    AscendC::GlobalTensor<uint64_t> weightScales_;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubGateFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubUpFp32List[UB_STAGES];
    AscendC::LocalTensor<float> ubActivationList[UB_STAGES];
    AscendC::LocalTensor<float> ubPerTokenScaleOutputList[SCALE_BUFFER_COUNT];
    AscendC::LocalTensor<float> ubRowAbsMax;
    AscendC::LocalTensor<float> ubPerTokenScaleInput;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
    uint32_t activationCode_{0};
    float activationParams1_{SwigluOaiActivation::DEFAULT_ALPHA};
    float activationParams2_{SituActivation::DEFAULT_BETA};
    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    CopyUbToGmDequantScale copyUbToGmDequantScale;
};

} // namespace detail

// A3 keeps its original dispatch policy and constructor/call contract.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_>
    : public detail::PerTokenSwigluImpl<EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>, CType_,
                                        LayoutPerTokenScale_, DType_, TileElemWiseMuls_, TileCopy_> {
    using Base = detail::PerTokenSwigluImpl<EpilogueAtlasA2PerTokenDequantSwigluQuant<UB_STAGES_>, CType_,
                                            LayoutPerTokenScale_, DType_, TileElemWiseMuls_, TileCopy_>;

public:
    using Base::Base;
};

// A2 INT8 keeps its original dispatch policy and constructor/call contract.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantSwigluQuantInt8<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_>
    : public detail::PerTokenSwigluImpl<EpilogueAtlasA2PerTokenDequantSwigluQuantInt8<UB_STAGES_>, CType_,
                                        LayoutPerTokenScale_, DType_, TileElemWiseMuls_, TileCopy_> {
    using Base = detail::PerTokenSwigluImpl<EpilogueAtlasA2PerTokenDequantSwigluQuantInt8<UB_STAGES_>, CType_,
                                            LayoutPerTokenScale_, DType_, TileElemWiseMuls_, TileCopy_>;

public:
    using Base::Base;
};

// A2 BF16 keeps its original dispatch policy and constructor/call contract.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileElemWiseMuls_,
          class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileElemWiseMuls_, TileCopy_>
    : public detail::PerTokenSwigluImpl<EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>, CType_,
                                        LayoutPerTokenScale_, DType_, TileElemWiseMuls_, TileCopy_> {
    using Base = detail::PerTokenSwigluImpl<EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<UB_STAGES_>, CType_,
                                            LayoutPerTokenScale_, DType_, TileElemWiseMuls_, TileCopy_>;

public:
    using Base::Base;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_SWIGLU_HPP
