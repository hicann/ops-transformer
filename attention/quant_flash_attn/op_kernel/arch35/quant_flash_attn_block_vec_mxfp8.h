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
 * \file quant_flash_attn_block_vec_mxfp8.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_H_

#include "kernel_operator.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "adv_api/activation/softmax.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "../../../common/op_kernel/init_output.h"
#include "../../../common/op_kernel/attn_sink_gs1.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <bool useDn>
__simd_vf__ inline void UpdateMinCheckValueVF(__ubuf__ float* dstUb, const float minValue, const float scaleValue,
                                              const float pScale)
{
    RegTensor<float> vregMin;
    RegTensor<float> vregPScale;
    RegTensor<float> vregLnPScale;
    MaskReg pregAll = CreateMask<uint16_t, MaskPattern::ALL>();
    Duplicate(vregMin, minValue);
    Duplicate(vregPScale, pScale);
    Ln(vregLnPScale, vregPScale, pregAll);
    if constexpr (useDn) {
        Muls(vregMin, vregMin, scaleValue, pregAll);
    }
    Muls(vregMin, vregMin, INV_LN2, pregAll);
    Truncate<float, RoundMode::CAST_CEIL>(vregMin, vregMin, pregAll);
    Muls(vregMin, vregMin, LN2, pregAll);
    Sub(vregMin, vregMin, vregLnPScale, pregAll);
    StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float*&)dstUb, vregMin, pregAll);
}

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
    bool isFd = false, bool useDn = false, bool isDAligned = true, bool hasVTail = false, bool hasSinkFlag = false>
class QuantFlashAttnBlockVecMxfp8 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = mBaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = mBaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = mBaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr uint32_t s2SplitSize =
        (dTemplateAlign64 == static_cast<uint32_t>(DTemplateType::Aligned256)) ? 128U : 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool HAS_V_TAIL = hasVTail;
    static constexpr bool USE_3_P_SLOTS =
        (static_cast<uint32_t>(dTemplateType) == 192U && static_cast<uint32_t>(dVTemplateType) == 128U);

    // attention sink 开关: 由 tiling key 决定, 编译期选择, 运行期无分支
    static constexpr bool HAS_SINK = hasSinkFlag;

    // sink 临时 UB 同步。8~11 已由 FlashDecode 的 LSE/MM2 buffer 占用，
    // 2/4 用于 FD 输出，12 为本类 sink 临时 buffer 保留的 Mutex flag。
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 12;
    // sink 暂存布局: [0,512) 原始值, [512,1024) 逐行展开值, [1024,1056) ln(pScale)
    static constexpr uint32_t SINK_STAGE_TMP_BYTES = 512U;
    static constexpr uint32_t SINK_STAGE_EXPANDED_BYTES = 512U;
    static constexpr uint32_t SINK_STAGE_LN_BYTES = 32U;
    static constexpr uint32_t SINK_STAGE_BYTES = SINK_STAGE_TMP_BYTES + SINK_STAGE_EXPANDED_BYTES + SINK_STAGE_LN_BYTES;
    static_assert(!HAS_SINK || (mBaseSize / CV_RATIO <= 64U),
                  "attention sink prologue supports at most 64 rows per vector block");

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    // ND 路径 gs1 行序与 query 布局相关: BNSD/NTD 为 GS1，BSH/BSND/TND 为 S1G。
    static constexpr bool IS_GS1 = (layout == LayOutTypeEnum::LAYOUT_BNSD) || (layout == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr MaskFormat MASK_LAYOUT = IS_GS1 ? MaskFormat::GS : MaskFormat::SG;

    static constexpr bool USE_DN = useDn;
    static constexpr UbFormat OUT_UB_FORMAT = USE_DN ? UbFormat::S1_ONLY : GetOutUbFormat<layout>();
    using OutGmCoord = std::conditional_t<USE_DN, GmCoordS1Only, GmCoordGs1Merge>;

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar_ = *((uint32_t*)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;

    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = std::conditional_t<HAS_V_TAIL, ConstInfoVTail_t, ConstInfo_t>;
    using RunInfoX = std::conditional_t<HAS_V_TAIL, RunInfoVTailX, ::RunInfoX>;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // 静态Tensor尺寸与TPipe版本一致
    static constexpr uint32_t L1_P_BUFCNT = (HAS_V_TAIL && !USE_3_P_SLOTS) ? 4U : 3U;
    static constexpr uint32_t L1_P_BUF_BYTES =
        mBaseSize * s2BaseSize * sizeof(INPUT_T) + mBaseSize * s2BaseSize / MXFP_GROUP_SIZE +
        (HAS_V_TAIL ? mBaseSize * MXFP8_V_TAIL_BOUNDARY * sizeof(bfloat16_t) : 0U);
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = (mBaseSize / CV_RATIO) * s2BaseSize * sizeof(T) / 2U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES =
        (mBaseSize / CV_RATIO) * static_cast<uint32_t>(dVTemplateType) * sizeof(T);
    static constexpr uint32_t UB_SOFTMAX_BUF_BYTES = 256U;
    static constexpr uint32_t UB_COMMON_BUF_BYTES = 512U;
    static constexpr uint32_t UB_BRDCST_BUF_BYTES = 2048U;
    static constexpr uint32_t UB_LSE_BUF_BYTES = (mBaseSize >> 1U) * sizeof(float) * 8U;
    static constexpr uint32_t UB_STAGE1_OUT_BYTES = 16896U;
    static constexpr uint32_t UB_STAGE1_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_STAGE2_OUT_ELEM_CNT = 64U * dTemplateAlign64;
    static constexpr uint32_t UB_MASK_BUF_BYTES = mBaseSize / CV_RATIO * (s2BaseSize >> 1U);

    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CC_MM1_READY_0 = 0U;
    static constexpr uint32_t CC_MM1_READY_1 = 1U;
    static constexpr uint32_t CC_MM2_READY = 2U;
    static constexpr uint32_t CC_L1P_0 = 4U;
    static constexpr uint32_t CC_MM1_FREE_0 = 8U;
    static constexpr uint32_t CC_MM1_FREE_1 = 9U;
    static constexpr uint32_t CC_MM2_FREE = 10U;
    static_assert(CC_L1P_0 + L1_P_BUFCNT <= CC_MM1_FREE_0, "L1 P cross-core flag ids overlap UB free flag ids");

    // VECOUT队列展开为显式V<->MTE3事件; stage2 0-1, stage1 2-3, lse 4, broadcast 5-6
    static constexpr uint32_t MTE3_V_EVENT0 = EVENT_ID0;
    static constexpr uint32_t MTE3_V_EVENT1 = EVENT_ID1;
    static constexpr uint32_t MTE3_V_EVENT2 = EVENT_ID2;
    static constexpr uint32_t MTE3_V_EVENT3 = EVENT_ID3;
    static constexpr uint32_t MTE3_V_EVENT4 = EVENT_ID4;
    static constexpr uint32_t MTE3_V_EVENT5 = EVENT_ID5;
    static constexpr uint32_t MTE3_V_EVENT6 = EVENT_ID6;
    static constexpr uint32_t V_MTE3_EVENT0 = EVENT_ID0;
    static constexpr uint32_t V_MTE3_EVENT1 = EVENT_ID1;
    static constexpr uint32_t V_MTE3_EVENT2 = EVENT_ID2;
    static constexpr uint32_t V_MTE3_EVENT3 = EVENT_ID3;
    static constexpr uint32_t V_MTE3_EVENT4 = EVENT_ID4;
    static constexpr uint32_t V_MTE3_EVENT5 = EVENT_ID5;
    static constexpr uint32_t V_MTE3_EVENT6 = EVENT_ID6;
    static constexpr uint32_t V_MTE2_EVENT0 = EVENT_ID0; // attenMask 消费完成->MTE2可覆盖
    static constexpr uint32_t MTE2_V_EVENT0 = EVENT_ID0; // attenMask GM->UB 完成->V可见

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<half> attentionOutInitGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<float> sinkGm_;

    GlobalTensor<int32_t> actualSeqLengthsGmQ_;
    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND), ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType* qActSeqLensParser_ = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType& qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    attenMaskGmType attenMaskGmInt_;

    postQuantGmType postQuantScaleGm_;
    postQuantGmType postQuantOffsetGm_;
    postQuantBf16GmType postQuantScaleBf16Gm_;
    postQuantBf16GmType postQuantOffsetBf16Gm_;
    quantGmType pScaleGm_;

    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // 跨核buffer与vec业务buffer均使用显式地址
    LocalTensor<uint8_t> ubBmm2ResBuf_;
    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> l1PBuffers_;
    uint32_t bmm1ResBufId_ = 0U;
    uint32_t l1PBufId_ = 0U;

    // sink 前置注入专用暂存区(1056B): [0,512) GM 原值, [512,1024) 逐行展开值,
    // [1024,1056) 预计算的 ln(pScale)。
    // 不能复用 commonBuf_: 它同时作为 VF scratch(apiTmpBuffer), 跨 pipe 的异步 VF
    // 会覆盖 sink 暂存内容, 表现为部分 task 的 sink 初值变成残留值(LSE 落在垃圾值)。
    LocalTensor<float> sinkStageBuf_;
    bool sinkLnReady_{false};

    LocalTensor<uint8_t> commonBuf_;
    LocalTensor<INPUT_T> stage1OutBuf_[UB_STAGE1_OUT_BUFCNT];
    LocalTensor<uint8_t> attenMaskInBuf_;
    LocalTensor<T> stage2OutBuf_;
    LocalTensor<float> softmaxMaxBuf_[PRELOAD_N + 1];
    LocalTensor<float> softmaxSumBuf_[PRELOAD_N + 1];
    LocalTensor<T> softmaxExpBuf_[PRELOAD_N + 1];
    LocalTensor<float> preLoopMaxBuf_;
    LocalTensor<float> preLoopSumBuf_;
    LocalTensor<float> firstLoopSumBuf_;
    LocalTensor<uint8_t> vselrIndexesBuf_[4];
    LocalTensor<float> maxBrdcst_;
    LocalTensor<float> sumBrdcst_;
    LocalTensor<float> softmaxLseBuf_;

    const ConstInfoX& constInfo_;
    T negativeFloatScalar_;
    float pScaleValue_{1.0f};
    bool isSkipMask_{false};
    bool isFullMask_{false};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32_LN2};
    // sink 组值常驻缓存: 每个 task(n2) 只做一次 GM->UB, 后续 M 块按窗口索引, 不再重复拷贝。
    bool sinkGroupCached_{false};
    int32_t sinkGroupKey_{-1};
    // DN 路径 (max0, sum0) 模板: 只依赖 sink 标量与 owner, task 内不变, 每 M 块仅拷贝。
    bool sinkTplReady_{false};
    int32_t sinkTplKey_{-1};
    bool sinkTplOwner_{false};

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecMxfp8(ConstInfoX& constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* sinks, __gm__ uint8_t* attenMask,
                                        __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut,
                                        __gm__ uint8_t* workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32_LN2;
        this->negativeFloatScalar_ = *((T*)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, sinks, attenMask, softmaxLse, attentionOut,
                     workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t* actualSeqQlenAddr, __gm__ uint8_t* actualSeqKvlenAddr,
                                        __gm__ uint8_t* pScale, __gm__ uint8_t* sinks, __gm__ uint8_t* attenMask,
                                        __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut,
                                        __gm__ uint8_t* workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T*)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float*)softmaxLse);
        }

        uint64_t actualLenQSize =
            (layout == LayOutTypeEnum::LAYOUT_TND) ? constInfo_.cuSeqLensQSize : constInfo_.seqUsedQSize;
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t*)actualSeqQlenAddr, actualLenQSize);

        if constexpr (HAS_MASK) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }
        if constexpr (isFp8) {
            if (pScale != nullptr) {
                pScaleGm_.SetGlobalBuffer((__gm__ float*)pScale);
                pScaleValue_ = this->pScaleGm_.GetValue(0);
            }
        }
        if constexpr (HAS_SINK) {
            sinkGm_.SetGlobalBuffer((__gm__ float*)sinks);
        }

        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                accumOutGm_.SetGlobalBuffer((__gm__ float*)workspace);
                softmaxFDSumGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize);
                softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float*)workspace + constInfo_.accumOutSize +
                                                constInfo_.logSumExpSize);
            }
        }
    }

    // task 级 sink 组值准备(ND): 每个 n2 的 G 个值只从 GM 读一次, 后续 M 块复用。
    // 返回 true 表示本次发起了 MTE2 拷贝(调用方随后取 V 锁时天然等待拷贝完成)。
    __aicore__ inline bool EnsureSinkGroup(const RunInfoX& runInfo)
    {
        int32_t key = static_cast<int32_t>(runInfo.n2Idx);
        if (sinkGroupCached_ && sinkGroupKey_ == key) {
            return false;
        }
        Mutex::Lock<PIPE_MTE2>(SYNC_SINKS_BUF_FLAG);
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = constInfo_.gSize * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = true;
        padParams.paddingValue = 0.0f;
        DataCopyPad(this->sinkStageBuf_, sinkGm_[runInfo.n2Idx * constInfo_.gSize], copyParams, padParams);
        Mutex::Unlock<PIPE_MTE2>(SYNC_SINKS_BUF_FLAG);
        sinkGroupCached_ = true;
        sinkGroupKey_ = key;
        return true;
    }

    // 前置(prologue)注入: 首个 S2 块前把 sink 折叠进 online-softmax 初值
    // (网格化 m0 = ceil(s/ln2)*ln2 - ln(pScale), sum0 = exp(s - m0) ∈ (0.5, 1],
    // 仅 owner 分片计入分母; 其余分片 sum0 = 0; 实际行, 其余行保持 (0, -inf))。
    // 首块 subLoop==0 走 Update 变体读该初值合并, sink 项经在线重缩放恒等于
    // exp(sink - max)。全路径(主路径/V-tail)统一前置, 无收尾注入。
    __aicore__ inline void ResetSoftmaxBuffer(const RunInfoX& runInfo)
    {
        constexpr uint32_t softmaxElemCount = 64;
        uint32_t slot = runInfo.mloop % (PRELOAD_N + 1);
        LocalTensor<float> sumUb = this->softmaxSumBuf_[slot];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[slot];
        // sink 初值行数必须与 softmax VF / 输出实际消费的行数一致:
        //   DN : actVecMSize 即"有效行对半切", 与 sub-block 输出行数一致, 直接使用;
        //   ND : VF 以 actMSizeAlign32/2 行为窗口, sub-block 输出行数 =
        //        min(actMSizeAlign32/2, actMSize - vecMbaseIdx);
        //        不可沿用 actVecMSize(=(actMSize+1)/2), 否则 actMSize 为奇数时
        //        后半有效行保留 (0, -inf) 默认初值, sink 静默丢失。
        uint32_t sinkRows = runInfo.actVecMSize;
        if constexpr (!USE_DN) {
            uint32_t alignedRows = runInfo.actMSizeAlign32 >> 1;
            uint32_t remainRows =
                runInfo.actMSize > runInfo.vecMbaseIdx ? (runInfo.actMSize - runInfo.vecMbaseIdx) : 0U;
            sinkRows = alignedRows < remainRows ? alignedRows : remainRows;
        }
        // FD 分片与普通 task 走同一套前置注入:
        //   所有分片共享 sink 网格 max m0(P 量化域一致), sink 的分母项 exp(sink-m0)
        //   仅由承接 task 起始 S2 的分片(isSinkOwner)计入, 其余分片 sum0=0; FD 合并按
        //   max 归一后 sink 项恰好出现一次, 不会随分片数重复累加。
        if (sinkRows == 0) {
            Duplicate(sumUb, 0.0f, softmaxElemCount);
            Duplicate(maxUb, negativeFloatScalar_, softmaxElemCount);
            PipeBarrier<PIPE_V>();
            return;
        }
        // sink prologue 只覆盖 [0, sinkRows), 其余 lane 保持 (0, -inf) 默认初值;
        // sinkRows == 64 时无需任何默认初始化, 省掉两条全宽 Duplicate。
        // Duplicate 目的地址必须 32B 对齐: 起点下取整到 8 个 float, 覆盖区间
        // [initStart, 64) 包含 [initStart, sinkRows), 这些行随后被 prologue 重写, 结果不变。
        if (sinkRows < softmaxElemCount) {
            const uint32_t initStart = sinkRows & ~7U;
            const uint32_t tailRows = softmaxElemCount - initStart;
            Duplicate(sumUb[initStart], 0.0f, tailRows);
            Duplicate(maxUb[initStart], negativeFloatScalar_, tailRows);
        }
        LocalTensor<float> sinkTmpUb = this->sinkStageBuf_;
        LocalTensor<float> sinkExpandedUb = sinkTmpUb[SINK_STAGE_TMP_BYTES / sizeof(float)];
        LocalTensor<float> sinkLnUb = sinkTmpUb[(SINK_STAGE_TMP_BYTES + SINK_STAGE_EXPANDED_BYTES) / sizeof(float)];
        // ln(pScale) 在 kernel 内只算一次; 后续 M 块直接 BRC 广播, 省掉每块一次全 lane Ln。
        if (!sinkLnReady_) {
            AttentionCommon::SinkLnScaleVF<float>((__ubuf__ float*)sinkLnUb.GetPhyAddr(), pScaleValue_);
            sinkLnReady_ = true;
        }
        if constexpr (USE_DN) {
            // DN: (max0, sum0) 只依赖 sink 标量与 owner, task 内不变。
            // 模板缓存在暂存区低 512B, 之后每 M 块仅两次向量拷贝, 不再重复 Truncate/Select/Exp。
            LocalTensor<float> maxTplUb = sinkTmpUb;
            LocalTensor<float> sumTplUb = sinkTmpUb[softmaxElemCount];
            int32_t key = static_cast<int32_t>(runInfo.realN2Idx);
            if (!sinkTplReady_ || sinkTplKey_ != key || sinkTplOwner_ != runInfo.isSinkOwner) {
                Duplicate(sinkTmpUb, static_cast<float>(sinkGm_.GetValue(runInfo.realN2Idx)), 1U);
                AttentionCommon::SinkBrcVF<float, float, softmaxElemCount>((__ubuf__ float*)sinkExpandedUb.GetPhyAddr(),
                                                                           (__ubuf__ float*)sinkTmpUb.GetPhyAddr(),
                                                                           softmaxElemCount);
                AttentionCommon::SinkPrologueGridVF<float, float>(
                    (__ubuf__ float*)sumTplUb.GetPhyAddr(), (__ubuf__ float*)maxTplUb.GetPhyAddr(),
                    (__ubuf__ float*)sinkExpandedUb.GetPhyAddr(), (__ubuf__ float*)sinkLnUb.GetPhyAddr(),
                    softmaxElemCount, runInfo.isSinkOwner);
                sinkTplReady_ = true;
                sinkTplKey_ = key;
                sinkTplOwner_ = runInfo.isSinkOwner;
            }
            Muls(maxUb, maxTplUb, 1.0f, sinkRows);
            Muls(sumUb, sumTplUb, 1.0f, sinkRows);
            // sink VF 的 store 必须对紧随其后的首块 Update VF load 可见。
            PipeBarrier<PIPE_V>();
            return;
        }
        // ND: 每个 n2 的组值常驻 UB, M 块按窗口 Gather, 不再逐窗口 GM 拷贝。
        EnsureSinkGroup(runInfo);
        // V 锁同时保证: (1) 首次拷贝已完成; (2) 下一个 task 的 MTE2 拷贝不会覆盖仍在读取的组值。
        Mutex::Lock<PIPE_V>(SYNC_SINKS_BUF_FLAG);
        if (constInfo_.gSize == 1U) {
            AttentionCommon::SinkBrcVF<float, float, softmaxElemCount>(
                (__ubuf__ float*)sinkExpandedUb.GetPhyAddr(), (__ubuf__ float*)sinkTmpUb.GetPhyAddr(), sinkRows);
        } else if constexpr (IS_GS1) {
            // 常驻组值布局: 直接用绝对 gs1 下标, idx=(gs1Start+r)/actS1Size ∈ [0,G)。
            AttentionCommon::SinkGs1DirectVF<float, float, softmaxElemCount>(
                (__ubuf__ float*)sinkExpandedUb.GetPhyAddr(), (__ubuf__ float*)sinkTmpUb.GetPhyAddr(),
                runInfo.gS1Idx + runInfo.vecMbaseIdx, runInfo.actS1Size, sinkRows);
        } else {
            // 头内/跨头/回绕统一走 mod 索引: idx=(gStart+r) mod G。
            AttentionCommon::SinkRotDirectVF<float, float, softmaxElemCount>(
                (__ubuf__ float*)sinkExpandedUb.GetPhyAddr(), (__ubuf__ float*)sinkTmpUb.GetPhyAddr(),
                (runInfo.gS1Idx + runInfo.vecMbaseIdx) % constInfo_.gSize, constInfo_.gSize, sinkRows);
        }
        AttentionCommon::SinkPrologueGridVF<float, float>(
            (__ubuf__ float*)sumUb.GetPhyAddr(), (__ubuf__ float*)maxUb.GetPhyAddr(),
            (__ubuf__ float*)sinkExpandedUb.GetPhyAddr(), (__ubuf__ float*)sinkLnUb.GetPhyAddr(), sinkRows,
            runInfo.isSinkOwner);
        // 对齐 flash_attn 参考实现: 屏障置于 Reset 完成之后。sink VF 对 (max,sum)
        // 的 store 必须对紧随其后的首块 Update VF load 可见, 否则部分 tile 会读到
        // Duplicate 的默认初值, sink 静默丢失。
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(SYNC_SINKS_BUF_FLAG);
    }

    // hasSink 为编译期开关(tiling key): sink 只通过 ResetSoftmaxBuffer 把
    // (max0,sum0) 前置注入 online-softmax 初值, P 的量化流程与原协议完全一致
    // (running max 域直接 cast e4m3)。
    // 有 sink: 首个 S2 子块改走 Update 变体消费该初值, 其余子块仍为 NoUpdate
    // (Cube 侧按子块 MXFP8 scale 合并, 不能重复并入 running state)。
    // 无 sink: 首个 S2 tile 的子块全部保持 NoUpdate, 与改动前逐位一致。
    template <bool hasAttenT, uint16_t ubNT, typename... A>
    __aicore__ inline void DnMxfp8Dispatch(bool isFirstSubLoop, A&&... a)
    {
        if constexpr (HAS_SINK) {
            // 首个 S2 子块消费 prologue 状态；同一 S2 tile 的后续子块仍是独立
            // NoUpdate，Cube 侧会按各子块的 MXFP8 scale 合并，不能把子块再累加一次。
            if (isFirstSubLoop) {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, hasAttenT, ubNT>(a...);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, hasAttenT, ubNT>(a...);
            }
        } else {
            // 无 sink: 保持原路径，首个 S2 tile 的子块均为 NoUpdate。
            FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, hasAttenT, ubNT>(a...);
        }
    }

    template <bool hasAttenT, uint16_t ubNT, typename... A>
    __aicore__ inline void DnMxfp8UpdateDispatch(A&&... a)
    {
        FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, hasAttenT, ubNT>(a...);
    }

    // hasSink 由 tiling key 编译期决定: 仅首个 S2 子块改用 Update 变体消费
    // prologue 注入的 (max0,sum0), P 的量化流程不变。
    template <typename T2_, typename pseT, uint32_t mB, uint32_t s2B, OriginNRange rng, bool hasMaskT, typename... A>
    __aicore__ inline void NdMxfp8Dispatch(bool isFirstSubLoop, A&&... a)
    {
        if constexpr (HAS_SINK) {
            // 只有首个 S2 子块消费 prologue 的 (sum0,max0)；后续子块保持
            // NoUpdate，避免把同一 tile 的两个子块重复并入 running state。
            if (isFirstSubLoop) {
                ProcessVec1VfMxfp8<T, T2_, pseT, true, mB, s2B, rng, hasMaskT, PseTypeEnum::PSE_NONE_TYPE, false, false,
                                   false, false, false>(a...);
            } else {
                ProcessVec1VfMxfp8<T, T2_, pseT, false, mB, s2B, rng, hasMaskT, PseTypeEnum::PSE_NONE_TYPE, false,
                                   false, false, false, false>(a...);
            }
        } else {
            // 无 sink: 保持原路径，首个 S2 tile 的子块均为 NoUpdate。
            ProcessVec1VfMxfp8<T, T2_, pseT, false, mB, s2B, rng, hasMaskT, PseTypeEnum::PSE_NONE_TYPE, false, false,
                               false, false, false>(a...);
        }
    }

    template <typename T2_, typename pseT, uint32_t mB, uint32_t s2B, OriginNRange rng, bool hasMaskT, typename... A>
    __aicore__ inline void NdMxfp8UpdateDispatch(A&&... a)
    {
        ProcessVec1VfMxfp8<T, T2_, pseT, true, mB, s2B, rng, hasMaskT, PseTypeEnum::PSE_NONE_TYPE, false, false, false,
                           false, false>(a...);
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_FREE);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_1);
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void ProcessVec1(RunInfoX runInfo, uint32_t subLoop)
    {
        uint32_t bmm1ResBufId = bmm1ResBufId_;
        bmm1ResBufId_ = (bmm1ResBufId_ + 1U) % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = l1PBufId_;
        if (subLoop % 2 == 0) {
            l1PBufId_ = (l1PBufId_ + 1U) % L1_P_BUFCNT;
        } else {
            pL1BufId = (l1PBufId_ + L1_P_BUFCNT - 1U) % L1_P_BUFCNT;
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_READY_0 + bmm1ResBufId);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0 + bmm1ResBufId);
            if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
            }
            return;
        }

        if constexpr (HAS_V_TAIL) {
            if (unlikely(runInfo.isVtailTask)) {
                if constexpr (USE_DN) {
                    ProcessVec1DnTail(pL1BufId, bmm1ResBufId, runInfo, subLoop);
                    return;
                } else if constexpr ((s2BaseSize >> 1) >= 128) {
                    ProcessVec1NdTail(pL1BufId, bmm1ResBufId, runInfo, subLoop);
                    return;
                }
            }
        }

        if constexpr (USE_DN) {
            ProcessVec1Dn(pL1BufId, bmm1ResBufId, runInfo, subLoop);
        } else {
            ProcessVec1Nd(pL1BufId, bmm1ResBufId, runInfo, subLoop);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        if constexpr (POST_QUANT) {
            totalOutputSize /= 2;
        }
        uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum; // 2 means c:v = 1:2
        // FA 计算前可复用跨核 BMM 区清零输出
        static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0U;
        static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = 32768U / sizeof(OUT_T);
        if constexpr (IsSameType<OUT_T, int8_t>::value) {
            GlobalTensor<half> attentionOutTmpGm;
            attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(attentionOutGm_.GetPhyAddr(0)));
            AttentionCommon::InitOutput<half, MTE3_V_EVENT0, ATTEN_OUT_POP_BUF_START_ADDR, 32768U / sizeof(half),
                                        false>(attentionOutTmpGm, totalOutputSize, vecCoreNum, 0);
        } else {
            AttentionCommon::InitOutput<OUT_T, MTE3_V_EVENT0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE,
                                        false>(attentionOutGm_, totalOutputSize, vecCoreNum, 0);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum; // 2 means c:v = 1:2
        static constexpr float LSE_INIT_VAL = 3e+99;
        static constexpr uint32_t LSE_POP_BUF_START_ADDR = 32768U;
        static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = 32768U / sizeof(float);
        AttentionCommon::InitOutput<float, MTE3_V_EVENT1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, false>(
            softmaxLseGm_, totalOutputSize, vecCoreNum, LSE_INIT_VAL);
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(uint32_t pL1BufId, uint32_t bmm1ResBufId, RunInfoX runInfo, uint32_t subLoop)
    {
        LocalTensor<uint8_t> outputBuf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        // 所有 S2 分片共享 sink 网格 max m0(保证各分片状态/P 量化域一致);
        // 但 sink 的分母项 exp(sink-m0) 只由 owner 分片(isSinkOwner)计入, FD 合并不会重复。
        if constexpr (HAS_SINK) {
            // 无 sink 保持改动前路径: 首块走 NoUpdate 变体直接覆盖, 不做任何 reset
            if (unlikely(runInfo.isFirstS2Loop) && subLoop == 0) {
                ResetSoftmaxBuffer(runInfo);
            }
        }
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo, subLoop);
        }
        if (unlikely(isFullMask_)) {
            maskLine = -256;
        }

        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = subLoop % DB;

        LocalTensor<float> sumUb = this->softmaxSumBuf_[softmaxBufIdx];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[softmaxBufIdx];
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_;
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_;
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_;
        LocalTensor<float> expUb = this->softmaxExpBuf_[expBufIdx];

        LocalTensor<T> mmRes = ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutBuf_[stage1Offset];
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        float descaleQK = 1.0;
        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }

        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(!isSkipMask_)) {
                DnMxfp8Dispatch<hasAtten, s2BaseSizeCur>(
                    (subLoop == 0), stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            } else {
                DnMxfp8Dispatch<false, s2BaseSizeCur>(
                    (subLoop == 0), stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            }
        } else {
            if (unlikely(!isSkipMask_)) {
                DnMxfp8UpdateDispatch<hasAtten, s2BaseSizeCur>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            } else {
                DnMxfp8UpdateDispatch<false, s2BaseSizeCur>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign / 2, s2CalcSize,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0 + bmm1ResBufId);

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        //-------------------------Data copy to l1-------------------------
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.ReinterpretCast<fp8_e8m0_t>()[pScaleL1Offset];
        constexpr uint64_t pScaleDataLen = (mBaseSize >> 1) * s2BaseSizeCur / MXFP_GROUP_SIZE;
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        uint64_t vecOffset = constInfo_.subBlockIdx * pScaleDataLen;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 2;
        const bool isSplitS2 = runInfo.actSingleLoopS2Size > s2SplitSize;
        // MXFP8 DN protocol: subLoop-0 starts with scale=1. After subLoop-1
        // determines the final running max, its pScale rescales subLoop-0.
        // Therefore copy only subLoop-1 back to the first K-half scale slot.
        if (isSplitS2 && (subLoop % 2 == 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor, {4, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.ReinterpretCast<INPUT_T>();
        int64_t subLoopOffset = s2BaseSizeCur * mBaseSize * subLoop;
        constexpr uint16_t elementSize = 32;
        constexpr uint32_t singleProcessSOuterSize = mBaseSize >> 1;
        constexpr uint32_t actCopyCount = (singleProcessSOuterSize + elementSize - 1) / elementSize;
        constexpr uint32_t actVec0Align32 = mBaseSize >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.actSingleLoopS2Size + 63) >> 6) << 6);
        for (uint32_t i = 0; i < actCopyCount; i++) {
            uint32_t dstOffset = 32 * s2BaseSizeCur * i + subLoopOffset;
            if (constInfo_.subBlockIdx == 1) {
                dstOffset += s2BaseSizeCur * actVec0Align32;
            }
            uint32_t srcOffset = i * (((s2BaseSizeCur >> 2) + 1) << 5);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor[srcOffset],
                     {4, s2BaseSizeCur / 4, s2BaseSizeCur / 4 + 2, 0});
        }
        //-----------------------------------------------------------------
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    __aicore__ inline void ProcessVec1DnTail(uint32_t pL1BufId, uint32_t bmm1ResBufId, RunInfoX runInfo,
                                             uint32_t subLoop)
    {
        LocalTensor<uint8_t> outputBuf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        // 所有 S2 分片共享 sink 网格 max m0(保证各分片状态/P 量化域一致);
        // 但 sink 的分母项 exp(sink-m0) 只由 owner 分片(isSinkOwner)计入, FD 合并不会重复。
        if constexpr (HAS_SINK) {
            // 无 sink 保持改动前路径: 首块走 NoUpdate 变体直接覆盖, 不做任何 reset
            if (unlikely(runInfo.isFirstS2Loop) && subLoop == 0) {
                ResetSoftmaxBuffer(runInfo);
            }
        }
        int64_t maskLine = 0;
        if constexpr (HAS_MASK) {
            maskLine = ComputeMaskLineDN(runInfo, subLoop);
        }
        if (unlikely(isFullMask_)) {
            maskLine = -256;
        }

        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = subLoop % DB;

        LocalTensor<float> sumUb = this->softmaxSumBuf_[softmaxBufIdx];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[softmaxBufIdx];
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_;
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_;
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_;
        LocalTensor<float> expUb = this->softmaxExpBuf_[expBufIdx];

        LocalTensor<T> mmRes = ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutBuf_[stage1Offset];
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        float descaleQK = 1.0;
        if constexpr (HAS_V_TAIL) {
            LocalTensor<bfloat16_t> pTailBf16Ub = stage1CastTensor.template ReinterpretCast<bfloat16_t>();

            // Match the score row pitch written by FixpipeMm1DnTail.
            uint16_t tailM = (uint16_t)(runInfo.actMSizeAlign32 >> 1);
            bool tailNeedAtten = false;
            if constexpr (HAS_MASK) {
                tailNeedAtten = !isSkipMask_;
            }
            // sink: 首块恒 Update 消费 prologue 注入的 (max0, sum0);
            // 无 sink: 与改动前一致, 首块 NoUpdate 直接覆盖。
            if constexpr (HAS_SINK) {
                FaVectorApi::ProcessVec1VfDnTailMxfp8<T, bfloat16_t, true>(
                    pTailBf16Ub, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, tailM, runInfo.kvTailLen,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, pScaleValue_, tailNeedAtten,
                    maskLine);
            } else if (unlikely(runInfo.isFirstS2Loop)) {
                FaVectorApi::ProcessVec1VfDnTailMxfp8<T, bfloat16_t, false>(
                    pTailBf16Ub, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, tailM, runInfo.kvTailLen,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, pScaleValue_, tailNeedAtten,
                    maskLine);
            } else {
                FaVectorApi::ProcessVec1VfDnTailMxfp8<T, bfloat16_t, true>(
                    pTailBf16Ub, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], pScaleSubLoop0Tensor,
                    preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, tailM, runInfo.kvTailLen,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, pScaleValue_, tailNeedAtten,
                    maskLine);
            }
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0 + bmm1ResBufId);
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
            // Copy BF16 P to the L1 tail region; C2 consumes 64 columns.
            {
                constexpr uint64_t pTailL1Offset =
                    (static_cast<uint64_t>(mBaseSize) * s2BaseSize + mBaseSize * s2BaseSize / MXFP_GROUP_SIZE) /
                    sizeof(bfloat16_t);
                LocalTensor<bfloat16_t> mm2ATailL1Tensor = outputBuf.ReinterpretCast<bfloat16_t>()[pTailL1Offset];
                if (likely(runInfo.actVecMSize != 0)) {
                    uint64_t tailDstOffset = static_cast<uint64_t>(constInfo_.subBlockIdx) * (mBaseSize >> 1) * 64U;
                    DataCopy(mm2ATailL1Tensor[tailDstOffset], pTailBf16Ub, {(uint16_t)(mBaseSize >> 5), 64U, 1U, 0U});
                }
            }
            SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
            if (runInfo.actSingleLoopS2Size <= s2SplitSize) {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
            }
            if (unlikely(runInfo.isLastS2Loop)) {
                SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
            }
            return;
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float>& sumUb, LocalTensor<float>& maxUb)
    {
        // sink 全路径前置注入(含 V-tail: DnTail 首块已恒 Update 消费初值), 无收尾注入。
        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                if (runInfo.isS2SplitCore) {
                    ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
                }
            }
        }

        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                    SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
                }
            } else {
                if (constInfo_.isSoftmaxLseEnable) {
                    SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
                }
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void UpdateMinCheckValue()
    {
        LocalTensor<float> minValueUb = commonBuf_.ReinterpretCast<float>();
        constexpr uint32_t minCheckEventId = EVENT_ID0;
        UpdateMinCheckValueVF<USE_DN>((__ubuf__ float*)minValueUb.GetPhyAddr(), *((float*)&minValue_),
                                      constInfo_.scaleValue, pScaleValue_);
        SetFlag<HardEvent::V_S>(minCheckEventId);
        WaitFlag<HardEvent::V_S>(minCheckEventId);
        float min = minValueUb.GetValue(0);
        minValue_ = static_cast<uint32_t>(*reinterpret_cast<int32_t*>(&min));
    }

    // sink 场景下, 无有效 key 的行(如 LU-causal 下 q 侧前段超出 kv 的行)在本 tile 内
    // 同样会拿到 sink 初值 (m0, sum0), 直接算 LSE 会得到有限值, 而 golden 该行为 inf 哨兵。
    // 判据与 RowInvalid 完全一致: 把这些行的 max 槽位置为 minValue_, 使
    // ComputeLseOutputVF 走 inf 分支; Vec2 的 RowInvalidUpdateVF 按 max==minValue_
    // 判据读同一缓冲区, 因此 O 也会被正确清 0。
    // 无效行只可能由 mask 产生, 故与 Vec2 的 RowInvalid 一致按 HAS_MASK 编译期门控,
    // 无 mask 时不进入该路径(对齐 FP8/HIF8 实现)。
    __aicore__ inline void MarkInvalidRowMaxForSink(LocalTensor<float>& maxUb, const RunInfoX& runInfo, uint32_t rows)
    {
        if constexpr (HAS_SINK && HAS_MASK) {
            if (rows == 0 || runInfo.actS1Size == 0) {
                return;
            }
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            if (s1FirstValidToken <= 0 && s1LastValidToken >= static_cast<int64_t>(runInfo.actS1Size) - 1) {
                return; // 全有效: 热路径不付出代价
            }
            // 行 -> s1 映射与 CalcBlockNeedRowInvalid 保持一致:
            //   S1G 行序为 s1*realGSize+g, GS1 行序为 s1 在 G 内循环(s1 = gs1 % actS1Size)。
            const uint32_t gs1Start = runInfo.gS1Idx + runInfo.vecMbaseIdx;
            const uint32_t realGSize = constInfo_.realGSize == 0 ? 1U : constInfo_.realGSize;
            // O(1) 块级预判(与 flash_mla CalcBlockNeedRowInvalid 同构): 只看区间端点即可
            // 判定本块是否可能含无效行, 整块有效时直接跳过下面的向量标记与两次 V 屏障。
            // 端点不越界 => 块内所有行都有效(不跨 G 时行 -> s1 单调); 跨 G 时端点判断
            // 不充分, 交给向量侧按取模逐行标记。
            {
                const uint32_t gs1Last = gs1Start + rows - 1U;
                if constexpr (IS_GS1) {
                    if (gs1Start / runInfo.actS1Size == gs1Last / runInfo.actS1Size) {
                        // 本块不跨 G: s1 区间为两端点取模
                        const int64_t s1Begin = static_cast<int64_t>(gs1Start % runInfo.actS1Size);
                        const int64_t s1End = static_cast<int64_t>(gs1Last % runInfo.actS1Size);
                        if (s1Begin >= s1FirstValidToken && s1End <= s1LastValidToken) {
                            return;
                        }
                    }
                } else {
                    const int64_t s1Begin = static_cast<int64_t>(gs1Start / realGSize);
                    const int64_t s1End = static_cast<int64_t>(gs1Last / realGSize);
                    if (s1Begin >= s1FirstValidToken && s1End <= s1LastValidToken) {
                        return;
                    }
                }
            }
            // 向量标记: 标量先算好行区间阈值, 向量侧只做 Arange/Div/Compare/Select,
            // 只占 V 流水, 不占标量流水。
            // 与 softmax VF 的 store / 后续 LSE VF 的 load 之间存在 UB store->load RAW 依赖,
            // 按本类既有约定用 V 屏障(而非原来的 PIPE_ALL)保证可见性。
            PipeBarrier<PIPE_V>();
            if constexpr (IS_GS1) {
                // GS1 行序为 g 外层、s1 内层, 一个 vector 块可跨多个 G 组, 行 -> s1 必须
                // 按取模计算(不能假设至多一次回绕), 具体见 MarkInvalidRowMaxGs1VF。
                const int32_t base = static_cast<int32_t>(gs1Start % runInfo.actS1Size);
                AttentionCommon::MarkInvalidRowMaxGs1VF<float, mBaseSize / CV_RATIO>(
                    (__ubuf__ float*)maxUb.GetPhyAddr(), rows, base, static_cast<int32_t>(runInfo.actS1Size),
                    static_cast<int32_t>(s1FirstValidToken), static_cast<int32_t>(s1LastValidToken),
                    *((float*)&minValue_));
            } else {
                const int64_t loThr =
                    s1FirstValidToken * static_cast<int64_t>(realGSize) - static_cast<int64_t>(gs1Start);
                const int64_t hiThr =
                    (s1LastValidToken + 1) * static_cast<int64_t>(realGSize) - static_cast<int64_t>(gs1Start);
                const int32_t loThrI32 = AttentionCommon::SinkClampToI32(loThr);
                const int32_t hiThrI32 = AttentionCommon::SinkClampToI32(hiThr);
                AttentionCommon::MarkInvalidRowMaxS1GVF<float, mBaseSize / CV_RATIO>(
                    (__ubuf__ float*)maxUb.GetPhyAddr(), rows, loThrI32, hiThrI32, *((float*)&minValue_));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumTmp, LocalTensor<float>& softmaxMaxTmp,
                                             RunInfoX& runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = USE_DN ? runInfo.actVecMSize : (runInfo.actMSizeAlign32 / 2);
        uint32_t gmDealRowCount;
        if constexpr (USE_DN) {
            gmDealRowCount = runInfo.actVecMSize;
        } else {
            uint32_t groupsOf32 = (runInfo.actMSize + 31) / 32;
            if (constInfo_.subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > runInfo.actMSize ? runInfo.actMSize : groupsOf32 * 16;
            } else {
                int32_t vec1RemainRows = runInfo.actMSize - 16 * groupsOf32;
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
        }
        if (gmDealRowCount == 0) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseBuf_;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);
        MarkInvalidRowMaxForSink(softmaxMaxTmp, runInfo, vecMSize);
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT4);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT4);

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            // LSE 输出改为 N-major 排布 [N2*G, T]: N 在外, T 在内
            // bN2Offset = n2Idx * G * T_total + prefixBS1, 内部按 gIdx * T_total + s1Idx 步进
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            if constexpr (USE_DN) {
                DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                         gmDealRowCount, constInfo_);
            } else {
                DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                               constInfo_);
            }
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(runInfo.bIdx);
            uint32_t s1Size = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo_.n2Size * constInfo_.gSize + runInfo.n2Idx * constInfo_.gSize;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                       constInfo_, s1Size);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            // DN 范式下任务按 realN2Idx 逐 head 分解(realGSize=1), 头部偏移须用 realN2Idx*realGSize
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.realN2Idx * constInfo_.realGSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, s1LeftPaddingSize);
        } else { // BNSD
            // DN 范式下任务按 realN2Idx 逐 head 分解(realGSize=1), 头部偏移须用 realN2Idx*realGSize
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.realN2Idx * constInfo_.realGSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo_, qActSeqLens, s1LeftPaddingSize);
        }

        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);
    }

    __aicore__ inline void ProcessVec1Nd(uint32_t pL1BufId, uint32_t bmm1ResBufId, RunInfoX runInfo, uint32_t subLoop)
    {
        LocalTensor<uint8_t> outputBuf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        // 所有 S2 分片共享 sink 网格 max m0; sink 分母项仅 owner 分片(isSinkOwner)计入。
        if constexpr (HAS_SINK) {
            // 无 sink 保持改动前路径: 首块走 NoUpdate 变体直接覆盖, 不做任何 reset
            if (unlikely(runInfo.isFirstS2Loop) && subLoop == 0) {
                ResetSoftmaxBuffer(runInfo);
            }
        }
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        uint32_t actVecMSizeAlign16 = runInfo.actMSizeAlign32 >> 1;

        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInBuf_;
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0);                            // attenMask AllocTensor
            AttenMaskCopyIn(attenMaskUb, 0, actVecMSizeAlign16, runInfo, subLoop); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
        // Loop 0拷贝一次全局, Loop 1用来计算update_mul
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_;
        // 两轮循环前的sum值
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_;
        // Loop 0的sum值传给loop 1用来计算更新的sum值
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_;
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonBuf_;

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutBuf_[stage1Offset];
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }
        if (unlikely(runInfo.isFirstS2Loop)) {
            if (unlikely(s2CalcSize == 128)) {
                NdMxfp8Dispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, EQ_128, HAS_MASK>(
                    (subLoop == 0), stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb,
                    preLoopSumUb, firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize <= 64)) {
                NdMxfp8Dispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, GT_0_AND_LTE_64, HAS_MASK>(
                    (subLoop == 0), stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb,
                    preLoopSumUb, firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize < 128)) {
                NdMxfp8Dispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, GT_64_AND_LTE_128, HAS_MASK>(
                    (subLoop == 0), stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb,
                    preLoopSumUb, firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else {
                if constexpr (s2BaseSizeCur == 256) {
                    NdMxfp8Dispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, GT_128_AND_LTE_256, HAS_MASK>(
                        (subLoop == 0), stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb,
                        maxUb, attenMaskUb, pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb,
                        preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride,
                        slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F,
                        queryScaleUb, deSCaleKValue, pScaleValue_);
                }
            }
        } else {
            if (unlikely(s2CalcSize == 128)) {
                NdMxfp8UpdateDispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, EQ_128, HAS_MASK>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize <= 64)) {
                NdMxfp8UpdateDispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, GT_0_AND_LTE_64, HAS_MASK>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else if (unlikely(s2CalcSize < 128)) {
                NdMxfp8UpdateDispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, GT_64_AND_LTE_128, HAS_MASK>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            } else {
                if constexpr (s2BaseSizeCur == 256) {
                    NdMxfp8UpdateDispatch<INPUT_T, pseShiftType, mBaseSize, s2BaseSizeCur, GT_128_AND_LTE_256,
                                          HAS_MASK>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                        firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift,
                        static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                        deSCaleKValue, pScaleValue_);
                }
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0 + bmm1ResBufId);
        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0); // attenMask FreeTensor
        }

        // ===================DataCopy to L1 ====================
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.ReinterpretCast<fp8_e8m0_t>()[pScaleL1Offset];
        uint64_t pScaleDataLen = actVecMSizeAlign16 * s2BaseSizeCur / MXFP_GROUP_SIZE;
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        uint64_t pScaleSubLoopOffset = pScaleDataLen * 2;
        uint16_t copyCount = actVecMSizeAlign16 / 16;
        uint64_t vecOffset = constInfo_.subBlockIdx * pScaleDataLen;
        uint16_t dstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 2;
        if ((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                // PScale s2方向block块大小为32，共256/32=8个，L1需满足16x2分形，重复拷贝4次
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
                DataCopy(mm2AScaleL1Tensor[pScaleSubLoopOffset + vecOffset + i * 32], pScaleSubLoop0Tensor[128],
                         {copyCount, 1, 0, pScaleDstStride});
            }
        } else if (unlikely(runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = outputBuf.ReinterpretCast<INPUT_T>();
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo_.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * actVecMSizeAlign16 +
                                s2BaseSizeCur * mBaseSize * subLoop;
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2BaseSizeCur / 32, (uint16_t)actVecMSizeAlign16, (uint16_t)(vec1Srcstride - actVecMSizeAlign16),
                      (uint16_t)(mBaseSize - actVecMSizeAlign16)});
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec1NdTail(uint32_t pL1BufId, uint32_t bmm1ResBufId, RunInfoX runInfo,
                                             uint32_t subLoop)
    {
        LocalTensor<uint8_t> outputBuf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        // 所有 S2 分片共享 sink 网格 max m0; sink 分母项仅 owner 分片(isSinkOwner)计入。
        if constexpr (HAS_SINK) {
            // 无 sink 保持改动前路径: 首块走 NoUpdate 变体直接覆盖, 不做任何 reset
            if (unlikely(runInfo.isFirstS2Loop) && subLoop == 0) {
                ResetSoftmaxBuffer(runInfo);
            }
        }
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        uint32_t actVecMSizeAlign16 = runInfo.actMSizeAlign32 >> 1;

        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInBuf_;
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0);                                // attenMask AllocTensor
            AttenMaskCopyInTail(attenMaskUb, 0, actVecMSizeAlign16, runInfo, subLoop); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
        // Loop 0拷贝一次全局, Loop 1用来计算update_mul
        LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf_;
        // 两轮循环前的sum值
        LocalTensor<float> preLoopSumUb = this->preLoopSumBuf_;
        // Loop 0的sum值传给loop 1用来计算更新的sum值
        LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf_;
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonBuf_;

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = ubBmm1ResBuffers_[bmm1ResBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        LocalTensor<INPUT_T> stage1CastTensor = stage1OutBuf_[stage1Offset];
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));
        constexpr int32_t pScaleByteOffset = 16640; // 16640: softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        if constexpr (HAS_V_TAIL) {
            LocalTensor<bfloat16_t> pTailBf16Ub = stage1CastTensor.template ReinterpretCast<bfloat16_t>();
            constexpr int32_t s2TailBaseSize = (s2BaseSizeCur > 128) ? 128 : s2BaseSizeCur;
            if (unlikely(runInfo.isFirstS2Loop)) {
                NdMxfp8Dispatch<bfloat16_t, pseShiftType, mBaseSize, s2TailBaseSize, GT_0_AND_LTE_64, HAS_MASK>(
                    (subLoop == 0), pTailBf16Ub, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb,
                    preLoopSumUb, firstLoopSumUb, subLoop, actVecMSizeAlign16, runInfo.kvTailLen, pseStride, slopes,
                    posShift, static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F,
                    queryScaleUb, deSCaleKValue, pScaleValue_);
            } else {
                NdMxfp8UpdateDispatch<bfloat16_t, pseShiftType, mBaseSize, s2TailBaseSize, GT_0_AND_LTE_64, HAS_MASK>(
                    pTailBf16Ub, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb, pseUb,
                    dropMaskUb, pScaleSubLoop0Tensor, apiTmpBuffer, pScaleUb, preLoopMaxUb, preLoopSumUb,
                    firstLoopSumUb, subLoop, actVecMSizeAlign16, runInfo.kvTailLen, pseStride, slopes, posShift,
                    static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb,
                    deSCaleKValue, pScaleValue_);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_FREE_0 + bmm1ResBufId);
        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0); // attenMask FreeTensor
        }

        // ===================DataCopy to L1 ====================
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT2 + static_cast<uint32_t>(stage1Offset));
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.ReinterpretCast<fp8_e8m0_t>()[pScaleL1Offset];
        uint64_t pScaleDataLen = actVecMSizeAlign16 * s2BaseSizeCur / MXFP_GROUP_SIZE;
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        uint64_t pScaleSubLoopOffset = pScaleDataLen * 2;
        uint16_t copyCount = actVecMSizeAlign16 / 16;
        uint64_t vecOffset = constInfo_.subBlockIdx * pScaleDataLen;
        uint16_t dstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 2;
        if ((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                // PScale s2方向block块大小为32，共256/32=8个，L1需满足16x2分形，重复拷贝4次
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
                DataCopy(mm2AScaleL1Tensor[pScaleSubLoopOffset + vecOffset + i * 32], pScaleSubLoop0Tensor[128],
                         {copyCount, 1, 0, pScaleDstStride});
            }
        } else if (unlikely(runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[vecOffset + i * 32], pScaleSubLoop0Tensor,
                         {copyCount, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = outputBuf.ReinterpretCast<INPUT_T>();
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo_.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * actVecMSizeAlign16 +
                                s2BaseSizeCur * mBaseSize * subLoop;
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2BaseSizeCur / 32, (uint16_t)actVecMSizeAlign16, (uint16_t)(vec1Srcstride - actVecMSizeAlign16),
                      (uint16_t)(mBaseSize - actVecMSizeAlign16)});
        }
        if constexpr (HAS_V_TAIL) {
            if (subLoop == 0 || runInfo.actSingleLoopS2Size <= s2BaseSizeCur) {
                // Copy the first four BF16 column chunks, including zero-padded tail columns.
                constexpr uint64_t pTailL1Offset =
                    (static_cast<uint64_t>(mBaseSize) * s2BaseSize + mBaseSize * s2BaseSize / MXFP_GROUP_SIZE) /
                    sizeof(bfloat16_t);
                LocalTensor<bfloat16_t> mm2ATailL1Tensor = outputBuf.ReinterpretCast<bfloat16_t>()[pTailL1Offset];
                // Subblock 1 follows the actual aligned rows of subblock 0.
                uint64_t dstOffset = static_cast<uint64_t>(constInfo_.subBlockIdx) * actVecMSizeAlign16 * 16U;
                if (likely(runInfo.actVecMSize != 0)) {
                    LocalTensor<bfloat16_t> pTailBf16Ub = stage1CastTensor.template ReinterpretCast<bfloat16_t>();
                    DataCopy(
                        mm2ATailL1Tensor[dstOffset], pTailBf16Ub,
                        {(uint16_t)(MXFP8_V_TAIL_BOUNDARY / 16), (uint16_t)actVecMSizeAlign16,
                         (uint16_t)(vec1Srcstride - actVecMSizeAlign16), (uint16_t)(mBaseSize - actVecMSizeAlign16)});
                }
            }
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2 + static_cast<uint32_t>(stage1Offset));

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(CC_L1P_0 + pL1BufId);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_FREE);
            return;
        }

        uint32_t vecMSize = USE_DN ? runInfo.actVecMSize : (runInfo.actMSizeAlign32 / 2);
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;
        constexpr float deSCaleVValue = 1.0f;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_;
        LocalTensor<T> mmRes = ubBmm2ResBuf_.template ReinterpretCast<T>();
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            constexpr float deSCalePreVValue = 1.0f;
            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                // 多块 FD 非 owner 分片(sum0=0)在大 sink 下 P 全下溢时, 末尾 running sum 精确为 0,
                // 与单块分支同样的 0/0 保护: 用 FLT_MIN 兜底, 分子同为 0 时输出 0。
                if constexpr (HAS_SINK) {
                    if (unlikely(runInfo.isS2SplitCore && !runInfo.isSinkOwner)) {
                        constexpr float SUM_FLOOR = 1.17549435e-38f; // FLT_MIN
                        Maxs(sumUb, sumUb, SUM_FLOOR, static_cast<int32_t>(vecMSize));
                        PipeBarrier<PIPE_V>();
                    }
                }
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_FREE);
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)];
                // 只有 FD 非 owner 分片(sum0=0)在大 sink 下 P 全下溢时, 分母才会精确为 0;
                // 其它任务的 sum0 = exp(sink-m0) >= 0.5, 恒不为 0, 因此这里用运行期条件把
                // 向量保护挡在热路径之外。用 FLT_MIN 兜底避免 0/0=NaN(分子同为 0 时输出 0)。
                if constexpr (HAS_SINK) {
                    if (unlikely(runInfo.isS2SplitCore && !runInfo.isSinkOwner)) {
                        constexpr float SUM_FLOOR = 1.17549435e-38f; // FLT_MIN
                        Maxs(sumUb, sumUb, SUM_FLOOR, static_cast<int32_t>(vecMSize));
                        PipeBarrier<PIPE_V>();
                    }
                }
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                          (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            uint32_t DealRowCount;
            if constexpr (USE_DN) {
                DealRowCount = runInfo.actVecMSize;
            } else {
                uint32_t groupsOf32 = (runInfo.actMSize + 31) / 32;
                if (constInfo_.subBlockIdx == 0) {
                    DealRowCount = groupsOf32 * 16 > runInfo.actMSize ? runInfo.actMSize : groupsOf32 * 16;
                } else {
                    int32_t vec1RemainRows = runInfo.actMSize - 16 * groupsOf32;
                    DealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
                }
            }
            if (DealRowCount == 0) {
                SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
                return;
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize, DealRowCount);
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize, gmDealRowCount);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX& runInfo, LocalTensor<OUTPUT_T>& attenOut,
                                     LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        if constexpr (FLASH_DECODE) {
            if (constInfo_.enableFlashDecode) {
                if (runInfo.isS2SplitCore) {
                    Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
                } else {
                    Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
                }
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize, gmDealRowCount);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX& runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                // 跨多个G
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T>& vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX& runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor = softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)][mStartVec];
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64), minValue_);
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo_.dSizeV, sizeof(float)); // w8后量化后的处理长度
                    RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, mDealSize, dSize, dStride,
                                              minValue_);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX& info, LocalTensor<OUTPUT_T>& attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        // mxfp8 colCount 只能为64或者128，与dDealSize相等
        FaUbTensor<OUTPUT_T, !isDAligned> ubTensor{
            .tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        if constexpr (USE_DN) {
            // DN分支下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
            OutGmCoord gmCoord{.bIdx = info.bIdx,
                               .n2Idx = info.realN2Idx,
                               .gIdx = 0,
                               .s1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                               .dIdx = 0,
                               .s1DealSize = gmDealRowCount,
                               .dDealSize = (uint32_t)constInfo_.dSizeV};
            CopyAttentionOut(ubTensor, gmCoord);
        } else {
            OutGmCoord gmCoord{.bIdx = info.bIdx,
                               .n2Idx = info.realN2Idx,
                               .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                               .dIdx = 0,
                               .gS1DealSize = gmDealRowCount,
                               .dDealSize = (uint32_t)constInfo_.dSizeV};
            CopyAttentionOut(ubTensor, gmCoord);
        }
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T, !isDAligned>& ubTensor, OutGmCoord& gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, OUT_UB_FORMAT> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(LocalTensor<float>& sumUb, LocalTensor<float>& maxUb, int64_t gmOffset,
                                               uint32_t gmDealRowCount)
    {
        int64_t calculateSize = gmDealRowCount * fp32BaseSize;
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst_;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, gmDealRowCount);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT5);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT5);
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst_;
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, gmDealRowCount);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT6);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT6);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX& runInfo, LocalTensor<float>& sumUb,
                                                       LocalTensor<float>& maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        uint32_t gmDealRowCount;
        if constexpr (USE_DN) {
            gmDealRowCount = runInfo.actVecMSize;
        } else {
            uint32_t groupsOf32 = (runInfo.actMSize + 31) / 32;
            if (constInfo_.subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > runInfo.actMSize ? runInfo.actMSize : groupsOf32 * 16;
            } else {
                int32_t vec1RemainRows = runInfo.actMSize - 16 * groupsOf32;
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
        }
        if (gmDealRowCount == 0) {
            return;
        }
        // FD 拆片同样要打无效行哨兵: 合并侧只在"合并 max == minValue_"时把 LSE 置 inf,
        // 而 owner 分片的 sink 网格 max m0 恒非哨兵; 不标记则全无效行的 LSE 会输出 sink 值。
        // 标记后该分片 partial max 恰为哨兵, 合并时 exp(max_i-M)=0 -> 分子分母贡献均为 0,
        // 全部分片都无效时合并 max 即为哨兵 -> LSE=inf; O 由分母 SUM_FLOOR 兜底保持 0。
        const uint32_t vecMSize = USE_DN ? runInfo.actVecMSize : (runInfo.actMSizeAlign32 >> 1);
        MarkInvalidRowMaxForSink(maxUb, runInfo, vecMSize);
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(sumUb, maxUb, gmOffset, gmDealRowCount);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX& runInfo, LocalTensor<T>& vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0 + (runInfo.loop % DB));
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT0 + (runInfo.loop % DB));
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(RunInfoX runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_READY);
        ProcessVec2OnUb(runInfo);
        return;
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t addrUb = 0U;
        ubBmm2ResBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUF_BYTES;
        ubBmm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);

        // FA实际地址按原TPipe顺序紧密排布; GetPersistUbSize仅作为FD基址的保留上界
        uint32_t addr = GetMmUbTotalSize();
        for (uint32_t i = 0; i < PRELOAD_N + 1; ++i) {
            softmaxSumBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addr, UB_SOFTMAX_BUF_BYTES / sizeof(float));
            addr += UB_SOFTMAX_BUF_BYTES;
        }
        for (uint32_t i = 0; i < PRELOAD_N + 1; ++i) {
            softmaxMaxBuf_[i] = LocalTensor<float>(TPosition::VECCALC, addr, UB_SOFTMAX_BUF_BYTES / sizeof(float));
            addr += UB_SOFTMAX_BUF_BYTES;
        }
        for (uint32_t i = 0; i < PRELOAD_N + 1; ++i) {
            softmaxExpBuf_[i] = LocalTensor<T>(TPosition::VECCALC, addr, UB_SOFTMAX_BUF_BYTES / sizeof(T));
            addr += UB_SOFTMAX_BUF_BYTES;
        }
        preLoopMaxBuf_ = LocalTensor<float>(TPosition::VECCALC, addr, UB_SOFTMAX_BUF_BYTES / sizeof(float));
        addr += UB_SOFTMAX_BUF_BYTES;
        preLoopSumBuf_ = LocalTensor<float>(TPosition::VECCALC, addr, UB_SOFTMAX_BUF_BYTES / sizeof(float));
        addr += UB_SOFTMAX_BUF_BYTES;
        firstLoopSumBuf_ = LocalTensor<float>(TPosition::VECCALC, addr, UB_SOFTMAX_BUF_BYTES / sizeof(float));
        addr += UB_SOFTMAX_BUF_BYTES;

        if constexpr (FLASH_DECODE) {
            maxBrdcst_ = LocalTensor<float>(TPosition::VECCALC, addr, UB_BRDCST_BUF_BYTES / sizeof(float));
            addr += UB_BRDCST_BUF_BYTES;
            sumBrdcst_ = LocalTensor<float>(TPosition::VECCALC, addr, UB_BRDCST_BUF_BYTES / sizeof(float));
            addr += UB_BRDCST_BUF_BYTES;
        }
        commonBuf_ = LocalTensor<uint8_t>(TPosition::VECCALC, addr, UB_COMMON_BUF_BYTES);
        addr += UB_COMMON_BUF_BYTES;
        softmaxLseBuf_ = LocalTensor<float>(TPosition::VECCALC, addr, UB_LSE_BUF_BYTES / sizeof(float));
        addr += UB_LSE_BUF_BYTES;
        if constexpr (HAS_SINK) {
            // sink 前置注入专用暂存(不与 VF scratch 复用); 无 sink 不分配, 保持原 UB 布局容量
            sinkStageBuf_ = LocalTensor<float>(TPosition::VECCALC, addr, SINK_STAGE_BYTES / sizeof(float));
            addr += SINK_STAGE_BYTES;
        }
        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)] =
                    LocalTensor<uint8_t>(TPosition::VECCALC, addr, 256U);
                addr += 256U;
                LocalTensor<uint8_t> indexes = vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)];
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        indexes.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
                UpdateMinCheckValue();
            } else {
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)] =
                    LocalTensor<uint8_t>(TPosition::VECCALC, addr, 128U);
                addr += 128U;
                vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)] =
                    LocalTensor<uint8_t>(TPosition::VECCALC, addr, 64U);
                addr += 64U;
                LocalTensor<uint8_t> indexes =
                    vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
                indexes.SetValue(0, 0x7f);
                for (int i = 0; i < 128; i++) {
                    indexes.SetValue(i, i * 2);
                }
                indexes = vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)];
                for (int i = 0; i < 64; i++) {
                    indexes.SetValue(i, i * 4);
                }
            }
        }

        stage2OutBuf_ = LocalTensor<T>(TPosition::VECCALC, addr, UB_STAGE2_OUT_ELEM_CNT);
        addr += UB_STAGE2_OUT_ELEM_CNT * sizeof(T);
        for (uint32_t i = 0; i < UB_STAGE1_OUT_BUFCNT; ++i) {
            stage1OutBuf_[i] =
                LocalTensor<uint8_t>(TPosition::VECCALC, addr, UB_STAGE1_OUT_BYTES).template ReinterpretCast<INPUT_T>();
            addr += UB_STAGE1_OUT_BYTES;
        }
        if constexpr (HAS_MASK) {
            attenMaskInBuf_ = LocalTensor<uint8_t>(TPosition::VECCALC, addr, UB_MASK_BUF_BYTES);
        }
    }

    static __aicore__ inline constexpr uint32_t GetMmUbTotalSize()
    {
        constexpr uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T) / 2U;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * static_cast<uint32_t>(dVTemplateType) * sizeof(T);
        return mm1ResultSize * 2U + mm2ResultSize;
    }

    // FA保留区字节数(紧跟跨核bmm结果区之后):
    // softmax状态(12*256) + preLoop(3*256) + brdcst(FLASH_DECODE时2*2048) + commonBuf(512)
    // + softmaxLseBuf(常开) + sink暂存(HAS_SINK时1056B) + vselr索引表(isFp8时, 按256上界对齐)
    static __aicore__ inline constexpr uint32_t GetPersistUbSize()
    {
        constexpr uint32_t softmaxSize = (12U + 3U) * 256U;
        constexpr uint32_t brdcstSize = FLASH_DECODE ? (2U * 2048U) : 0U;
        constexpr uint32_t commonSize = 512U;
        constexpr uint32_t sinkStageSize = HAS_SINK ? SINK_STAGE_BYTES : 0U; // sink 前置注入专用暂存区
        constexpr uint32_t lseSize = (mBaseSize >> 1U) * sizeof(float) * 8U;
        constexpr uint32_t vselrSize = isFp8 ? 256U : 0U;
        return softmaxSize + brdcstSize + commonSize + sinkStageSize + lseSize + vselrSize;
    }

    // FA瞬态区字节数: stage2Out + stage1Out*2 + attenMask, FD静态Tensor业务区(见FD block
    // GetFdTotalUbSize)须完全落在该区间内, 供static_assert校验
    static __aicore__ inline constexpr uint32_t GetTransientUbSize()
    {
        constexpr uint32_t stage2Size = 64U * dTemplateAlign64 * sizeof(T);
        constexpr uint32_t stage1Size = 2U * 16896U; // 16896: 16640 + 256(pScale)
        constexpr uint32_t maskSize = HAS_MASK ? (mBaseSize / CV_RATIO * (s2BaseSize >> 1U)) : 0U;
        return stage2Size + stage1Size + maskSize;
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT0);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT1);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT2);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT3);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT4);
        if constexpr (FLASH_DECODE) {
            SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT5);
            SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT6);
        }
        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT0); // attenMask初始空闲
        }
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT1);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT2);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT3);
        WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT4);
        if constexpr (FLASH_DECODE) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT5);
            WaitFlag<AscendC::HardEvent::MTE3_V>(MTE3_V_EVENT6);
        }
        if constexpr (HAS_MASK) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(V_MTE2_EVENT0);
        }
    }

    template <uint32_t s2MaskBase>
    __aicore__ inline void AttentionmaskCopyInLocal(LocalTensor<uint8_t>& attenMaskUb, MaskInfo& info)
    {
        if constexpr (MASK_LAYOUT == MaskFormat::GS) {
            int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
            int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
            uint32_t attenMaskS2Stride = AttentionCommon::Align(info.s2dealNum, 32U) + 32 * info.attenMaskDstStride;
            uint32_t headS1Count = (s1StartIdx == 0) ? 0 : (info.s1Size - s1StartIdx);
            uint32_t midGCount = (info.gs1dealNum > headS1Count) ? (info.gs1dealNum - headS1Count) / info.s1Size : 0;
            if (info.gs1dealNum <= info.s1Size || midGCount == 0) {
                if (s1StartIdx + info.gs1dealNum > info.s1Size) {
                    AttentionmaskDataCopy<uint8_t>(attenMaskUb, attenMaskGmInt_, info, s1StartIdx, info.s1Size, false);
                    LocalTensor<uint8_t> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskS2Stride];
                    AttentionmaskDataCopy<uint8_t>(attenMaskSecUb, attenMaskGmInt_, info, 0, s1EndIdx, false);
                } else {
                    AttentionmaskDataCopy<uint8_t>(attenMaskUb, attenMaskGmInt_, info, s1StartIdx, s1EndIdx, false);
                }
                SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT0);
                WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT0);
            } else {
                uint32_t tailS1Size = (info.gs1dealNum - headS1Count) % info.s1Size;
                LocalTensor<uint8_t> attenMaskSecUb = attenMaskUb[headS1Count * attenMaskS2Stride];
                AttentionmaskDataCopy<uint8_t>(attenMaskSecUb, attenMaskGmInt_, info, 0, info.s1Size, false);
                SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT0);
                WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT0);
                if (headS1Count > 0) {
                    DataCopy(attenMaskUb, attenMaskUb[info.s1Size * attenMaskS2Stride],
                             headS1Count * attenMaskS2Stride);
                }
                for (uint32_t i = 1; i < midGCount; i++) {
                    DataCopy(attenMaskUb[(headS1Count + i * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
                             info.s1Size * attenMaskS2Stride);
                }
                if (tailS1Size > 0) {
                    DataCopy(attenMaskUb[(headS1Count + midGCount * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
                             tailS1Size * attenMaskS2Stride);
                }
            }
            PipeBarrier<PIPE_V>();
        } else {
            uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
            uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
            uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
            uint32_t headGSize = info.gs1dealNum;
            if (s1Count > 1) {
                headGSize = (info.gs1StartIdx % info.gSize == 0) ? 0 : (info.gSize - info.gs1StartIdx % info.gSize);
            }
            uint32_t remainRowCount = info.gs1dealNum - headGSize;
            uint32_t midS1Count = remainRowCount / info.gSize;
            uint32_t tailGSize = remainRowCount % info.gSize;
            uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
            uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;
            if (headGSize > 0) {
                AttentionmaskDataCopy<uint8_t>(attenMaskUb, attenMaskGmInt_, info, s1StartIdx, s1StartIdx + 1, false);
                s1StartIdx++;
            }
            if (remainRowCount > 0) {
                uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, false);
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
                dataCopyParams.blockLen = info.s2dealNum;
                dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
                dataCopyParams.dstStride = (info.gSize - 1) * attenMaskS2Stride / 32 + info.attenMaskDstStride;
                DataCopyPadExtParams<uint8_t> padParams{true, 0,
                                                        static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
                DataCopyPad(attenMaskUb[headGSize * attenMaskS2Stride], attenMaskGmInt_[maskOffset], dataCopyParams,
                            padParams);
            }
            SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT0);
            WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT0);
            MaskUbCopyS1G<uint8_t, s2MaskBase>(attenMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
        }
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2RealSize = subLoop == 0 ? s2SplitSize : runInfo.actSingleLoopS2Size - s2SplitSize;
        }

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.realGSize;
        maskInfo.s2StartIdx = (subLoop == 0) ? runInfo.s2Idx : (runInfo.s2Idx + s2BaseSizeCur);
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar_;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        if (unlikely(!IsSkipMask)) {
            AttentionmaskCopyInLocal<s2BaseSizeCur>(attenMaskUb, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
        }
    }

    __aicore__ inline void AttenMaskCopyInTail(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                               RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2RealSize = subLoop == 0 ? s2SplitSize : runInfo.actSingleLoopS2Size - s2SplitSize;
        }

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.realGSize;
        maskInfo.s2StartIdx = (subLoop == 0) ? runInfo.s2Idx : (runInfo.s2Idx + s2BaseSizeCur);
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar_;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        if constexpr (HAS_V_TAIL) {
            // Use the same tail width for the attention mask and the softmax VF.
            constexpr uint32_t s2TailBaseSize = (s2BaseSizeCur > 128) ? 128 : s2BaseSizeCur;
            maskInfo.s2dealNum = runInfo.kvTailLen;
            maskInfo.nBaseSize = s2TailBaseSize;
            maskInfo.attenMaskDstStride = (s2TailBaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
            bool isSkipMaskTail = IsSkipAttentionmask(maskInfo);
            if (unlikely(!isSkipMaskTail)) {
                AttentionmaskCopyInLocal<s2TailBaseSize>(attenMaskUb, maskInfo);
            } else {
                Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2TailBaseSize);
            }
            return;
        }
    }

    __aicore__ inline int64_t ComputeMaskLineDN(RunInfoX& runInfo, uint32_t subLoop)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2RealSize = subLoop == 0 ? s2SplitSize : runInfo.actSingleLoopS2Size - s2SplitSize;
        }
        int64_t nextToken = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        uint32_t s1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        uint32_t s2StartIdx = (subLoop == 0) ? runInfo.s2Idx : (runInfo.s2Idx + s2BaseSizeCur);
        if (static_cast<int64_t>(s2StartIdx + s2RealSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            isSkipMask_ = true;
        } else {
            isSkipMask_ = false;
        }
        int64_t maskLine = nextToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(s2StartIdx);
        if (maskLine < 0) {
            isFullMask_ = -maskLine >= (mBaseSize / 2);
        } else {
            isFullMask_ = maskLine >= s2BaseSizeCur;
        }
        return maskLine;
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecMxfp8Dummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecMxfp8Dummy(ConstInfoX& constInfo){};
    __aicore__ inline void InitCrossCoreSync() {}
    __aicore__ inline void UnInitCrossCoreSync() {}
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_H_
