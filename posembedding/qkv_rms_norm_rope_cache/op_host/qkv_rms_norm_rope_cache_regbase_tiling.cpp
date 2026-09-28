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
 * \file qkv_rms_norm_rope_cache_regbase_tiling.cpp
 * \brief QkvRmsNormRopeCache arch35(Ascend950 / DAV_3510)tiling
 *
 * 与 A2 的 QkvRmsNormRopeCacheTilingDs 共用同一套 shape/dtype/attr 校验
 * (直接复用基类的 GetShapeAttrsInfoInner 与 11 个 Check*Valid),只重写:
 *   - 切核:按 (B*S) token 均分,每核连续一段 token(不区分 Q/K/V 核组)
 *   - UB 反推:按 regbase 内核的 buffer 布局重新计算 ubFactor
 *   - tiling key:10000(避开 A2 的默认 tiling 结构体注册)
 */

#include "qkv_rms_norm_rope_cache_tiling.h"
#include <string>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

bool QkvRmsNormRopeCacheTilingRegbase::IsCapable()
{
    // isRegbase_ 由 base 的 GetPlatformInfo 设;platformInfo 为空时它从 CompileInfo 取
    // (CompileInfo 是 TilingPrepare 阶段用保证非空的 platformInfo 算好的),故恒可靠。
    return isRegbase_;
}

void QkvRmsNormRopeCacheTilingRegbase::CalBlockTiling()
{
    const int64_t tokenSum = batchSize_ * seqLength_;
    int64_t usedCoreNum = static_cast<int64_t>(coreNum_);
    if (usedCoreNum > tokenSum) {
        usedCoreNum = tokenSum; // 不启用空核
    }
    blockFactor_ = (tokenSum + usedCoreNum - 1) / usedCoreNum; // 每核 token 数,末核吸收余数
    blockDim_ = (tokenSum + blockFactor_ - 1) / blockFactor_;  // ceil 后实际用到的核数
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CalUbTiling()
{
    const int64_t qkvBytes = static_cast<int64_t>(ge::GetSizeByDataType(qkvDtype_));
    const int64_t kCacheBytes = (isKQuant_ != 0) ? INT8_BYTES : qkvBytes;
    const int64_t vCacheBytes = (isVQuant_ != 0) ? INT8_BYTES : qkvBytes;
    const int64_t dimBytes = qkvDim_;

    // 与 token 数无关的常驻 buffer:gamma(q_gamma + k_gamma)+ 量化因子(scale/offset × K/V)
    gammaUbBytes_ = 2 * dimBytes * qkvBytes;
    quantUbBytes_ = 2 * (numHeadK_ + numHeadV_) * dimBytes * FLOAT32_BYTES;

    // 每个 token 占用的 UB(与 arch35 内核的 buffer 布局一一对应)
    // 单 token 的各个 buffer 字节数也在这里一次算准,随 tiling 下发给内核,内核不重算
    //
    // ⚠️ cosSinUbBytes_ 的语义是【cos 或 sin 的单份】,不是"cos+sin 合计":
    // 内核里 cos / sin 是两个**独立的 queue**(regbase.h InitUbBuffers:
    //   InitBuffer(inQueueCos_, 1, cosSinUbBytes_);
    //   InitBuffer(inQueueSin_, 1, cosSinUbBytes_);),
    // 而每个 queue 只装一个 token 的 D 个元素 —— 所以两处相加才是 cos+sin。
    // 曾经把它写成 2*dimBytes*qkvBytes(合计一份)却仍按"单份"下发给两个 queue,
    // 内核实物就比记账多出 `qkvBytes*dimBytes*ubFactor_` 字节 → ubFactor 大的形态
    // 撑爆 UB,报 VEC_ERROR(实测 ubFactor=40 的 fp16 形态必挂,39 及以下正常)。
    int64_t perTokenBytes = numHead_ * dimBytes * qkvBytes; // x(整行 qkv)
    cosSinUbBytes_ = dimBytes * qkvBytes;                   // cos(或 sin)单份
    qOutUbBytes_ = numHeadQ_ * dimBytes * qkvBytes;         // q_out / q_out_before_quant
    kProtoUbBytes_ = numHeadK_ * dimBytes * qkvBytes;       // k_out_before_quant
    vProtoUbBytes_ = numHeadV_ * dimBytes * qkvBytes;       // v_out_before_quant
    kCacheUbBytes_ = numHeadK_ * dimBytes * kCacheBytes;    // k_cache 散射源
    vCacheUbBytes_ = numHeadV_ * dimBytes * vCacheBytes;    // v_cache 散射源
    perTokenBytes +=
        2 * cosSinUbBytes_ + qOutUbBytes_ + kProtoUbBytes_ + vProtoUbBytes_ + kCacheUbBytes_ + vCacheUbBytes_;

    const int64_t fixedBytes = gammaUbBytes_ + quantUbBytes_;
    const int64_t available = static_cast<int64_t>(ubSize_) - UB_RESERVED_BYTES - fixedBytes;
    OP_CHECK_IF(perTokenBytes <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "perTokenUbBytes",
                                                      std::to_string(perTokenBytes).c_str(),
                                                      "per-token UB cost must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(available <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "residentUbBytes",
                                                      std::to_string(-available).c_str(),
                                                      "ubSize is too small for the resident gamma/quant buffers"),
                return ge::GRAPH_FAILED);

    ubFactor_ = available / perTokenBytes;
    // A2 的同义检查在共享的 _tiling.cpp 里(裸 OP_LOGE);这是 A5 新增的独立校验入口,
    // 按红线 R6 用结构化宏上报(不牵连 A2 的既有实现)。
    OP_CHECK_IF(
        ubFactor_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubFactor", std::to_string(ubFactor_).c_str(),
                                              "per-token UB cost exceeds ubSize; token shape is too big"),
        return ge::GRAPH_FAILED);
    if (ubFactor_ > blockFactor_) {
        ubFactor_ = blockFactor_; // 单核 token 数不足一轮时不留空转
    }
    inUbBytes_ = ubFactor_ * numHead_ * dimBytes * qkvBytes;
    cosSinUbBytes_ *= ubFactor_;
    qOutUbBytes_ *= ubFactor_;
    kProtoUbBytes_ *= ubFactor_;
    vProtoUbBytes_ *= ubFactor_;
    kCacheUbBytes_ *= ubFactor_;
    vCacheUbBytes_ *= ubFactor_;
    return ge::GRAPH_SUCCESS;
}

// A5(ascend950)侧收紧:cos/sin 的第一维必须恰为 B*S。
//
// 基类共享的 CheckCosSinValid(A2/A3 同样在用)额外放行了 batch 内广播形态 [B, D],
// 但该形态:
//   ① proto / README / aclnn 三处公开资料均未声明,只承诺 [B*S, D];
//   ② 内核一律按【全局 token 行】读 cos/sin(tokenStart * D),并未实现广播 ——
//      而 cos/sin 只有 [0, B) 行有效:S>1 时 tokenStart 会落在 [B, B*S),
//      读到 cos 张量之外的内存。
ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckCosSinExact()
{
    const gert::StorageShape *cosShapePtr = context_->GetInputShape(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosShapePtr);
    const gert::StorageShape *sinShapePtr = context_->GetInputShape(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinShapePtr);
    const gert::Shape &cosShape = cosShapePtr->GetStorageShape();
    const gert::Shape &sinShape = sinShapePtr->GetStorageShape();

    // 秩已由基类 CheckCosSinValid 校为 2,此处可直接取 dim 0/1
    const int64_t expect = batchSize_ * seqLength_;
    OP_CHECK_IF(
        cosShape.GetDim(SHAPE_IDX_BS) != expect || sinShape.GetDim(SHAPE_IDX_BS) != expect,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos and sin",
                                              Ops::Base::ToString(cosShape) + " and " + Ops::Base::ToString(sinShape),
                                              "the 0th dim of input cos and sin must be the same and equal to B*S"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::DoOpTiling()
{
    // 复用 A2 的 shape / dtype / attr 校验入口,支持面与 A2 严格一致
    OP_CHECK_IF(GetShapeAttrsInfoInner() == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfoInner failed."), return ge::GRAPH_FAILED);
    // 基类放行了 [B,D] 广播形态但内核读不了,A5 侧单独拒掉
    OP_CHECK_IF(CheckCosSinExact() == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "cos or sin shape is invalid."), return ge::GRAPH_FAILED);

    CalBlockTiling();
    if (CalUbTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    tilingKey_ = static_cast<uint64_t>(TILING_KEY_REGBASE_PA_NZ);

    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqLength(seqLength_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_qkvDim(qkvDim_);
    tilingData_.set_numHeadQ(numHeadQ_);
    tilingData_.set_numHeadK(numHeadK_);
    tilingData_.set_numHeadV(numHeadV_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_blockNum(blockNum_); // index 合法上界的另一半,内核散射前据此判越界
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_isOutputQkv(isOutputQkv_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_inUbBytes(inUbBytes_);
    tilingData_.set_cosSinUbBytes(cosSinUbBytes_);
    tilingData_.set_qOutUbBytes(qOutUbBytes_);
    tilingData_.set_kProtoUbBytes(kProtoUbBytes_);
    tilingData_.set_vProtoUbBytes(vProtoUbBytes_);
    tilingData_.set_kCacheUbBytes(kCacheUbBytes_);
    tilingData_.set_vCacheUbBytes(vCacheUbBytes_);
    tilingData_.set_gammaUbBytes(gammaUbBytes_);
    tilingData_.set_quantUbBytes(quantUbBytes_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkSpaceSize);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheTilingRegbase::DumpTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "tilingKey_:    %lu", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "coreNum_:  %lu", coreNum_);
    OP_LOGD(context_->GetNodeName(), "ubSize_:  %lu", ubSize_);
    OP_LOGD(context_->GetNodeName(), "isRegbase_:  %ld", static_cast<int64_t>(isRegbase_));
    OP_LOGD(context_->GetNodeName(), "batchSize_:  %ld", batchSize_);
    OP_LOGD(context_->GetNodeName(), "seqLength_:  %ld", seqLength_);
    OP_LOGD(context_->GetNodeName(), "numHead_:  %ld", numHead_);
    OP_LOGD(context_->GetNodeName(), "qkvDim_:  %ld", qkvDim_);
    OP_LOGD(context_->GetNodeName(), "numHeadQ_:  %ld", numHeadQ_);
    OP_LOGD(context_->GetNodeName(), "numHeadK_:  %ld", numHeadK_);
    OP_LOGD(context_->GetNodeName(), "numHeadV_:  %ld", numHeadV_);
    OP_LOGD(context_->GetNodeName(), "blockNum_:  %ld", blockNum_);
    OP_LOGD(context_->GetNodeName(), "blockSize_:  %ld", blockSize_);
    OP_LOGD(context_->GetNodeName(), "blockFactor_:  %ld", blockFactor_);
    OP_LOGD(context_->GetNodeName(), "blockDim_:  %ld", blockDim_);
    OP_LOGD(context_->GetNodeName(), "ubFactor_:  %ld", ubFactor_);
    OP_LOGD(context_->GetNodeName(), "epsilon_:  %f", epsilon_);
    OP_LOGD(context_->GetNodeName(), "reciprocal_:  %f", reciprocal_);
    OP_LOGD(context_->GetNodeName(), "isOutputQkv_:  %ld", isOutputQkv_);
    OP_LOGD(context_->GetNodeName(), "isKQuant_:  %ld", isKQuant_);
    OP_LOGD(context_->GetNodeName(), "isVQuant_:  %ld", isVQuant_);
}

REGISTER_OPS_TILING_TEMPLATE(QkvRmsNormRopeCache, QkvRmsNormRopeCacheTilingRegbase, TEMPLATE_REGBASE_PRIORITY);
} // namespace optiling
