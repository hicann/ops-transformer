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
 * \file und_gen_qkv_rms_norm_rope_cache_tiling.cpp
 * \brief regbase（arch35 / Ascend 950）模板 tiling
 */

#include "und_gen_qkv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {

bool UndGenQkvRmsNormRopeCacheRegbaseTiling::IsCapable()
{
    // 本算子当前仅有一套 regbase 模板，覆盖全部已校验的 shape
    return true;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::CalBlockTiling()
{
    // 按输出 token 维 total 切分：每个 out_t 计算独立，slot_mapping[out_t] 唯一保证
    // KV Cache 无写冲突，因此无需 SyncAll。
    //
    // 这里用余数分配而不是 blockFactor = CeilDiv(total, coreNum)：后者在 total 略大于
    // coreNum 时会大幅少用核。例如 total=113、coreNum=56，CeilDiv 得 blockFactor=3，
    // 实际只能用 CeilDiv(113,3)=38 个核，白扔 18 个。余数分配下核数拉满，
    // 核间负载差恒为 1 个 token。
    const int64_t usedCoreNum = std::min(coreNum_, totalTokens_);
    OP_CHECK_IF(usedCoreNum <= 0,
                OP_LOGE(context_->GetNodeName(), "usedCoreNum must be positive, got %ld (coreNum=%ld, total=%ld).",
                        usedCoreNum, coreNum_, totalTokens_),
                return ge::GRAPH_FAILED);

    const int64_t baseFactor = totalTokens_ / usedCoreNum;
    const int64_t formerCoreNum = totalTokens_ % usedCoreNum;

    tilingData_.set_usedCoreNum(usedCoreNum);
    tilingData_.set_formerCoreNum(formerCoreNum);
    tilingData_.set_blockFactor(formerCoreNum > 0 ? baseFactor + 1 : baseFactor);
    tilingData_.set_tailBlockFactor(baseFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::CalUbTiling()
{
    // UB 只切 token 维：N*D 单 token 最大 20*128*4B=10KB，一定放得下，不需要再切 N/D。
    // 计算与写出逐 token，所以计算区与 ubFactor 无关；只有三个 DMA 队列随 ubFactor 变化。
    // 各 buffer 的含义与公式见 tiling.h 的 "UB 划分" 注释，kernel 侧必须保持一致。
    const int64_t halfDim = headDim_ / DIM_NUM_TWO;

    // RMSNorm + MRoPE 融合在 VF 里做，fp32 的 x、合并后的 cos/sin、RoPE 交叉项全部留在 vreg，
    // 计算区只剩 4 份 gamma 的 float 版本和 gather 索引两块小 buffer。
    const int64_t residentBytes =
        WEIGHT_NUM * headDim_ * BF16_BYTES +                                       // wInQue（单 buffer）
        WEIGHT_NUM * headDim_ * FLOAT32_BYTES +                                    // wFp32Buf
        Ops::Base::CeilAlign(halfDim * UINT32_BYTES, BLOCK_ALIGN_BYTES) +          // gatherIdxBuf
        IDX_REGION_NUM * IDX_WINDOW_TOKENS * INT64_BYTES;                          // idxBuf（索引滑窗）

    const int64_t perTokenBytes =
        DOUBLE_BUFFER * numHead_ * headDim_ * BF16_BYTES +                      // qkvInQue
        DOUBLE_BUFFER * MROPE_AXIS_NUM * headDim_ * FLOAT32_BYTES +             // cosSinInQue
        DOUBLE_BUFFER * numHead_ * headDim_ * BF16_BYTES;                       // outQue

    OP_CHECK_IF(residentBytes + perTokenBytes > ubSize_,
                OP_LOGE(context_->GetNodeName(),
                        "UB is not enough for even one token: resident(%ld) + perToken(%ld) > ubSize(%ld).",
                        residentBytes, perTokenBytes, ubSize_),
                return ge::GRAPH_FAILED);

    // 每核最多也只处理 blockFactor 个 token，ubFactor 再大也用不上
    // 上限 MAX_UB_FACTOR 来自 kernel 侧 und/gen 位图的宽度，见 tiling.h
    const int64_t ubFactor =
        std::min({(ubSize_ - residentBytes) / perTokenBytes, tilingData_.get_blockFactor(), MAX_UB_FACTOR});
    OP_CHECK_IF(ubFactor <= 0,
                OP_LOGE(context_->GetNodeName(), "ubFactor must be positive, got %ld.", ubFactor),
                return ge::GRAPH_FAILED);
    // 上面的 min 已经保证不会超，这里再复核一次：kernel 侧 undMask_ 是 uint64_t，
    // ubFactor 一旦大于位图宽度，CopyIn 的 1ULL << i 就是未定义行为且不会报错，
    // 板上表现为 gamma 选错、精度随机劣化。宁可在 tiling 阶段就失败。
    OP_CHECK_IF(ubFactor > MAX_UB_FACTOR,
                OP_LOGE(context_->GetNodeName(),
                        "ubFactor(%ld) exceeds the und/gen bitmap width(%ld) used by the kernel.", ubFactor,
                        MAX_UB_FACTOR),
                return ge::GRAPH_FAILED);
    tilingData_.set_ubFactor(ubFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::DoOpTiling()
{
    tilingData_.set_totalTokens(totalTokens_);
    tilingData_.set_undLen(undLen_);
    tilingData_.set_genLen(genLen_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_numHeadQ(numHeadQ_);
    tilingData_.set_numHeadK(numHeadK_);
    tilingData_.set_numHeadV(numHeadV_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_maxPos(maxPos_);
    tilingData_.set_blockNum(blockNum_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_hasGen(static_cast<int64_t>(hasGen_));
    tilingData_.set_hasCatIndices(static_cast<int64_t>(hasCatIndices_));
    tilingData_.set_mropeSectionT(mropeSection_[DIM_ZERO]);
    tilingData_.set_mropeSectionH(mropeSection_[DIM_ONE]);
    tilingData_.set_mropeSectionW(mropeSection_[DIM_TWO]);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_reciprocal(reciprocal_);

    OP_CHECK_IF(CalBlockTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalBlockTiling failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalUbTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalUbTiling failed."),
                return ge::GRAPH_FAILED);

    // 当前只有一套模板，tilingKey 固定为 0；后续按 hasGen/hasCatIndices 等分支扩展
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

void UndGenQkvRmsNormRopeCacheRegbaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "TilingData: totalTokens=%ld, undLen=%ld, genLen=%ld, numHead=%ld(Q=%ld,K=%ld,V=%ld), headDim=%ld, "
            "maxPos=%ld, blockNum=%ld, blockSize=%ld, hasGen=%ld, hasCatIndices=%ld, mropeSection=[%ld,%ld,%ld], "
            "epsilon=%f, reciprocal=%f, usedCoreNum=%ld, formerCoreNum=%ld, blockFactor=%ld, tailBlockFactor=%ld, "
            "ubFactor=%ld",
            tilingData_.get_totalTokens(), tilingData_.get_undLen(), tilingData_.get_genLen(),
            tilingData_.get_numHead(), tilingData_.get_numHeadQ(), tilingData_.get_numHeadK(),
            tilingData_.get_numHeadV(), tilingData_.get_headDim(), tilingData_.get_maxPos(),
            tilingData_.get_blockNum(), tilingData_.get_blockSize(), tilingData_.get_hasGen(),
            tilingData_.get_hasCatIndices(), tilingData_.get_mropeSectionT(), tilingData_.get_mropeSectionH(),
            tilingData_.get_mropeSectionW(), tilingData_.get_epsilon(), tilingData_.get_reciprocal(),
            tilingData_.get_usedCoreNum(), tilingData_.get_formerCoreNum(), tilingData_.get_blockFactor(),
            tilingData_.get_tailBlockFactor(), tilingData_.get_ubFactor());
}

ge::graphStatus UndGenQkvRmsNormRopeCacheRegbaseTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(static_cast<uint32_t>(tilingData_.get_usedCoreNum()));
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // 各核独立写各自输出，无跨核归约，不需要 workspace
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(UndGenQkvRmsNormRopeCache, UndGenQkvRmsNormRopeCacheRegbaseTiling, 1000);
} // namespace optiling
