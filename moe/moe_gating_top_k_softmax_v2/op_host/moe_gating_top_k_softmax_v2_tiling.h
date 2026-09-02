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
 * \file moe_gating_top_k_softmax_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_V2_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2, MoeGatingTopKSoftmaxV2TilingData);

BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxV2EKFullLoadTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, colAlign);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlignB16);
TILING_DATA_FIELD_DEF(uint32_t, kAlignB32);
TILING_DATA_FIELD_DEF(uint32_t, kAlignT);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);
TILING_DATA_FIELD_DEF(uint32_t, blockTail);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, softmaxFlag);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, formerSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, formerBlockTailSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, tailBlockTailSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, formerTopkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, formerBlockTailTopkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, tailBlockTailTopkTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101010, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101011, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101020, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101021, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101030, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101031, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101110, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101111, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101120, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101121, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101130, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_101131, MoeGatingTopKSoftmaxV2EKFullLoadTilingData);

BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxV2KFullLoadTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);
TILING_DATA_FIELD_DEF(uint32_t, blockTail);
TILING_DATA_FIELD_DEF(uint32_t, ubLoop);
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
TILING_DATA_FIELD_DEF(uint32_t, ubFormerAlign);
TILING_DATA_FIELD_DEF(uint32_t, ubTail);
TILING_DATA_FIELD_DEF(uint32_t, ubTailAlign);
TILING_DATA_FIELD_DEF(uint32_t, softmaxFlag);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, ubFormerSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, ubTailSoftmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkFormerTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTailTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_102011, MoeGatingTopKSoftmaxV2KFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_102021, MoeGatingTopKSoftmaxV2KFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_102031, MoeGatingTopKSoftmaxV2KFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_102111, MoeGatingTopKSoftmaxV2KFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_102121, MoeGatingTopKSoftmaxV2KFullLoadTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_102131, MoeGatingTopKSoftmaxV2KFullLoadTilingData);

BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxV2PerfTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, colBytesAlign);
TILING_DATA_FIELD_DEF(uint32_t, colAlign);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);
TILING_DATA_FIELD_DEF(uint32_t, blockTail);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, topKValuesMask);
TILING_DATA_FIELD_DEF(uint32_t, topKIndicesMask);
TILING_DATA_FIELD_DEF(uint32_t, bufferElemSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103011, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103021, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103031, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103111, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103121, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103131, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103012, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103022, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103032, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103112, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103122, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103132, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103013, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103023, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103033, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103113, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103123, MoeGatingTopKSoftmaxV2PerfTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103133, MoeGatingTopKSoftmaxV2PerfTilingData);

BEGIN_TILING_DATA_DEF(MoeGatingTopKSoftmaxV2PerfRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(uint32_t, colBytesAlign);
TILING_DATA_FIELD_DEF(uint32_t, colAlign);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFormer);
TILING_DATA_FIELD_DEF(uint32_t, blockTail);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubFormer);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(uint32_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, topKValuesMask);
TILING_DATA_FIELD_DEF(uint32_t, topKIndicesMask);
TILING_DATA_FIELD_DEF(uint32_t, bufferElemSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxFlag);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103010, MoeGatingTopKSoftmaxV2PerfRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103020, MoeGatingTopKSoftmaxV2PerfRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103030, MoeGatingTopKSoftmaxV2PerfRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103110, MoeGatingTopKSoftmaxV2PerfRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103120, MoeGatingTopKSoftmaxV2PerfRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(MoeGatingTopKSoftmaxV2_103130, MoeGatingTopKSoftmaxV2PerfRegbaseTilingData);

const int64_t BLOCK_SIZE = 32;
const int64_t ALIGN_NUM = 32;
const int32_t BLOCK_B32_SIZE = 8;   // 一个BLOCK能容纳的元素数，其中一个元素4字节
const int32_t REPEAT_B32_SIZE = 64; // 一次Repeat能处理的元素数，其中一个元素取4字节

const int SYSTEM_WORKSPACE = 16 * 1024 * 1024;
const int TOPK_SOFTMAX_TILING_KEY_BASE_ALL = 100000;
const int TOPK_SOFTMAX_TILING_KEY_BASE_SCENE = 1000;
const int TOPK_SOFTMAX_TILING_KEY_BASE_RENORM = 100;
const int TOPK_SOFTMAX_TILING_KEY_BASE_DTYPE = 10;
const int TOPK_SOFTMAX_TILING_KEY_BASE_HALF = 2;
const int TOPK_SOFTMAX_TILING_KEY_BASE_BF16 = 3;
const int TOPK_SOFTMAX_TILING_KEY_BASE_COL_SMALLER_THAN_8 = 1;
const int TOPK_SOFTMAX_TILING_KEY_BASE_COL_FROM_8_TO_64 = 2;
const int TOPK_SOFTMAX_TILING_KEY_BASE_COL_BIGGER_THAN_64 = 3;

inline int dtypeKey(const ge::DataType &dtype)
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return 1;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TOPK_SOFTMAX_TILING_KEY_BASE_HALF;
    } else if (dtype == ge::DataType::DT_BF16) {
        return TOPK_SOFTMAX_TILING_KEY_BASE_BF16;
    }
    return 0;
}

inline int colNumKey(const uint32_t &col)
{
    if (col <= BLOCK_B32_SIZE) {
        return TOPK_SOFTMAX_TILING_KEY_BASE_COL_SMALLER_THAN_8;
    } else if (col <= REPEAT_B32_SIZE) {
        return TOPK_SOFTMAX_TILING_KEY_BASE_COL_FROM_8_TO_64;
    } else {
        return TOPK_SOFTMAX_TILING_KEY_BASE_COL_BIGGER_THAN_64;
    }
}

class MoeGatingTopKSoftmaxV2BaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeGatingTopKSoftmaxV2BaseTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus CheckOutShape(const gert::Shape &, gert::Shape &, bool, const char *);
    ge::graphStatus CheckInShape(const gert::Shape &);
    ge::graphStatus CheckOptionalAttr(gert::Shape &);

protected:
    int coreNum;
    uint64_t ubSize;
    ge::DataType dtype;
    uint32_t row;
    uint32_t col;
    int32_t k;
    int32_t renorm;
    int32_t softmaxFlag;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

inline uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    if (factor == 0) {
        return value;
    }
    return (value + factor - 1) / factor;
}

inline int64_t calcUbAlignBufferSize(const uint32_t curRowInUb, const uint32_t col, const int typeSize)
{
    return CeilDiv(col * typeSize, BLOCK_SIZE) * BLOCK_SIZE * curRowInUb;
}

inline uint32_t calcGatingAlignCol(const uint32_t col, const ge::DataType dtypeLocal)
{
    (void)dtypeLocal;
    // 对齐成32个数处理
    return CeilDiv(col, ALIGN_NUM) * ALIGN_NUM;
}

struct MoeGatingTopKSoftmaxV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_V2_H_
