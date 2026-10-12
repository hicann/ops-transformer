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
 * \file nsa_compress_grad_tiling.h
 * \brief
 */
#ifndef NSA_COMPRESS_GRAD_TILING_H
#define NSA_COMPRESS_GRAD_TILING_H

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <utility>

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <register/tilingdata_base.h>
#include <register/op_def_registry.h>
#include <tiling/tiling_api.h>
#include "op_host/tiling_base.h"

namespace optiling {

enum class TilingKeyInfo : int {
    KEY_FP16 = 10000,
    KEY_BF16 = 10001,
};

BEGIN_TILING_DATA_DEF(NsaCompressGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, numOfBlock);
TILING_DATA_FIELD_DEF(uint32_t, numOfHead);
TILING_DATA_FIELD_DEF(uint32_t, dimOfHead);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, blockStride);
TILING_DATA_FIELD_DEF(uint32_t, seqLenType);
TILING_DATA_FIELD_DEF(uint32_t, headToProcess);
TILING_DATA_FIELD_DEF(uint32_t, headRemainder);
TILING_DATA_FIELD_DEF(uint64_t, ubMaxSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NsaCompressGrad, NsaCompressGradTilingData)

struct NsaCompileInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    platform_ascendc::SocVersion socVersion;
};

class NsaCompressGradTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit NsaCompressGradTiling(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~NsaCompressGradTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;

    // 获取平台信息，如 CoreNum, UB/L1/L0C 大小
    ge::graphStatus GetPlatformInfo() override;
    // 获取 input, output, attr 信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算数据切分信息 TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算高阶 API 的 TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 计算 TilingKey
    uint64_t GetTilingKey() const override;
    // 计算 WorkSpace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存 Tiling 数据
    ge::graphStatus PostTiling() override;
    // 检查Tiling 数据有效性
    bool CheckTilingData();

    void Reset();

private:
    const gert::Shape GetShapeOfInput(const size_t index);
    bool PreparePlatformInfo();

    NsaCompressGradTilingData tilingData_;

    uint32_t aicNum_ = 1;
    uint32_t aivNum_ = 1;
    uint32_t tSeqLen_;
    uint64_t ubMaxSize_;
};

} // namespace optiling
#endif // NSA_COMPRESS_GRAD_TILING_H
