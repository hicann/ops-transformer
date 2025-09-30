/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_rotate_half.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_ROTATE_HALF_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_ROTATE_HALF_H
#include "register/op_def_registry.h"
#include "rotary_position_embedding_tiling.h"

namespace optiling {

class RopeRotateHalfTilingClass : public RotaryPosEmbeddingMembaseTilingClass {
public:
    explicit RopeRotateHalfTilingClass(gert::TilingContext *context) : RotaryPosEmbeddingMembaseTilingClass(context)
    {
    }

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        if (socVersion_ == platform_ascendc::SocVersion::ASCEND910B && inputMode_ != MODE_ROTATE_INTERLEAVED) {
            return true;
        }
        return false;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_ROTATE_HALF_H
