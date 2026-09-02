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
 * \file matmul_all_reduce_tiling_310_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
#define MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
#include "../matmul_all_reduce_tiling_base.h"
namespace optiling {
class MatmulAllReduceTiling310General : public MatmulAllReduceTilingBase {
public:
    explicit MatmulAllReduceTiling310General(gert::TilingContext *context)
        : MatmulAllReduceTilingBase(context)
    {}
    ~MatmulAllReduceTiling310General() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    CutResult GetTilingResult() override;

    void DoMatmulTiling310(matmul_tiling::MultiCoreMatmulTiling &mm1, AscendC::tiling::TCubeTiling &cubeTiling,
                           Mc2Tiling::Mc2L2cacheTilePara &l2cacheTiling);

    void DoWeightAntiQuantTiling();

    void GetL2CacheParm(uint64_t &l2CacheSize, uint64_t &singleMatrixSize, uint32_t &tileSize, uint32_t &tileLimit,
                        bool useNewPara) override;

    void SetTransLength(matmul_tiling::MultiCoreMatmulTiling &mm1, AscendC::tiling::TCubeTiling &cubeTiling);

private:
    bool isTransB_ = false;
    bool isWeightQuant_ = false;
    AntiQuantType antiQuantT_ = AntiQuantType::NONE;
    bool hasAntiQuantOffset_ = false;
};
} // namespace optiling
#endif // MATMUL_ALL_REDUCE_TILING_310_GENERAL_H
