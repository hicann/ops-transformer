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
 * \file quant_grouped_matmul_inplace_add_basic_api_tiling.h
 * \brief
 */
#ifndef QUANT_GROUPED_MATMUL_INPLACE_ADD_BASIC_API_TILING_H
#define QUANT_GROUPED_MATMUL_INPLACE_ADD_BASIC_API_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

#include "../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.h"
#include "../../op_kernel/arch35/quant_grouped_matmul_inplace_add_tiling_data.h"
#include "../quant_grouped_matmul_inplace_add_host_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
namespace optiling {

class QGMMInplaceAddBasicApiTiling : public GroupedQmmBasicApiTiling {
public:
    explicit QGMMInplaceAddBasicApiTiling(gert::TilingContext *context);
    ~QGMMInplaceAddBasicApiTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmBasicApiTiling::Reset(context);
        tilingData_ = QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData();
    }

protected:
    const char *GetOpType() const override
    {
        return "QuantGroupedMatmulInplaceAdd";
    }
    bool IsCapable() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool SetQuantModeForQGmmInplaceAdd();
    bool CheckCoreNum() const override;
    int64_t LogQuantParams() const;

    QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData tilingData_;
};

} // namespace optiling

#endif // QUANT_GROUPED_MATMUL_INPLACE_ADD_BASIC_API_TILING_H
