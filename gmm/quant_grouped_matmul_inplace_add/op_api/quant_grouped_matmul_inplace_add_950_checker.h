/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_950_CHECKER_H
#define OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_950_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "quant_grouped_matmul_inplace_add_util.h"
#include "../../grouped_matmul/op_api/grouped_matmul_util.h"

namespace QGmmInPlaceAdd {
template <typename T>
class AclnnQuantGroupedMatmulInplaceAddDAV3510Checker {
public:
    explicit AclnnQuantGroupedMatmulInplaceAddDAV3510Checker(const gmm::GroupedMatmulParamsBase<T> &gmmParams)
        : gmmParams_(gmmParams) {};
    ~AclnnQuantGroupedMatmulInplaceAddDAV3510Checker() {};
    aclnnStatus CheckQuantGroupedMatmulInplaceAddDAV3510() const;
    aclnnStatus CheckHif8QuantParamsShape() const;
    void SetInputName(const std::string &xName, const std::string &weightName, const std::string &perTokenScaleName,
                      const std::string &scaleName, const std::string &groupTensorName);

private:
    aclnnStatus CheckGeneralQuantShape() const;
    aclnnStatus CheckQuantCasesFormat() const;
    aclnnStatus CheckHif8QuantParams() const;
    aclnnStatus CheckTensorListSizeForEachInput() const;

private:
    gmm::GroupedMatmulParamsBase<T> gmmParams_;
    std::string xName_ = "x1";
    std::string weightName_ = "x2";
    std::string scaleName_ = "scale1Optional";
    std::string perTokenScaleName_ = "scale2";
    std::string groupTensorName_ = "groupList";
    std::string biasName_ = "bias";
    std::string yName_ = "yRef";
};
} // namespace QGmmInPlaceAdd
#endif
