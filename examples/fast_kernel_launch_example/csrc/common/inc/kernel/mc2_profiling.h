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
 * \file mc2_profiling.h
 * \brief
 */

#ifndef MC2_PROFILING_H
#define MC2_PROFILING_H

#include <ATen/ATen.h>

#include "acl/acl_prof.h"
#include "acl/acl.h"

constexpr int64_t NCL_DIM_NUM = 3;
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_DIM_NUM = 5;

static inline aclFormat GetFormat(int64_t dimNum)
{
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case NCL_DIM_NUM:
            format = ACL_FORMAT_NCL;
            break;
        case NCHW_DIM_NUM:
            format = ACL_FORMAT_NCHW;
            break;
        case NCDHW_DIM_NUM:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }
    return format;
}

static inline aclDataType ConvertType(const at::ScalarType& scalar_type)
{
    switch (scalar_type) {
        case at::kFloat:
            return ACL_FLOAT;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16; // 修正：使用 ACL_BF16 (值为27)
        case at::kInt:
            return ACL_INT32;
        case at::kLong:
            return ACL_INT64;
        case at::kChar:
            return ACL_INT8;
        case at::kByte:
            return ACL_UINT8;
        case at::kBool:
            return ACL_BOOL;
        case at::kShort:
            return ACL_INT16;

        default:
            return ACL_DT_UNDEFINED; // 处理未识别的类型
    }
}

#define INIT_ACL_TENSOR_ARRAY(tensors1, ...) aclTensor* tensors1[] = {__VA_ARGS__}
#define INPUT(x) \
    aclprofTensor \
    { \
        (uint32_t)0, static_cast<uint32_t>(GetFormat(static_cast<int64_t>(x.sizes().size()))), \
            static_cast<uint32_t>(ConvertType(x.scalar_type())), static_cast<uint32_t>(x.sizes().size()), \
        { \
            static_cast<uint32_t>(x.sizes().size() > 0 ? x.sizes().data()[0] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 1 ? x.sizes().data()[1] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 2 ? x.sizes().data()[2] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 3 ? x.sizes().data()[3] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 4 ? x.sizes().data()[4] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 5 ? x.sizes().data()[5] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 6 ? x.sizes().data()[6] : 0), \
                static_cast<uint32_t>(x.sizes().size() > 7 ? x.sizes().data()[7] : 0) \
        } \
    }

#define OUTPUT(z) \
    aclprofTensor \
    { \
        (uint32_t)1, static_cast<uint32_t>(GetFormat(static_cast<int64_t>(z.sizes().size()))), \
            static_cast<uint32_t>(ConvertType(z.scalar_type())), static_cast<uint32_t>(z.sizes().size()), \
        { \
            static_cast<uint32_t>(z.sizes().size() > 0 ? z.sizes().data()[0] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 1 ? z.sizes().data()[1] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 2 ? z.sizes().data()[2] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 3 ? z.sizes().data()[3] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 4 ? z.sizes().data()[4] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 5 ? z.sizes().data()[5] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 6 ? z.sizes().data()[6] : 0), \
                static_cast<uint32_t>(z.sizes().size() > 7 ? z.sizes().data()[7] : 0) \
        } \
    }

#define INIT_ACL_PROF_TENSOR_INFO(opName, opType, blockdim, kernelType, tensorInfo, stream, ...) \
    uint64_t opNameId = aclprofStr2Id(opName); \
    uint64_t opTypeId = aclprofStr2Id(opType); \
    struct aclprofTensor _macro_prof_tensors[] = {__VA_ARGS__}; \
    tensorInfo = {opNameId,   opTypeId, 0,      sizeof(_macro_prof_tensors) / sizeof(aclprofTensor), \
                  kernelType, blockdim, stream, _macro_prof_tensors}

#endif // MC2_PROFILING_H
