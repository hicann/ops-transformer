/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kv_compress_epilog.h"
#include "kv_compress_epilog.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

// arch22 quant_mode=1 (int8) 场景常量, 与 op_host tiling 层校验保持一致
namespace {
constexpr int64_t QUANT_MODE_INT8_ACLNN = 1;
constexpr int64_t DEFAULT_QUANT_GROUP_SIZE_ACLNN = 64;
constexpr int64_t D_LENGTH_MODE3_ACLNN = 512;
constexpr int64_t KVCACHE_COL_MODE3_ACLNN = 608;

constexpr size_t CACHE_VIEW_DIM_NUM = 4;
constexpr size_t CACHE_BLOCKNUM_DIM = 0;
constexpr size_t CACHE_BLOCKSIZE_DIM = 1;
constexpr size_t CACHE_HEAD_DIM = 3;

// cacheRef 的布局契约: 必须连续, 或仅 blockNum(维0)维非连续(分页)。
// [blockNum, blockSize, 1, headDim] 须满足:
//   - headDim 维 stride == 1(kernel 按行内连续字节散写);
//   - blockSize 维 stride == headDim(块内行紧密排布);
//   - blockNum 维 stride >= blockSize * headDim(块不重叠, == 时即完全连续)
// 长度为 1 的轴其 stride 不参与寻址(索引恒为 0), 跳过校验:
bool CheckCacheLayout(const aclTensor* t, const char* name)
{
    const auto& viewShape = t->GetViewShape();
    if (viewShape.GetDimNum() != CACHE_VIEW_DIM_NUM) {
        return true;
    }
    const auto& strides = t->GetViewStrides();
    if (strides.size() != CACHE_VIEW_DIM_NUM) {
        return true; // 无 stride 信息, 按连续布局处理
    }
    const int64_t blockNum = viewShape.GetDim(CACHE_BLOCKNUM_DIM);
    const int64_t blockSize = viewShape.GetDim(CACHE_BLOCKSIZE_DIM);
    const int64_t headDim = viewShape.GetDim(CACHE_HEAD_DIM);
    const int64_t headStride = strides[CACHE_HEAD_DIM];
    const int64_t rowStride = strides[CACHE_BLOCKSIZE_DIM];
    const int64_t blockStride = strides[CACHE_BLOCKNUM_DIM];
    // dim2 固定为 1, 其 stride 不参与寻址, 不校验。
    if (headDim > 1 && headStride != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s: stride of headDim(dim3) must be 1 (contiguous within a row), got %ld.",
                name, static_cast<long>(headStride));
        return false;
    }
    if (blockSize > 1 && rowStride != headDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s: stride of blockSize(dim1) must equal headDim(%ld), got %ld. Only the blockNum(dim0) dimension "
                "may be non-contiguous.",
                name, static_cast<long>(headDim), static_cast<long>(rowStride));
        return false;
    }
    if (blockNum > 1 && blockStride < blockSize * headDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s: stride of blockNum(dim0) must be >= blockSize * headDim(%ld), got %ld. Blocks must not overlap.",
                name, static_cast<long>(blockSize * headDim), static_cast<long>(blockStride));
        return false;
    }
    return true;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(aclTensor* cacheRef, const aclTensor* x, const aclTensor* slotMapping,
                                                  int64_t quantGroupSize, int64_t quantMode, bool roundScale,
                                                  float xScale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnKvCompressEpilog,
                   DFX_IN(cacheRef, x, slotMapping, quantGroupSize, quantMode, roundScale, xScale), DFX_OUT());

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(cacheRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(slotMapping, return ACLNN_ERR_PARAM_NULLPTR);

    // 平台分流: arch22 (A2/A3) 执行 quant_mode=1 (int8) 专属参数校验;
    const bool isArch22 = (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201);
    if (isArch22) {
        if (quantMode != QUANT_MODE_INT8_ACLNN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "quant_mode %lld is not supported on arch22 (A2/A3); only quant_mode=1 (INT8) is supported",
                    static_cast<long long>(quantMode));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (quantGroupSize != DEFAULT_QUANT_GROUP_SIZE_ACLNN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quant_group_size must be 64 when quant_mode==1, got %lld",
                    static_cast<long long>(quantGroupSize));
            return ACLNN_ERR_PARAM_INVALID;
        }

        auto xShape = x->GetViewShape();
        auto cacheShape = cacheRef->GetViewShape();
        auto slotShape = slotMapping->GetViewShape();

        // x: 2D [bs, 512], bf16, 非空
        if (x->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "empty tensor is not supported (x)!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (xShape.GetDimNum() != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be 2D [bs, d] when quant_mode==1, got dimNum=%u",
                    static_cast<unsigned>(xShape.GetDimNum()));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (xShape.GetDim(1) != D_LENGTH_MODE3_ACLNN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "d must be 512 when quant_mode==1, got %ld",
                    static_cast<long>(xShape.GetDim(1)));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x->GetDataType() != DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dtype must be bfloat16, got %d", static_cast<int>(x->GetDataType()));
            return ACLNN_ERR_PARAM_INVALID;
        }

        // cache: 4D [blockNum, blockSize, 1, headDim], uint8, blockNum>0, headDim>=608,
        //        blockStride >= blockSize*headDim (分页视图)
        if (cacheShape.GetDimNum() != 4 || cacheShape.GetDim(2) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cache must be 4D with dim2 == 1 when quant_mode==1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cacheRef->GetDataType() != DataType::DT_UINT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cache dtype must be uint8, got %d",
                    static_cast<int>(cacheRef->GetDataType()));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cacheShape.GetDim(0) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cache blockNum must be > 0, got %ld",
                    static_cast<long>(cacheShape.GetDim(0)));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cacheRef->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "empty tensor is not supported (cache)!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t headDim = cacheShape.GetDim(3);
        if (headDim < KVCACHE_COL_MODE3_ACLNN) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cache headDim must be >= 608 when quant_mode==1, got %ld",
                    static_cast<long>(headDim));
            return ACLNN_ERR_PARAM_INVALID;
        }
        {
            auto cacheStrides = cacheRef->GetViewStrides();
            const int64_t blockSize = cacheShape.GetDim(1);
            const int64_t blockStride = cacheStrides[0];
            if (blockStride < blockSize * headDim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cache blockStride must be >= blockSize * headDim, got %ld < %ld",
                        static_cast<long>(blockStride), static_cast<long>(blockSize * headDim));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        // slot_mapping: int32/int64, 1D [bs] 与 x bs 一致, 非空
        if (slotMapping->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "empty tensor is not supported (slot_mapping)!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (slotMapping->GetDataType() != DataType::DT_INT32 && slotMapping->GetDataType() != DataType::DT_INT64) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "slot_mapping dtype only supports int32/int64, got %d",
                    static_cast<int>(slotMapping->GetDataType()));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (slotShape.GetDimNum() != 1 || slotShape.GetDim(0) != xShape.GetDim(0)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "slot_mapping length must equal x bs, got dimNum=%u len=%ld, x bs=%ld",
                    static_cast<unsigned>(slotShape.GetDimNum()), static_cast<long>(slotShape.GetDim(0)),
                    static_cast<long>(xShape.GetDim(0)));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // cacheRef 布局契约校验: 连续或仅 blockNum 维非连续(分页), 其余布局直接报错。
    CHECK_RET(CheckCacheLayout(cacheRef, "cacheRef"), ACLNN_ERR_PARAM_INVALID);

    auto kvCacheView = uniqueExecutor->CreateView(cacheRef, cacheRef->GetViewShape(), cacheRef->GetStorageShape(),
                                                  cacheRef->GetViewStrides(), cacheRef->GetViewOffset());
    CHECK_RET(kvCacheView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // x/slotMapping 的 kernel 按 view 逻辑 shape 连续寻址, 非连续输入先统一转连续(连续时零拷贝透传),
    // 保证按用户视角的逻辑 shape 语义计算。
    const aclTensor* xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::KvCompressEpilog(const_cast<aclTensor*>(kvCacheView), xContiguous, slotMappingContiguous,
                                         quantGroupSize, quantMode, roundScale, xScale, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvCompressEpilog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvCompressEpilog);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
