/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_util.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/fill.h"
#include "level0/padv3.h"
#include "level0/mul.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "common/op_api_def.h"
#include "common/op_host/op_api/cube_util.h"
#include "common/op_host/math_util.h"
#include "mat_mul_v3/op_host/op_api/matmul.h"
#include "aclnn_mm_white_list.h"

using namespace std;
using namespace op;
using namespace Ops::Transformer;

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

namespace {
static const int64_t SPLIT_K_MULTI = 8;
static const int64_t MKN_MAX = 8000000000;
static const int64_t MN_MULTI = 50;
static const int64_t N_KEQAL1_LIMIT = 4000;
static const int64_t MM_KEQAL1_LIMIT = 10000;
static const size_t MM_DIM = 2;
static const int32_t INNER_AXIS = 1;
static const int32_t OUTER_AXIS = 2;
static const int64_t DIM_EQUAL_ONE = 1;
static const uint64_t SMALL_SHAPE_LIMIT = 524288UL;
static const uint32_t kDeqScaleMul = 0xFFFFE000;
static const uint64_t BASIC_BLOCK_SIZE_256 = 256;
static const uint64_t NUM_HALF = 2;
static const uint64_t FP32_HF32_DTYPE_SIZE = 4;
static const uint64_t BASIC_BLOCK_K_256_BYTE = 256;
static const uint64_t BASIC_BLOCK_SIZE_32 = 32;
static const uint64_t CACHELINE = 512;
static const uint64_t NUM_TWO = 2;
static const int64_t DIMS_TWO = 2;
static const int64_t HALF_ALIGN_UNIT = 256;
static const int64_t ALIGN_UNIT = 512;
static const int64_t M_DIM_SELF_IDX = 0;
static const int64_t K_DIM_SELF_IDX = 1;
static const int64_t N_DIM_SELF_IDX = 1;
static const int64_t ALIGN_UNIT_16 = 16;
static const int64_t ALIGN_UNIT_128 = 128;
static const int64_t MIN_V3_SHAPE_310 = 2048;
static const int64_t MAX_V3_SHAPE_310 = 5504;
static const int64_t SINGLE_CORE_SPLIT_K = 27392;
static const int64_t BLOCK_CUBE = 16;
static const int64_t BLOCK_BYTE_SIZE = 32;
static const uint64_t MB = 1024UL * 1024UL;

static const std::initializer_list<op::DataType> V100_DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = (mat2Float || selfFloat) ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static inline bool CheckKEqual1Support(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E &&
           GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType)
{
    bool bf16flag = CheckSocVersionIsSupportBf16();
    if (bf16flag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
        }
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        }
    }

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat2->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(mat2->GetDataType()).GetString());
        return false;
    }
    if (out != nullptr && cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
        self->GetDataType() == op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
        return false;
    }
    // self和mat2的dtype不相等时，会做promote处理。
    bool dtype_match = self->GetDataType() == mat2->GetDataType();
    if (!dtype_match) {
        OP_LOGW(
            "Self's dtype [%s] and mat2's dtype [%s] are not equal. Promotion of Data Type will be applied",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* self, const aclTensor* mat2, bool transposeX2 = false)
{
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    int64_t mat2KDim = transposeX2 ? mat2Shape.GetDim(K_DIM_SELF_IDX) : mat2Shape.GetDim(M_DIM_SELF_IDX);
    int64_t selfKDim = selfShape.GetDim(K_DIM_SELF_IDX); // self固定不转置
    if (mat2KDim != selfKDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different, self Kdim[%ld], mat2 Kdim[%ld].",
            selfKDim, mat2KDim);
        return false;
    }
    return true;
}

static bool CheckShapeValidWithTrans(const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2)
{
    OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    auto selfKDim = transSelf > 0 ? selfShape.GetDim(0) : selfShape.GetDim(1);
    auto mat2KDim = transMat2 > 0 ? mat2Shape.GetDim(1) : mat2Shape.GetDim(0);
    if (selfKDim != mat2KDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different.");
        return false;
    }
    return true;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape outShape = {selfShape.GetDim(0), mat2Shape.GetDim(1)};
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    if (out->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return out;
    }
    FVector<int64_t> fillShape = GetShape(out);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static const aclTensor* ProcessEmptyTensorWithTrans(
    const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    auto mDim = transSelf > 0 ? selfShape.GetDim(1) : selfShape.GetDim(0);
    auto nDim = transMat2 > 0 ? mat2Shape.GetDim(0) : mat2Shape.GetDim(1);
    op::Shape outShape = {mDim, nDim};
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    if (out->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return out;
    }
    FVector<int64_t> fillShape = GetShape(out);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static bool CheckSupportSingleSplitKBf16(
    const aclTensor* self, const aclTensor* mat2, const DataType selfDtype, const DataType mat2Dtype)
{
    // 判决门限
    // 1. 输入数据类型为bf16
    // 2. 在K轴非256字节对齐场景下，输入数据大小不超过INT32最大值
    // 3. K轴大于27392
    // 4. M、N中最大不超过K轴的一半
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        return false;
    }
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();

    int64_t splitKMultiThres = 2;
    int64_t kDim = selfShape.GetDim(1);
    bool dtypeCorrect = (selfDtype == DataType::DT_BF16) && (mat2Dtype == DataType::DT_BF16);
    int64_t checkSize = 0;

    int dtypeASize = ge::GetSizeByDataType(selfDtype);
    int dtypeBSize = ge::GetSizeByDataType(mat2Dtype);
    if ((kDim * dtypeASize) % 256 != 0) { // check if inner-dim is aligned to 256 byte
        checkSize += selfShape[selfShape.GetDimNum() - 1] * selfShape[selfShape.GetDimNum() - DIMS_TWO] * dtypeASize;
        checkSize += mat2Shape[mat2Shape.GetDimNum() - 1] * mat2Shape[mat2Shape.GetDimNum() - DIMS_TWO] * dtypeBSize;
    }

    bool checkMemSize = checkSize <= INT32_MAX;
    return dtypeCorrect && checkMemSize && kDim >= SINGLE_CORE_SPLIT_K &&
           kDim >= splitKMultiThres * std::max(selfShape.GetDim(0), mat2Shape.GetDim(1));
}

// 1980/1951 支持fp16进 fp16/fp32出，非对齐case 只能NZ进出，对齐case支持ND进出
static aclnnStatus SetMatmulOpSupportInfo(
    const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo, int8_t cubeMathType)
{
    // 判断传入L0接口，用于计算的Dtype
    SetMmSupportDType(mmOpInfo, cubeMathType);

    // 判断当前Shape是否支持使用ND输入输出
    SetMmSupportFormat(self, mat2, mmOpInfo);

    TensorInfo SpTensor_sefl = {self, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.self_format};
    TensorInfo SpTensor_mat2 = {mat2, mmOpInfo.support_info.mat2_dtype, mmOpInfo.support_info.output_format};

    if (IsSplitk(&SpTensor_sefl, &SpTensor_mat2)) {
        mmOpInfo.supporSplitK = true;
        if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
            mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        }
    }

    if (CheckSupportSingleSplitKBf16(self, mat2, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.mat2_dtype)) {
        OP_LOGI("Hit mat_mul_v3 ND bf16 single core splitK case channel.");
        mmOpInfo.support_info.output_format = Format::FORMAT_ND;
        mmOpInfo.support_info.self_format = Format::FORMAT_ND;
        mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
        mmOpInfo.supporSplitK = true;
    }

    // self=nd, mat2=nz不支持切K
    bool isNdNzIn =
        self->GetStorageFormat() == Format::FORMAT_ND && mat2->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = isNdNzIn ? Format::FORMAT_FRACTAL_NZ : mmOpInfo.support_info.mat2_format;
    mmOpInfo.support_info.output_dtype =
        isNdNzIn ? mmOpInfo.support_info.mat2_dtype : mmOpInfo.support_info.output_dtype;
    return ACLNN_SUCCESS;
}

static MmOpInfo GetMatmulOpInfoWithTrans(
    const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2, int8_t cubeMathType)
{
    // 获取m、k、n轴的大小
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t mDim = transSelf > 0 ? selfShape.GetDim(1) : selfShape.GetDim(0);
    int64_t kDim = transSelf > 0 ? selfShape.GetDim(0) : selfShape.GetDim(1);
    int64_t nDim = transMat2 > 0 ? mat2Shape.GetDim(0) : mat2Shape.GetDim(1);

    // Dtype和Format初始化
    MmOpInfo mmOpInfo;
    mmOpInfo.ori_info.self_dtype = self->GetDataType();
    mmOpInfo.ori_info.self_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    mmOpInfo.ori_info.mat2_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.output_dtype = self->GetDataType();
    mmOpInfo.ori_info.output_format = op::Format::FORMAT_ND;

    mmOpInfo.shapeInfo.kDim = kDim;
    mmOpInfo.shapeInfo.nDim = nDim;
    mmOpInfo.shapeInfo.mDim = mDim;

    mmOpInfo.shapeInfo.transposeX1 = transSelf > 0;
    mmOpInfo.shapeInfo.transposeX2 = transMat2 > 0;
    mmOpInfo.support_info = mmOpInfo.ori_info;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    mmOpInfo.opImplModeEnum = (cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32) ? 0x40 : 0x1;
    mmOpInfo.enableHf32 = (cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32);
    OP_LOGD(
        "opImplModeEnum=%ld, enableHf32=%d, cubeMathType=%d", mmOpInfo.opImplModeEnum, mmOpInfo.enableHf32,
        cubeMathType);

    SetMatmulOpSupportInfo(self, mat2, mmOpInfo, cubeMathType);
    GetMmInfo(mmOpInfo);
    return mmOpInfo;
}

inline bool CheckCacheL2(uint64_t sizeA, uint64_t sizeB, uint64_t sizeC)
{
    constexpr double cacheRatio = 0.6666;
    uint64_t l2Size = GetL2Size(GetCurrentPlatformInfo().GetSocLongVersion());
    uint64_t cacheL2Size = static_cast<uint64_t>(ceil(cacheRatio * l2Size) * MB);
    uint64_t totalSize = sizeA + sizeB + sizeC;
    return !(totalSize < (l2Size * MB) && sizeA < cacheL2Size && sizeB < cacheL2Size && sizeC < cacheL2Size);
}

inline static bool CheckNetShape(uint64_t mDim, uint64_t nDim, uint64_t kDim, bool isFloat32)
{
    constexpr uint64_t mStep = 16384UL;
    constexpr uint64_t mLimitSize = 524288UL;
    constexpr uint64_t netShapeArg1 = 7168UL;
    constexpr uint64_t netShapeArg2 = 576UL;
    constexpr uint64_t netShapeArg3 = 256UL;
    // netshape1: k = 7168, n = 576, m = 32k -- 512k step 16k
    // netshape4: k = 7168, n = 576, m = 262144
    if (!isFloat32 && kDim == netShapeArg1 && nDim == netShapeArg2 &&
        (mDim <= mLimitSize && mDim % mStep == 0 && mDim != mStep)) {
        return true;
    }
    // netshape2: is float32 k = 7168, n = 256, m = 32k -- 512k step 16k
    if (isFloat32 && kDim == netShapeArg1 && nDim == netShapeArg3 &&
        (mDim <= mLimitSize && mDim % mStep == 0 && mDim != mStep)) {
        return true;
    }
    // netshape3: N=7168, k = 576 & 1152 m = 32k -- 512k step 16K, is surported by CheckSize()
    return false;
}

inline static bool CheckSize(uint64_t mDim, uint64_t nDim)
{
    constexpr uint64_t limitSizeM = 1152UL;
    constexpr uint64_t limitSizeN = 960UL;
    return (mDim >= limitSizeM && nDim >= limitSizeN);
}

inline static bool CheckKForNZ(int64_t kDim) {
    // 测试发现k < 640 时性能较v2 NZ劣化
    constexpr int64_t threshold = 640;
    // 当前k范围在现有条件下仅走基础模板, 后续修改tiling需要重新评估
    return (kDim >= threshold && kDim < SINGLE_CORE_SPLIT_K);
}

inline static bool CheckSupportMultiSplitK(
    const int64_t mDim, const int64_t kDim, const int64_t nDim, const bool transposeX1, const bool transposeX2)
{
    constexpr int64_t splitKThres = 27392L;
    constexpr int64_t deterThresOutSize = 4L;
    constexpr int64_t mbSize = 1024L * 1024L;
    constexpr int64_t splitKMultiThres = 8L;
    bool splitKScene = IsSmallMNMultiSplitK(
        static_cast<uint64_t>(mDim), static_cast<uint64_t>(kDim), static_cast<uint64_t>(nDim), transposeX1,
        transposeX2);
    return ((kDim >= splitKThres) && (mDim * nDim <= deterThresOutSize * mbSize) &&
            kDim >= splitKMultiThres * std::max(mDim, nDim)) ||
           splitKScene;
}

inline static bool CheckN1ToV3Case(const int64_t kDim, const int64_t nDim, const bool transposeX2)
{
    constexpr int64_t minKDim = 100000L;
    constexpr int64_t nDimOne = 1L;
    if (!transposeX2) {
        return false;
    }
    bool dimScene = (nDim == nDimOne && kDim > minKDim);
    return dimScene;
}

inline static bool CheckMFrom1To256(
    const int64_t mDim, const int64_t kDim, const int64_t nDim, const bool transposeX1, const bool transposeX2)
{
    constexpr int64_t minMDim = 1L;
    constexpr int64_t maxMDim = 256L;
    constexpr int64_t kDimValue = 7168L;
    constexpr int64_t nDimValue = 256L;
    if (transposeX1 || !transposeX2) {
        return false;
    }
    bool mDimScene = (minMDim <= mDim && mDim <= maxMDim && kDim == kDimValue && nDim == nDimValue);
    return mDimScene;
}

inline static bool CheckSmallMNSupportMultiSplitK(
    const int64_t mDim, const int64_t kDim, const int64_t nDim, const bool transposeX1, const bool transposeX2)
{
    // 如果MN <= 64, K >= 6144 时走多核切K
    constexpr int64_t DIM64 = 64L;
    constexpr int64_t DIM6144 = 6144L;
    if (mDim <= DIM64 && nDim <= DIM64 && kDim >= DIM6144) {
        return true;
    }

    constexpr int64_t minKDim = 1000L;
    constexpr int64_t maxKDim = 4608L;
    constexpr int64_t maxMnDim = 256L;
    constexpr int64_t fp32Align = 8L;
    if (!transposeX1 || transposeX2) {
        return false;
    }
    bool dimScene = (kDim >= minKDim && kDim <= maxKDim && mDim <= maxMnDim && nDim <= maxMnDim);
    bool dimAlignScene = (mDim % fp32Align != 0 && nDim % fp32Align == 0);
    return dimScene && dimAlignScene;
}

inline static bool CheckSmallMNSupportMultiSplitKFp16Bf16(
    const int64_t mDim, const int64_t kDim, const int64_t nDim, const bool transposeX1, const bool transposeX2)
{
    constexpr int64_t maxKDim = 4608L;
    constexpr int64_t minKDim = 4000L;
    constexpr int64_t maxMnDim = 256L;
    constexpr int64_t minMnDim = 128L;
    constexpr int64_t fp16Bf16Align = 8L;
    if (transposeX2 || !transposeX1) {
        return false;
    }
    bool dimScene =
        (kDim >= minKDim && kDim <= maxKDim && mDim > minMnDim && mDim <= maxMnDim && nDim > minMnDim &&
         nDim <= maxMnDim);
    bool dimAlignScene = (mDim % fp16Bf16Align != 0 && nDim % fp16Bf16Align == 0);
    return dimScene && dimAlignScene;
}

inline static bool CheckFixpipeBoundCase(
    const int64_t mDim, const int64_t kDim, const int64_t nDim, const aclTensor* x1, const bool transposeX1,
    const bool transposeX2)
{
    int64_t dtypeASize = ge::GetSizeByDataType(x1->GetDataType());
    CHECK_RET(dtypeASize != 0L, false);
    // 考虑L0C上的较优的切分时N=256，不感知dtype， 因此限制N的大小为n < 256
    bool smallBtensor = nDim < HALF_ALIGN_UNIT;
    bool bigAtensor = mDim > ALIGN_UNIT * static_cast<int64_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    bool isFixpipeBound = (kDim <= HALF_ALIGN_UNIT) &&
                          (nDim % (HALF_ALIGN_UNIT / dtypeASize) != 0 && ((HALF_ALIGN_UNIT / dtypeASize) % nDim != 0));
    bool isScalarBound = (nDim < (32 / dtypeASize)) && (kDim < (32 / dtypeASize)); // 32 means block size
    // this fp32 case whill hit BL1FullloadWithFixpipe ，scalar bound(when v3 solved ,can del4te)
    bool isScalarBoundInV3FixpipeOPT =
        !transposeX1 && (kDim <= 80 && kDim % 8 == 0) &&
        (transposeX2 || (nDim <= 80 && nDim % 8 == 0)); // 80 is experience value. 8 means block / sizeof(float)
    bool isMte2Bound = (dtypeASize == 2L);              // 2L means float16 and bfloat16
    if (smallBtensor && bigAtensor && isFixpipeBound && !isScalarBound && !isScalarBoundInV3FixpipeOPT &&
        !isMte2Bound) {
        return true;
    }
    return false;
}

inline static bool IsSupportedInnerDim(
    const op::Shape& shapeX1, int dtypeASize, const op::Shape& shapeX2, int dtypeBSize)
{
    int64_t checkSize = 0;
    if ((shapeX1[shapeX1.GetDimNum() - 1] * dtypeASize) % 256 != 0) { // check if inner-dim is aligned to 256 byte
        checkSize +=
            shapeX1[shapeX1.GetDimNum() - 1] * shapeX1[shapeX1.GetDimNum() - 2] * dtypeASize; // 2: outer axis idx
    }
    if ((shapeX2[shapeX2.GetDimNum() - 1] * dtypeBSize) % 256 != 0) { // check if inner-dim is aligned to 256 byte
        checkSize +=
            shapeX2[shapeX2.GetDimNum() - 1] * shapeX2[shapeX2.GetDimNum() - 2] * dtypeBSize; // 2: outer axis idx
    }
    return checkSize <= INT32_MAX; // roughly 64GB of memory is required when matrix area approaches int32_max
}

inline static bool CheckSupportVnchwconv(int64_t outerDim, int64_t innerDim, int dtypeSize)
{
    if (dtypeSize == 0) {
        return false;
    }
    const std::vector<uint64_t> supportNd2nzGm2l0 = {32, 64, 96, 128, 160, 192, 224, 256, 384}; // check if is inclusive
    constexpr int64_t innerThres = 512L;                        // inner size should below 512
    constexpr int64_t align256B = 256L;                         // whether the size is 256B aligned
    bool is256BAlign = (innerDim * dtypeSize) % align256B == 0; // 256 means 256B align
    bool supportNd2NzOnTheWay =
        std::find(supportNd2nzGm2l0.begin(), supportNd2nzGm2l0.end(), innerDim * dtypeSize) != supportNd2nzGm2l0.end();
    // outersize over 8192 and (innersize below 384B when it is even or below 192B when it is odd) for vnchw.
    bool willFitVnchwCond = outerDim > 8192 && (innerDim > 1) &&
                            (innerDim * dtypeSize <= 192 || (innerDim * dtypeSize <= 384 && innerDim % 2 == 0) ||
                             (innerDim * dtypeSize <= innerThres && innerDim % 4 == 0));
    bool willInnerSizeEqualC0 = (innerDim == (32L / dtypeSize));
    // if the shape size is (1, 512B] and not aligned to 256B and not inclusive, matmul op can be changed as matmulv3.
    return (willFitVnchwCond && !is256BAlign && !supportNd2NzOnTheWay && !willInnerSizeEqualC0);
}

static bool IsMiddleSizedShape(int64_t m, int64_t k, int64_t n)
{
    constexpr int64_t baseBlock256 = 256L;
    constexpr int64_t baseBlock128 = 128L;
    constexpr double threshold = 0.7;
    auto isMiddleSizedDim = [](int64_t dim) -> bool {
        return dim >= 1024 && dim <= 10368; // currently define "middle-sized" dimension interval as [1024, 10368]
    };
    auto isMMiddleSizedDim = [](int64_t dim) -> bool {
        return dim >= 1024 && dim <= 18000; // currently define "m-middle-sized" dimension interval as [1024, 18000]
    };
    auto isMultiple1280Dim = [](int64_t dim) -> bool {
        return dim % 1280 == 0;
    };
    if (!(isMMiddleSizedDim(m) && isMiddleSizedDim(k) && isMiddleSizedDim(n))) {
        OP_LOGD("Not V3 middle-size case, dim value not within range");
        return false;
    }
    if (!(isMiddleSizedDim(m) || (isMultiple1280Dim(k) && isMultiple1280Dim(n)))) {
        OP_LOGD("Not V3 middle-size case, dim value not within multiple 1280 optimize range");
        return false;
    }

    int64_t coreNum = static_cast<int64_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    CHECK_COND(coreNum > 0L, false, "Invalid AI core num [%ld]", coreNum);

    auto getAverageMADCount = [m, k, n, coreNum]() -> int64_t { // currently not apply to fp32 hf32
        int64_t cntMAD = CeilDiv(m, BLOCK_CUBE) * CeilDiv(n, BLOCK_CUBE) * CeilDiv(k, BLOCK_CUBE);
        return cntMAD / coreNum;
    };
    if (getAverageMADCount() <= 90000L) { // 90000 is threshold
        OP_LOGD("Not V3 middle-size case, MAD count not great enough");
        return false;
    }

    auto getCoreUtilization = [m, n, coreNum](int64_t baseM, int64_t baseN) -> double {
        int64_t cnt = CeilDiv(m, baseM) * CeilDiv(n, baseN);
        return static_cast<double>(cnt) / CeilAlign(cnt, coreNum);
    };
    if (getCoreUtilization(baseBlock128, baseBlock256) <= threshold &&
        getCoreUtilization(baseBlock256, baseBlock128) <= threshold) { // 128 256 base block, 0.7 threshold
        OP_LOGD("Not V3 middle-size case, AI core utilization not great enough");
        return false;
    }

    return true;
}

inline static bool CheckSmallKSupportSingleSplitKFp16Bf16(
    const int64_t mDim, const int64_t kDim, const int64_t nDim)
{
    static const int64_t SINGLE_CORE_SPLIT_SMALL_K = 1536;
    static const int64_t SINGLE_CORE_SPLIT_SMALL_MN = 384;
    static const int64_t SINGLE_CORE_SPLIT_LARGE_MN = 128 * 384;
    static const uint64_t L2_CACHE_ALLOWANCE = 8 * MB;
    // 获取L2缓存大小
    uint64_t l2Size = GetL2Size(GetCurrentPlatformInfo().GetSocLongVersion());
    // 条件1： M,N 被128整除，K=1536
    bool isSmallKwithLargeMN = (kDim == SINGLE_CORE_SPLIT_SMALL_K) &&
                            (mDim % ALIGN_UNIT_128 == 0 && nDim % ALIGN_UNIT_128 == 0);
    // 条件2： N>>M 且 M = 384 或 M>>N 且 N = 384 ;
    bool isLargeNSmallM = (mDim == SINGLE_CORE_SPLIT_SMALL_MN && nDim >= SINGLE_CORE_SPLIT_LARGE_MN);
    bool isLargeMSmallN = (nDim == SINGLE_CORE_SPLIT_SMALL_MN && mDim >= SINGLE_CORE_SPLIT_LARGE_MN);
    // 条件3： M * N <= (L2 - 8) Mb， 以保证过程矩阵拷出不会溢出L2缓存;
    int64_t l2CacheLimitation = (l2Size * MB - L2_CACHE_ALLOWANCE) / FP32_HF32_DTYPE_SIZE;
    bool isL2Enough = nDim * mDim <= l2CacheLimitation;
    return (isSmallKwithLargeMN && (isLargeNSmallM || isLargeMSmallN) && isL2Enough);
}

static bool CheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const MmOpInfo& mmOpInfo, const bool transposeX1,
    const bool transposeX2)
{
    op::Shape shapeX1 = x1->GetViewShape();
    op::Shape shapeX2 = x2->GetViewShape();
    int64_t innerADim = shapeX1[shapeX1.GetDimNum() - 1];
    int64_t innerBDim = shapeX2[shapeX2.GetDimNum() - 1];
    int64_t outerADim = shapeX1[shapeX1.GetDimNum() - 2]; // 2: dim index
    int64_t outerBDim = shapeX2[shapeX2.GetDimNum() - 2]; // 2: dim index
    int64_t mDim = transposeX1 ? innerADim : outerADim;
    int64_t kDim = transposeX1 ? outerADim : innerADim;
    int64_t nDim = transposeX2 ? outerBDim : innerBDim;
    int dtypeASize = ge::GetSizeByDataType(x1->GetDataType());
    int dtypeBSize = ge::GetSizeByDataType(x2->GetDataType());
    uint64_t sizeA = mDim * kDim * dtypeASize;
    uint64_t sizeB = nDim * kDim * dtypeBSize;
    uint64_t sizeC = mDim * nDim * dtypeASize;
    int64_t x1DtypeFlag = (x1->GetDataType() == DataType::DT_FLOAT) ? FP32_FLAG : FP16_BF16_FLAG;

    bool hasBias = bias != nullptr;
    std::initializer_list<int64_t> checkCase = {
        mDim,
        kDim,
        nDim,
        static_cast<int64_t>(transposeX1),
        static_cast<int64_t>(transposeX2),
        static_cast<int64_t>(hasBias),
        x1DtypeFlag};
    OP_LOGI(
        "Checking V3 Case: m = %ld, k = %ld, n = %ld, trans1 = %ld, trans2 = %ld, bias = %ld, dtype-flag = %ld", mDim,
        kDim, nDim, static_cast<int64_t>(transposeX1), static_cast<int64_t>(transposeX2), static_cast<int64_t>(hasBias),
        x1DtypeFlag);

    std::function<bool(const std::initializer_list<std::initializer_list<int64_t>>)> caseCheckFun =
        [checkCase](const std::initializer_list<std::initializer_list<int64_t>>& list) -> bool {
        return std::any_of(list.begin(), list.end(), [checkCase](std::initializer_list<int64_t> oneCase) {
            return checkCase.size() == oneCase.size() && std::equal(oneCase.begin(), oneCase.end(), checkCase.begin());
        });
    };
    // 部分shape满足以下条件切换matmulV3：B3\B4 && FP16 && 满足shape白名单
    if (mmOpInfo.support_info.mat2_format == ge::FORMAT_FRACTAL_NZ && x1DtypeFlag != FP32_FLAG) {
        bool hitB3Shape =
            ((GetCurrentPlatformInfo().GetSocLongVersion() == SOC_B3) &&
             caseCheckFun(matmul_ascendc_list::ASCEND_C_NDNZ_WHITE_LIST_B3));
        bool hitB4Shape =
            ((GetCurrentPlatformInfo().GetSocLongVersion() == SOC_B4) &&
             caseCheckFun(matmul_ascendc_list::ASCEND_C_NDNZ_WHITE_LIST_B4));
        if (x1->GetDataType() == DataType::DT_FLOAT16 && (hitB3Shape || hitB4Shape)) {
            OP_LOGI("Hit mat_mul_v3 NDNZ case channel.");
            return true;
        }

        if (!CheckKForNZ(kDim)) {
            OP_LOGI("Not hit requirement of K for mat_mul_v3 NDNZ.");
            return false;
        }

        if ((x1DtypeFlag == FP16_BF16_FLAG) && IsMiddleSizedShape(mDim, kDim, nDim)) {
            OP_LOGI("Hit mat_mul_v3 middle-sized shape channel.");
            return true;
        }

        OP_LOGD("check Hit mat_mul_v3 NDNZ l2 channel.sizeA:%lu, sizeB:%lu, sizeC:%lu",
            sizeA, sizeB, sizeC);
        if ((CheckSize(mDim, nDim) || CheckNetShape(mDim, nDim, kDim, x2->GetDataType() == DataType::DT_FLOAT))
            && CheckCacheL2(sizeA, sizeB, sizeC)) {
            OP_LOGI("Hit mat_mul_v3 NDNZ l2 channel.");
            return true;
        }

        OP_LOGI("Not hit mat_mul_v3 NDNZ case channel.");
        return false;
    }
    if (mmOpInfo.support_info.mat2_format != ge::FORMAT_ND) {
        return false;
    }

    if (caseCheckFun(matmul_ascendc_list::ASCEND_C_BLACK_LIST)) {
        return false;
    }
    bool hitWhiteList = false;
    if (GetCurrentPlatformInfo().GetSocLongVersion() == SOC_B3 ||
        GetCurrentPlatformInfo().GetSocLongVersion() == SOC_C3) {
        hitWhiteList = caseCheckFun(matmul_ascendc_list::ASCEND_C_WHITE_LIST_B3);
    } else if (
        GetCurrentPlatformInfo().GetSocLongVersion() == SOC_B4 ||
        GetCurrentPlatformInfo().GetSocLongVersion() == SOC_C4) {
        hitWhiteList = caseCheckFun(matmul_ascendc_list::ASCEND_C_WHITE_LIST_B4);
    } else {
        hitWhiteList = caseCheckFun(matmul_ascendc_list::ASCEND_C_WHITE_LIST);
    }
    if (hitWhiteList) {
        OP_LOGI("Hit mat_mul_v3 case channel.");
        return true;
    }

    if ((x1DtypeFlag == FP16_BF16_FLAG) && IsMiddleSizedShape(mDim, kDim, nDim)) {
        OP_LOGI("Hit mat_mul_v3 middle-sized shape channel.");
        return true;
    }

    if (!hasBias && CheckN1ToV3Case(kDim, nDim, transposeX2)) {
        OP_LOGI("Hit mat_mul_v3 nDim is 1 shape channel.");
        return true;
    }

    if (CheckMFrom1To256(mDim, kDim, nDim, transposeX1, transposeX2) && (x1DtypeFlag == FP32_FLAG)) {
        OP_LOGI("Hit mat_mul_v3 mDim from 1 to 256 channel.");
        return true;
    }

    if (CheckSmallMNSupportMultiSplitK(mDim, kDim, nDim, transposeX1, transposeX2)) {
        if (x1DtypeFlag == FP32_FLAG) {
            OP_LOGI("Hit mat_mul_v3 small m&n deterministic multicoresplitk channel.");
            return true;
        }
    }

    if (CheckSmallMNSupportMultiSplitKFp16Bf16(mDim, kDim, nDim, transposeX1, transposeX2)) {
        if (x1DtypeFlag == FP16_BF16_FLAG) {
            OP_LOGI("Hit mat_mul_v3 small m&n deterministic multicoresplitk channel for fp16 and bf16.");
            return true;
        }
    }

    if (CheckSupportMultiSplitK(mDim, kDim, nDim, transposeX1, transposeX2)) {
        if ((x1->GetDataType() != DataType::DT_FLOAT16 && x1->GetDataType() != DataType::DT_BF16 &&
             x1->GetDataType() != DataType::DT_FLOAT) ||
            (x2->GetDataType() != DataType::DT_FLOAT16 && x2->GetDataType() != DataType::DT_BF16 &&
             x2->GetDataType() != DataType::DT_FLOAT)) {
            OP_LOGI("MatMulV3 deterministic multicoresplitk only hit fp16&bf16&fp32");
            return false;
        }
        OP_LOGI("Hit mat_mul_v3 deterministic multicoresplitk channel.");
        return true;
    }

    if (CheckFixpipeBoundCase(mDim, kDim, nDim, x1, transposeX1, transposeX2)) {
        OP_LOGI("Hit mat_mul_v3 fixpipe scence.");
        return true;
    }

    if (!IsSupportedInnerDim(shapeX1, dtypeASize, shapeX2, dtypeBSize)) {
        return false;
    }

    OP_LOGD(
        "check Hit mat_mul_v3 splitK channel.supportSplitK:%ld, kDim:%ld", static_cast<int64_t>(mmOpInfo.supporSplitK),
        kDim);
    if ((x1->GetDataType() == DataType::DT_FLOAT16 || x1->GetDataType() == DataType::DT_BF16) &&
        (x2->GetDataType() == DataType::DT_FLOAT16 || x2->GetDataType() == DataType::DT_BF16) && !hasBias) {
        bool splitK = mmOpInfo.supporSplitK && kDim >= SINGLE_CORE_SPLIT_K;
        if (splitK || CheckSmallKSupportSingleSplitKFp16Bf16(mDim, kDim, nDim)) {
            OP_LOGI("Hit mat_mul_v3 singleCoreSplitK channel.");
            return true;
        }
    }

    OP_LOGD("check Hit mat_mul_v3 l2 channel.sizeA:%lu, sizeB:%lu, sizeC:%lu", sizeA, sizeB, sizeC);
    if ((CheckSize(mDim, nDim) || CheckNetShape(mDim, nDim, kDim, x2->GetDataType() == DataType::DT_FLOAT)) &&
        CheckCacheL2(sizeA, sizeB, sizeC)) {
        OP_LOGI("Hit mat_mul_v3 l2 channel.");
        return true;
    }
    if ((CheckSupportVnchwconv(outerADim, innerADim, dtypeASize) ||
         CheckSupportVnchwconv(outerBDim, innerBDim, dtypeBSize)) &&
        innerADim != 1 && innerBDim != 1) { // avoid innerdim = 1 goes in mmv3
        OP_LOGI("Hit mat_mul_v3 unalign optimization process when innersize of each matrix below 512B.");
        return true;
    }

    OP_LOGI("Current MatMul operator still does not support MatMulV3 in current soc version, remains MatMulV2.");
    return false;
}

static bool CheckAscendCScenario(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const MmOpInfo& mmOpInfo, const bool transposeX1,
    const bool transposeX2)
{
    if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) ||
        mmOpInfo.support_info.self_format != ge::FORMAT_ND) {
        OP_LOGI("Not mat_mul_v3 case for unsupported SOC version or unsupported Format.");
        return false;
    }
    bool alwaysUseV3 = false;
    return (alwaysUseV3 || CheckHitV3Shape(x1, x2, bias, mmOpInfo, transposeX1, transposeX2));
}

static bool CheckAscendCScenario2(
    const aclTensor* x1, const aclTensor* x2, const MmOpInfo& mmOpInfo, const bool transposeX1, const bool transposeX2)
{
    if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P)) {
        return false;
    }
    if (x1->GetDataType() != DataType::DT_FLOAT16 || x2->GetDataType() != DataType::DT_FLOAT16) {
        return false;
    }
    if (mmOpInfo.support_info.self_format != ge::FORMAT_ND ||
        mmOpInfo.support_info.mat2_format != ge::FORMAT_FRACTAL_NZ ||
        mmOpInfo.support_info.output_format != ge::FORMAT_ND) {
        return false;
    }
    op::Shape shapeX1 = x1->GetViewShape();
    op::Shape shapeX2 = x2->GetViewShape();
    int64_t mDim = transposeX1 ? shapeX1[shapeX1.GetDimNum() - 1] : shapeX1[shapeX1.GetDimNum() - 2];
    int64_t kDim = transposeX1 ? shapeX1[shapeX1.GetDimNum() - 2] : shapeX1[shapeX1.GetDimNum() - 1];
    int64_t nDim = transposeX2 ? shapeX2[shapeX2.GetDimNum() - 2] : shapeX2[shapeX2.GetDimNum() - 1];
    if (mDim > MAX_V3_SHAPE_310 || mDim < MIN_V3_SHAPE_310 || kDim > MAX_V3_SHAPE_310 || kDim < MIN_V3_SHAPE_310 ||
        nDim > MAX_V3_SHAPE_310 || nDim < MIN_V3_SHAPE_310) {
        return false;
    }
    return true;
}

static const aclTensor* GetGemmV3Op(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* c, bool transposeX1, bool transposeX2, bool enableHf32,
    aclOpExecutor* executor)
{
    OP_LOGI("Hit gemmv3 scenario.");
    const aclTensor* mmOut = l0op::GemmV3Nd(x1, x2, c, transposeX1, transposeX2, enableHf32, executor);
    return mmOut;
}

static const aclTensor* GetMatMulOp(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, MmOpInfo& mmOpInfo, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const bool enableHf32, const int64_t opImplModeEnum,
    aclOpExecutor* executor)
{
    if (CheckAscendCScenario(x1, x2, bias, mmOpInfo, transposeX1, transposeX2) ||
        CheckAscendCScenario2(x1, x2, mmOpInfo, transposeX1, transposeX2)) {
        OP_LOGI("Hit matmul_v3 scenario.");
        const aclTensor* mmOut =
            l0op::MatMulV3Nd(x1, x2, bias, transposeX1, transposeX2, offsetX, enableHf32, executor);
        return mmOut;
    } else if (
        mmOpInfo.support_info.output_dtype == DataType::DT_FLOAT &&
        mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT16 &&
        mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT16 && bias == nullptr) {
        // This is Split K Mode; Check if MatMul using Nd in Nd Out
        const aclTensor* mmOut =
            (mmOpInfo.support_info.self_format == ge::FORMAT_ND &&
             mmOpInfo.support_info.output_format == ge::FORMAT_ND) ?
                l0op::MatMulNdFp162Fp32(
                    x1, x2, nullptr, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor) :
                l0op::MatMulNzFp162Fp32(
                    x1, x2, nullptr, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
        return mmOut;
    } else {
        if (mmOpInfo.support_info.self_format == ge::FORMAT_ND) {
            const aclTensor* mmOut =
                (mmOpInfo.support_info.mat2_format == ge::FORMAT_ND) ?
                    l0op::MatMulNd(x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor) :
                    l0op::MatMulNdNz(
                        x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
            return mmOut;
        } else {
            OP_LOGD("self format is not ND.");
            if (mmOpInfo.support_info.output_format == ge::FORMAT_ND) {
                OP_LOGD("Output format is ND, call MatMulNzNzNd.");
                const aclTensor* mmOut = l0op::MatMulNzNzNd(
                    x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
                return mmOut;
            }
            const aclTensor* mmOut =
                l0op::MatMulNz(x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
            return mmOut;
        }
    }
}

static inline int64_t ComputePadNum(int64_t kDim, int64_t dataSize)
{
    return CeilAlign(kDim, CeilDiv(ALIGN_UNIT, dataSize)) - kDim;
}

static inline const aclTensor* GetPadTensor(int64_t padNum, int64_t padDim, aclOpExecutor* executor)
{
    // pad: top bottom left right
    size_t dims = 4;
    std::vector<int64_t> padVec(dims, 0);

    padVec[padDim] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), dims);
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static bool CheckStreamKSKTiling(MmOpInfo& mmOpInfo)
{
    // 判断k轴是否大于32*512 / DtypeSize_, 小于就不走stream-k-sk
    uint64_t kAlign =
        static_cast<uint64_t>(CeilAlign(mmOpInfo.shapeInfo.kDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t aiCoreCnt = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.aiCoreCnt));
    uint64_t dtypeASize = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.shapeInfo.dtypeASize));
    uint64_t kThreshold = aiCoreCnt * NUM_HALF * BASIC_BLOCK_K_256_BYTE / dtypeASize;
    if (kAlign < kThreshold) {
        return false;
    }

    uint64_t alignValue = BASIC_BLOCK_SIZE_256;
    //
    if (mmOpInfo.shapeInfo.dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32) {
        alignValue = BASIC_BLOCK_SIZE_32; // 如果是Fp32 基本块判断要用32
    }
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.mDim, static_cast<int64_t>(alignValue)));
    uint64_t nCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.nDim, static_cast<int64_t>(alignValue)));
    return !(mCnt * nCnt > aiCoreCnt / NUM_HALF);
}

static bool CheckStreamKDPSKTiling(MmOpInfo& mmOpInfo)
{
    uint64_t aiCoreCnt = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.aiCoreCnt));
    uint64_t dtypeASize = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.shapeInfo.dtypeASize));
    uint64_t kThreshold = aiCoreCnt * BASIC_BLOCK_K_256_BYTE / dtypeASize;
    uint64_t kDim = static_cast<uint64_t>(mmOpInfo.shapeInfo.kDim);
    // 如果k轴小于32*256/DtypeSize_ 或 mn轴不是256对齐 或 输入是fp32类型，不走stream-k-dpsk
    if (mmOpInfo.shapeInfo.mDim % BASIC_BLOCK_SIZE_256 != 0UL ||
        mmOpInfo.shapeInfo.nDim % BASIC_BLOCK_SIZE_256 != 0UL || kDim < kThreshold ||
        (dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32)) {
        return false;
    }
    // 如果mn用256切分的份数小于核数 或者 取余核数为0或大于一半的核数，则不使用stream-k-dpsk
    uint64_t mCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.mDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t nCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.nDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t tatalMNCnt = mCnt * nCnt;
    if ((tatalMNCnt < aiCoreCnt) || (tatalMNCnt % aiCoreCnt == 0UL) || (tatalMNCnt % aiCoreCnt > aiCoreCnt / NUM_TWO)) {
        return false;
    }
    return true;
}

// 判断shape是否支持GemmV3
static bool CheckShapeSupport(MmOpInfo& mmOpInfo)
{
    // 判断是否不走stream-k
    bool notSkTiling = !CheckStreamKSKTiling(mmOpInfo) && !CheckStreamKDPSKTiling(mmOpInfo);
    // 判断m和n是否大于等于512
    bool mnValid = mmOpInfo.shapeInfo.mDim >= static_cast<int64_t>(CACHELINE) &&
                   mmOpInfo.shapeInfo.nDim >= static_cast<int64_t>(CACHELINE);
    // 判断k是否大于256
    bool kValid = mmOpInfo.shapeInfo.kDim > static_cast<int64_t>(BASIC_BLOCK_SIZE_256);
    // 满足条件shape范围
    bool shapeAswt = notSkTiling && mnValid && kValid;
    if (!shapeAswt) {
        OP_LOGI("Not support this shape in gemmV3.");
        return false;
    }
    OP_LOGD("Check shape success in gemmV3.");
    return true;
}
} // namespace

namespace Ops {
namespace Transformer {
op::Shape SwapLastTwoDimValue(const op::Shape tensorShape)
{
  op::Shape swapedShape = tensorShape;
  int64_t dimNum = tensorShape.GetDimNum();
  if (dimNum >= static_cast<int64_t>(MM_DIM)) {
      int64_t lastDim = tensorShape.GetDim(dimNum - 1);
      // dimNum - 1, 这里1指的是取最后一维的dim值。dimNum - 2, 这里2指的是取倒数第二维的dim值
      swapedShape.SetDim(dimNum - 1, tensorShape.GetDim(dimNum - 2));
      // dimNum - 2, 这里2指的是取倒数第二维的dim值
      swapedShape.SetDim(dimNum - 2, lastDim);
  }
  else {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimNum is not supported , which is %ld.", dimNum);
  }
  return swapedShape;
}

MmOpInfo GetMatmulOpInfo(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    // 获取m、k、n轴的大小
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t mDim = selfShape.GetDim(M_DIM_SELF_IDX);
    int64_t kDim = selfShape.GetDim(K_DIM_SELF_IDX);
    int64_t nDim = mat2Shape.GetDim(N_DIM_SELF_IDX);

    // Dtype和Format初始化
    MmOpInfo mmOpInfo;
    mmOpInfo.ori_info.self_dtype = self->GetDataType();
    mmOpInfo.ori_info.self_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    mmOpInfo.ori_info.mat2_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.output_dtype = self->GetDataType();
    if (FP16FP32_KEEP_DTYPE == cubeMathType) {
        mmOpInfo.ori_info.output_dtype = DataType::DT_FLOAT;
    }
    mmOpInfo.ori_info.output_format = op::Format::FORMAT_ND;

    mmOpInfo.shapeInfo.kDim = kDim;
    mmOpInfo.shapeInfo.nDim = nDim;
    mmOpInfo.shapeInfo.mDim = mDim;
    mmOpInfo.shapeInfo.transposeX1 = false;
    mmOpInfo.shapeInfo.transposeX2 = false;
    mmOpInfo.shapeInfo.dtypeASize = ge::GetSizeByDataType(self->GetDataType());
    mmOpInfo.shapeInfo.dtypeBSize = ge::GetSizeByDataType(mat2->GetDataType());
    OP_LOGD(
        "mDim=%ld, kDim=%ld, nDim=%ld, dtypeASize=%ld, dtypeBSize=%ld", mDim, kDim, nDim, mmOpInfo.shapeInfo.dtypeASize,
        mmOpInfo.shapeInfo.dtypeBSize);
    mmOpInfo.support_info = mmOpInfo.ori_info;

    // 不同芯片能力不同
    // 1980 1951 shape是否对齐
    // fp16 fp32 选择，1980 vector支持fp32
    SetMatmulOpSupportInfo(self, mat2, mmOpInfo, cubeMathType);

    // 获取aicore数目
    mmOpInfo.aiCoreCnt = GetCurrentPlatformInfo().GetCubeCoreNum();

    bool inputFp32Flag = mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT &&
                         mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    mmOpInfo.opImplModeEnum =
        (inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32))) ? 0x40 : 0x1;
    mmOpInfo.enableHf32 = inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32));
    OP_LOGD(
        "opImplModeEnum=%ld, enableHf32=%d, cubeMathType=%d, inputFp32Flag= %d", mmOpInfo.opImplModeEnum,
        mmOpInfo.enableHf32, cubeMathType, inputFp32Flag);
    // Log mm info
    GetMmInfo(mmOpInfo);
    return mmOpInfo;
}

bool ContiguousAndCast(
    const aclTensor*& contiguousInput, const aclTensor*& castOut, bool& transposeFlag, op::DataType dtype,
    aclOpExecutor* executor)
{
    auto contiguousOut = contiguousInput;
    if (IsTransposeLastTwoDims(contiguousInput)) {
        contiguousOut = executor->CreateView(
            contiguousInput, SwapLastTwoDimValue(contiguousInput->GetViewShape()), contiguousInput->GetViewOffset());
        transposeFlag = true;
    } else {
        contiguousOut = l0op::Contiguous(contiguousInput, executor);
    }
    CHECK_RET(contiguousOut != nullptr, false);

    // cast
    castOut = l0op::Cast(contiguousOut, dtype, executor);
    CHECK_RET(castOut != nullptr, false);
    return true;
}

const aclTensor* ExecMmOp(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType, aclOpExecutor* executor)
{
    return ExecMmOpWithBias(self, mat2, nullptr, cubeMathType, executor);
}

/*
计算MatMul的workSize， 内涵MatMul算子的构图流程
*/
const aclTensor* ExecMmOpWithBias(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, int8_t cubeMathType, aclOpExecutor* executor,
    bool transposeX2)
{
    /*
                  self            mat2
                   |               |
              contiguous       contiguous
                   |               |
                 cast             cast
                   |               |
                  pad             pad
                   |               |
                transpose      transpose
                   |               |
                transdata      transdata
                    \              /
                        matmul_op
                            |
                        transdata
                            |
                          cast
                            |
                         output
  */
    CHECK_RET(self != nullptr, nullptr);
    CHECK_RET(mat2 != nullptr, nullptr);
    CHECK_RET(CheckDtypeValid(self, mat2, bias, nullptr, cubeMathType), nullptr);
    CHECK_RET(CheckShapeValid(self, mat2, transposeX2), nullptr);
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), nullptr);
    // 空Tensor处理逻辑
    if (self->IsEmpty() || mat2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(self, mat2, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        // output cast
        if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
             GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) &&
            cubeMathType == FP16FP32_KEEP_DTYPE) {
            auto castOut = l0op::Cast(emptyOut, DataType::DT_FLOAT, executor);
            CHECK_RET(castOut != nullptr, nullptr);
            return castOut;
        }
        return emptyOut;
    }
    OP_LOGI("Format of self orign  is  [%s].", op::ToString(self->GetStorageShape()).GetString());
    OP_LOGI("Format of mat2 orign  is  [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // 内部只处理ND格式，reformat，全部转成ND
    self = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(self != nullptr, nullptr);

    if (mat2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat2 StorageFormat not FORMAT_FRACTAL_NZ.");
        mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
        CHECK_RET(mat2 != nullptr, nullptr);
    }
    OP_LOGI("Format of mat2 is  [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // 解析当前规格matmulop支持的dtype、format能力
    MmOpInfo mmOpInfo = GetMatmulOpInfo(self, mat2, cubeMathType);

    // weightNZ转置属性刷新
    mmOpInfo.shapeInfo.transposeX2 = mmOpInfo.shapeInfo.transposeX2 || transposeX2;

    // 左输入非连续转连续
    auto selfCastOut = self;
    bool selfCastRes = ContiguousAndCast(
        self, selfCastOut, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastRes, nullptr);

    // 右输入非连续转连续
    auto mat2CastOut = mat2;
    auto mat2StorageShape = mat2->GetStorageShape();
    bool mat2CastRes = ContiguousAndCast(
        mat2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);
    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat2 GetStorageFormat FORMAT_FRACTAL_NZ.");
        aclTensor* mat2ShapeSet = const_cast<aclTensor*>(mat2CastOut);
        mat2ShapeSet->SetStorageShape(mat2StorageShape); // 对NZ的场景用原来的stroageShape刷新
    }
    OP_LOGI("Format of mat2StorageShape is  [%s].", op::ToString(mat2StorageShape).GetString());
    // bias非连续转连续以及转换dtype
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        contiguousBias = ContiguousBias(self, bias, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }

    // k,m,n=1特殊场景
    auto selfReshapeOutput = selfCastOut;
    auto mat2ReshapeOutput = mat2CastOut;
    bool ifKEqual1 = IfKEqual1(selfCastOut, mmOpInfo, mmOpInfo.shapeInfo.transposeX1, bias);
    if (mmOpInfo.support_info.self_dtype == DataType::DT_BF16 ||
        mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16) {
        ifKEqual1 = ifKEqual1 && checkBF16SizeValid(mat2CastOut, mmOpInfo.shapeInfo.transposeX2) &&
                    CheckKEqual1Support() && checkBF16MMValid(selfCastOut, mat2CastOut, mmOpInfo.shapeInfo.transposeX2);
    }
    if (ifKEqual1) {
        aclnnStatus kEqual1SelfToMKRes =
            IfKEqual1SelfToMK(selfCastOut, selfReshapeOutput, mmOpInfo.shapeInfo.transposeX1, executor);
        CHECK_RET(kEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus kEqual1Mat2ToKNRes =
            IfKEqual1Mat2ToKN(mat2CastOut, mat2ReshapeOutput, mmOpInfo.shapeInfo.transposeX2, executor);
        CHECK_RET(kEqual1Mat2ToKNRes == ACLNN_SUCCESS, nullptr);
        OP_LOGI("Hit MatMul or BatchMatmul k=1 scenario, trans matmul to mul to calculate");
    } else {
        aclnnStatus mEqual1SelfToMKRes = IfMEqual1SelfToMK(
            selfCastOut, selfReshapeOutput, mmOpInfo.support_info.self_format, mmOpInfo.shapeInfo.transposeX1,
            executor);
        CHECK_RET(mEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus nEqual1Mat2ToNKRes = IfNEqual1Mat2ToNK(
            mat2CastOut, mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, mmOpInfo.shapeInfo.transposeX2,
            executor);
        CHECK_RET(nEqual1Mat2ToNKRes == ACLNN_SUCCESS, nullptr);
    }

    auto selfTransdataOut = l0op::TransData(selfReshapeOutput, mmOpInfo.support_info.self_format, 0, executor);
    CHECK_RET(selfTransdataOut != nullptr, nullptr);
    OP_LOGI("Format of self is selfTransdataOut [%s].", op::ToString(selfTransdataOut->GetStorageShape()).GetString());
    auto mat2TransdataOut = l0op::TransData(mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, 0, executor);
    CHECK_RET(mat2TransdataOut != nullptr, nullptr);
    OP_LOGI("Format of mat2 is mat2TransdataOut [%s].", op::ToString(mat2TransdataOut->GetStorageShape()).GetString());

    const aclTensor* mmOut = nullptr;
    if (ifKEqual1) {
        mmOut = l0op::Mul(selfTransdataOut, mat2TransdataOut, executor);
    } else {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, contiguousBias, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.enableHf32, mmOpInfo.opImplModeEnum, executor);
    }
    CHECK_RET(mmOut != nullptr, nullptr);

    auto mmTransdataOut = l0op::TransData(mmOut, mmOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(mmTransdataOut != nullptr, nullptr);

    // output cast
    auto castOut = l0op::Cast(mmTransdataOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);

    return castOut;
}

/*
计算MatMul，输入根据 transSelf和transMat2决定是否转置
*/
const aclTensor* ExecMmOpWithTrans(
    const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2, int8_t cubeMathType,
    aclOpExecutor* executor)
{
    CHECK_RET(CheckDtypeValid(self, mat2, nullptr, nullptr, cubeMathType), nullptr);
    CHECK_RET(CheckShapeValidWithTrans(self, mat2, transSelf, transMat2), nullptr);
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), nullptr);
    // 空Tensor处理逻辑
    if (self->IsEmpty() || mat2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensorWithTrans(self, mat2, transSelf, transMat2, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }

    // 内部只处理ND格式
    // reformat，全部转成ND
    self = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(self != nullptr, nullptr);
    mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    CHECK_RET(mat2 != nullptr, nullptr);

    // 解析当前规格matmulop支持的dtype、format能力
    MmOpInfo mmOpInfo = GetMatmulOpInfoWithTrans(self, mat2, transSelf, transMat2, cubeMathType);

    // 左输入 非连续转连续 cast
    auto contiguousSelf = l0op::Contiguous(self, executor);
    CHECK_RET(contiguousSelf != nullptr, nullptr);
    auto selfCastOut = l0op::Cast(contiguousSelf, mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastOut != nullptr, nullptr);

    // 右输入 非连续转连续 cast
    auto contiguousMat2 = l0op::Contiguous(mat2, executor);
    CHECK_RET(contiguousMat2 != nullptr, nullptr);
    auto mat2CastOut = l0op::Cast(contiguousMat2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastOut != nullptr, nullptr);

    const aclTensor* bias = nullptr;
    auto selfReshapeOutput = selfCastOut;
    auto mat2ReshapeOutput = mat2CastOut;
    bool ifKEqual1 = IfKEqual1(selfCastOut, mmOpInfo, mmOpInfo.shapeInfo.transposeX1, bias);
    if (mmOpInfo.support_info.self_dtype == DataType::DT_BF16 ||
        mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16) {
        ifKEqual1 = ifKEqual1 && checkBF16SizeValid(mat2CastOut, mmOpInfo.shapeInfo.transposeX2) &&
                    CheckKEqual1Support() && checkBF16MMValid(selfCastOut, mat2CastOut, mmOpInfo.shapeInfo.transposeX2);
        ;
    }
    if (ifKEqual1) {
        aclnnStatus kEqual1SelfToMKRes =
            IfKEqual1SelfToMK(selfCastOut, selfReshapeOutput, mmOpInfo.shapeInfo.transposeX1, executor);
        CHECK_RET(kEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus kEqual1Mat2ToKNRes =
            IfKEqual1Mat2ToKN(mat2CastOut, mat2ReshapeOutput, mmOpInfo.shapeInfo.transposeX2, executor);
        CHECK_RET(kEqual1Mat2ToKNRes == ACLNN_SUCCESS, nullptr);
        OP_LOGI("Hit MatMul or BatchMatmul k=1 scenario, trans matmul to mul to calculate");
    } else {
        aclnnStatus mEqual1SelfToMKRes = IfMEqual1SelfToMK(
            selfCastOut, selfReshapeOutput, mmOpInfo.support_info.self_format, mmOpInfo.shapeInfo.transposeX1,
            executor);
        CHECK_RET(mEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus nEqual1Mat2ToNKRes = IfNEqual1Mat2ToNK(
            mat2CastOut, mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, mmOpInfo.shapeInfo.transposeX2,
            executor);
        CHECK_RET(nEqual1Mat2ToNKRes == ACLNN_SUCCESS, nullptr);
    }

    auto selfTransdataOut = l0op::TransData(selfReshapeOutput, mmOpInfo.support_info.self_format, 0, executor);
    CHECK_RET(selfTransdataOut != nullptr, nullptr);
    auto mat2TransdataOut = l0op::TransData(mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, 0, executor);
    CHECK_RET(mat2TransdataOut != nullptr, nullptr);

    const aclTensor* mmOut = nullptr;
    if (ifKEqual1) {
        mmOut = l0op::Mul(selfTransdataOut, mat2TransdataOut, executor);
    } else {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, nullptr, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.enableHf32, mmOpInfo.opImplModeEnum, executor);
    }
    CHECK_RET(mmOut != nullptr, nullptr);

    auto mmTransdataOut = l0op::TransData(mmOut, mmOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(mmTransdataOut != nullptr, nullptr);

    // output cast
    auto castOut = l0op::Cast(mmTransdataOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);

    return castOut;
}

bool CheckGemmV3Support(const aclTensor* mat1, const aclTensor* mat2, MmOpInfo& mmOpInfo, int8_t cubeMathType)
{
    CHECK_RET(mat1 != nullptr, false);
    CHECK_RET(mat2 != nullptr, false);
    CHECK_RET(CheckDtypeValid(mat1, mat2, nullptr, nullptr, cubeMathType), false);
    CHECK_RET(CheckShapeValid(mat1, mat2), false);
    CHECK_RET(CheckMathType(mat1, mat2, cubeMathType), false);
    // 空Tensor不由gemmV3处理
    if (mat1->IsEmpty() || mat2->IsEmpty()) {
        OP_LOGI("mat1 or mat2 is empty, does not support GemmV3.");
        return false;
    }

    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        mat1->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat1 or mat2 StorageFormat is FORMAT_FRACTAL_NZ, does not support GemmV3.");
        return false;
    }
    // 当前支持平台
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
        OP_LOGI("Current SOC version does not support GemmV3.");
        return false;
    }

    // 解析当前规格matmulop支持的dtype format能力
    mmOpInfo = GetMatmulOpInfo(mat1, mat2, cubeMathType);

    // 当前支持shape范围
    return CheckShapeSupport(mmOpInfo);
}

const aclTensor* ExecGemmV3Op(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* c, MmOpInfo& mmOpInfo, aclOpExecutor* executor)
{
    OP_LOGD("Format of self orign is [%s].", op::ToString(self->GetStorageShape()).GetString());
    OP_LOGD("Format of mat2 orign is [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // gemmv3只处理ND格式，reformat全部转成ND
    self = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(self != nullptr, nullptr);

    mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    CHECK_RET(mat2 != nullptr, nullptr);

    OP_LOGI("Format of mat2 is [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // 左输入非连续转连续
    auto selfCastOut = self;
    bool selfCastRes = ContiguousAndCast(
        self, selfCastOut, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastRes, nullptr);

    // 右输入非连续转连续
    auto mat2CastOut = mat2;
    bool mat2CastRes = ContiguousAndCast(
        mat2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);

    // 输入c非连续转连续以及转换dtype
    auto contiguousC = c;
    if (contiguousC != nullptr) {
        contiguousC = ContiguousBias(self, c, executor);
        CHECK_RET(contiguousC != nullptr, nullptr);
    }

    // GEMMV3 output fp32
    mmOpInfo.ori_info.output_dtype = DataType::DT_FLOAT;

    // Invoke GemmV3 l0 api
    const aclTensor* mmOut = GetGemmV3Op(
        selfCastOut, mat2CastOut, contiguousC, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.shapeInfo.transposeX2,
        mmOpInfo.enableHf32, executor);

    CHECK_RET(mmOut != nullptr, nullptr);

    // output cast
    auto castOut = l0op::Cast(mmOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);

    return castOut;
}

bool IsInputSupportFp32() {
  if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
      op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93) {
    return false;
  }
  return true;
}

bool CheckBatchDimBroadcast(size_t batch1DimNum, size_t batch2DimNum, const op::Shape& batch1, const op::Shape& batch2) {
    size_t batchIndex = MM_DIM;
    while (batch1DimNum > batchIndex && batch2DimNum > batchIndex) {
        if (batch1[batch1DimNum - batchIndex - 1] != 1 && batch2[batch1DimNum - batchIndex - 1] != 1 &&
            batch1[batch1DimNum - batchIndex - 1] != batch2[batch1DimNum - batchIndex - 1]) {
            return false;
        }
        batchIndex++;
    }
    return true;
}

// bmm 相对于 mm 取坐标需偏移
int64_t GetOffSet(int64_t DimNum) {
  int64_t rightMove = 0;
  // bmm DimNum 为 3, mm DimNum 为 2 ，bmm需要相对于mm向后偏移一位取行列值，默认rightMove为 0
  rightMove = DimNum == 3 ? 1 : 0;
  return rightMove;
}

// 检查单Tensor是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupport(const aclTensor *tensor) {
  if (!IsInputSupportFp32()) {
    auto iter = std::find(V100_DTYPE_SUPPORT.begin(), V100_DTYPE_SUPPORT.end(), tensor->GetDataType());
    return iter != V100_DTYPE_SUPPORT.end();
  }
  auto iter = std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(), tensor->GetDataType());
  return iter != DTYPE_SUPPORT_LIST.end();
}

// 检查是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupportBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2) {
  bool matMulDtypeCorrect = CheckDtypeSupport(mat1) && CheckDtypeSupport(mat2);
  if (mat1->GetDataType() == DataType::DT_BF16) {
    return matMulDtypeCorrect &&
           (self->GetDataType() == DataType::DT_BF16 || self->GetDataType() == DataType::DT_FLOAT);
  }
  return CheckDtypeSupport(self) && matMulDtypeCorrect;
}

// 如果beta==1 && alpha == 1 && self.shape[0] == mat2.shape[1] && 不属于切k，直接走matmul的bias模式
bool NeedToConvertBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2,
                       const aclScalar *beta, const aclScalar *alpha) {
  int64_t mat1DimNum = static_cast<int64_t>(mat1->GetViewShape().GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(mat1DimNum);

  TensorInfo Tensor_matl = {mat1, mat1->GetDataType(), Format::FORMAT_ND};
  TensorInfo Tensor_mat2 = {mat2, mat2->GetDataType(), Format::FORMAT_ND};

  bool isSplitK = false;
  if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
      op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93) {
    isSplitK = IsSplitk(&Tensor_matl, &Tensor_mat2);;
  }
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  bool canBeBiasFlag = false;
  // bmm (DimNum==3) only apply the case of batch == 1
  bool batchIsOne = !(mat1DimNum == 3 && mat1->GetViewShape().GetDim(0) != 1);

  if (selfDimNum == 1) {
    canBeBiasFlag = (mat2->GetViewShape().GetDim(1 + rightMove) == self->GetViewShape().GetDim(0)) &&
                     CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
    // When input tensor is a 2 dimentional tensor
  } else if (selfDimNum == 2) {
    canBeBiasFlag = (selfShape.GetDim(0) == 1) && (selfShape.GetDim(1) == mat2Shape.GetDim(1 + rightMove)) &&
                     CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
  }
  OP_LOGI("Current Shape's canBeBiasFlag = %ld", static_cast<int64_t>(canBeBiasFlag));
  return (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
         (std::abs(beta->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
         !isSplitK && canBeBiasFlag;
}

// Nz fp16 in fp32 out experimental rules
bool GetNzSplitKFlag(const aclTensor *self, const aclTensor *mat2, const Format selfSuppFormat, const Format outSuppFormat) {
  if ((selfSuppFormat == Format::FORMAT_ND) && (outSuppFormat == Format::FORMAT_ND)) {
    return true;
  }
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(selfDimNum);

  int64_t m = selfShape.GetDim(rightMove);
  int64_t k = selfShape.GetDim(rightMove + 1);
  int64_t n = mat2Shape.GetDim(rightMove + 1);
  bool mn_multi = m > n ? m < (MN_MULTI * n) : n < (MN_MULTI * m);
  return (m * n * k < MKN_MAX) && mn_multi;
}

bool IsSplitk(const TensorInfo* self, const TensorInfo* mat2) {
  op::Shape selfShape = self->tensor->GetViewShape();
  op::Shape mat2Shape = mat2->tensor->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(selfDimNum);
  bool NzSplitKFlag = true;
  // only apply on mm now
  if (!rightMove) {
    NzSplitKFlag = GetNzSplitKFlag(self->tensor, mat2->tensor, self->format, mat2->format);
  }

  int64_t k_dim = selfShape.GetDim(1 + rightMove);
  bool dtype_correct = (self->dataType == DataType::DT_FLOAT16) && (mat2->dataType == DataType::DT_FLOAT16);
  return dtype_correct && k_dim >= SPLIT_K_MULTI * std::max(selfShape.GetDim(rightMove), mat2Shape.GetDim(1 + rightMove)) && NzSplitKFlag;
}

bool IsFormatSupportNd(const aclTensor *self, const aclTensor *mat2) {
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t dimNum = selfShape.GetDimNum();
    auto isAligin = [selfShape, mat2Shape, dimNum]() {
      return (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F));
    };
    if (isAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
      return true;
    }
    return false;
  }
  if ((self->GetDataType() == DataType::DT_FLOAT16 && mat2->GetDataType() == DataType::DT_FLOAT16) ||
      (self->GetDataType() == DataType::DT_BF16 && mat2->GetDataType() == DataType::DT_BF16)) {
    return IsNdToNzOnTheFly(self, mat2);
  }
  return true;
}

bool IsSupportNzNzNd(const aclTensor* self, const aclTensor* mat2) {
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t dimNum = selfShape.GetDimNum();
  auto isNAligin = [mat2Shape, dimNum]() { return ((static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F)
                   == 0); };
  if (isNAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
    return true;
  }
  return false;
}

bool IsNdToNzOnTheFly(const aclTensor *self, const aclTensor *mat2) {
  uint64_t kInnerAxisMinLimit = 128U;
  uint64_t kInnerAxisMaxLimit = 65535U;
  uint64_t kAxisLengthOne = 1U;
  // 如果self或mat2的维度数量小于2，则不符合判断是否16对齐的条件，返回失败
  if (self->GetViewShape().GetDimNum() < 2 || mat2->GetViewShape().GetDimNum() < 2) {
    return false;
  }
  bool isTransposeSelf = IsTransposeLastTwoDims(self);
  bool isTransposeMat2 = IsTransposeLastTwoDims(mat2);
  uint64_t selfInnerAxis = isTransposeSelf ?
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)) :
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
  uint64_t mat2InnerAxis = isTransposeMat2 ?
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2)) :
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));

  uint64_t selfOuterAxis = isTransposeSelf ?
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1)) :
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2));
  uint64_t mat2OuterAxis = isTransposeMat2 ?
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1)) :
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2));
  uint64_t mAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)); //倒数第2维
  uint64_t kAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
  uint64_t nAxis = static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));
  if (selfInnerAxis * selfOuterAxis <= kInnerAxisMaxLimit &&
      mat2InnerAxis * mat2OuterAxis <= kInnerAxisMaxLimit) {
    // too small tensor size
    return true;
  }
  OP_LOGD("Check IsNdToNzOnTheFly, if k=1 scenerio then remains ND.");
  if (kAxis == kAxisLengthOne) {
    return true;
  }

  if (IsSmallMNMultiSplitK(mAxis, kAxis, nAxis, isTransposeSelf, isTransposeMat2)) {
    OP_LOGD("Hit small mn multi split k.");
    return true;
  }

  return ((selfInnerAxis >= kInnerAxisMinLimit && selfInnerAxis <= kInnerAxisMaxLimit) ||
          (selfInnerAxis < kInnerAxisMinLimit && ((selfInnerAxis & 0xF) == 0))) &&
          ((mat2InnerAxis >= kInnerAxisMinLimit && mat2InnerAxis <= kInnerAxisMaxLimit) ||
          (mat2InnerAxis < kInnerAxisMinLimit && ((mat2InnerAxis & 0xF) == 0)));
}

bool IsSmallMNMultiSplitK(const uint64_t mDim, const uint64_t kDim, const uint64_t nDim,
                          const bool transposeX1, const bool transposeX2) {
  constexpr uint64_t align128 = 128;
  constexpr uint64_t numTwo = 2;
  constexpr uint64_t smallMNsplitKThres = 15000;
  bool kIsEnoughMultiCore = kDim >= smallMNsplitKThres;
  bool mnIsNotEnoughCore = (std::ceil(mDim / align128) * std::ceil(nDim / align128) <
                            static_cast<int64_t>(GetCurrentPlatformInfo().GetCubeCoreNum() / numTwo));
  // M/N轴在内轴的场景切m/n不影响MTE2搬运效率，M/N可以切小保证多核能开启，属于cube_bound场景
  return kIsEnoughMultiCore && mnIsNotEnoughCore && !(!transposeX1 && transposeX2);
}

bool IsTransposeLastTwoDims(const aclTensor *tensor) {
  // 当输入tensor的shape小于2或者大于6的时候，返回错误
  if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
    return false;
  }
  int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
  int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
  // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
  if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
    int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2);
    // 多batch连续，3是batch索引
    for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 3; batchDim >= 0; batchDim--) {
    if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
        return false;
      }
      tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
    }
    if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
      return false;
    }
    return true;
  }
  return false;
}

aclnnStatus SetMmSupportDType(MmOpInfo &mmOpInfo, int8_t cubeMathType) {
  bool dtypeMismatch = mmOpInfo.ori_info.self_dtype != mmOpInfo.ori_info.mat2_dtype;
  bool tensorFloat = mmOpInfo.ori_info.self_dtype == DataType::DT_FLOAT ||
                     mmOpInfo.ori_info.mat2_dtype == DataType::DT_FLOAT;
  bool tensorBfloat16 = mmOpInfo.ori_info.self_dtype == DataType::DT_BF16 ||
                        mmOpInfo.ori_info.mat2_dtype == DataType::DT_BF16;

  if (!IsInputSupportFp32()) {
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
  } else if (IsInputSupportFp32() && cubeMathType == USE_FP16 && (!tensorBfloat16)) {
    // FP16
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT16;
  } else if (IsInputSupportFp32() && dtypeMismatch && (tensorFloat || tensorBfloat16)) {
    // BF16或者存在FP32输入则全部dtype统一到FP32
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT;
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus SetMmSupportFormat(const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo) {
  if (IsFormatSupportNd(self, mat2)) {
    OP_LOGD("Matmul support NDNDND");
    mmOpInfo.support_info.output_format = Format::FORMAT_ND;
    mmOpInfo.support_info.self_format = Format::FORMAT_ND;
    mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
  } else {
    OP_LOGD("Matmul do not support NDNDND");
    // if 310p and n%16==0
    bool is310p = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    if (IsSupportNzNzNd(self, mat2) && is310p) {
      mmOpInfo.support_info.output_format = Format::FORMAT_ND;
      mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
      mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
      return ACLNN_SUCCESS;
    }
    mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus GetMmInfo(MmOpInfo mmOpInfo) {
  OP_LOGI(
    "Self tensor input's ori dtype = %s and format = %s; Mat2 tensor input's ori dtype = %s and format = %s;"
    "Output tensor's ori dtype = %s and ori format = %s;"
    "Self tensor input's Npu dtype = %s and Npu format = %s; Mat2 tensor input's Npu dtype = %s and Npuformat = %s;"
    "Output tensor's Npu dtype = %s and Npu format = %s.",
    op::ToString(mmOpInfo.ori_info.self_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.self_format).GetString(),
    op::ToString(mmOpInfo.ori_info.mat2_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.mat2_format).GetString(),
    op::ToString(mmOpInfo.ori_info.output_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.output_format).GetString(),
    op::ToString(mmOpInfo.support_info.self_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.self_format).GetString(),
    op::ToString(mmOpInfo.support_info.mat2_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.mat2_format).GetString(),
    op::ToString(mmOpInfo.support_info.output_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.output_format).GetString());
  return ACLNN_SUCCESS;
}

aclIntArray* NeedTransPerm(const aclTensor *x, aclOpExecutor *executor) {
  op::Shape shape = x->GetViewShape();
  int64_t dimSize = x->GetViewShape().GetDimNum();
  std::vector<int64_t> valuePerm(dimSize, 0);
  for (int64_t i = 0; i < dimSize; i++) {
    valuePerm[i] = shape[i];
  }
  std::swap(valuePerm[dimSize - INNER_AXIS], valuePerm[dimSize - OUTER_AXIS]);
  return executor->AllocIntArray(valuePerm.data(), dimSize);
}

bool checkBF16SizeValid(const aclTensor *&mat2, const bool &transX2Flag) {
  //校验N轴是否在优化范围内
  int64_t nDimNumWhenNoTrans = mat2->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimNumWhenTrans = mat2->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t nDim = transX2Flag ? mat2->GetViewShape().GetDim(nDimNumWhenTrans) :
                 mat2->GetViewShape().GetDim(nDimNumWhenNoTrans);
  if (nDim > N_KEQAL1_LIMIT) {
    return false;
  }
  return true;
}

bool checkBF16MMValid(const aclTensor *&self, const aclTensor *&mat2, const bool &transX2Flag) {
  //校验MN轴是否在优化范围内
  int64_t mDimNumWhenNoTrans = self->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t mDimNumWhenTrans = self->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t mDim = transX2Flag ? self->GetViewShape().GetDim(mDimNumWhenTrans) :
                 self->GetViewShape().GetDim(mDimNumWhenNoTrans);
  int64_t nDimNumWhenNoTrans = mat2->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimNumWhenTrans = mat2->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t nDim = transX2Flag ? mat2->GetViewShape().GetDim(nDimNumWhenTrans) :
                 mat2->GetViewShape().GetDim(nDimNumWhenNoTrans);
  if(mDim * nDim < MM_KEQAL1_LIMIT){
    return false;
  }
  return true;
}

bool IfKEqual1(const aclTensor *&selfInput, const MmOpInfo& mmOpInfo, const bool &transX1Flag, const aclTensor *&bias) {
  // 不支持nz场景
  if (mmOpInfo.support_info.self_format == Format::FORMAT_FRACTAL_NZ ||
      mmOpInfo.support_info.mat2_format == Format::FORMAT_FRACTAL_NZ) {
    return false;
  }
  OP_LOGD("Check MatMul or BatchMatmul k=1 scenario, and support_info is not NZ");
  if (mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.self_dtype ||
      mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.mat2_dtype) {
    return false;
  }
  // 判断是否带bias
  if (bias != nullptr) {
    return false;
  }
  // 判断k轴是否满足切mul需求(等于1)
  int64_t kDimNumWhenNoTrans = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t kDimNumWhenTrans = selfInput->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t kDim = transX1Flag ? selfInput->GetViewShape().GetDim(kDimNumWhenTrans) :
                 selfInput->GetViewShape().GetDim(kDimNumWhenNoTrans);
  if (kDim != DIM_EQUAL_ONE) {
    return false;
  }
  return true;
}

aclnnStatus IfKEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput, bool &transX1Flag,
                              aclOpExecutor *executor) {
  auto x1Perm = NeedTransPerm(selfInput, executor);
  selfReshapeOutput = transX1Flag ? l0op::Reshape(selfInput, x1Perm, executor) : selfInput;
  CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  transX1Flag = false;
  return ACLNN_SUCCESS;
}

aclnnStatus IfKEqual1Mat2ToKN(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput, bool &transX2Flag,
                              aclOpExecutor *executor) {
  auto x2Perm = NeedTransPerm(mat2Input, executor);
  mat2ReshapeOutput = transX2Flag ? l0op::Reshape(mat2Input, x2Perm, executor) : mat2Input;
  CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  transX2Flag = false;
  return ACLNN_SUCCESS;
}

aclnnStatus IfMEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput,
                              const Format selfInputFormat, bool &transX1Flag, aclOpExecutor *executor) {
  // 不支持nz场景
  if (selfInputFormat == Format::FORMAT_FRACTAL_NZ) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Check MatMul or BatchMatmul m=1 scenario, and support_info is not NZ");
  // 首先判断m轴是否已经为外轴，是外轴则return
  if (!transX1Flag) {
    return ACLNN_SUCCESS;
  }
  // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
  int64_t mDimNumWhenInner = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t mDimSize = selfInput->GetViewShape().GetDim(mDimNumWhenInner);
  if (mDimSize != DIM_EQUAL_ONE) {
    return ACLNN_SUCCESS;
  }
  auto x1Perm = NeedTransPerm(selfInput, executor);
  selfReshapeOutput = l0op::Reshape(selfInput, x1Perm, executor);
  transX1Flag = false;
  CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGI("Hit MatMul or BatchMatmul m=1 and m is inner scenario, trans m axis to outer");
  return ACLNN_SUCCESS;
}

aclnnStatus IfNEqual1Mat2ToNK(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput,
                              const Format mat2InputFormat, bool &transX2Flag, aclOpExecutor *executor) {
  // 不支持nz场景。
  if (mat2InputFormat == Format::FORMAT_FRACTAL_NZ) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Check MatMul or BatchMatmul n=1 scenario, and support_info is not NZ");
  // 首先判断n轴是否已经为外轴，是外轴则return
  if (transX2Flag) {
    return ACLNN_SUCCESS;
  }
  // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
  int64_t nDimNumWhenInner = mat2Input->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimSize = mat2Input->GetViewShape().GetDim(nDimNumWhenInner);
  if (nDimSize != DIM_EQUAL_ONE) {
    return ACLNN_SUCCESS;
  }
  auto x2Perm = NeedTransPerm(mat2Input, executor);
  mat2ReshapeOutput = l0op::Reshape(mat2Input, x2Perm, executor);
  transX2Flag = true;
  CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGI("Hit MatMul or BatchMatmul n=1 and n is inner scenario, trans n axis to outer");
  return ACLNN_SUCCESS;
}

uint64_t TransDequantScaleToM1(const float deqScale) {
  union {
    float scaleFloat;
    uint32_t scaleInt;
  } dequantScale;
  dequantScale.scaleFloat = deqScale;
  uint64_t fixpipeDeqScale = static_cast<uint64_t>(dequantScale.scaleInt) & kDeqScaleMul;
  return fixpipeDeqScale;
}

uint32_t GetL2Size(const std::string& socLongVersion) {
  constexpr int32_t l2Size2 = 192;
  constexpr int32_t l2Size4 = 96;
  return (socLongVersion == SOC_B4 || socLongVersion == SOC_C4) ? l2Size4 : l2Size2;
}

uint64_t GetL1Size([[maybe_unused]] const std::string& socLongVersion) {
  constexpr int64_t l1Size = 512;
  //支持芯片固定512K,后续开放别的芯片修改此处
  return l1Size;
}

op::FVector<int64_t> GetShape(const aclTensor *tensor) {
  op::FVector<int64_t> shape;
  if (tensor == nullptr) {
    shape.push_back(1);
    OP_LOGW("The input tensor of Func GetShape is nullptr");
    return shape;
  }
  if (tensor->GetViewShape().GetDimNum() == 0U) {
    shape.push_back(1);
  } else {
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    for (size_t idx = 0U; idx < dimNum; idx++) {
      int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
      shape.push_back(tmpVal);
    }
  }
  return shape;
}

const aclTensor *ContiguousBias(const aclTensor *self, const aclTensor *bias, aclOpExecutor *executor) {
    auto contiguousBias = l0op::Contiguous(bias, executor);
    CHECK_RET(contiguousBias != nullptr, nullptr);
    // bias为bf16时cast为fp32保证精度
    if ((contiguousBias->GetDataType() == DataType::DT_BF16)||
        self->GetDataType() == DataType::DT_FLOAT) {
        contiguousBias = l0op::Cast(contiguousBias, op::DataType::DT_FLOAT, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }
    return contiguousBias;
}

} // namespace Transformer
} // namespace Ops
