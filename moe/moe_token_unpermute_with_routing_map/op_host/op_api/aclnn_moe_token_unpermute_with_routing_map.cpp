/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/sort.h"
#include "level0/zero_op.h"
#include "level0/masked_scatter.h"
#include "level0/inplace_index_add.h"
#include "level0/mul.h"

#include "moe_token_unpermute_with_routing_map.h"
#include "aclnn_moe_token_unpermute_with_routing_map.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace
{
static constexpr int64_t INDEX_SHAPE = 1;
static constexpr int64_t TRANSPOSE_SHAPE_SIZE = 2;

// FLOAT类型在2139095040时为inf，不能sort
static constexpr int64_t MAX_SORT_SHAPE_DIM = 2139095040;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,  op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                             op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                        op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT16,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_95_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_BF16,    op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64,   op::DataType::DT_BOOL};

static bool IsAICoreSupport(const aclTensor* self)
{
    // 根据芯片类型和输入self类型判断是否走aicore
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), ASCEND910_95_AICORE_DTYPE_SUPPORT_LIST);
    } else if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
               GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        if (CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
            return true;
        }
    } else {
        if (CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST)) {
            return true;
        }
    }
    return false;
}

static const std::initializer_list<DataType> dtype_list = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                           op::DataType::DT_BF16};

static const std::initializer_list<DataType> routing_map_dtype_list = {op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<DataType> indice_dtype_list = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* permuteTokens,
                                const aclTensor* sortedIndices,
                                const aclTensor* probsOptional,
                                const aclIntArray* restoreShapeOptional)
{
    OP_CHECK_NULL(permuteTokens, return false);
    OP_CHECK_NULL(sortedIndices, return false);
    if(probsOptional == nullptr && restoreShapeOptional == nullptr){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The probsOptional and restoreShapeOptional should not be nullptr at the same time.");
        return false;
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* permuteTokens,
                                   const aclTensor* sortedIndices,
                                   const aclTensor* probsOptional,
                                   const aclTensor* unpermutedTokens, 
                                   const aclTensor* outIndex, 
                                   const aclTensor* permuteTokenId, 
                                   const aclTensor* permuteProbs)
{
    // 检查gradY的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokens, dtype_list, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndices, indice_dtype_list, return false);
    
    // 检查输入和输出的数据类型是否一致
    if (probsOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, dtype_list, return false);
        if (permuteProbs != nullptr) {
            OP_CHECK_DTYPE_NOT_MATCH(permuteProbs, probsOptional->GetDataType(), return false);
        }
    }

    if (unpermutedTokens != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(unpermutedTokens, permuteTokens->GetDataType(), return false);
    }
    
    if (outIndex != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outIndex, sortedIndices->GetDataType(), return false);
    }

    if (permuteTokenId != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokenId, sortedIndices->GetDataType(), return false);
    }

    return true;
}

static bool CheckShapeValid(const aclTensor* permuteTokens, const aclTensor* sortedIndices, const aclTensor* unpermutedTokens)
{
    auto permuteTokenspDimNum = permuteTokens->GetViewShape().GetDimNum();
    OP_CHECK(
        permuteTokenspDimNum == TRANSPOSE_SHAPE_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of permuteTokens should be two, but got %lu.", permuteTokenspDimNum),
        return false);

    auto sortedIndicesDimNum = sortedIndices->GetViewShape().GetDimNum();
    OP_CHECK(
        sortedIndicesDimNum == INDEX_SHAPE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of sortedIndices should be one, but got %lu.", sortedIndicesDimNum),
        return false);

    auto unpermutedTokensDimNum = unpermutedTokens->GetViewShape().GetDimNum();
    OP_CHECK(
        unpermutedTokensDimNum == TRANSPOSE_SHAPE_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of unpermutedTokensDimNum should be two, but got %lu.", unpermutedTokensDimNum),
        return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* permuteTokens,
                               const aclTensor* sortedIndices,
                               const aclTensor* probsOptional, 
                               const aclIntArray* restoreShapeOptional,
                               aclTensor* unpermutedTokens, aclTensor* outIndex, aclTensor* permuteTokenId, aclTensor* permuteProbs)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(permuteTokens, sortedIndices, probsOptional, restoreShapeOptional), 
                           ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(permuteTokens, sortedIndices, probsOptional, unpermutedTokens, outIndex, permuteTokenId, permuteProbs), 
                              ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的Shape是否符合要求
    CHECK_RET(CheckShapeValid(permuteTokens, sortedIndices, unpermutedTokens), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static void ViewDataType(const aclTensor* input, const op::DataType dtype)
{
    if (input == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "view data type error! it is null.");
        return;
    }
    auto tmpTensor = const_cast<aclTensor*>(input);
    tmpTensor->SetDataType(dtype);
    input = tmpTensor;
}
}  // namespace

aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(const aclTensor* permutedTokens,
                                                                 const aclTensor* sortedIndices,
                                                                 const aclTensor* routingMapOptional,
                                                                 const aclTensor* probsOptional, 
                                                                 bool paddedMode,
                                                                 const aclIntArray* restoreShapeOptional, 
                                                                 aclTensor* unpermutedTokens, aclTensor* outIndex, aclTensor* permuteTokenId, aclTensor* permuteProbs,
                                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMoeTokenUnpermuteWithRoutingMap,
                   DFX_IN(permutedTokens, sortedIndices, routingMapOptional, probsOptional, paddedMode, restoreShapeOptional),
                   DFX_OUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs));

    auto ret = CheckParams(permutedTokens, sortedIndices, probsOptional, 
                           restoreShapeOptional, unpermutedTokens, outIndex, permuteTokenId, permuteProbs);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    if (permutedTokens->IsEmpty() || sortedIndices->IsEmpty() ) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，将输入转换成连续的tensor
    auto permuteTokensContiguous = l0op::Contiguous(permutedTokens, uniqueExecutor.get());
    CHECK_RET(permuteTokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sortedIndicesContiguous = l0op::Contiguous(sortedIndices, uniqueExecutor.get());
    CHECK_RET(sortedIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    //yqw
    const aclTensor *routingMapOptionalContiguous = nullptr;
    if (routingMapOptional != nullptr){
        routingMapOptionalContiguous = l0op::Contiguous(routingMapOptional, uniqueExecutor.get());
        CHECK_RET(routingMapOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* source = nullptr;
    std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> permuteprob;
    if (probsOptional != nullptr){
        auto probsOptionalContiguous = l0op::Contiguous(probsOptional, uniqueExecutor.get());
        CHECK_RET(probsOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        permuteprob = l0op::MoeTokenUnpermuteWithRoutingMap(permuteTokensContiguous, sortedIndicesContiguous, routingMapOptionalContiguous, probsOptionalContiguous, 
                                                            paddedMode, restoreShapeOptional, uniqueExecutor.get()); //传参
        if(paddedMode == true){
            int64_t tensorSize = static_cast<int64_t>(permuteTokensContiguous->GetViewShape().GetDimNum());
            std::vector<int64_t> tensorShape(2);
            tensorShape[0] = (permuteTokensContiguous->GetViewShape())[0];
            tensorShape[1] = 1;
            auto outShape = uniqueExecutor->AllocIntArray(tensorShape.data(), tensorSize);
            auto permuteprobReshape = l0op::Reshape(std::get<3>(permuteprob), outShape, uniqueExecutor.get());
            CHECK_RET(permuteprobReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
            source = l0op::Mul(permuteTokensContiguous, permuteprobReshape, uniqueExecutor.get());
        }     
    } else {
        if (paddedMode == false){
            permuteprob = l0op::MoeTokenUnpermuteWithRoutingMap(permuteTokensContiguous, sortedIndicesContiguous, routingMapOptionalContiguous, nullptr, 
                                                                paddedMode, restoreShapeOptional, uniqueExecutor.get());
        }
        source = permuteTokensContiguous;
    }

    if (paddedMode == true) {
        const aclTensor* indexAddRes = nullptr;
        constexpr bool descending = false;
        constexpr bool stable = true;

        auto unpermutedTokensContiguous = l0op::Contiguous(unpermutedTokens, uniqueExecutor.get());
        CHECK_RET(unpermutedTokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor* unpermutedTokensOut = nullptr;
        unpermutedTokensOut = l0op::ZerosLike(unpermutedTokensContiguous, uniqueExecutor.get());
        CHECK_RET(unpermutedTokensOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor* indicesViewFloat =
            uniqueExecutor.get()->CreateView(sortedIndicesContiguous, sortedIndicesContiguous->GetViewShape(), 0);
        ViewDataType(indicesViewFloat, op::DataType::DT_FLOAT);
        auto sortResult = l0op::Sort(indicesViewFloat, -1, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
        auto sortIndex = std::get<1>(sortResult);//输出1
        auto sortValues = std::get<0>(sortResult);//输出2

        CHECK_RET(sortValues != nullptr && sortIndex != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto sortValuesI32 = uniqueExecutor.get()->CreateView(sortValues, sortedIndicesContiguous->GetViewShape(),
                                                                sortValues->GetViewOffset());
        ViewDataType(sortValuesI32, op::DataType::DT_INT32);
        // 当设备类型为A2或A3且index为int32类型时，切为InplaceIndexAddWithSorted算子
        bool useNewOp = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) &&
                         unpermutedTokens->GetViewShape().GetDim(0) < MAX_SORT_SHAPE_DIM &&
                        (unpermutedTokensOut->GetDataType() == op::DataType::DT_BF16 || unpermutedTokensOut->GetDataType() == op::DataType::DT_FLOAT16);
        if (useNewOp) {
            indexAddRes =
                l0op::InplaceIndexAddWithSorted(unpermutedTokensOut, 0, sortValuesI32, sortIndex,
                                                source, nullptr, uniqueExecutor.get());
        } else if (IsAICoreSupport(unpermutedTokensOut)) {
            indexAddRes =
                l0op::InplaceIndexAddAiCore(unpermutedTokensOut, 0, sortedIndicesContiguous,
                                            source, nullptr, uniqueExecutor.get());
        } else {
            indexAddRes =
                l0op::InplaceIndexAddAiCpu(unpermutedTokensOut, 0, sortedIndicesContiguous,
                                           source, nullptr, uniqueExecutor.get());
        }
        CHECK_RET(indexAddRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        //输出3
        if (std::get<3>(permuteprob) != nullptr) {
            auto permuteProbsResult = l0op::ViewCopy(std::get<3>(permuteprob), permuteProbs, uniqueExecutor.get());
            CHECK_RET(permuteProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        //输出2
        auto sortIndexViewCopyResult = l0op::ViewCopy(sortIndex, outIndex, uniqueExecutor.get());
        CHECK_RET(sortIndexViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);        
        //输出1
        auto sortValuesViewCopyResult = l0op::ViewCopy(sortValuesI32, permuteTokenId, uniqueExecutor.get());
        CHECK_RET(sortValuesViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        //输出0
        auto unpermutedTokensResult = l0op::ViewCopy(indexAddRes, unpermutedTokens, uniqueExecutor.get());
        CHECK_RET(unpermutedTokensResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }else{
        auto unpermutedTokensOpOut = std::get<0>(permuteprob);
        CHECK_RET(unpermutedTokensOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto unpermutedTokensResult = l0op::ViewCopy(unpermutedTokensOpOut, unpermutedTokens, uniqueExecutor.get());
        CHECK_RET(unpermutedTokensResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto permuteProbsOpOut = std::get<3>(permuteprob);
        CHECK_RET(permuteProbsOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if(permuteProbs != nullptr){
            auto permuteProbsResult = l0op::ViewCopy(permuteProbsOpOut, permuteProbs, uniqueExecutor.get());
            CHECK_RET(permuteProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnMoeTokenUnpermuteWithRoutingMap);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif