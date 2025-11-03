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
 * \file nsa_select_attention_infer_case.cpp
 * \brief nsa_select_attention_infer 测试用例.
 */

#include "nsa_select_attention_infer_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling_base/tiling_templates_registry.h"


extern "C" __global__ __aicore__ void nsa_select_attention_infer NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM;

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus NsaSelectAttentionInfer(gert::TilingContext *context);
} // namespace optiling

using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using NsaSelectAttentionInferCase = ops::adv::tests::NsaSelectAttentionInfer::NsaSelectAttentionInferCase;
using NsaSelectAttentionInferCase = ops::adv::tests::NsaSelectAttentionInfer::NsaSelectAttentionInferCase;


ASCENDC_EXTERN_C ge::graphStatus TilingForNsaSelectAttentionInferStub(gert::TilingContext *context)
{
        auto *nsaSelectAttentionInferCase = static_cast<NsaSelectAttentionInferCase *>(Case::GetCurrentCase());
    if (nsaSelectAttentionInferCase != nullptr) {
    NsaSelectAttentionInferCase::DoTilingParam p;
    p.ctx = context;
    p.ret = ge::GRAPH_SUCCESS;
    p.actSeqSelQLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(6));
    p.actSeqSelKVLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(7));
    p.blocktableTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(5));
    p.topkIndicesTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(3));
    if (!nsaSelectAttentionInferCase->DoOpTiling(p)) {
        return p.ret;
    }
    return nsaSelectAttentionInferCase->nsaSelectAttentionInferTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

using NsaSelectAttentionInferKernelFunc = void(*) NSA_SELECT_ATTENTION_INFER_KERNEL_PARAM;
using namespace ops::adv::tests::NsaSelectAttentionInfer;
using Platform = ops::adv::tests::utils::Platform;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;

enum class KernelParams : uint32_t {
    INPUT = 0,
    WEIGHT,
    SLOT_MAPPING,
    OUTPUT_CACHE,
    ACT_SEQ_LEN,
    BLOCK_TABLE
};

bool RunNsaSelectAttentionInfer(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                             std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    auto kernelFunc = (NsaSelectAttentionInferCase::NsaSelectAttentionInferKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), 
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(), 
                inputs[7]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

NsaSelectAttentionInferCase::NsaSelectAttentionInferCase()
    : NsaSelectAttentionInferCase("Undefined", true, "", OpInfo(), NsaSelectAttentionInferParam(), 0)
{
}

NsaSelectAttentionInferCase::NsaSelectAttentionInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo opinfo,
                                                   NsaSelectAttentionInferParam param, int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opinfo)), mParam(std::move(param))
{
    mOpInfo.mName = "NsaSelectAttentionInfer";
    mNsaSelectAttentionInferOriginTilingFuncName = "TilingNsaSelectAttentionInfer";

    mNsaSelectAttentionInferKernelFunc = (void *)nsa_select_attention_infer;
}

bool NsaSelectAttentionInferCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool NsaSelectAttentionInferCase::InitParam()
{
    if (!mParam.Init()) {
        return false;
    }
    return true;
}

bool NsaSelectAttentionInferCase::InitOpInfo()
{
    bool rst = this->InitOpInfoCtx();
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(NsaSelectAttentionInfer).Tiling(TilingForNsaSelectAttentionInferStub);

    return true;
}

bool NsaSelectAttentionInferCase::InitOpInfoCtx()
{
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.query, &mParam.key, &mParam.value, &mParam.topkIndices, &mParam.blocktable, &mParam.actualQSeqLengths, &mParam.actualKvSeqLengths});
    rst = rst && mCtx.SetOutputs({&mParam.attentionOut});
    rst = rst && mCtx.SetAttrs({{"input_layout", mParam.inputLayout},
                                {"num_heads", mParam.headSize},
                                {"num_key_value_heads", mParam.headSizeV},
                                {"selected_block_size", mParam.selectedBlockSize},
                                {"selected_block_count", mParam.selectedBlockCount},
                                {"block_size", mParam.blockSize},
                                {"scale_value", mParam.scaleValue},
                                {"sparse_mode", mParam.sparseMode}});

    rst = rst && mCtx.SetKernelRunCbf(RunNsaSelectAttentionInfer);
    rst = rst && mCtx.SetKernelMainFunc((void *)mNsaSelectAttentionInferKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool NsaSelectAttentionInferCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    nsaSelectAttentionInferTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym(mNsaSelectAttentionInferOriginTilingFuncName.c_str());
    if (nsaSelectAttentionInferTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsaCompressWithCache(%p)", nsaSelectAttentionInferTilingFunc);
        return false;
    }
    return true;
}

bool NsaSelectAttentionInferCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaSelectAttentionInferCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (mPreTilingRunCbf != nullptr) {
        mPreTilingRunCbf(tilingParam);
    }
    if (tilingParam.actSeqSelQLenTensor != nullptr) {
        tilingParam.actSeqSelQLenTensor->SetData(gert::TensorData{mParam.actualSeqQLenList.data()});
    }
    if (tilingParam.actSeqSelKVLenTensor != nullptr) {
        tilingParam.actSeqSelKVLenTensor->SetData(gert::TensorData{mParam.actualSeqKVLenList.data()});
    }
    
    tilingParam.ret = Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(tilingParam.ctx);
    return true;
}