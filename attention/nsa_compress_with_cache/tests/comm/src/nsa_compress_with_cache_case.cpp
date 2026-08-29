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
 * \file nsa_compress_with_cache_case.cpp
 * \brief nsa_compress_with_cache 测试用例.
 */

#include "nsa_compress_with_cache_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/tests_utils_log.h"
#include "tests/utils/platform.h"
#include "tiling/ncwc/tiling_data.h"
#include "op_host/tiling_templates_registry.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

extern "C" __global__ __aicore__ void nsa_compress_with_cache_fp16 NSA_COMPRESS_WITH_CACHE_KERNEL_PARAM;
extern "C" __global__ __aicore__ void nsa_compress_with_cache_bf16 NSA_COMPRESS_WITH_CACHE_KERNEL_PARAM;

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus NsaCompressWithCache(gert::TilingContext *context);
} // namespace optiling

using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using NsaCompressWithCacheCase = ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheCase;
using NsaCompressWithCacheCase = ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheCase;

namespace {
const size_t NSA_COMPRESS_WITH_CACHE_SLOT_MAPPING_INPUT_INDEX = 2UL;
const size_t NSA_COMPRESS_WITH_CACHE_ACT_SEQ_LENGTH_INPUT_INDEX = 4UL;
const size_t NSA_COMPRESS_WITH_CACHE_BLOCK_TABLE_INPUT_INDEX = 5UL;

ASCENDC_EXTERN_C ge::graphStatus TilingForNsaCompressWithCacheStub(gert::TilingContext *context)
{
    auto *nsaCompressWithCacheCase = static_cast<NsaCompressWithCacheCase *>(Case::GetCurrentCase());
    if (nsaCompressWithCacheCase != nullptr) {
        NsaCompressWithCacheCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actSeqLenTensor = const_cast<gert::Tensor *>(
            context->GetOptionalInputTensor(NSA_COMPRESS_WITH_CACHE_ACT_SEQ_LENGTH_INPUT_INDEX));
        p.slotMappingTensor = const_cast<gert::Tensor *>(
            context->GetOptionalInputTensor(NSA_COMPRESS_WITH_CACHE_SLOT_MAPPING_INPUT_INDEX));
        p.blockTableTensor = const_cast<gert::Tensor *>(
            context->GetOptionalInputTensor(NSA_COMPRESS_WITH_CACHE_BLOCK_TABLE_INPUT_INDEX));
        if (!nsaCompressWithCacheCase->DoOpTiling(p)) {
            return p.ret;
        }
        return nsaCompressWithCacheCase->mNsaCompressWithCacheOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}
} // namespace

using NsaCompressWithCacheKernelFunc = void(*) NSA_COMPRESS_WITH_CACHE_KERNEL_PARAM;
using namespace ops::adv::tests::NsaCompressWithCache;
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

bool RunNsaCompressWithCache(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                             std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    auto kernelFunc = (NsaCompressWithCacheCase::NsaCompressWithCacheKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, 1, inputs[static_cast<int>(KernelParams::INPUT)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::WEIGHT)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::SLOT_MAPPING)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::OUTPUT_CACHE)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::ACT_SEQ_LEN)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::BLOCK_TABLE)]->GetDevData(),
                outputs[0]->GetDevData(), // outputCache
                workspace, tilingData);
    return true;
}

NsaCompressWithCacheCase::NsaCompressWithCacheCase()
    : NsaCompressWithCacheCase("Undefined", true, "", OpInfo(), NsaCompressWithCacheParam(), 0)
{
}

NsaCompressWithCacheCase::NsaCompressWithCacheCase(const char *name, bool enable, const char *dbgInfo, OpInfo opinfo,
                                                   NsaCompressWithCacheParam param, int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opinfo)), mParam(std::move(param))
{
    mOpInfo.mName = "NsaCompressWithCache";
    mNsaCompressWithCacheOriginTilingFuncName = "NsaCompressWithCache";

    mNsaCompressWithCacheKernelFunc = (void *)nsa_compress_with_cache_fp16;
}

bool NsaCompressWithCacheCase::Run()
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

bool NsaCompressWithCacheCase::InitParam()
{
    if (!mParam.Init()) {
        return false;
    }
    return true;
}

bool NsaCompressWithCacheCase::InitOpInfo()
{
    bool rst = this->InitOpInfoCtx();
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(NsaCompressWithCache).Tiling(TilingForNsaCompressWithCacheStub);

    return true;
}

bool NsaCompressWithCacheCase::InitOpInfoCtx()
{
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.mInput, &mParam.mWeight, &mParam.mSlotMapping, &mParam.mOutputCache,
                                 &mParam.mActSeqLenOptional, &mParam.mBlockTableOptional});
    rst = rst && mCtx.SetOutputs({&mParam.mOutputCache});
    rst = rst && mCtx.SetAttrs({{"layoutOptional", mParam.mLayoutOptional},
                                {"compressBlockSize", mParam.mCompressBlockSize},
                                {"compressStride", mParam.mCompressStride},
                                {"actSeqLenType", mParam.mActSeqLenType},
                                {"pageBlockSize", mParam.mPageBlockSize}});

    rst = rst && mCtx.SetKernelRunCbf(RunNsaCompressWithCache);
    rst = rst && mCtx.SetKernelMainFunc((void *)mNsaCompressWithCacheKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool NsaCompressWithCacheCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    mNsaCompressWithCacheOriginTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym(
        mNsaCompressWithCacheOriginTilingFuncName.c_str());
    if (mNsaCompressWithCacheOriginTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsaCompressWithCache(%p)", mNsaCompressWithCacheOriginTilingFunc);
        return false;
    }
    return true;
}

bool NsaCompressWithCacheCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaCompressWithCacheCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (mPreTilingRunCbf != nullptr) {
        mPreTilingRunCbf(tilingParam);
    }
    tilingParam.slotMappingTensor->SetData(gert::TensorData{mParam.mSlotMappingList.data()});
    if (tilingParam.actSeqLenTensor != nullptr) {
        tilingParam.actSeqLenTensor->SetData(gert::TensorData{mParam.mActSeqLenList.data()});
    }
    if (tilingParam.blockTableTensor != nullptr) {
        tilingParam.blockTableTensor->SetData(gert::TensorData{mParam.mBlockTableList.data()});
    }
    tilingParam.ret = Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(tilingParam.ctx);
    return true;
}