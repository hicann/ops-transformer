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
 * \file ncai_case.h
 * \brief NsaCompressAttentionInfer 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::ncaInfer {

/**
 * 算子 NsaCompressAttentionInfer 参数
 */

class NcaInferCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    class ncaInferParam {
    public:
        /* 设置参数 */
        int64_t B = 0;
        int64_t S1 = 0;
        int64_t S2 = 0;
        int64_t N1 = 0;
        int64_t G = 0;
        int64_t D = 0;
        int64_t N2 = 0;

        // attr
        int64_t numHeads = 1;
        int64_t numKeyValueHeads = 1;
        int64_t selectBlockSize = 1;
        int64_t selectBlockCount = 1;
        int64_t compressBlockSize = 1;
        int64_t compressStride = 1;
        float scaleValue = 1.0;
        std::string layoutOptional = "TND";
        int64_t pageBlockSize = 1;
        int64_t sparseMode = 0;

        std::vector<int64_t> actualQSeqLenOptionalLenList = {};
        std::vector<int64_t> actualCmpKvSeqLenOptionalLenList = {};
        std::vector<int64_t> actualSelKvSeqLenOptionalLenList = {};
        ncaInferParam();
        ncaInferParam(int64_t pB, int64_t pS1, int64_t pS2, int64_t pN1, int64_t pG, int64_t pD, int64_t pN2,
                      int64_t pnumHeads, int64_t pnumKeyValueHeads, int64_t pselectBlockSize, int64_t pselectBlockCount,
                      int64_t pcompressBlockSize, int64_t pcompressStride, float pscaleValue, std::string pInputLayout,
                      int64_t ppageBlockSize, int64_t pSparseMode);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actQSeqLenTensor = nullptr;
        gert::Tensor *actSeqCmpKVLenTensor = nullptr;
        gert::Tensor *actSeqSelKVLenTensor = nullptr;
    };

    Tensor query, key, value, attentionMaskOptional, blockTableOptional, actualQSeqLenOptional,
        actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topkMaskOptional, output, topkIndicesOut;
    OpInfo mOpInfo;
    Context mCtx;
    ncaInferParam mParam;
    gert::OpImplRegisterV2::TilingKernelFunc NcaiTilingFunc = nullptr;
    NcaInferCase();
    NcaInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, ncaInferParam param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam &tilingParam);
    template <class T>
    static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
    {
        if (hostData.empty()) {
            return true;
        }
        int64_t expMinSize = hostData.size() * sizeof(T);
        if (tensor.AllocDevData(0, expMinSize) == nullptr) {
            printf("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};

} // namespace ops::adv::tests::ncaInfer
