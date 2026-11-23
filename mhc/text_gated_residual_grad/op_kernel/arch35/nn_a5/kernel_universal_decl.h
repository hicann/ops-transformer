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
 * \file kernel_universal_decl.h
 * \brief Declaration of the HcGradNnGemm primary template. Shared by the aggregate header
 * kernel_universal.h and every kernel_xx.h specialization fragment, so that fragments stay
 * self-contained without including kernel_universal.h back.
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

/**
 * @class HcGradNnGemm
 * @brief
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class HcGradNnGemm {
    static_assert(always_false_v<BlockEpilogue_> && always_false_v<BlockMmad_>,
                  "KernelStreamk is not implemented for this BlockEpilogue or BlockMmad");
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
