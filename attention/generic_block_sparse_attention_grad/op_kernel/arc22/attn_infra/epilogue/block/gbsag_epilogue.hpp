/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSAG_EPILOGUE_HPP
#define GBSAG_EPILOGUE_HPP

#include "../../../attn_infra/gbsag_base_defs.hpp"

namespace NpuArch::Epilogue::Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace NpuArch::Epilogue::Block

#include "../../../attn_infra/epilogue/block/gbsag_epilogue_fag_pre.hpp"
#endif // EPILOGUE_BLOCK_GBSAG_EPILOGUE_HPP
