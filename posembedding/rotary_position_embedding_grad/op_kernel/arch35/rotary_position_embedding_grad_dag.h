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
 * \file rotary_position_embedding_grad_dag.h
 * \brief rotary_position_embedding_grad dag
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_DAG_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_DAG_H__

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace RotaryPositionEmbeddingGrad {
using namespace Ops::Base;
template <typename T, typename PromteT>
struct RotaryPositionEmbeddingGradDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using Mul0 = Bind<Vec::Mul<PromteT>, Cast0, Cast1>;
    using Reduce0 = Bind<Vec::ReduceSumOp<PromteT>, Mul0>;
    using Cast2 = Bind<Vec::Cast<T, PromteT, 1>, Reduce0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace RotaryPositionEmbeddingGrad

#endif
