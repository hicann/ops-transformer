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
 * \file kv_compress_epilog_tiling.h
 * \brief KvCompressEpilog 架构无关的 tiling 公共定义：输入/属性索引、行列布局公共常量、
 *        TilingData 结构注册与 CompileInfo。arch22 (arch22/kv_compress_epilog_tiling_arch22.h) 与
 *        arch35 (arch35/kv_compress_epilog_tiling_arch35.h) 各自实现本平台 tiling 类并仅依赖本头文件，
 *        两个架构的代码互不包含。分派入口见 kv_compress_epilog_tiling.cpp。
 */

#ifndef KV_COMPRESS_EPILOG_TILING_H
#define KV_COMPRESS_EPILOG_TILING_H

#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {

// ---------- 输入/输出/属性索引（arch22 / arch35 相同 schema） ----------
constexpr int64_t KV_COMPRESS_CACHE_INPUT_INDEX = 0;
constexpr int64_t X_INPUT_INDEX = 1;
constexpr int64_t SLOT_MAPPING_INDEX = 2;
constexpr int64_t KV_COMPRESS_CACHE_OUTPUT_INDEX = 0;

constexpr int64_t QUANT_GROUP_SIZE_ATTR_INDEX = 0;
constexpr int64_t QUANT_MODE_ATTR_INDEX = 1;
constexpr int64_t ROUND_SCALE_ATTR_INDEX = 2;
constexpr int64_t SCALE_ATTR_INDEX = 3;

// ---------- 两个架构共用的布局常量 ----------
constexpr int64_t DEFAULT_QUANT_GROUP_SIZE = 64;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 0;
// rope 段长度（x 尾轴后 64 列，两个架构一致）
constexpr int64_t SLICE_SIZE = 64;
// UB/GM 对齐粒度（字节）
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MIN_ROW_PER_CORE = 1;
constexpr int64_t X_BUFFER_ALIGN = 16;
constexpr int64_t X_ELEMENT_BYTES = 2;
constexpr int64_t TMP_BUFFER_ELEMENT_BYTES = 4;
// 4D-only cache contract: cache 逻辑 shape 必须恰为 4D [blockNum, blockSize, 1, headDim]。
//   - dim0 blockNum: 仅此维支持非连续(分页); dim1 blockSize: 每 block 的 token 数;
//   - dim2 固定为 1 (每 token 写出一个压缩向量, 该轴不支持为变量);
//   - dim3 headDim: 每 token 行宽(行步长), 须 >= 算子写出的 kvCacheCol。
constexpr size_t CACHE_VIEW_DIM_NUM = 4;
constexpr size_t CACHE_BLOCKNUM_DIM = 0;
constexpr size_t CACHE_BLOCKSIZE_DIM = 1;
constexpr size_t CACHE_ONE_DIM = 2; // 倒数第二维, 必须 == 1
constexpr int64_t CACHE_ONE_DIM_VALUE = 1;

// ---------- TilingData Structure ----------
// CANN 反射工厂按算子名唯一注册, 多平台变体须共用同一结构（chunk_gated_delta_rule 等先例同此约定），
// 否则 ttk/DYN 编译按算子名解析到的结构体与 kernel 实际使用的不一致。
BEGIN_TILING_DATA_DEF(KvCompressEpilogTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, kvCacheCol);
TILING_DATA_FIELD_DEF(int64_t, kvCacheRowStride);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockStride);
// arch22 专用(x 行 GM 步长); arch35 不使用, tiling 侧显式置 0, 与 arch22 共用同一结构定义
TILING_DATA_FIELD_DEF(int64_t, xRowStride);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, concatCol);
TILING_DATA_FIELD_DEF(int64_t, padCol);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(int64_t, perGroupSize);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(float, scalesAttr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCompressEpilog, KvCompressEpilogTilingData)

// ---------- CompileInfo Structure ----------
struct KvCompressEpilogCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // KV_COMPRESS_EPILOG_TILING_H
