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
constexpr int64_t KV_COMPRESS_CACHE_OUTPUT_INDEX = 0;
constexpr int64_t X_INPUT_INDEX = 1;
constexpr int64_t SLOT_MAPPING_INDEX = 2;

constexpr int64_t QUANT_GROUP_SIZE_ATTR_INDEX = 0;
constexpr int64_t QUANT_MODE_ATTR_INDEX = 1;
constexpr int64_t ROUND_SCALE_ATTR_INDEX = 2;
constexpr int64_t SCALE_ATTR_INDEX = 3;

// ---------- 两个架构共用的布局常量 ----------
// quant_mode=3: MXFP8 全行量化（全部 d 元素，per-group amax scale 存 bf16，可选 scale 2 的幂取整；
//               cache dtype 须为 float8_e4m3fn/float8_e5m2，group 仅支持 32）
constexpr int64_t QUANT_MODE_MXFP8 = 3;
// quant_mode=4: MXFP4 packed 全行量化（全部 d 元素，两元素一字节 E2M1，共享指数 scale 编码为 bf16；
//               cache dtype 须为 uint8，group 支持 16/32）
constexpr int64_t QUANT_MODE_MXFP4 = 4;

// mode 3/4 布局与约束常量
constexpr int64_t MXFP_GROUP_SIZE_16 = 16;
constexpr int64_t MXFP_GROUP_SIZE_32 = 32;
constexpr int64_t MXFP_D_LENGTH_MAX = 8192;
// mode3/4 的 d 对齐约束（v1.12 新增）：d 须为 128 的倍数——规避不对齐 dataCol 下的
// 已知 kernel 缺陷类（r7/r8 P1），并将 d%group 之外的合法形状收敛到对齐几何。
constexpr int64_t MXFP_D_LENGTH_ALIGN = 128;
constexpr int64_t MXFP_CACHE_RANK_2D = 2;
constexpr int64_t MXFP_CACHE_RANK_4D = 4;
// cache 布局标记：1 = layout1 平铺 2D [N, kvCacheCol]；2 = layout2 分页 4D
constexpr int64_t MXFP_LAYOUT_1 = 1;
constexpr int64_t MXFP_LAYOUT_2 = 2;
constexpr int64_t MXFP_LAYOUT2_CACHE_COL_DIM = 3;
constexpr int64_t MXFP_X_BUFFER_ALIGN = 16;
constexpr int64_t MXFP_X_ELEMENT_BYTES = 2;
constexpr int64_t MXFP_DOUBLE_BUFFER = 2;

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
// ===== 以下 6 字段仅 mode 3/4 路径（TilingKey 2xxx/21xx）写入并读取；=====
// ===== legacy 路径（key 0）不读不写，其值无意义。=====
TILING_DATA_FIELD_DEF(int64_t, layout);
// cache 布局标记，由 cache rank 派生：1 = layout1 平铺 2D [N, kvCacheCol]；
// 2 = layout2 分页 4D [blockNum, kvCacheBlockSize, 1, layout2CacheCol]。
// legacy(key 0) 路径不写（单布局，无需标记）。
TILING_DATA_FIELD_DEF(int64_t, cacheRows);
// cache 逻辑总行数（token 槽位数），单位：行。
// layout1 = dim0(N)；layout2 = blockNum * kvCacheBlockSize。
// 仅用于 slot 越界静默跳过判断（slot 值在 [-1, cacheRows) 之外跳过）。
TILING_DATA_FIELD_DEF(int64_t, dataCol);
// 单行量化数据字节数，单位：字节。
// mode3(MXFP8) = d；mode4(MXFP4 packed，2 元素/字节) = d/2。
TILING_DATA_FIELD_DEF(int64_t, scaleBytes);
// 单行 scale 总字节数 = scaleCol * 2（每个 scale 为一个 bf16），单位：字节。
// 注意与 scaleCol（scale 个数）区分：个数 vs 字节数。
TILING_DATA_FIELD_DEF(int64_t, layout2TokenBundleSize);
// 【仅 layout2】单个 token 交错 bundle（[dataCol 数据 | scaleBytes scale]）的总大小，
// 单位：字节。block 内第 pos 个 token 的写入起始偏移 =
//     block * kvCacheBlockStride + pos * layout2TokenBundleSize。
// 约束 layout2TokenBundleSize <= kvCacheRowStride，bundle 之后的行尾间隙不写。
TILING_DATA_FIELD_DEF(int64_t, layout2CacheCol);
// 【仅 layout2】cache dim3 的物理列宽（storage shape 最后一维），单位：字节。
// 约束：layout2CacheCol >= layout2TokenBundleSize；块尾间隙不写。
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCompressEpilog, KvCompressEpilogTilingData)

// ---------- CompileInfo Structure ----------
struct KvCompressEpilogCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // KV_COMPRESS_EPILOG_TILING_H
