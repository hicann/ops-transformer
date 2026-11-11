/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_TILING_H
#define POOL_KEY_INDEXER_TILING_H

#include <vector>
#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_util.h"

namespace optiling {
enum class PkiDataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

// ------------------ Input/Output Index ------------------
constexpr uint32_t PKI_QUERY_INDEX = 0;
constexpr uint32_t PKI_POOL_KEY_INDEX = 1;
constexpr uint32_t PKI_WEIGHTS_INDEX = 2;
constexpr uint32_t PKI_POOL_TAIL_K_INDEX = 3;
constexpr uint32_t PKI_ACTUAL_SEQ_Q_INDEX = 4;
constexpr uint32_t PKI_ACTUAL_SEQ_K_INDEX = 5;
constexpr uint32_t PKI_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t PKI_Q_DESCALE_INDEX = 7;
constexpr uint32_t PKI_K_DESCALE_INDEX = 8;
constexpr uint32_t PKI_SPARSE_INDICES_INDEX = 0;
constexpr uint32_t PKI_SPARSE_VALUES_INDEX = 1;

// ------------------ Attribute Index ------------------
constexpr uint32_t PKI_ATTR_TOPK = 0;
constexpr uint32_t PKI_ATTR_POOL_SIZE = 1;
constexpr uint32_t PKI_ATTR_LAYOUT_Q = 2;
constexpr uint32_t PKI_ATTR_LAYOUT_K = 3;
constexpr uint32_t PKI_ATTR_MASK_MODE = 4;
constexpr uint32_t PKI_ATTR_QUANT_MODE = 5;
constexpr uint32_t PKI_ATTR_RETURN_VALUE = 6;
constexpr uint32_t PKI_ATTR_KEY_STRIDE0 = 7;
constexpr uint32_t PKI_ATTR_K_DESCALE_STRIDE0 = 8;
// sparse_indices_mode(索引输出布局模式: 仅 aclnnPoolKeyIndexerV2 暴露, V1 兼容签名不含;
// 旧 GE 图的运行时属性容器可能不含该项, 读取须以 GetAttrNum/判空守护)
constexpr uint32_t PKI_ATTR_SPARSE_INDICES_MODE = 9;

// ------------------ Limits ------------------
constexpr uint32_t PKI_HEAD_DIM = 128;
constexpr uint32_t PKI_N2_FIXED = 1;
constexpr uint32_t PKI_N1_LIMIT = 64;
constexpr uint32_t PKI_POOL_SIZE_LIMIT = 128;
// PA block_size 上限: PA 寻址按运行时 blockSize 动态计算, 支持 16 对齐的任意 blockSize
constexpr uint32_t PKI_BLOCK_SIZE_LIMIT = 2048;
constexpr uint32_t PKI_BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t PKI_TOPK_DEFAULT = 2048;
// quant_mode host-side values (match def.cpp attribute defaults, may be negative)
// Mapping to template values: tpl = host + 1 (see template_tiling_key.h PKI_TPL_QUANT_*)
constexpr int32_t PKI_QUANT_NONE = -1;
constexpr int32_t PKI_QUANT_FP8_PER_TOKEN = 0;
constexpr int32_t PKI_QUANT_MXFP8 = 1;
// int8 对称量化: q/k=INT8, weights/q_descale/k_descale=FP16(三平台)
constexpr int32_t PKI_QUANT_INT8 = 2;
constexpr int32_t PKI_MASK_DEFAULT = 0;
constexpr int32_t PKI_MASK_CAUSAL = 3;
// mxFP8 scale 布局: 每 MX_SCALE_GROUP_SIZE 个 D 元素共享一个
// E8M0 scale, 尾维 [headDim/MX_SCALE_GROUP_ALIGN, MX_E8M0_SCALE_PACK_NUM] 打包
constexpr uint32_t PKI_MX_SCALE_GROUP_SIZE = 32;   // 每 32 个 D 元素一个 scale
constexpr uint32_t PKI_MX_E8M0_SCALE_PACK_NUM = 2; // E8M0 尾维打包数(2 个 e8m0 打包寻址)
constexpr uint32_t PKI_MX_SCALE_SHAPE_ALIGN = 64;  // headDim 对齐约束(当前 headDim 恒为 128)

// ------------------ Tiling Constants ------------------
// ------------------ Index Output Layout Mode (尾块追加模式) ------------------
// 0=legacy: tail 固定偏移 topk, 短序列 [expandTotal, topk) 为 -1 缝隙(历史布局)
// 1=compact: tail 紧跟有效展开 token, -1 全部位于行尾(紧凑布局)
// V1 op(PoolKeyIndexer)恒 legacy; V2 op(PoolKeyIndexerV2)按 sparse_indices_mode 属性选择
constexpr uint32_t PKI_SPARSE_INDICES_MODE_LEGACY = 0U;
constexpr uint32_t PKI_SPARSE_INDICES_MODE_COMPACT = 1U;
// 以下 S1/S2/M 基本块常量为 arch35/host 侧口径; arch22 kernel 以自有硬编码覆盖, 不消费这些字段。
constexpr uint32_t PKI_S1_BASE_SIZE = 4;
constexpr uint32_t PKI_S1_BASE_SIZE_SMALL = 2;
constexpr uint32_t PKI_S2_BASE_SIZE = 128;
constexpr uint32_t PKI_M_BASE_SIZE = 256;
constexpr uint32_t PKI_M_BASE_SIZE_SMALL = 128;
constexpr uint32_t PKI_D_BASE_BLOCK = 128;
constexpr uint32_t PKI_TOPK_6K = 6144;
constexpr uint32_t PKI_GM_ALIGN_BYTES = 512;
// arch35 量化键宽(对齐 QLIV2, 模式感知; 须与 kernel common.h scoreType 保持一致,
// arch22 分区不受影响, 其 scoreElemSize 在 tiling.cpp 单独算):
//   mode0/1(FP8/MXFP8) = 2B uint16 bf16 键; mode2(INT8) = 4B uint32 fp32 直接稳定排序
constexpr uint32_t PKI_SCORE_T_SIZE_FP8 = 2;
constexpr uint32_t PKI_SCORE_T_SIZE_INT8 = 4;
constexpr uint32_t PKI_SCORE_T_SIZE = 4; // arch35 非量化/arch22 键宽

// Trunk lengths for TopK
constexpr uint32_t PKI_TRUNK_LEN_16K = 16384;
constexpr uint32_t PKI_TRUNK_LEN_12K = 12288;
constexpr uint32_t PKI_TRUNK_LEN_8K = 8192;
constexpr uint32_t PKI_TRUNK_LEN_4K = 4096;
constexpr uint32_t PKI_TRUNK_LEN_2K = 2048;
constexpr uint32_t PKI_TOPK_2K = 2048;
constexpr uint32_t PKI_TOPK_3K = 3072;
constexpr uint32_t PKI_TOPK_4K = 4096;
constexpr uint32_t PKI_TOPK_5K = 5120;

// ------------------ TilingData Definition ------------------
BEGIN_TILING_DATA_DEF(PoolKeyIndexerTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(int64_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, topk)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, s1BaseSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSizeMax)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
TILING_DATA_FIELD_DEF(uint32_t, trunkLen)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(uint32_t, quantMode)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
TILING_DATA_FIELD_DEF(uint32_t, layoutQ)
TILING_DATA_FIELD_DEF(uint32_t, layoutK)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint64_t, wsOffScore)
TILING_DATA_FIELD_DEF(uint64_t, wsOffLdScore)
TILING_DATA_FIELD_DEF(uint64_t, wsOffLdIdx)
TILING_DATA_FIELD_DEF(float, qkScale) // 1/sqrt(headDim), score 缩放系数
// 索引输出布局模式: 0=legacy / 1=compact(见 ConstInfo::sparseIndicesMode)。
// V1 tiling 恒填 0; V2 op 复用本结构体, 由 sparse_indices_mode 属性填充。
TILING_DATA_FIELD_DEF(uint32_t, sparseIndicesMode)
END_TILING_DATA_DEF
// 注: REGISTER_TILING_DATA_CLASS 在 tiling.cpp 内执行(与 infershape/tiling 的
// 注册约定一致, 头文件仅保留结构体定义)

// ------------------ CompileInfo ------------------
// 图模式必需: 注册 TilingParse 后框架才生成 compile info JSON 供 FE 解析, 未注册图编译报错
struct PoolKeyIndexerCompileInfo {
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
};

// ------------------ TilingInfo ------------------
struct PoolKeyIndexerTilingInfo {
    const char* opName = nullptr;
    fe::PlatFormInfos* platformInfo = nullptr;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 1;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t headDim = PKI_HEAD_DIM;
    uint32_t topk = PKI_TOPK_DEFAULT;
    uint32_t poolSize = 16;
    uint32_t sparseCount = 0;
    int32_t maskMode = PKI_MASK_CAUSAL;
    int32_t quantMode = PKI_QUANT_NONE;
    bool returnValue = false;
    PkiDataLayout layoutQ = PkiDataLayout::BSND;
    PkiDataLayout layoutK = PkiDataLayout::BSND;
    bool pageAttention = false;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    uint32_t usedCoreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    ge::DataType queryDtype = ge::DT_FLOAT16;
    ge::DataType keyDtype = ge::DT_FLOAT16;
    ge::DataType weightsDtype = ge::DT_FLOAT16;
    // 索引输出布局模式(尾块追加模式): 0=legacy / 1=compact, 来自 sparse_indices_mode 属性
    // (仅 V2 接口/图模式可设置; V1 缺省 legacy)
    uint32_t sparseIndicesMode = PKI_SPARSE_INDICES_MODE_LEGACY;
};

// ------------------ Tiling Class ------------------
class PoolKeyIndexerTiling {
public:
    explicit PoolKeyIndexerTiling(gert::TilingContext* context)
        : context_(context)
    {}
    ge::graphStatus DoTiling(PoolKeyIndexerTilingInfo* tilingInfo);

private:
    gert::TilingContext* context_ = nullptr;
    PoolKeyIndexerTilingData tilingData_;
    std::vector<uint32_t> keyStridesVec_;
    // k_descale 运行时 stride(PA 0 轴非连续支持, 与 keyStridesVec_ 同源时机捕获)
    std::vector<uint32_t> keyDequantScaleStridesVec_;
    // pool_key 是否按"0 池"空张量编码(BSND dim1=0 / TND·PA dim0=0)。
    // 在空张量形态校验处置位, 供后续 int8 k_descale 缺省校验使用
    // (此时 info.s2Size 尚未解析, 不可用其判 0 池)
    bool keyZeroPool_ = false;

    ge::graphStatus ParseAndCheckParams(PoolKeyIndexerTilingInfo& info);
    ge::graphStatus CalcTilingParams(PoolKeyIndexerTilingInfo& info);
    ge::graphStatus CalcWorkspaceSize(PoolKeyIndexerTilingInfo& info, uint64_t& workspaceSize);
    ge::graphStatus CheckKeyContiguous(const PoolKeyIndexerTilingInfo& info) const;
    uint32_t GetTrunkLen(uint32_t sparseCount);
    void SetTilingData(PoolKeyIndexerTilingInfo& info);
};

} // namespace optiling
#endif // POOL_KEY_INDEXER_TILING_H
