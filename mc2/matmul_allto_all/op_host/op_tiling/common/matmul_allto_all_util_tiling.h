/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_ALLTO_ALL_UTIL_TILING_H
#define MATMUL_ALLTO_ALL_UTIL_TILING_H

#include "op_host/op_tiling/mc2_tiling_utils.h"

namespace MC2Tiling {

using namespace std;
using namespace ge;
using namespace gert;

// 参数范围
const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16};
constexpr uint64_t K_MAX_VALUE = 65535UL;
constexpr uint64_t MAX_INT32_VALUE = 2147483647UL;
constexpr size_t MAX_GROUP_NAME_LEN = 127;
constexpr int64_t RANK_DEFAULT_NUM = -1;
constexpr uint32_t MAX_CCU_RANKSIZE = 8;
// FOR NON_QUANT
const std::vector<uint32_t> NON_QUANT_X_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16};
// FOR QUANT
const std::vector<uint32_t> KC_QUANT_X_DTYPE_LIST = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
const std::vector<uint32_t> KC_QUANT_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
const std::vector<uint32_t> MX_QUANT_X_DTYPE_LIST = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1};
const std::vector<uint32_t> MX_QUANT_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
// 维度范围
constexpr uint32_t TWO_DIMS = 2;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;

// input index
constexpr size_t INPUT_X1_INDEX = 0;
constexpr size_t INPUT_X2_INDEX = 1;
constexpr size_t INPUT_BIAS_INDEX = 2;
constexpr size_t INPUT_X1_SCALE_INDEX = 3;
constexpr size_t INPUT_X2_SCALE_INDEX = 4;
constexpr size_t INPUT_COMM_SCALE_INDEX = 5;
constexpr size_t INPUT_X1_OFFSET_INDEX = 6;
constexpr size_t INPUT_X2_OFFSET_INDEX = 7;

// output index
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t ALLTO_ALL_OUT_INDEX = 1; // AlltoAllMatmul的第二个输出

// attr index,对于MatmulAlltoAll和AlltoAllMatmul相同的index
constexpr size_t ATTR_GROUP_INDEX = 0;
constexpr size_t ATTR_WORLD_SIZE_INDEX = 1;
constexpr size_t ATTR_ALLTO_ALL_AXES_INDEX = 2;
constexpr size_t ATTR_Y_DTYPE_INDEX = 3;
constexpr size_t ATTR_X1_QUANTMODE_INDEX = 4;
constexpr size_t ATTR_X2_QUANTMODE_INDEX = 5;
constexpr size_t ATTR_COMMON_QUANTMODE_INDEX = 6;
// MatmulAlltoAll的差异的Attr
constexpr size_t ATTR_COMMON_QUANTDTYPE_INDEX = 7;
constexpr size_t ATTR_X1_TRANSPOSE_INDEX = 8;
constexpr size_t ATTR_X2_TRANSPOSE_INDEX = 9;
constexpr size_t ATTR_GROUP_SIZE_INDEX = 10;
constexpr size_t ATTR_COMM_MODE_INDEX = 11;
// AlltoAllMatmul的差异的Attr
constexpr size_t ALLTOALLMATMUL_ATTR_X1_QUANTDTYPE_INDEX = 7;
constexpr size_t ALLTOALLMATMUL_ATTR_COMMON_QUANTDTYPE_INDEX = 8;
constexpr size_t ALLTOALLMATMUL_ATTR_X1_TRANSPOSE_INDEX = 9;
constexpr size_t ALLTOALLMATMUL_ATTR_X2_TRANSPOSE_INDEX = 10;
constexpr size_t ALLTOALLMATMUL_ATTR_GROUP_SIZE_INDEX = 11;
constexpr size_t ALLTOALLMATMUL_ATTR_COMM_MODE_INDEX = 12;
constexpr size_t ALLTOALLMATMUL_ATTR_ALLTO_ALL_OUT_FLAG_INDEX = 13;

// 用来存放MatmulAlltoAll和AllToAllMatmul错位的属性信息的处理，
// 比如说MatmulAllToAll存放x1转置(ATTR_X1_TRANSPOSE_INDEX)的位置是8，
// AllToAllMatmul(ALLTOALLMATMUL_ATTR_X1_TRANSPOSE_INDEX)的是9
struct OpAttrIndexSchema {
    size_t x1Transpose;
    size_t x2Transpose;
    size_t commMode;
    // 可以继续添加其他错位属性索引
};

// 上面结构体的实例化
const OpAttrIndexSchema MATMUL_ALLTOALL_INDEX_SCHEMA = {ATTR_X1_TRANSPOSE_INDEX, ATTR_X2_TRANSPOSE_INDEX,
                                                        ATTR_COMM_MODE_INDEX};

const OpAttrIndexSchema ALLTOALL_MATMUL_INDEX_SCHEMA = {ALLTOALLMATMUL_ATTR_X1_TRANSPOSE_INDEX,
                                                        ALLTOALLMATMUL_ATTR_X2_TRANSPOSE_INDEX,
                                                        ALLTOALLMATMUL_ATTR_COMM_MODE_INDEX};

// 维护一个输入输出shape的维度的结构体，仅支持x1,x2,output都是二维的情况
struct Matrix2DShapes {
    uint64_t x1Dim0;
    uint64_t x1Dim1;
    uint64_t x2Dim0;
    uint64_t x2Dim1;
    uint64_t yDim0;
    uint64_t yDim1;
};

constexpr uint32_t X1_QUANTMODE_VALUE = 3;
constexpr uint32_t X2_QUANTMODE_VALUE = 2;

// 定义量化模式枚举，直接取量化组合
enum class QuantMode : uint8_t {
    NON_QUANT = 0, // 非量化模式
    KC_QUANT = 1,  // K量化模式,对应x1, x2分别为perToken量化与perchannel量化
    MX_QUANT = 2,
    ERROR = 255 // 特殊设置，表示不支持的类型组合
};

// 封装输入参数，主要为输入获取得到的参数
struct TilingContextInfo {
    std::string group = "group"; // group属性
    QuantMode quantMode = QuantMode::NON_QUANT;
    mc2tiling::TilingArgs args_;
    bool allToAllOutFlag =
        false; // AlltoAllMatmul用于存放alltoAllFlag的标识,AlltoAll在前，为true表示当前存在alltoall的对应地址
    ge::DataType hcclGeType;
    uint64_t x1KcDynQuantDTypeVal = 0;
};

// 封装Tiling过程中推导得到的参数
struct TilingInferredInfo {
    uint64_t mmResultLen =
        0UL; // 存储计算MM的地址大小，仅对于MatmulAlltoAll，因为先执行完Matmul之后需要有空间存放计算地址
    uint64_t commLen = 0UL; // 存储通信结果的临时空间，仅对于AlltoAllMatmul，需要有空间存放重排的地址（和kernel侧约定）
    uint64_t permuteLen =
        0UL; // 重排空间大小,对于AlltoAllMatmul来说，当alltoAllout存在时，就有一个额外的alltoall地址传递给kernel侧，不需要额外分配
    uint64_t x1ScaleOptionalLen = 0UL; // 存储x1ScaleOptional的地址大小
    uint64_t quantOutLen = 0UL;        // 存储quantOut的空间
    uint64_t commScaleLen = 0UL;       // 存储Scale通信的空间
    uint64_t permuteScaleLen = 0UL;    // 存储Scale重排的空间
    uint32_t biasLen = 0UL;            // 存储偏移的地址大小
    uint32_t tileM = 0UL;              // 头块大小
    uint32_t tileCnt = 0UL;            // 头块数量
    uint32_t tailM = 0UL;              // 尾块大小
    uint32_t tailCnt = 0UL;            // 尾块数量
};

class MatmulAlltoAllTilingUtil {
public:
    static QuantMode GetQuantMode(const gert::TilingContext *context, const char *opName);
    static ge::graphStatus GetAndValidateRankSize(const gert::TilingContext *context, const char *opName,
                                                  const char *group, int64_t &rankDim);
    static ge::graphStatus GetAndConvertCommMode(const gert::TilingContext *context, const char *opName,
                                                 const TilingContextInfo &contextInfo,
                                                 const OpAttrIndexSchema &indexSchema, uint8_t &commMode);
    static void GetMatrix2DShapes(const gert::TilingContext *context, Matrix2DShapes &shapes);
    static ge::graphStatus CheckAttrsInfo(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema);
    static ge::graphStatus CheckShapeInfo(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema);
    static ge::graphStatus CheckKcQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                                 const OpAttrIndexSchema &indexSchema);
    static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context, const char *opName);
    static ge::graphStatus CheckNonQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    static ge::graphStatus CheckKcQuantTensorDataType(const gert::TilingContext *context, const char *opName);
    static ge::graphStatus SetAttrsInfo(const gert::TilingContext *context, const char *opName,
                                        TilingContextInfo &contextInfo, const OpAttrIndexSchema &indexSchema);
    static ge::graphStatus SetShapeInfo(const gert::TilingContext *context, TilingContextInfo &contextInfo);
    static ge::graphStatus SetDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                           TilingContextInfo &contextInfo);
    static ge::graphStatus SetKcDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                             TilingContextInfo &contextInfo);
};

// Builder模式成员函数通常使用小写字母开头
class Mc2CcTilingConfigBuilder {
private:
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig;
    static constexpr uint32_t SUCCESS = 0;
    std::set<std::string> errorSet{};

    uint32_t reduceType = 0;
    uint8_t dstDataType = 0;
    uint8_t srcDataType = 0;
    uint8_t stepSize = 0;
    uint8_t localRankDataToLocalDst = 0;
    // 0,1代表false 2代表true
    uint8_t srcDataFromWindow = 0;
    uint32_t debugMode = 0;
    uint16_t commBlockNum = 0;
    uint16_t queueNum = 0;
    uint8_t commEngine = 0;

    // 标识是否参与build操作,为false则跳过对应的set方法调用
    bool hasReduceType = false;
    bool hasStepSize = false;
    bool hasLocalRankDataToLocalDst = false;
    bool hasSrcDataFromWindow = false;
    bool hasDebugMode = false;
    bool hasCommBlockNum = false;
    bool hasQueueNum = false;
    bool hasCommEngine = false;

    bool hasGotInit = false;

public:
    struct ConfigFile {
        static constexpr const char *GROUP_NAME = "groupName";
        static constexpr const char *OP_TYPE = "opType";
        static constexpr const char *ALG_CONFIG = "algConfig";
        static constexpr const char *REDUCE_TYPE = "reduceType";
        static constexpr const char *STEP_SIZE = "stepSize";
        static constexpr const char *SKIP_LOCAL_RANK_COPY = "skipLocalRankCopy";
        static constexpr const char *SKIP_BUFFER_WINDOW_COPY = "skipBufferWindowCopy";
        static constexpr const char *DEBUG_MODE = "debugMode";
        static constexpr const char *COMM_BLOCK_NUM = "commBlockNum";
        static constexpr const char *QUEUE_NUM = "queueNum";
        static constexpr const char *COMM_ENGINE = "commEngine";
    };

    struct AlgConfigType {
        static constexpr const char *ALL_REDUCE = "AllReduce=level0:doublering";
        static constexpr const char *ALL_GATHER = "AllGather=level0:doublering";
        static constexpr const char *REDUCE_SCATTER = "ReduceScatter=level0:doublering";
        static constexpr const char *ALL_TO_ALL = "AlltoAll=level0:fullmesh;level1:pairwise";
        static constexpr const char *BATCH_WRITE = "BatchWrite=level0:fullmesh";
    };

    static constexpr const char *GROUP = "group";
    static constexpr const uint8_t CONSTANTS_ZERO = 0;
    static constexpr const uint8_t CONSTANTS_ONE = 1;
    static constexpr const uint8_t CONSTANTS_TWO = 2;

    // 构造函数
    Mc2CcTilingConfigBuilder(const std::string &groupName, uint32_t opType, const std::string &algConfig);

    // 链式配置方法（声明),通过change和with方法来修改hccl中的对应属性
    Mc2CcTilingConfigBuilder &withReduceType(const char *opName, mc2tiling::HcclReduceOp reduceType,
                                             ge::DataType dstDataType, ge::DataType srcDataType);
    Mc2CcTilingConfigBuilder &withStepSize(uint8_t stepSize);
    Mc2CcTilingConfigBuilder &isLocalRankDataToLocalDst(bool flag);
    Mc2CcTilingConfigBuilder &isSrcDataFromWindow(bool fromWindow);
    Mc2CcTilingConfigBuilder &withDebugMode(uint8_t debugMode);
    Mc2CcTilingConfigBuilder &withCommBlockNum(uint16_t commBlockNum);
    Mc2CcTilingConfigBuilder &withQueueNum(uint16_t queueNum);
    Mc2CcTilingConfigBuilder &withCommEngine(uint8_t commEngine);

    AscendC::Mc2CcTilingConfig build();
    bool isSuccess() const;
    std::string errorMsg() const;
    void setInitTilingData(::Mc2InitTiling initTiling);

    // 模版方法
    template <typename... Args>
    void setCcTilingData(Args &&...args)
    {
        (mc2CcTilingConfig.GetTiling(args), ...);
    }

    // 静态工厂
    static Mc2CcTilingConfigBuilder create(const std::string &groupName, mc2tiling::AicpuComType opType,
                                           const std::string &algConfig);
};

inline bool IsArrayEqual(std::vector<uint32_t> &arr1, const std::vector<uint32_t> &arr2, uint32_t size)
{
    if (arr1.size() < size || arr2.size() < size) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

}; // namespace MC2Tiling

#endif // MATMUL_ALLTO_ALL_UTIL_TILING_H
