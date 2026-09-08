/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

// ACLNN_CMD 宏将 GetWorkspaceSize 函数名绑定为 "<api>GetWorkspaceSize"(字符串拼接),
// 无法覆盖 GWS 与 run 函数名不同前缀的场景: aclnnPoolKeyIndexerTensorGetWorkspaceSize
// (Tensor 变体, ValueDepend 输入为 aclTensor*) 与 run 函数共享 aclnnPoolKeyIndexer。
// 本辅助函数按显式函数名调用, 执行语义与 ACLNN_CMD 一致。
// 函数地址经名称缓存(线程安全), 避免每次调用重复 dlsym。
static void *CachedOpApiFuncAddr(const char *apiName)
{
    static std::mutex mu;
    static std::unordered_map<std::string, void *> cache;
    std::lock_guard<std::mutex> guard(mu);
    auto it = cache.find(apiName);
    if (it != cache.end()) {
        return it->second;
    }
    void *addr = GetOpApiFuncAddr(apiName);
    cache.emplace(apiName, addr);
    return addr;
}

template <typename... Ts>
static void AclnnCallNamed(const char *gwsApiName, const char *runApiName, Ts &...args)
{
    auto device = DecodeDevice(args...);
    const c10::OptionalDeviceGuard device_guard(device);
    const auto getWorkspaceSizeFuncAddr = CachedOpApiFuncAddr(gwsApiName);
    const auto opApiFuncAddr = CachedOpApiFuncAddr(runApiName);
    const auto initMemAddr = CachedOpApiFuncAddr("InitHugeMemThreadLocal");
    const auto unInitMemAddr = CachedOpApiFuncAddr("UnInitHugeMemThreadLocal");
    const auto releaseMemAddr = CachedOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, gwsApiName, " or ", runApiName,
                " not in opapi lib.");
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;
    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }
    ApplyDeterministicConfig();
    auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);
    auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
    if (workspace_status != 0) {
        ReleaseConvertTypes(converted_params);
        TORCH_CHECK(false, "call ", gwsApiName, " failed, detail:", aclGetRecentErrMsg());
    }
    at::Tensor workspace_tensor;
    void *workspace_addr = nullptr;
    if (workspace_size != 0) {
        at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte));
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }
    ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor, opApiFuncAddr,
                     releaseMemFunc, runApiName]() -> int {
        typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        ReleaseConvertTypes(converted_params);
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        TORCH_CHECK(api_ret == 0, "call ", runApiName, " failed, detail:", aclGetRecentErrMsg());
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name(runApiName);
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
}

const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;

inline TensorWrapper PkiMakeWrapper(const at::Tensor &tensor)
{
    return {tensor, ConvertToAclDataType(tensor.scalar_type())};
}

inline bool PkiIsFp8Tensor(const at::Tensor &tensor)
{
    return tensor.scalar_type() == at::kFloat8_e4m3fn;
}

inline bool PkiIsE8M0Tensor(const at::Tensor &tensor)
{
    return tensor.scalar_type() == at::kFloat8_e8m0fnu;
}

constexpr int64_t PKI_E8M0_SCALE_PACK_NUM = 2;

// PKI 量化 dtype 修正(参考 QLIv2 FixQLIV2AclDtypes 模式):
//   quantMode=0: q/k=FP8_E4M3FN, descale=FLOAT —— aclnn 按输入 dtype 自动识别,
//                仅校验 dtype 匹配;
//   quantMode=1: descale=FLOAT8_E8M0 —— 需显式设置 scale dtype(ACL_FLOAT8_E8M0),
//                并校验 2 元素 E8M0 打包对齐(Cube 按 bf16 视图成对加载)。
inline void FixPkiAclDtypes(int64_t quantMode, TensorWrapper &queryWrapper, TensorWrapper &poolKeyWrapper,
                            TensorWrapper &qScaleWrapper, TensorWrapper &kScaleWrapper)
{
    if (quantMode == 0 || quantMode == 1) {
        TORCH_CHECK(PkiIsFp8Tensor(queryWrapper.tensor_),
                    "When quant_mode is 0 or 1, query must be torch.float8_e4m3fn, but got ",
                    queryWrapper.tensor_.scalar_type());
        TORCH_CHECK(PkiIsFp8Tensor(poolKeyWrapper.tensor_),
                    "When quant_mode is 0 or 1, pool_key must be torch.float8_e4m3fn, but got ",
                    poolKeyWrapper.tensor_.scalar_type());
    }
    if (quantMode == 0) {
        TORCH_CHECK(qScaleWrapper.tensor_.scalar_type() == at::kFloat,
                    "When quant_mode is 0, q_descale must be torch.float, but got ",
                    qScaleWrapper.tensor_.scalar_type());
        TORCH_CHECK(kScaleWrapper.tensor_.scalar_type() == at::kFloat,
                    "When quant_mode is 0, k_descale must be torch.float, but got ",
                    kScaleWrapper.tensor_.scalar_type());
    } else if (quantMode == 1) {
        TORCH_CHECK(PkiIsE8M0Tensor(qScaleWrapper.tensor_),
                    "When quant_mode is 1, q_descale must be torch.float8_e8m0fnu, but got ",
                    qScaleWrapper.tensor_.scalar_type());
        TORCH_CHECK(PkiIsE8M0Tensor(kScaleWrapper.tensor_),
                    "When quant_mode is 1, k_descale must be torch.float8_e8m0fnu, but got ",
                    kScaleWrapper.tensor_.scalar_type());
        // Cube loads E8M0 scales in two-byte groups through a bfloat16_t view.
        TORCH_CHECK(qScaleWrapper.tensor_.storage_offset() % PKI_E8M0_SCALE_PACK_NUM == 0,
                    "When quant_mode is 1, q_descale storage offset must satisfy 2-element E8M0 packing alignment, "
                    "but got ",
                    qScaleWrapper.tensor_.storage_offset());
        TORCH_CHECK(kScaleWrapper.tensor_.storage_offset() % PKI_E8M0_SCALE_PACK_NUM == 0,
                    "When quant_mode is 1, k_descale storage offset must satisfy 2-element E8M0 packing alignment, "
                    "but got ",
                    kScaleWrapper.tensor_.storage_offset());
        qScaleWrapper.dtype = ACL_FLOAT8_E8M0;
        kScaleWrapper.dtype = ACL_FLOAT8_E8M0;
    }
}

// Derive output shapes from query shape + layout.
//   BSND: query (B,S1,N1,D) -> indices (B,S1,topk+poolSize-1) / values (B,S1,topk/poolSize)
//   TND:  query (T1,N1,D)   -> indices (T1,topk+poolSize-1)     / values (T1,topk/poolSize)
// Note: PKI output has no N2 dimension (N2 is fixed to 1), unlike LIV2 which emits keyHeadNum.
std::tuple<at::Tensor, at::Tensor> ConstructPoolKeyIndexerOutputTensor(const at::Tensor &query, int64_t topk,
                                                                       int64_t poolSize,
                                                                       const std::string &queryLayoutStr,
                                                                       bool returnValue)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater than 0, but shape[", i,
                    "] is ", query.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    TORCH_CHECK(poolSize > 0, "pool_size should be greater than 0, but now is ", poolSize);
    TORCH_CHECK(topk % poolSize == 0, "topk(", topk, ") should be divisible by pool_size(", poolSize, ").");

    int64_t indicesLen = topk + poolSize - 1;
    int64_t valuesLen = topk / poolSize;

    at::SmallVector<int64_t, SIZE> indicesShape;
    at::SmallVector<int64_t, SIZE> valuesShape;

    if (queryLayoutStr == "BSND") {
        indicesShape = {query.size(DIM_0), query.size(DIM_1), indicesLen};
        valuesShape = {query.size(DIM_0), query.size(DIM_1), valuesLen};
    } else {
        indicesShape = {query.size(DIM_0), indicesLen};
        valuesShape = {query.size(DIM_0), valuesLen};
    }

    at::Tensor sparseIndicesOut = at::empty(indicesShape, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        // Spec requires sparseValuesOut to be FLOAT (not query dtype)
        sparseValuesOut = at::empty(valuesShape, query.options().dtype(at::kFloat));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::vector<bool> IsContiguousAxes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguousStride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguousStride[i] = contiguousStride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguousStride[i]);
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor> PoolKeyIndexer(
    const at::Tensor &query, const at::Tensor &poolKey, const at::Tensor &weights, const at::Tensor &poolTailK,
    const c10::optional<at::Tensor> &actualSeqQ, const c10::optional<at::Tensor> &actualSeqK,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &qDescale,
    const c10::optional<at::Tensor> &kDescale, c10::string_view layoutQ, c10::string_view layoutK, int64_t topk,
    int64_t poolSize, int64_t maskMode, int64_t quantMode, bool returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(poolKey.numel() > 0, "Tensor pool_key is empty.")
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.")
    TORCH_CHECK(poolTailK.numel() > 0, "Tensor pool_tail_k is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    std::tuple<at::Tensor, at::Tensor> poolKeyIndexerOutput =
        ConstructPoolKeyIndexerOutputTensor(query, topk, poolSize, queryLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(poolKeyIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(poolKeyIndexerOutput);

    at::Device outputDevice = query.device();
    (void)outputDevice;

    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    // pool_key 0轴非连续支持(参考 compressor 方案): eager/aclnn 链路的
    // tiling context 不上报运行时 stride, 故在 torch extension 层直读
    // at::Tensor::stride(0) 作为 key_stride0 属性下传; 非 0 轴必须连续。
    // PA_BBND 下 0 轴 stride >= blockSize*N2*D 时按非连续寻址, 否则等价连续。
    // 非 PA(BSND/TND) 布局: kernel KeyNd2Nz 连续寻址且 tiling 不消费
    // key_stride0 属性(该属性仅 PA 分支接线), 0 轴非连续输入会静默错算,
    // 必须在此显式拒绝(图模式由框架上报运行时 stride 经 tiling 拦截)。
    int64_t keyStride0 = -1;
    if (!poolKey.is_contiguous()) {
        auto contiguousAxes = IsContiguousAxes(poolKey);
        bool isPaBbnd = (keyLayoutStr == "PA_BBND");
        TORCH_CHECK(isPaBbnd,
                    "pool_key non-contiguous tensor is only supported with layout_k='PA_BBND' "
                    "(0-axis non-contiguous), but layout_k is '",
                    keyLayoutStr, "'; pass a contiguous pool_key or use PA_BBND layout");
        for (int64_t i = 1; i < static_cast<int64_t>(contiguousAxes.size()); i++) {
            TORCH_CHECK(contiguousAxes[i], "pool_key only supports non-contiguous tensor on the 0-axis, axis[", i,
                        "] stride is not contiguous");
        }
        keyStride0 = poolKey.stride(0);
        if (isPaBbnd) {
            int64_t contiguousStride0 = poolKey.size(1) * poolKey.size(2) * poolKey.size(3);
            TORCH_CHECK(keyStride0 >= contiguousStride0, "pool_key stride0(", keyStride0,
                        ") must be >= contiguous stride(", contiguousStride0, ") in PA_BBND scenarios");
        }
    }

    // 量化路径 dtype 修正(quantMode=0/1): mxFP8 的 E8M0 scale 需显式 ACL dtype;
    // 非量化路径直接透传 at::Tensor(ConvertType 自动转换)。
    const bool qDescaleDefined = qDescale.has_value() && qDescale.value().defined();
    const bool kDescaleDefined = kDescale.has_value() && kDescale.value().defined();
    const bool isQuantPath = (quantMode == 0 || quantMode == 1);

    // k_descale 0轴非连续支持(与 pool_key 的 key_stride0 属性同机制):
    // eager/aclnn 链路的 tiling context 不上报运行时 stride, 故在 torch extension
    // 层直读 at::Tensor::stride(0) 作为 k_descale_stride0 属性下传; 非 0 轴必须连续。
    // 仅 PA_BBND 布局支持 0 轴非连续(kernel GetKeyScale 按 block_table 逐物理块
    // 寻址, 块基址 = blockId * keyDequantScaleStride0); 非 PA 布局 kernel 连续
    // 寻址且 tiling 不消费该属性, 非连续输入会静默错算, 必须显式拒绝。
    int64_t kDescaleStride0 = -1;
    if (kDescaleDefined && !kDescale.value().is_contiguous()) {
        bool isPaBbndKds = (keyLayoutStr == "PA_BBND");
        TORCH_CHECK(isPaBbndKds,
                    "k_descale non-contiguous tensor is only supported with layout_k='PA_BBND' "
                    "(0-axis non-contiguous), but layout_k is '",
                    keyLayoutStr, "'; pass a contiguous k_descale or use PA_BBND layout");
        auto kdsContiguousAxes = IsContiguousAxes(kDescale.value());
        for (int64_t i = 1; i < static_cast<int64_t>(kdsContiguousAxes.size()); i++) {
            TORCH_CHECK(kdsContiguousAxes[i], "k_descale only supports non-contiguous tensor on the 0-axis, axis[", i,
                        "] stride is not contiguous");
        }
        kDescaleStride0 = kDescale.value().stride(0);
        if (isPaBbndKds) {
            // PA: 属性须 >= 连续 stride(0 轴非连续只允许加 padding);
            // 连续值因 quant_mode 而异(mode=0: blockSize*N2*D/32_e8m0_per_token... 此处
            // 按元素单位校验下界, tiling 侧做精确校验)
            TORCH_CHECK(kDescaleStride0 > 0, "k_descale stride0(", kDescaleStride0, ") must be positive");
        }
    }

    // 量化路径 dtype 修正(quantMode=0/1): mxFP8 的 E8M0 scale 需显式 ACL dtype;
    // 非量化路径直接透传 at::Tensor(ConvertType 自动转换)。
    if (isQuantPath) {
        TORCH_CHECK(qDescaleDefined && kDescaleDefined, "quant_mode=", quantMode,
                    " requires both q_descale and k_descale to be provided");
    } else if (qDescaleDefined || kDescaleDefined) {
        TORCH_CHECK(false, "quant_mode=-1 requires q_descale and k_descale to be null");
    }
    // 量化路径: wrapper 生命周期与所在分支的 ACLNN_CMD 调用一致
    auto queryWrapper = PkiMakeWrapper(query);
    auto poolKeyWrapper = PkiMakeWrapper(poolKey);
    auto qScaleWrapper = qDescaleDefined ? PkiMakeWrapper(qDescale.value()) : PkiMakeWrapper(poolKey);
    auto kScaleWrapper = kDescaleDefined ? PkiMakeWrapper(kDescale.value()) : PkiMakeWrapper(poolKey);
    if (isQuantPath) {
        FixPkiAclDtypes(quantMode, queryWrapper, poolKeyWrapper, qScaleWrapper, kScaleWrapper);
    }

    // pool_tail_k / actual_seq_q / actual_seq_k 为 ValueDepend(OPTIONAL) 输入,
    // 生成的 aclnn API 有两个变体, 按输入设备分流:
    //   - aclnnPoolKeyIndexerGetWorkspaceSize (aclIntArray*, host 值):
    //     全部值输入在 CPU 时使用。tiling 期可读 host 值做校验(前缀和单调性 /
    //     tail 范围), 无任何拷贝; aclgraph 捕获期无 D2H, 值烘焙为图常量
    //     (捕获后改 CPU 值不生效, 需重新捕获)。
    //   - aclnnPoolKeyIndexerTensorGetWorkspaceSize (aclTensor*, run 函数共享
    //     aclnnPoolKeyIndexer): 任一值输入在 NPU 时使用。无阻塞 D2H —— 消除
    //     eager 每次调用的同步点, 并使 NPUGraph/aclgraph 捕获成为可能
    //     (device 输入为图输入, 捕获后修改 buffer 内容 replay 自动生效)。
    //     代价: tiling 跳过这些输入的 host 值校验(与 GE 图模式口径一致),
    //     kernel 运行期从 GM 读值。混布输入(CPU+NPU)统一搬至计算设备,
    //     host->device 拷贝仅 eager 可用(捕获期内禁止)。
    const bool actualSeqQDefined = actualSeqQ.has_value() && actualSeqQ.value().defined();
    const bool actualSeqKDefined = actualSeqK.has_value() && actualSeqK.value().defined();
    const bool valueInputsAllHost = poolTailK.is_cpu() && (!actualSeqQDefined || actualSeqQ.value().is_cpu()) &&
                                    (!actualSeqKDefined || actualSeqK.value().is_cpu());

    if (valueInputsAllHost) {
        // Convert 1-D int64 tensors to at::IntArrayRef so ConvertType produces aclIntArray*.
        at::Tensor poolTailKCpu = poolTailK.to(at::kLong).cpu().contiguous();
        at::IntArrayRef poolTailKArr(poolTailKCpu.data_ptr<int64_t>(), poolTailKCpu.numel());

        at::Tensor actualSeqQCpu;
        c10::optional<at::IntArrayRef> actualSeqQArr = c10::nullopt;
        if (actualSeqQDefined) {
            actualSeqQCpu = actualSeqQ.value().to(at::kLong).cpu().contiguous();
            actualSeqQArr = at::IntArrayRef(actualSeqQCpu.data_ptr<int64_t>(), actualSeqQCpu.numel());
        }

        at::Tensor actualSeqKCpu;
        c10::optional<at::IntArrayRef> actualSeqKArr = c10::nullopt;
        if (actualSeqKDefined) {
            actualSeqKCpu = actualSeqK.value().to(at::kLong).cpu().contiguous();
            actualSeqKArr = at::IntArrayRef(actualSeqKCpu.data_ptr<int64_t>(), actualSeqKCpu.numel());
        }

        if (isQuantPath) {
            // 量化路径: TensorWrapper 传递(使 FixPkiAclDtypes 修正后的 dtype 生效)
            ACLNN_CMD(aclnnPoolKeyIndexer, queryWrapper, poolKeyWrapper, weights, poolTailKArr, actualSeqQArr,
                      actualSeqKArr, blockTable, qScaleWrapper, kScaleWrapper, topk, poolSize, queryLayoutPtr,
                      keyLayoutPtr, maskMode, quantMode, returnValue, keyStride0, kDescaleStride0, sparseIndicesOut,
                      sparseValuesOut);
        } else {
            ACLNN_CMD(aclnnPoolKeyIndexer, query, poolKey, weights, poolTailKArr, actualSeqQArr, actualSeqKArr,
                      blockTable, qDescale, kDescale, topk, poolSize, queryLayoutPtr, keyLayoutPtr, maskMode, quantMode,
                      returnValue, keyStride0, kDescaleStride0, sparseIndicesOut, sparseValuesOut);
        }

        return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
    }

    // Tensor 变体: 值输入统一为计算设备上的 int64 连续 tensor。
    // (NPU 上的 dtype 转换/连续化为异步算子, 可被图捕获; CPU 输入的 H2D 仅 eager。)
    const at::Device computeDevice = query.device();
    at::Tensor poolTailKT = poolTailK.to(at::kLong).to(computeDevice).contiguous();
    c10::optional<at::Tensor> actualSeqQT = c10::nullopt;
    if (actualSeqQDefined) {
        actualSeqQT = actualSeqQ.value().to(at::kLong).to(computeDevice).contiguous();
    }
    c10::optional<at::Tensor> actualSeqKT = c10::nullopt;
    if (actualSeqKDefined) {
        actualSeqKT = actualSeqK.value().to(at::kLong).to(computeDevice).contiguous();
    }

    if (isQuantPath) {
        AclnnCallNamed("aclnnPoolKeyIndexerTensorGetWorkspaceSize", "aclnnPoolKeyIndexer", queryWrapper, poolKeyWrapper,
                       weights, poolTailKT, actualSeqQT, actualSeqKT, blockTable, qScaleWrapper, kScaleWrapper, topk,
                       poolSize, queryLayoutPtr, keyLayoutPtr, maskMode, quantMode, returnValue, keyStride0,
                       kDescaleStride0, sparseIndicesOut, sparseValuesOut);
    } else {
        AclnnCallNamed("aclnnPoolKeyIndexerTensorGetWorkspaceSize", "aclnnPoolKeyIndexer", query, poolKey, weights,
                       poolTailKT, actualSeqQT, actualSeqKT, blockTable, qDescale, kDescale, topk, poolSize,
                       queryLayoutPtr, keyLayoutPtr, maskMode, quantMode, returnValue, keyStride0, kDescaleStride0,
                       sparseIndicesOut, sparseValuesOut);
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pool_key_indexer", &PoolKeyIndexer, "pool_key_indexer");
}
} // namespace op_api
