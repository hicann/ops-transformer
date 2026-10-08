/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_channel_builder.cpp
 * \brief Host 侧 HCCL channel 创建工具，通过 CommChannelBuilderManager 对外暴露接口
 *
 * CommChannelBuilder 直接引用 mc2/common/op_kernel/apace 下的 comm_channel_builder.h 原版实现，
 * CommChannelBuilderManager 负责适配 torch_extension：
 * - 从 group name 获取 HcclComm handle
 * - 动态加载 HCCL 函数指针 (InitHcclEngineCtxFunctions)
 * - 持有 CommContext 成员，调用原版 CreateDeviceContext 填充
 * - 用 at::from_blob 将 device 指针包装为 at::Tensor 返回
 */

#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "hccl_common.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/utils/comm_channel_builder.h"

namespace op_api {

using ApaceCommUdmaContext = Apace::AivComm::CommUdmaContext;
using ApaceCommUbmemContext = Apace::AivComm::CommUbmemContext;

struct CommContext {
    ApaceCommUdmaContext udmaCtx;
    ApaceCommUbmemContext ubmemCtx;
};

// 与 device 侧 quant_reduce_scatter_context.h 的 HCCL_MAX_RANK_SIZE 保持一致
constexpr uint32_t QUANT_MTE_MAX_RANK_SIZE = 1024U;

// 低比特量化通信算子公共 context 的 host 侧布局，字段与 device 侧
// quant_reduce_scatter_context.h 的 QuantReduceScatterContext 一一对应
struct QuantMteContext {
    uint32_t rankId = 0;                                // 本 rank 在通信组内的 rank id
    uint32_t rankSizePerServer = 0;                     // 单服务器内 rank 数
    uint64_t hcclBuffer_[QUANT_MTE_MAX_RANK_SIZE] = {}; // 各 rank 内置 HCCL buffer 的映射地址
};
static_assert(sizeof(QuantMteContext) == 8200, "QuantMteContext layout must match QuantReduceScatterContext");

using DefaultCommChannelBuilder = CommChannelBuilder<ChannelMode::URMA, ChannelMode::UBMEM>;

class CommChannelBuilderManager {
public:
    explicit CommChannelBuilderManager(const std::string& group)
        : group_(group)
    {
        InitHcclEngineCtxFunctions();

        auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm_);
        TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", aclnnRet);

        builder_ = std::make_unique<DefaultCommChannelBuilder>(comm_);
    }

    uint32_t GetRankId() const
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        return builder_->GetRankId();
    }

    uint32_t GetRankSize() const
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        return builder_->GetRankSize();
    }

    /*!
     * \brief 创建 device context 并包装为 at::Tensor 返回
     *
     * 内部流程：
     * 1. 调用原版 CreateDeviceContext(hostCtx, size, tag, dataCtx, barrierCtx) 填充 CommContext
     * 2. 用 at::from_blob 将返回的 device 指针包装为 at::Tensor
     */
    at::Tensor CreateContext(const std::string& ctxTag)
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        CommContext hostCtx = {};
        void* devCtx = builder_->CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag.c_str(), &hostCtx.udmaCtx,
                                                     &hostCtx.ubmemCtx);

        TORCH_CHECK(devCtx != nullptr, "CreateDeviceContext failed, ctxTag: ", ctxTag);

        at::Tensor context = at::from_blob(devCtx, {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
        return context;
    }

    /*!
     * \brief 获取或创建 quant MTE 算子族公共 context 的 host 侧内容
     *
     * 固定 ctxTag 保证 eager/静态图/动态图共享同一份 context。HcclEngineCtxGet 命中时
     * 直接读进程内 host 缓存；未命中时交换各 rank 内置 buffer 地址（UB_MEM 协议）、
     * 创建 engine context 并拷贝。全程仅调用 HCCL/ACL C 接口，不经过 PyTorch
     * dispatcher，可在 FakeTensorMode（torchair GE converter 在 AOT 编译期处于
     * fake mode）下安全调用。
     */
    QuantMteContext BuildOrLookupHostContext(const std::string& ctxTag)
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        const CommEngine engine = CommEngine::COMM_ENGINE_AIV;

        void* ctx = nullptr;
        uint64_t ctxSize = 0;
        if (HcclEngineCtxGetFunc(comm_, ctxTag.c_str(), engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
            TORCH_CHECK(ctx != nullptr, "HcclEngineCtxGet returned null context, tag: ", ctxTag);
            TORCH_CHECK(ctxSize >= sizeof(QuantMteContext), "Invalid quant MTE context size, expected at least ",
                        sizeof(QuantMteContext), ", got ", ctxSize);
            return LookupHostContext(ctxTag);
        }

        TORCH_CHECK(builder_->Init(), "Init CommChannelBuilder failed, tag: ", ctxTag);

        uint64_t remoteAddrs[QUANT_MTE_MAX_RANK_SIZE] = {};
        const std::string memTag = ctxTag + "_winbuf";
        // quant MTE kernel 以裸 DataCopy 读写各 rank 内置 buffer（不含 URMA channel 句柄），
        // 远端地址必须为本侧可直接访问的映射地址，因此必须走 UB_MEM 协议（与
        // Mc2Context/bandwidth_test 及旧 CommContextManager channel 路径一致）。
        // URMA(UBC_CTP) 的 HcclChannelGetHcclBuffer 返回远端原始 VA，仅可经 WriteNbi
        // 等 channel API 访问，裸 DataCopy 会触发 device 侧 MTE trap。
        TORCH_CHECK(builder_->AllocRegAndBuildChannels(ChannelMode::UBMEM, memTag.c_str(), nullptr, remoteAddrs),
                    "Build quant MTE HCCL buffer address table failed, tag: ", ctxTag);

        auto hcclRet = HcclEngineCtxCreateFunc(comm_, ctxTag.c_str(), engine, sizeof(QuantMteContext), &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclEngineCtxCreate failed, tag: ", ctxTag, ", ret: ", hcclRet);
        TORCH_CHECK(ctx != nullptr, "HcclEngineCtxCreate returned null context, tag: ", ctxTag);

        const uint32_t rankId = builder_->GetRankId();
        const uint32_t rankSize = builder_->GetRankSize();
        TORCH_CHECK(rankSize > 0 && rankSize <= QUANT_MTE_MAX_RANK_SIZE, "Invalid quant MTE rank size: ", rankSize);
        TORCH_CHECK(rankId < rankSize, "Invalid quant MTE rank id ", rankId, " for rank size ", rankSize);

        uint32_t* layers = nullptr;
        uint32_t layerNum = 0;
        hcclRet = HcclRankGraphGetLayersFunc(comm_, &layers, &layerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && layers != nullptr && layerNum > 0,
                    "HcclRankGraphGetLayers failed, ret: ", hcclRet);

        uint32_t rankSizePerServer = 0;
        hcclRet = HcclRankGraphGetRankSizeByLayerFunc(comm_, layers[0], &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && rankSizePerServer > 0 && rankSizePerServer <= rankSize,
                    "HcclRankGraphGetRankSizeByLayer failed, ret: ", hcclRet);

        QuantMteContext hostContext = {};
        hostContext.rankId = rankId;
        hostContext.rankSizePerServer = rankSizePerServer;
        for (uint32_t i = 0; i < rankSize; ++i) {
            hostContext.hcclBuffer_[i] = remoteAddrs[i];
        }

        hcclRet = HcclEngineCtxCopyFunc(comm_, engine, ctxTag.c_str(), &hostContext, sizeof(hostContext), 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclEngineCtxCopy failed, tag: ", ctxTag, ", ret: ", hcclRet);

        CacheHostContext(ctxTag, hostContext);
        return hostContext;
    }

    /*!
     * \brief 创建 quant MTE 算子族公共 context 并以 NPU tensor 返回（eager 路径）
     */
    std::tuple<at::Tensor, int64_t> CreateQuantMteContext(const std::string& ctxTag)
    {
        QuantMteContext hostContext = BuildOrLookupHostContext(ctxTag);
        return {WrapQuantMteContext(hostContext), static_cast<int64_t>(GetHcclBufferSize())};
    }

    /*!
     * \brief 以 host int32 列表返回 context 内容（FakeTensorMode 安全，供 GE converter 使用）
     *
     * torchair 的 fx2ge converter 在 AOT 编译期处于 FakeTensorMode，任何经过 PyTorch
     * dispatcher 的 tensor 创建/拷贝（如 at::empty / copy_ / Tensor.cpu）都会触发
     * fake mode 断言；converter 只需要 context 的 int32 内容构造 ge.Const，因此
     * 走本接口直接取 host 内容，不产生任何 tensor op。
     */
    std::tuple<std::vector<int32_t>, int64_t> GetQuantMteContextData(const std::string& ctxTag)
    {
        QuantMteContext hostContext = BuildOrLookupHostContext(ctxTag);
        const auto* begin = reinterpret_cast<const int32_t*>(&hostContext);
        std::vector<int32_t> data(begin, begin + sizeof(QuantMteContext) / sizeof(int32_t));
        return {std::move(data), static_cast<int64_t>(GetHcclBufferSize())};
    }

    /*!
     * \brief 获取 HCCL 内置 buffer 大小
     */
    uint64_t GetHcclBufferSize()
    {
        void* buf = nullptr;
        uint64_t bufSize = 0;
        auto hcclRet = HcclGetHcclBufferFunc(comm_, &buf, &bufSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && buf != nullptr, "HcclGetHcclBuffer failed, ret: ", hcclRet);
        return bufSize;
    }

private:
    /*!
     * \brief 将 host 侧 context 内容包装为 NPU tensor 返回
     *
     * 必须用 at::empty(PrivateUse1) 走 torch_npu 分配器创建真正的 NPU tensor（NPUTensorImpl，
     * 携带有效 device index）。at::from_blob(PrivateUse1) 得到的是普通 TensorImpl，aclnn
     * 转换链（IsOpInputBaseFormat -> NPUBridge::GetNpuTensorImpl）对其 static_cast 后读取
     * npu 专属字段属于未定义行为，会引发 host 侧段错误。
     * 与 CommContextManager::CreateContext + CopyContextToTensor 的既有可用实现保持一致。
     */
    static at::Tensor WrapQuantMteContext(const QuantMteContext& hostContext)
    {
        constexpr int64_t contextElemNum = static_cast<int64_t>(sizeof(QuantMteContext) / sizeof(int32_t));
        at::Tensor context = at::empty({contextElemNum}, at::TensorOptions()
                                                             .dtype(at::kInt)
                                                             .device(c10::DeviceType::PrivateUse1)
                                                             .memory_format(c10::MemoryFormat::Contiguous));
        at::Tensor hostTensor = at::from_blob(const_cast<QuantMteContext*>(&hostContext), {contextElemNum}, at::kInt);
        context.copy_(hostTensor);
        return context;
    }

    /*!
     * \brief 进程内缓存 host 侧 context 内容（key: group + ctxTag）
     *
     * 内容（rankId/rankSizePerServer/各 rank 内置 buffer 地址）在 (group, ctxTag)
     * 确定后不变，缓存安全；engine ctx 生命周期与进程一致，HcclEngineCtxGet 命中时
     * 本进程必然已走过创建路径并写入缓存，因此命中路径无需回读 device 内存。
     */
    static std::map<std::string, QuantMteContext>& HostContextCache()
    {
        static std::map<std::string, QuantMteContext> cache;
        return cache;
    }

    static std::mutex& HostContextCacheMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    QuantMteContext LookupHostContext(const std::string& ctxTag) const
    {
        std::lock_guard<std::mutex> guard(HostContextCacheMutex());
        auto iter = HostContextCache().find(MakeHostContextKey(ctxTag));
        TORCH_CHECK(iter != HostContextCache().end(), "Quant MTE host context cache miss, tag: ", ctxTag);
        return iter->second;
    }

    void CacheHostContext(const std::string& ctxTag, const QuantMteContext& hostContext) const
    {
        std::lock_guard<std::mutex> guard(HostContextCacheMutex());
        HostContextCache()[MakeHostContextKey(ctxTag)] = hostContext;
    }

    std::string MakeHostContextKey(const std::string& ctxTag) const
    {
        return group_ + "|" + ctxTag;
    }

    HcclComm comm_ = nullptr;
    std::string group_;
    std::unique_ptr<DefaultCommChannelBuilder> builder_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommChannelBuilderManager>(m, "CommChannelBuilderManager")
        .def(py::init<const std::string&>(), py::arg("group"))
        .def("get_rank_id", &CommChannelBuilderManager::GetRankId)
        .def("get_rank_size", &CommChannelBuilderManager::GetRankSize)
        .def("create_context", &CommChannelBuilderManager::CreateContext, py::arg("ctx_tag"))
        .def("create_quant_mte_context", &CommChannelBuilderManager::CreateQuantMteContext, py::arg("ctx_tag"))
        .def("get_quant_mte_context_data", &CommChannelBuilderManager::GetQuantMteContextData, py::arg("ctx_tag"))
        .def("get_hccl_buffer_size", &CommChannelBuilderManager::GetHcclBufferSize);
}

} // namespace op_api
