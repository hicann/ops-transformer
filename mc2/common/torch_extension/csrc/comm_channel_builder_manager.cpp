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
#include <string>
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

using DefaultCommChannelBuilder = CommChannelBuilder<ChannelMode::URMA, ChannelMode::UBMEM>;

class CommChannelBuilderManager {
public:
    explicit CommChannelBuilderManager(const std::string &group)
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
    at::Tensor CreateContext(const std::string &ctxTag)
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        CommContext hostCtx = {};
        void *devCtx = builder_->CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag.c_str(), &hostCtx.udmaCtx,
                                                     &hostCtx.ubmemCtx);

        TORCH_CHECK(devCtx != nullptr, "CreateDeviceContext failed, ctxTag: ", ctxTag);

        at::Tensor context = at::from_blob(devCtx, {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
        return context;
    }

    /*!
     * \brief 获取 HCCL 内置 buffer 大小
     */
    uint64_t GetHcclBufferSize()
    {
        void *buf = nullptr;
        uint64_t bufSize = 0;
        auto hcclRet = HcclGetHcclBufferFunc(comm_, &buf, &bufSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && buf != nullptr, "HcclGetHcclBuffer failed, ret: ", hcclRet);
        return bufSize;
    }

private:
    HcclComm comm_ = nullptr;
    std::string group_;
    std::unique_ptr<DefaultCommChannelBuilder> builder_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommChannelBuilderManager>(m, "CommChannelBuilderManager")
        .def(py::init<const std::string &>(), py::arg("group"))
        .def("get_rank_id", &CommChannelBuilderManager::GetRankId)
        .def("get_rank_size", &CommChannelBuilderManager::GetRankSize)
        .def("create_context", &CommChannelBuilderManager::CreateContext, py::arg("ctx_tag"))
        .def("get_hccl_buffer_size", &CommChannelBuilderManager::GetHcclBufferSize);
}

} // namespace op_api
