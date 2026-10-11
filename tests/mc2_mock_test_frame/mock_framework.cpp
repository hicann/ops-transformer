/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Mock HCCL Server — PyTorch C++ Extension (pybind11)
 *
 * Exposes MockContextBuilder and MockHcclServer to Python.
 * The MockServer accepts per-rank input tensors to simulate
 * real collective communication semantics.
 *
 * Usage:
 *   import mock_hccl_ext
 *   ctx = mock_hccl_ext.MockContext(rank_num=2, rank_id=0, device_id=0)
 *   ctx.build()
 *
 *   # rank_inputs: list of npu tensors, one per rank (including local rank)
 *   server = mock_hccl_ext.MockServer(
 *       workspace_ptr=ctx.workspace_ptr(),
 *       rank_inputs=[tensor_rank0, tensor_rank1],
 *       local_rank_id=0,
 *       device_id=0)
 *   server.start()
 *   ...
 *   server.stop()
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mock_framework.h"

namespace py = pybind11;
using namespace mock_hccl;

// ============================================================
// MockContext — Python-facing context builder
// ============================================================
class MockContext {
public:
    MockContext(uint32_t rankNum, uint32_t rankId, int deviceId)
        : rankNum_(rankNum),
          rankId_(rankId),
          deviceId_(deviceId)
    {}

    ~MockContext()
    {
        destroy();
    }

    bool build()
    {
        devWindowMem_ = DevMalloc(WINDOW_TOTAL_SIZE);
        if (!devWindowMem_)
            return false;

        size_t wsAllocSize = MIN_WORKSPACE_SIZE + 512;
        devWorkspaceRaw_ = DevMalloc(wsAllocSize);
        if (!devWorkspaceRaw_)
            return false;
        devWorkspaceAligned_ = AlignUp512(devWorkspaceRaw_);

        MockHcclContext hostCtx;
        memset(&hostCtx, 0, sizeof(hostCtx));
        hostCtx.rankId = rankId_;
        hostCtx.rankNum = rankNum_;
        hostCtx.winSize = WINDOW_TOTAL_SIZE;
        hostCtx.workSpace = reinterpret_cast<uint64_t>(devWorkspaceAligned_);
        hostCtx.workSpaceSize = MIN_WORKSPACE_SIZE;
        for (uint32_t i = 0; i < rankNum_; i++) {
            hostCtx.windowsIn[i] = reinterpret_cast<uint64_t>(devWindowMem_);
            hostCtx.windowsOut[i] = reinterpret_cast<uint64_t>(devWindowMem_);
        }

        devContext_ = DevMalloc(sizeof(MockHcclContext));
        if (!devContext_)
            return false;
        aclrtMemcpy(devContext_, sizeof(hostCtx), &hostCtx, sizeof(hostCtx), ACL_MEMCPY_HOST_TO_DEVICE);
        return true;
    }

    void destroy()
    {
        DevFree(devContext_);
        devContext_ = nullptr;
        DevFree(devWorkspaceRaw_);
        devWorkspaceRaw_ = nullptr;
        DevFree(devWindowMem_);
        devWindowMem_ = nullptr;
        devWorkspaceAligned_ = nullptr;
    }

    void clear_flags()
    {
        if (!devWindowMem_)
            return;
        void* flagArea = OffsetPtr(devWindowMem_, FLAG_OFFSET_BYTES);
        aclrtMemset(flagArea, FLAG_AREA_SIZE, 0, FLAG_AREA_SIZE);
    }

    at::Tensor as_tensor()
    {
        if (!devContext_)
            throw std::runtime_error("Context not built");
        auto opts = at::TensorOptions().dtype(at::kChar).device(at::kPrivateUse1, deviceId_);
        size_t nbytes = sizeof(MockHcclContext);
        return at::from_blob(devContext_, {static_cast<int64_t>(nbytes)}, opts);
    }

    int64_t context_ptr() const
    {
        return reinterpret_cast<int64_t>(devContext_);
    }
    int64_t workspace_ptr() const
    {
        return reinterpret_cast<int64_t>(devWorkspaceAligned_);
    }
    int64_t window_ptr() const
    {
        return reinterpret_cast<int64_t>(devWindowMem_);
    }
    uint32_t rank_num() const
    {
        return rankNum_;
    }

private:
    uint32_t rankNum_, rankId_;
    int deviceId_;
    void* devContext_{nullptr};
    void* devWorkspaceRaw_{nullptr};
    void* devWorkspaceAligned_{nullptr};
    void* devWindowMem_{nullptr};
};

// ============================================================
// PyMockServer — Python-facing HCCL server wrapper
//
// Accepts a list of torch.Tensor (one per rank) and delegates
// to MockHcclServer for real collective simulation.
// ============================================================
class PyMockServer {
public:
    /**
     * @param workspacePtr  int64 device address of 512B-aligned workspace
     * @param rankInputs    list of npu tensors, one per rank
     * @param localRankId   rank ID of the kernel under test
     * @param deviceId      NPU device ID
     */
    PyMockServer(int64_t workspacePtr, std::vector<at::Tensor> rankInputs, uint32_t localRankId, int deviceId)
        : deviceId_(deviceId)
    {
        // Hold references to tensors so they stay alive
        tensorRefs_ = std::move(rankInputs);

        // Extract device pointers → RankData
        std::vector<RankData> rankData;
        rankData.reserve(tensorRefs_.size());
        for (auto& t : tensorRefs_) {
            rankData.push_back({t.data_ptr(), (size_t)(t.nbytes())});
        }

        server_ = std::make_unique<MockHcclServer>(reinterpret_cast<void*>(workspacePtr), std::move(rankData),
                                                   localRankId, deviceId);
    }

    ~PyMockServer()
    {
        stop();
    }

    void start()
    {
        server_->Start();
    }
    void stop()
    {
        server_->Stop();
    }

    uint32_t msg_count() const
    {
        return server_->GetMsgCount();
    }
    bool is_finalized() const
    {
        return server_->IsFinalized();
    }

    bool wait_for_finalize(uint32_t slot, uint32_t timeoutMs)
    {
        return server_->WaitForFinalize(slot, timeoutMs);
    }

private:
    int deviceId_;
    std::vector<at::Tensor> tensorRefs_; // prevent GC
    std::unique_ptr<MockHcclServer> server_;
};

// ============================================================
// PyMultiRankContext — Python-facing multi-rank context builder
//
// Allocates N independent windows + N contexts with cross-referenced
// windowsIn[]. For use with multi-stream concurrent kernel launches.
// ============================================================
class PyMultiRankContext {
public:
    PyMultiRankContext(uint32_t rankNum, int deviceId)
        : rankNum_(rankNum),
          deviceId_(deviceId)
    {}

    ~PyMultiRankContext()
    {
        destroy();
    }

    bool build()
    {
        return ctx_.Build(rankNum_);
    }
    void destroy()
    {
        ctx_.Destroy();
    }
    void clear_flags()
    {
        ctx_.ClearAllFlags();
    }

    at::Tensor context_tensor(uint32_t rank)
    {
        void* addr = ctx_.GetContextAddr(rank);
        if (!addr)
            throw std::runtime_error("Context not built for rank " + std::to_string(rank));
        auto opts = at::TensorOptions().dtype(at::kChar).device(at::kPrivateUse1, deviceId_);
        return at::from_blob(addr, {static_cast<int64_t>(sizeof(MockHcclContext))}, opts);
    }

    // D2H read flag values from rank's window.
    // Returns list of int32 flag values at FLAG_OFFSET + [0..num_flags-1]
    std::vector<int32_t> read_flags(uint32_t rank, uint32_t num_flags = 8)
    {
        std::vector<int32_t> result(num_flags, -1);
        void* win = ctx_.GetWindowMem(rank);
        if (!win)
            return result;
        // flags are at (int32_t*)win + FLAG_OFFSET, where FLAG_OFFSET = 180*1024*1024/4
        constexpr int64_t FLAG_OFFSET = 180LL * 1024 * 1024 / sizeof(int32_t);
        void* flagAddr = reinterpret_cast<void*>(reinterpret_cast<int32_t*>(win) + FLAG_OFFSET);
        size_t bytes = num_flags * sizeof(int32_t);
        aclrtMemcpy(result.data(), bytes, flagAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        return result;
    }

    // D2H read first N bytes from rank's window data area (for debugging)
    std::vector<uint8_t> read_window_data(uint32_t rank, size_t offset, size_t nbytes)
    {
        std::vector<uint8_t> result(nbytes, 0);
        void* win = ctx_.GetWindowMem(rank);
        if (!win)
            return result;
        void* src = OffsetPtr(win, offset);
        aclrtMemcpy(result.data(), nbytes, src, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);
        return result;
    }

    uint32_t rank_num() const
    {
        return rankNum_;
    }

private:
    uint32_t rankNum_;
    int deviceId_;
    MultiRankMockContext ctx_;
};

// ============================================================
// pybind11 module
// ============================================================
PYBIND11_MODULE(mock_hccl_ext, m)
{
    m.doc() = "Mock HCCL server extension for single-rank MC2 testing";

    py::class_<MockContext>(m, "MockContext")
        .def(py::init<uint32_t, uint32_t, int>(), py::arg("rank_num") = 1, py::arg("rank_id") = 0,
             py::arg("device_id") = 0)
        .def("build", &MockContext::build)
        .def("destroy", &MockContext::destroy)
        .def("as_tensor", &MockContext::as_tensor)
        .def("clear_flags", &MockContext::clear_flags)
        .def("context_ptr", &MockContext::context_ptr)
        .def("workspace_ptr", &MockContext::workspace_ptr)
        .def("window_ptr", &MockContext::window_ptr)
        .def("rank_num", &MockContext::rank_num);

    py::class_<PyMockServer>(m, "MockServer")
        .def(py::init<int64_t, std::vector<at::Tensor>, uint32_t, int>(), py::arg("workspace_ptr"),
             py::arg("rank_inputs"), py::arg("local_rank_id") = 0, py::arg("device_id") = 0)
        .def("start", &PyMockServer::start)
        .def("stop", &PyMockServer::stop)
        .def("msg_count", &PyMockServer::msg_count)
        .def("is_finalized", &PyMockServer::is_finalized)
        .def("wait_for_finalize", &PyMockServer::wait_for_finalize, py::arg("slot") = 0, py::arg("timeout_ms") = 5000);

    py::class_<PyMultiRankContext>(m, "MultiRankContext")
        .def(py::init<uint32_t, int>(), py::arg("rank_num"), py::arg("device_id") = 0)
        .def("build", &PyMultiRankContext::build)
        .def("destroy", &PyMultiRankContext::destroy)
        .def("clear_flags", &PyMultiRankContext::clear_flags)
        .def("context_tensor", &PyMultiRankContext::context_tensor, py::arg("rank"))
        .def("read_flags", &PyMultiRankContext::read_flags, py::arg("rank"), py::arg("num_flags") = 8)
        .def("read_window_data", &PyMultiRankContext::read_window_data, py::arg("rank"), py::arg("offset") = 0,
             py::arg("nbytes") = 64)
        .def("rank_num", &PyMultiRankContext::rank_num);
}
