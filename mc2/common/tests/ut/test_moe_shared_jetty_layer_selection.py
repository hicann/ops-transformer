# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise production layer selection with local rank graphs and threaded collectives.

Requires a C++17 compiler; Torch, CANN and NPU hardware are not required.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "torch_extension/csrc/moe_shared_jetty_context.cpp"
)


def definition(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if marker.startswith("struct "):
        end += 1
    return source[start:end]


PRELUDE = r"""
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using HcclResult = int;
using EndpointAttrBwCoeff = uint32_t;
using EndpointAttrDieId = uint32_t;
constexpr int HCCL_SUCCESS = 0;
constexpr uint32_t GLOBAL_LANE_ROOT_RANK = 0;
constexpr uint32_t WORLD_SIZE = 4;
template<class... Args> void fail(Args... args) {
    std::ostringstream message; (message << ... << args);
    throw std::runtime_error(message.str());
}
#define TORCH_CHECK(c, ...) do { if (!(c)) fail(__VA_ARGS__); } while (false)
#define ASCEND_LOGI(...) ((void)0)
#define ASCEND_LOGW(...) ((void)0)
enum class CommProtocol { COMM_PROTOCOL_UBC_CTP = 4, COMM_PROTOCOL_UB_RTP = 9 };
enum EndpointAttr { ENDPOINT_ATTR_BW_COEFF, ENDPOINT_ATTR_DIE_ID };
struct EndpointDesc {
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    struct { int type = 0; unsigned char raws[8]{}; } commAddr;
    struct { int locType = 0; unsigned char raws[8]{}; } loc;
};
struct CommLink {
    EndpointDesc srcEndpointDesc, dstEndpointDesc;
    struct { CommProtocol linkProtocol; } linkAttr;
};
struct HcclChannelDesc {
    CommProtocol channelProtocol;
    uint32_t remoteRank;
    EndpointDesc localEndpoint, remoteEndpoint;
};
HcclResult HcclChannelDescInit(HcclChannelDesc*, uint32_t) { return 0; }
thread_local uint32_t myRank;
std::string scenario;
EndpointDesc endpoint(uint32_t layer, uint32_t rank) {
    EndpointDesc e; e.commAddr.raws[0] = layer; e.commAddr.raws[1] = rank; return e;
}
HcclResult HcclRankGraphGetLayersFunc(int, uint32_t** layers, uint32_t* count) {
    thread_local std::vector<uint32_t> ids;
    ids = myRank % 2 == 0 ? std::vector<uint32_t>{0, 1} : std::vector<uint32_t>{1, 0};
    if (scenario == "different_layers" && myRank == 1) ids = {1};
    if (scenario == "empty_peer" && myRank == 1) ids.clear();
    if (scenario == "empty_root" && myRank == 0) ids.clear();
    *layers = ids.data(); *count = ids.size(); return 0;
}
HcclResult HcclRankGraphGetRanksByLayerFunc(int, uint32_t layer, uint32_t** ranks, uint32_t* count) {
    thread_local std::vector<uint32_t> ids;
    ids = {0, 1, 2, 3};
    if (scenario == "partial_coverage" && myRank == 1 && layer == 0) ids.pop_back();
    *ranks = ids.data(); *count = ids.size(); return 0;
}
HcclResult HcclRankGraphGetLinksFunc(int, uint32_t layer, uint32_t src, uint32_t dst,
                                   CommLink** links, uint32_t* count) {
    TORCH_CHECK(src == myRank, "queried a remote rank's outgoing links");
    thread_local std::vector<CommLink> values;
    values.clear();
    const bool missing = src == 1 && dst == 2 &&
        ((scenario == "missing_link" && layer == 0) || scenario == "all_missing");
    if (!missing) {
        auto remote = endpoint(layer, dst);
        if (scenario == "reciprocal_mismatch" && layer == 0 && src == 1 && dst == 2)
            remote.commAddr.raws[2] = 1;
        values.push_back({endpoint(layer, src), remote, {CommProtocol::COMM_PROTOCOL_UBC_CTP}});
    }
    if (scenario == "ubg" && layer == 1 && src == 1 && dst == 2)
        values.push_back({endpoint(layer, src), endpoint(layer, dst), {CommProtocol::COMM_PROTOCOL_UB_RTP}});
    *links = values.data(); *count = values.size(); return 0;
}
HcclResult HcclRankGraphGetEndpointInfoFunc(int, uint32_t rank, const EndpointDesc* e,
                                          EndpointAttr attr, uint32_t, void* value) {
    TORCH_CHECK(rank == myRank && e->commAddr.raws[1] == rank, "queried remote attributes");
    *static_cast<uint32_t*>(value) = attr == ENDPOINT_ATTR_BW_COEFF ? 1 : 0;
    return 0;
}
class Collective {
    std::mutex mutex;
    std::condition_variable cv;
    uint32_t arrived = 0, generation = 0;
    std::vector<std::vector<char>> input{WORLD_SIZE};
    std::vector<char> output;
public:
    template<class T> std::vector<T> gather(const std::vector<T>& data) {
        std::unique_lock<std::mutex> guard(mutex);
        TORCH_CHECK(!data.empty(), "empty collective");
        const auto previous = generation;
        const auto bytes = data.size() * sizeof(T);
        input[myRank].resize(bytes);
        std::memcpy(input[myRank].data(), data.data(), bytes);
        if (++arrived == WORLD_SIZE) {
            output.clear();
            for (const auto& part : input) {
                TORCH_CHECK(part.size() == bytes, "collective payload sizes differ");
                output.insert(output.end(), part.begin(), part.end());
            }
            arrived = 0; ++generation; cv.notify_all();
        } else {
            TORCH_CHECK(cv.wait_for(guard, std::chrono::seconds(2), [&] { return generation != previous; }),
                        "collective order mismatch or timeout");
        }
        std::vector<T> result(output.size() / sizeof(T));
        std::memcpy(result.data(), output.data(), output.size());
        return result;
    }
};
Collective collective;
class Harness {
public:
"""

MEMBERS = r"""
    int comm_ = 0;
    uint32_t rankSize_ = WORLD_SIZE;
    struct { uint32_t epRankId = myRank, sharedJettyNum = 32; } context_;
    std::vector<SharedLinkGroup> sharedLinkGroups_;
    template<class T> std::vector<T> AllGatherMetadata(const std::vector<T>& data) {
        return collective.gather(data);
    }
"""

EPILOGUE = r"""
};
int main(int argc, char** argv) {
    scenario = argv[1];
    std::vector<std::string> errors(WORLD_SIZE);
    std::vector<int> selected(WORLD_SIZE, -1);
    std::vector<std::thread> threads;
    for (uint32_t rank = 0; rank < WORLD_SIZE; ++rank) threads.emplace_back([&, rank] {
        myRank = rank;
        try {
            Harness host;
            host.SelectSharedLayer();
            TORCH_CHECK(host.sharedLinkGroups_.size() == 1, "unexpected lane count");
            selected[rank] = host.sharedLinkGroups_[0].channelDescs[0].localEndpoint.commAddr.raws[0];
        } catch (const std::exception& error) { errors[rank] = error.what(); }
    });
    for (auto& thread : threads) thread.join();
    const bool rejected = scenario == "all_missing" || scenario == "ubg" ||
                          scenario == "empty_root" || scenario == "empty_peer";
    const int expected = scenario == "different_order" ? 0 : 1;
    for (uint32_t rank = 0; rank < WORLD_SIZE; ++rank) {
        if (rejected) {
            TORCH_CHECK(selected[rank] == -1 && !errors[rank].empty(), "ranks did not reject together");
            TORCH_CHECK(errors[rank] == errors[0], "different errors across ranks");
            TORCH_CHECK(errors[rank].find("timeout") == std::string::npos, errors[rank]);
            if (scenario == "ubg")
                TORCH_CHECK(errors[rank].find("does not support UBG") != std::string::npos, errors[rank]);
        } else {
            TORCH_CHECK(errors[rank].empty(), "rank=", rank, " ", errors[rank]);
            TORCH_CHECK(selected[rank] == expected, "ranks selected different layers");
        }
    }
    std::cout << "PASS " << scenario << '\n';
}
"""


class MoeSharedJettyLayerSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = SOURCE.read_text(encoding="utf-8")
        structs = [
            definition(source, "struct " + name + " {")
            for name in (
                "SharedLinkGroup",
                "GlobalLane",
                "RootLaneMetadata",
                "SelectedEndpointMetadata",
                "LayerCandidateMetadata",
                "UnsupportedProtocolMetadata",
            )
        ]
        methods = []
        for marker in (
            "static void CheckHccl(",
            "static bool IsSameEndpoint(",
            "static bool IsEndpointLess(",
            "std::vector<CommLink> GetProtocolLinks(",
            "bool TryGetLocalEndpointInfo(",
            "SelectedEndpointMetadata SelectLocalEndpoint(",
            "SharedLinkGroup MakeSharedLinkGroup(",
            "bool LayerCoversAllRanks(",
            "bool SelectSharedLinks(",
            "void CheckSupportedProtocols(",
            "void SelectSharedLayer(",
        ):
            method = definition(source, marker)
            if marker.startswith("bool TryGet"):
                method = "template<typename T>\n" + method
            methods.append(method)
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            raise unittest.SkipTest(
                "C++17 compiler required for shared layer regression"
            )
        cls.directory = tempfile.TemporaryDirectory(prefix="shared_jetty_layers_")
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(cls.directory.name) / "selection.cpp"
        path.write_text(
            PRELUDE + "\n".join(structs) + MEMBERS + "\n".join(methods) + EPILOGUE,
            encoding="utf-8",
        )
        cls.executable = Path(cls.directory.name) / "selection.exe"
        result = subprocess.run(
            [compiler, "-std=c++17", "-pthread", str(path), "-o", str(cls.executable)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def test_collective_layer_selection(self):
        for scenario in (
            "different_order",
            "different_layers",
            "partial_coverage",
            "missing_link",
            "reciprocal_mismatch",
            "all_missing",
            "empty_root",
            "empty_peer",
            "ubg",
        ):
            with self.subTest(scenario=scenario):
                result = subprocess.run(
                    [str(self.executable), scenario],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
