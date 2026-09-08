# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source-level contracts for the AIV/AIC expert pipeline.

The device-only implementation is excluded from kernel-test host compilation,
so these tests protect the orchestration order directly. NPU precision and
profiler tests remain the behavioral acceptance tests.
"""

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "op_kernel"
    / "allto_allv_grouped_mat_mul_aiv_mode.h"
)
COMM_SOURCE = SOURCE.parent / "allto_allv_grouped_mat_mul_aiv_comm.h"
CATLASS_SOURCE = SOURCE.parent / "allto_allv_grouped_mat_mul_catlass.h"
HOST_TILING_HEADER = (
    SOURCE.parent.parent
    / "op_host"
    / "op_tiling"
    / "allto_allv_grouped_mat_mul_tiling.h"
)


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    body_start = source.index("{", start)
    depth = 0
    for index in range(body_start, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1 : index]
    raise AssertionError(f"unterminated function body: {signature}")


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_catlass_uses_external_headers() -> None:
    source = CATLASS_SOURCE.read_text(encoding="utf-8")
    includes = [
        line.split('"')[1]
        for line in source.splitlines()
        if line.startswith('#include "catlass/')
    ]
    assert len(includes) == 9
    assert "template_linear_algebra" not in source
    assert "catlass/gemm/kernel/basic_matmul.hpp" in includes


def test_a2_ep8_accepts_both_fixed_and_runtime_dynamic_peer_windows() -> None:
    source = COMM_SOURCE.read_text(encoding="utf-8")
    init = _function_body(source, "__aicore__ inline bool Init(")
    window = _function_body(source, "__aicore__ inline GM_ADDR Window(")

    assert "GetRemoteRankAddrs" in window
    assert "a2Context_->windowsIn[rank]" in window
    assert "RequiresA2DynamicWindowTable" not in source
    assert "a2Context_->multiFlag != 0U" in init
    assert "a2Context_->data == nullptr" in init
    assert "a2Context_->multiFlag == 0U" in window
    assert "a2Context_->data[rank].localInput.addr" in window
    assert "a2Context_->data[rank].remoteInput.addr" in window


def test_peer_window_allocation_excludes_workspace_expert_ready_bytes() -> None:
    comm_source = COMM_SOURCE.read_text(encoding="utf-8")
    mode_source = SOURCE.read_text(encoding="utf-8")
    runtime_layout = _function_body(
        comm_source, "A2AVGMM_HOST_DEVICE bool BuildRuntimeControlLayout("
    )
    host_layout = _function_body(
        comm_source, "A2AVGMM_HOST_DEVICE bool BuildWindowLayout("
    )
    device_layout = _function_body(
        mode_source, "__aicore__ inline bool BuildDeviceWindowLayout("
    )

    assert "layout.peerControlBytes" in runtime_layout
    assert "control.peerControlBytes" in host_layout
    assert "control.totalBytes" not in host_layout
    # Device and Host intentionally share the same layout helper.  Keeping
    # the peer-window boundary in BuildWindowLayout prevents the two sides
    # from drifting when the prefix table size changes.
    assert "BuildWindowLayout(" in device_layout
    assert "control.totalBytes" not in device_layout


def test_nonquant_tiling_key_precedes_quant_base_include() -> None:
    """The quant base pulls in a legacy 0/1-only tiling-key declaration."""

    source = HOST_TILING_HEADER.read_text(encoding="utf-8")
    key_include = source.index(
        '#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"'
    )
    base_include = source.index('#include "allto_allv_grouped_mat_mul_tiling_base.h"')
    assert key_include < base_include


def test_shared_mm_precedes_counted_cross_core_expert_waits() -> None:
    body = _function_body(_source(), "__aicore__ inline void ProcessAic(")

    shared_mm = body.index("RunSharedExpertGemm")
    expert_loop = body.index("for (uint32_t expertIdx")
    wait_ready = body.index("WaitAicExpertReady", expert_loop)
    routed_gemm = body.index("RunExpertGemm", expert_loop)

    assert shared_mm < expert_loop
    assert expert_loop < wait_ready < routed_gemm


def test_shared_mm_starts_after_local_payload_publish_but_before_peer_wait() -> None:
    source = _source()
    aiv_body = _function_body(source, "__aicore__ inline void ProcessAiv(")
    publish = aiv_body.index("control.publishSlotsOffset")
    shared_ready = aiv_body.index("SignalAicSharedExpertReady", publish)
    peer_wait = aiv_body.index("WaitPeerPhase", shared_ready)

    aic_body = _function_body(source, "__aicore__ inline void ProcessAic(")
    wait_local_payload = aic_body.index("WaitAicSharedExpertReady")
    shared_mm = aic_body.index("RunSharedExpertGemm")

    assert publish < shared_ready < peer_wait
    assert wait_local_payload < shared_mm


def test_aiv_signals_each_expert_after_the_collective_copy_barrier() -> None:
    body = _function_body(_source(), "__aicore__ inline void ProcessAiv(")
    expert_loop = body.index("for (uint32_t expertIdx")
    final_release = body.index("control.releaseSlotsOffset")
    copy_expert = body.index("CopyExpertFromSource", expert_loop)
    copy_barrier = body.index("SyncAll<true>", copy_expert)
    publish_ready = body.index("SignalAicExpertReady", copy_barrier)

    assert expert_loop < copy_expert < copy_barrier < publish_ready < final_release


def test_small_or_single_nonempty_workloads_use_one_shot_routed_ready() -> None:
    source = _source()
    assert "ShouldUseExpertOverlap" in source
    assert "tiling.expertOverlapMode" in source

    aic_body = _function_body(source, "__aicore__ inline void ProcessAic(")
    choose_aic = aic_body.index("ShouldUseExpertOverlap")
    fallback_wait = aic_body.index("WaitAicRoutedDataReady", choose_aic)
    expert_loop = aic_body.index("for (uint32_t expertIdx", fallback_wait)
    expert_wait = aic_body.index("WaitAicExpertReady", expert_loop)
    assert choose_aic < fallback_wait < expert_loop < expert_wait

    aiv_body = _function_body(source, "__aicore__ inline void ProcessAiv(")
    choose_aiv = aiv_body.index("ShouldUseExpertOverlap")
    expert_loop = aiv_body.index("for (uint32_t expertIdx", choose_aiv)
    per_expert_signal = aiv_body.index("SignalAicExpertReady", expert_loop)
    fallback_signal = aiv_body.index("SignalAicRoutedDataReady", per_expert_signal)
    assert choose_aiv < expert_loop < per_expert_signal < fallback_signal


def test_one_shot_fallback_uses_one_post_copy_barrier() -> None:
    body = _function_body(_source(), "__aicore__ inline void ProcessAiv(")
    choose = body.index("const bool expertOverlap")
    release = body.index("control.releaseSlotsOffset", choose)
    pipeline = body.index("if (expertOverlap) {", choose)
    fallback = body.index("} else {", pipeline)

    assert body.count("for (uint32_t expertIdx", choose, release) == 2
    assert body.count("SyncAll<true>", choose, release) == 2

    fallback_body = body[fallback:release]
    copy_loop = fallback_body.index("for (uint32_t expertIdx")
    copy = fallback_body.index("CopyExpertFromSource", copy_loop)
    barrier = fallback_body.index("SyncAll<true>", copy)
    ready = fallback_body.index("SignalAicRoutedDataReady", barrier)
    assert copy_loop < copy < barrier < ready


def test_expert_ready_uses_megamoe_counted_cross_core_protocol() -> None:
    source = _source()
    signal = _function_body(source, "__aicore__ inline void SignalAicExpertReady(")
    wait = _function_body(source, "__aicore__ inline void WaitAicExpertReady(")

    assert "CrossCoreSetFlag<0x2, PIPE_MTE3>" in signal
    assert "CrossCoreWaitFlag<0x2>" in wait
    assert "ExpertReadyFlagId" in signal
    assert "ExpertReadyFlagId" in wait
    aiv = _function_body(source, "__aicore__ inline void ProcessAiv(")
    aic = _function_body(source, "__aicore__ inline void ProcessAic(")
    for body, action in [(aiv, "SignalAicExpertReady"), (aic, "RunExpertGemm")]:
        drain = body.index("NeedsExpertEventDrain(readySequence)")
        assert body.index(action) < drain < body.index("SyncAll<false>", drain)


def test_invocation_epoch_stays_in_the_hccl_peer_window() -> None:
    source = _source()
    load_epoch = _function_body(
        source, "__aicore__ inline int32_t LoadInvocationEpoch("
    )

    assert "context.Window(context.RankId())" in load_epoch
    assert "workspace" not in load_epoch
    assert "PublishInvocationEpoch" not in source


def test_prefix_lookup_uses_only_adjacent_cumsum_entries() -> None:
    mode_source = _source()
    peer_lookup = _function_body(
        mode_source, "__aicore__ inline bool GetPeerSourceTokenOffset("
    )
    comm_source = COMM_SOURCE.read_text(encoding="utf-8")
    prefix_range = _function_body(comm_source, "A2AVGMM_HOST_DEVICE bool PrefixRange(")

    assert "for (" not in peer_lookup
    assert "peerPrefix.GetValue(target)" in peer_lookup
    assert "peerPrefix.GetValue(target - 1U)" in peer_lookup
    assert "for (" not in prefix_range
    assert "inclusivePrefix[index]" in prefix_range
    assert "inclusivePrefix[index - 1U]" in prefix_range


def test_sparse_experts_skip_before_producer_and_consumer_notification():
    source = _source()
    for name, action in [
        ("ProcessAiv", "SignalAicExpertReady"),
        ("ProcessAic", "WaitAicExpertReady"),
    ]:
        body = _function_body(source, "__aicore__ inline void " + name + "(")
        skip = body.index("expert.tokenCount == 0U")
        assert skip < body.index("continue;", skip) < body.index(action)
        assert action + "(readySequence)" in body
        assert "NeedsExpertEventDrain(readySequence)" in body
        assert body.index(action) < body.index("++readySequence")


def test_oneshot_path_does_not_scan_expert_metadata_on_all_cores():
    body = _function_body(_source(), "__aicore__ inline void ProcessAiv(")
    one_shot = body[
        body.index("    } else {", body.index("const bool expertOverlap")) :
    ]
    one_shot = one_shot[: one_shot.index("SignalAicRoutedDataReady();")]
    assert "BuildExpertMetaForIndex" not in one_shot
    assert "expert.tokenCount" not in one_shot
    assert "CopyExpertFromSource<T>" in one_shot
    assert "subBlockIdx == 0U && blockIdx < workerNum" in one_shot
    assert one_shot.count("AscendC::SyncAll<true>();") == 1
