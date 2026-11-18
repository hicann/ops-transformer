# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fresh-process proof of installed custom binaries for all three Torch entries.

This is an execution check, not a benchmark. Missing runtime dependencies remain
errors; neither metadata CPU execution nor operator JIT is a permitted fallback.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
import traceback


def verify(args):
    report = Path(args.report_dir).absolute()
    report.mkdir(parents=True, exist_ok=True)
    vendor = Path(args.vendor_root).absolute()
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    os.environ["CANNBOTDSL_NATIVE_BINARY_MODE"] = "require"
    os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"] = str(report / "native-hits.json")
    os.environ["GDN_METADATA_BINARY_REPORT"] = str(report / "metadata-hits.json")
    os.environ["CANNBOTDSL_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="empty-operator-cache-", dir=report
    )

    import torch
    import torch_npu  # noqa: F401
    from cann_ops_transformer import flash_gdn, flash_gdn_fwd, flash_gdn_metadata
    from cann_ops_transformer.ops.attention.flash_gdn.op_dsl import (
        flash_gdn_metadata as metadata_impl,
    )

    def forbidden(*unused_args, **unused_kwargs):
        raise AssertionError(
            "operator compilation is forbidden during installed-binary proof"
        )

    metadata_impl.compile_aicpu_kernel = forbidden
    torch.npu.set_device(args.device)
    q = torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16, device=args.device)
    q[..., 0] = 1
    v = torch.zeros_like(q)
    v[0, 0, :, 0] = torch.tensor([2.0, 4.0], device=args.device)
    g = torch.full((1, 1, 2), -0.6931471805599453, device=args.device)
    beta = torch.full_like(g, 0.5)
    with torch.inference_mode():
        md = flash_gdn_metadata(
            batch_size=1, nheads_qk=1, nheads_v=1, seqlens=2, layout="BNSD"
        )
        assert md.device == q.device and md.dtype == torch.int32
        assert md.detach().cpu().tolist()[1] == 3
        route = flash_gdn(
            q, q, v, g, beta, md, layout="BNSD", scale=1.0, output_final_state=True
        )
        fwd = flash_gdn_fwd(
            q, q, v, g, beta, md, layout="BNSD", scale=1.0, output_final_state=True
        )
    torch.npu.synchronize()
    assert len(route) == 3 and len(fwd) == 16
    expected = torch.zeros_like(v, device="cpu").float()
    expected[0, 0, :, 0] = torch.tensor([1.0, 2.25])
    expected_state = torch.zeros(1, 1, 128, 128)
    expected_state[0, 0, 0, 0] = 2.25
    for result in (route, fwd):
        torch.testing.assert_close(
            result[0].cpu().float(), expected, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            result[1].cpu(), expected_state, atol=1e-5, rtol=1e-5
        )
        assert all(value is None for value in result[2:])
    native = json.loads((report / "native-hits.json").read_text())
    metadata = json.loads((report / "metadata-hits.json").read_text())
    assert native["counts"].get("installed_hit", 0) >= 1
    assert not native["counts"].get("installed_miss", 0)
    assert not native["counts"].get("backend_pipeline", 0), (
        "A compiler backend ran during the proof"
    )
    hits = [
        event for event in metadata["events"] if event["event"] == "installed_aicpu_hit"
    ]
    launches = [
        event
        for event in metadata["events"]
        if event["event"] == "aicpu_launch_validated"
    ]
    assert hits and launches
    assert not any(event["event"].startswith("jit") for event in metadata["events"])
    assert all(Path(event["library"]).is_relative_to(vendor) for event in hits)
    maps = Path("/proc/self/maps").read_text().splitlines()
    toolkit = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    runtime_names = ("libhccl.so", "libhcomm.so", "libruntime.so", "libascendcl.so")
    runtime = sorted(
        {
            line.split()[-1]
            for line in maps
            if any(name in line.split()[-1] for name in runtime_names)
        }
    )
    assert any("libhccl.so" in path for path in runtime), "HCCL runtime was not loaded"
    assert all(Path(path).resolve().is_relative_to(toolkit) for path in runtime), (
        runtime
    )
    mapped = [line for line in maps if str(vendor) in line]
    assert mapped, "No library from the installed custom vendor was mapped"
    assert any("libcust_opmaster_rt2.0.so" in line for line in mapped), (
        "Custom forward Host provider was not mapped"
    )
    assert any("flash_gdn_metadata_kernel.so" in line for line in mapped), (
        "Custom AICPU shim was not mapped"
    )
    (report / "mapped-custom-libraries.txt").write_text("\n".join(mapped) + "\n")
    return dict(
        status="PASS",
        device=str(q.device),
        vendor_root=str(vendor),
        native_mode="require",
        empty_cache=True,
        jit_forbidden=True,
        entries=["flash_gdn_metadata", "flash_gdn", "flash_gdn_fwd"],
        output=[1.0, 2.25],
        final_state=2.25,
        native_installed_hits=native["counts"]["installed_hit"],
        aicpu_installed_hits=len(hits),
        aicpu_validated_launches=len(launches),
        toolkit_root=str(toolkit),
        runtime_libraries=runtime,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    try:
        result = verify(args)
    except Exception as error:
        result = dict(
            status="FAIL_OR_ENVIRONMENT_BLOCKED",
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
    root = Path(args.report_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
