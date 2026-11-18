# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Portable AICPU-body checks and fail-closed installed-resource contracts.

The portable body is a unit-test oracle, never a backend of the public operator.
These tests do not claim to execute the AICPU device binary on an NPU.
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from cannbotdsl.aicpu import I32, X86Buffer
from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel


@pytest.fixture(scope="module")
def producer():
    path = Path(__file__).resolve().parents[2] / "op_dsl/flash_gdn_metadata.py"
    spec = importlib.util.spec_from_file_location("_gdn_aicpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def binary(producer, tmp_path_factory):
    directory = os.environ.get("GDN_AICPU_TEST_BINARY")
    if directory:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["source_sha256"] == producer._source_digest()
        return CompiledAicpuKernel(
            str(root / manifest["library"]),
            producer.flash_gdn_metadata_kernel.spec,
            name=manifest["kernel_name"],
            launch_mode="interface",
            x86_so_path=str(root / "flash_gdn_metadata_kernel_x86.so"),
        )
    return producer.build_aicpu(
        tmp_path_factory.mktemp("metadata-binary"), build_x86=True
    )


def portable(binary, producer, *, batch=2, length=9, layout="TND", cu=None, used=None):
    out = X86Buffer([-99] * (producer.HEADER_SIZE + batch), I32)
    cu_buffer = X86Buffer(cu if cu is not None else [0], I32)
    used_buffer = X86Buffer(used if used is not None else [0], I32)
    result = binary.run_x86(
        cu_seqlens=SimpleNamespace(x86_buffer=cu_buffer),
        seqused=SimpleNamespace(x86_buffer=used_buffer),
        metadata=SimpleNamespace(x86_buffer=out),
        capacity=len(out.data),
        batch=batch,
        q_heads=1,
        v_heads=2,
        seqlens=length,
        layout_id=producer.LAYOUT_IDS[layout],
        has_cu=int(cu is not None),
        has_used=int(used is not None),
    )
    assert result == 0
    return list(out.data)


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "layout,cu,used,lengths",
    [
        ("TND", [0, 4, 9], None, [4, 5]),
        ("TND", None, [4, 5], [4, 5]),
        ("TND", [0, 4, 9], [4, 5], [4, 5]),
        ("BSND", None, None, [9, 9]),
        ("BNSD", None, [3, 7], [3, 7]),
        ("BSND", [0, 3, 10], [3, 7], [3, 7]),
    ],
)
def test_compiled_aicpu_body_matches_host_reference(
    producer, binary, layout, cu, used, lengths
):
    words = portable(binary, producer, layout=layout, cu=cu, used=used)
    header = producer.make_header(2, 1, 2, 9, 128, 128, layout)
    assert words == header + lengths


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "cu,used",
    [
        ([1, 4, 9], None),
        ([0, 5, 4], None),
        ([0, 4, 8], None),
        ([0, 4, 9], [3, 6]),
        (None, [0, 9]),
        (None, None),
    ],
)
def test_aicpu_body_reports_invalid_lengths(producer, binary, cu, used):
    assert portable(binary, producer, cu=cu, used=used)[producer.STATUS_INDEX] != 0


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([0, 4, 9], dtype=torch.float32),
        torch.tensor([[0, 4, 9]], dtype=torch.int32),
        torch.empty(6, dtype=torch.int32)[::2],
    ],
)
def test_meta_validates_length_tensor_specs(producer, bad):
    with pytest.raises(ValueError, match="cu_seqlens"):
        producer.flash_gdn_metadata(
            bad, batch_size=2, nheads_qk=1, nheads_v=2, seqlens=9, _meta_only=True
        )


def test_shape_only_metadata_is_not_a_cpu_backend(producer):
    cu = torch.empty(3, dtype=torch.int32, device="meta")
    result = producer.flash_gdn_metadata(
        cu, batch_size=2, nheads_qk=1, nheads_v=2, seqlens=9
    )
    assert result.device.type == "meta" and result.shape == (14,)
    with torch.device("meta"):
        result = producer.flash_gdn_metadata(
            batch_size=1, nheads_qk=1, nheads_v=1, seqlens=3, _meta_only=True
        )
    assert result.device.type == "meta"


def test_missing_runtime_does_not_use_cpu_fallback(producer, monkeypatch):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False
    )
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        producer.flash_gdn_metadata(batch_size=1, nheads_qk=1, nheads_v=1, seqlens=3)


def test_require_mode_never_compiles(producer, monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(tmp_path))
    monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path))

    def forbidden(*args, **kwargs):
        pytest.fail("require mode attempted JIT")

    monkeypatch.setattr(producer, "compile_aicpu_kernel", forbidden)
    with pytest.raises(RuntimeError, match="JIT is forbidden"):
        producer._get_aicpu_kernel()


def test_report_bounds_memory_and_file(producer, monkeypatch, tmp_path):
    path = tmp_path / "events.json"
    monkeypatch.setenv("GDN_METADATA_BINARY_REPORT", str(path))
    monkeypatch.setattr(producer, "_EVENTS", [])
    for sequence in range(256):
        producer._report("unit", "kernel.so", sequence=sequence)
    saved = json.loads(path.read_text())["events"]
    assert len(producer._EVENTS) == len(saved) == 64
    assert [event["sequence"] for event in saved] == list(range(192, 256))
    assert saved == producer._EVENTS
    assert not list(tmp_path.glob(".gdn-metadata-report-*"))


def test_report_disabled_does_not_append(producer, monkeypatch, tmp_path):
    monkeypatch.delenv("GDN_METADATA_BINARY_REPORT", raising=False)
    monkeypatch.setattr(producer, "_EVENTS", [])
    producer._report("unit", tmp_path / "not-created.so")
    assert producer._EVENTS == []
    assert not list(tmp_path.iterdir())


def test_report_is_bounded_with_concurrent_writers(producer, monkeypatch, tmp_path):
    path = tmp_path / "events.json"
    monkeypatch.setenv("GDN_METADATA_BINARY_REPORT", str(path))
    monkeypatch.setattr(producer, "_EVENTS", [])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda sequence: producer._report(
                    "unit", "kernel.so", sequence=sequence
                ),
                range(128),
            )
        )
    saved = json.loads(path.read_text())["events"]
    assert len(producer._EVENTS) == len(saved) == 64
    assert saved == producer._EVENTS
    assert len({event["sequence"] for event in saved}) == 64


@pytest.mark.ascendc_toolchain
def test_installed_library_is_loaded_without_compilation(
    producer, binary, monkeypatch, tmp_path
):
    resource = tmp_path / "vendor" / producer._RESOURCE
    resource.mkdir(parents=True)
    built = Path(binary.so_path).parent
    for name in (Path(binary.so_path).name, "manifest.json"):
        shutil.copy2(built / name, resource / name)
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(tmp_path / "vendor"))
    monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("GDN_METADATA_BINARY_REPORT", str(tmp_path / "hit.json"))

    def forbidden(*args, **kwargs):
        pytest.fail("installed metadata attempted JIT")

    monkeypatch.setattr(producer, "compile_aicpu_kernel", forbidden)
    loaded = producer._get_aicpu_kernel()
    assert Path(loaded.so_path) == resource / Path(binary.so_path).name
    assert (
        json.loads((tmp_path / "hit.json").read_text())["events"][-1]["event"]
        == "installed_aicpu_hit"
    )
    manifest = json.loads((resource / "manifest.json").read_text())
    manifest["library_sha256"] = "0" * 64
    (resource / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        producer._get_aicpu_kernel()
