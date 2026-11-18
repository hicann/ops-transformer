# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GDN ABI3: AICPU-generated device metadata and checked installed-binary loading.

Host helpers validate scalar/tensor contracts and the produced result. They are
not a CPU implementation of the public metadata operator. Real output is written
by flash_gdn_metadata_kernel on AICPU; Meta execution returns shape only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

import torch
from cannbotdsl.aicpu import I32, aicpu_kernel, current_raw_stream, gm_in, gm_out
from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_aicpu_kernel

MAGIC = 0x47444E31
ABI_VERSION = 3
# Fixed ABI3 header size in int32 words, not bytes. Per-sequence lengths follow
# at metadata[HEADER_SIZE + b]; both the AICPU producer and forward reader use it.
HEADER_SIZE = 12
STATUS_INDEX = 11
LAYOUT_IDS = {"TND": 0, "BSND": 1, "BNSD": 2}
CHUNK_SIZE = 64
HEAD_DIM = 128
_LOCK = threading.RLock()
_HANDLES = {}
_EVENTS = []
_RESOURCE = Path("op_impl/aicpu/flash_gdn")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_header(
    batch_size, nheads_qk, nheads_v, seqlens, headdim_qk, headdim_v, layout
):
    require(layout in LAYOUT_IDS, "layout must be TND, BSND, or BNSD")
    for name, value in (
        ("batch_size", batch_size),
        ("nheads_qk", nheads_qk),
        ("nheads_v", nheads_v),
        ("seqlens", seqlens),
    ):
        require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 < value < 2**31,
            f"{name} must be a positive int32-sized integer",
        )
    require(nheads_v % nheads_qk == 0, "nheads_v must be a multiple of nheads_qk")
    require(batch_size + HEADER_SIZE < 2**31, "metadata size exceeds int32")
    require(
        layout == "TND" or batch_size * seqlens < 2**31,
        "storage token offsets exceed int32",
    )
    if headdim_qk != HEAD_DIM or headdim_v != HEAD_DIM:
        raise NotImplementedError("FlashGDN supports Dk=Dv=128 currently")
    return [
        MAGIC,
        ABI_VERSION,
        batch_size,
        nheads_qk,
        nheads_v,
        seqlens,
        headdim_qk,
        headdim_v,
        LAYOUT_IDS[layout],
        CHUNK_SIZE,
        (seqlens + CHUNK_SIZE - 1) // CHUNK_SIZE,
        0,
    ]


def validate_length_specs(cu_seqlens, seqused, batch):
    for name, value, count in (
        ("cu_seqlens", cu_seqlens, batch + 1),
        ("seqused", seqused, batch),
    ):
        if value is not None:
            require(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.int32
                and tuple(value.shape) == (count,)
                and value.is_contiguous(),
                f"{name} must be contiguous INT32 [{count}]",
            )


def read_lengths(cu_seqlens, seqused, batch, length, layout):
    """Host validation/reference only; never used to fill the device output."""
    validate_length_specs(cu_seqlens, seqused, batch)
    lengths = None
    if cu_seqlens is not None:
        cu = cu_seqlens.detach().cpu().tolist()
        require(cu[0] == 0, "cu_seqlens must start at zero")
        lengths = [end - start for start, end in zip(cu, cu[1:])]
        require(
            all(0 < n <= length for n in lengths), "cu_seqlens lengths out of bounds"
        )
    if seqused is not None:
        used = seqused.detach().cpu().tolist()
        require(all(0 < n <= length for n in used), "seqused lengths out of bounds")
        require(
            lengths is None or used == lengths, "seqused must agree with cu_seqlens"
        )
        lengths = used
    if lengths is None:
        require(
            layout != "TND" or batch == 1,
            "multi-sequence TND requires cu_seqlens or seqused",
        )
        lengths = [length] * batch
    if layout == "TND":
        require(
            sum(lengths) == length, "TND sequence lengths must cover all packed tokens"
        )
    return lengths


class FlashGdnMetadataArgs:
    cu_seqlens: gm_in(I32)
    seqused: gm_in(I32)
    metadata: gm_out(I32)
    capacity: I32
    batch: I32
    q_heads: I32
    v_heads: I32
    seqlens: I32
    layout_id: I32
    has_cu: I32
    has_used: I32


@aicpu_kernel
def flash_gdn_metadata_kernel(args: FlashGdnMetadataArgs):
    if args.capacity < HEADER_SIZE:
        return 0
    args.metadata[0] = MAGIC
    args.metadata[1] = ABI_VERSION
    args.metadata[2] = args.batch
    args.metadata[3] = args.q_heads
    args.metadata[4] = args.v_heads
    args.metadata[5] = args.seqlens
    args.metadata[6] = HEAD_DIM
    args.metadata[7] = HEAD_DIM
    args.metadata[8] = args.layout_id
    args.metadata[9] = CHUNK_SIZE
    args.metadata[10] = (args.seqlens + CHUNK_SIZE - 1) // CHUNK_SIZE
    args.metadata[STATUS_INDEX] = 0
    if args.batch < 1 or args.seqlens < 1 or args.q_heads < 1 or args.v_heads < 1:
        args.metadata[STATUS_INDEX] = 6
        return 0
    if args.v_heads % args.q_heads != 0:
        args.metadata[STATUS_INDEX] = 6
        return 0
    if args.capacity < HEADER_SIZE + args.batch:
        args.metadata[STATUS_INDEX] = 5
        return 0
    if args.layout_id < 0 or args.layout_id > 2:
        args.metadata[STATUS_INDEX] = 6
        return 0
    status = 0
    total = 0
    if args.has_cu == 1:
        if args.cu_seqlens[0] != 0:
            status = 1
    for b in range(args.batch):
        length = args.seqlens
        if args.has_cu == 1:
            length = args.cu_seqlens[b + 1] - args.cu_seqlens[b]
        elif args.has_used == 1:
            length = args.seqused[b]
        elif args.layout_id == 0 and args.batch != 1:
            status = 3
        if length <= 0 or length > args.seqlens:
            status = 2
        if args.has_cu == 1 and args.has_used == 1:
            if args.seqused[b] != length:
                status = 4
        args.metadata[HEADER_SIZE + b] = length
        total += length
    if args.layout_id == 0 and total != args.seqlens:
        status = 3
    args.metadata[STATUS_INDEX] = status
    # Report invalid input through metadata, not a nonzero AICPU task return.
    return 0


def _source_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_aicpu(output, *, build_x86=False):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    compiled = compile_aicpu_kernel(
        flash_gdn_metadata_kernel,
        workdir=str(output),
        launch_mode="interface",
        npu_arch="dav-3510",
        build_x86=build_x86,
    )
    library = Path(compiled.so_path)
    manifest = dict(
        schema=1,
        operator="flash_gdn_metadata",
        abi=ABI_VERSION,
        source_sha256=_source_digest(),
        library=library.name,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        kernel_name=compiled.name,
        launch_mode="interface",
        npu_arch="dav-3510",
        toolkit_home=os.environ.get("ASCEND_HOME_PATH"),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return compiled


def _resources():
    seen = set()
    for variable in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
        for value in os.environ.get(variable, "").split(os.pathsep):
            if not value:
                continue
            root = Path(value).absolute()
            candidates = [root / _RESOURCE, root / "built-in" / _RESOURCE]
            vendors = root if root.name == "vendors" else root / "vendors"
            if vendors.is_dir():
                candidates.extend(
                    child / _RESOURCE
                    for child in sorted(vendors.iterdir())
                    if child.is_dir()
                )
            for directory in candidates:
                if str(directory) not in seen:
                    seen.add(str(directory))
                    yield directory


def _report(event, library, **fields):
    path = os.environ.get("GDN_METADATA_BINARY_REPORT")
    if not path:
        return
    with _LOCK:
        _EVENTS.append(dict(event=event, library=str(library), **fields))
        del _EVENTS[:-64]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".gdn-metadata-report-", dir=out.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump({"events": _EVENTS}, stream, indent=2)
            os.replace(name, out)
        finally:
            Path(name).unlink(missing_ok=True)


def _get_aicpu_kernel():
    mode = os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE", "prefer")
    require(mode in ("prefer", "require", "off"), "invalid Native binary mode")
    with _LOCK:
        if mode != "off":
            for directory in _resources():
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text())
                require(
                    manifest.get("schema") == 1
                    and manifest.get("abi") == ABI_VERSION
                    and manifest.get("operator") == "flash_gdn_metadata"
                    and manifest.get("source_sha256") == _source_digest(),
                    "installed GDN AICPU metadata manifest does not match this module",
                )
                name = manifest["library"]
                require(Path(name).name == name, "invalid GDN AICPU library filename")
                library = directory / name
                require(
                    library.is_file()
                    and hashlib.sha256(library.read_bytes()).hexdigest()
                    == manifest.get("library_sha256"),
                    "GDN AICPU library checksum mismatch",
                )
                key = str(library)
                if key not in _HANDLES:
                    _HANDLES[key] = CompiledAicpuKernel(
                        key,
                        flash_gdn_metadata_kernel.spec,
                        name=manifest["kernel_name"],
                        launch_mode="interface",
                    )
                _report("installed_aicpu_hit", library)
                return _HANDLES[key]
        if mode == "require":
            raise RuntimeError(
                "installed flash_gdn metadata AICPU binary is required; JIT is forbidden"
            )
        key = "jit:" + _source_digest()
        if key not in _HANDLES:
            cache = Path(os.environ.get("CANNBOTDSL_CACHE_DIR", tempfile.gettempdir()))
            _HANDLES[key] = build_aicpu(cache / "flash_gdn_metadata" / _source_digest())
        _report("jit_aicpu_compile_or_cache", _HANDLES[key].so_path)
        return _HANDLES[key]


def flash_gdn_metadata(
    cu_seqlens=None,
    seqused=None,
    *,
    batch_size,
    nheads_qk,
    nheads_v,
    seqlens,
    headdim_qk=128,
    headdim_v=128,
    enable_grad=False,
    layout="TND",
    _meta_only=False,
):
    require(isinstance(enable_grad, bool), "enable_grad must be bool")
    if enable_grad:
        raise NotImplementedError("GDN backward is not implemented")
    header = make_header(
        batch_size, nheads_qk, nheads_v, seqlens, headdim_qk, headdim_v, layout
    )
    validate_length_specs(cu_seqlens, seqused, batch_size)
    size = HEADER_SIZE + batch_size
    if _meta_only or any(t is not None and t.is_meta for t in (cu_seqlens, seqused)):
        return torch.empty(size, dtype=torch.int32, device="meta")
    lengths = read_lengths(cu_seqlens, seqused, batch_size, seqlens, layout)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError(
            "FlashGDN metadata requires an Ascend NPU/AICPU runtime; no CPU fallback"
        )
    device = torch.device("npu", torch.npu.current_device())
    values = []
    for value in (cu_seqlens, seqused):
        if value is not None:
            require(
                value.device.type == "cpu" or value.device == device,
                "length tensors must be CPU or on the current NPU",
            )
            value = value.to(device).contiguous()
        values.append(value)
    metadata = torch.empty(size, dtype=torch.int32, device=device)
    stream = torch.npu.current_stream(device)
    for value in (*values, metadata):
        if value is not None:
            value.record_stream(stream)
    kernel = _get_aicpu_kernel()
    kernel.launch(
        current_raw_stream(device.index),
        cu_seqlens=0 if values[0] is None else values[0].data_ptr(),
        seqused=0 if values[1] is None else values[1].data_ptr(),
        metadata=metadata.data_ptr(),
        capacity=size,
        batch=batch_size,
        q_heads=nheads_qk,
        v_heads=nheads_v,
        seqlens=seqlens,
        layout_id=LAYOUT_IDS[layout],
        has_cu=int(values[0] is not None),
        has_used=int(values[1] is not None),
    )
    words = metadata.detach().cpu().tolist()
    require(
        words[STATUS_INDEX] == 0, f"GDN metadata AICPU status {words[STATUS_INDEX]}"
    )
    require(words == header + lengths, "GDN metadata AICPU output mismatch")
    _report("aicpu_launch_validated", kernel.so_path, device=str(device), words=size)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-aicpu", required=True)
    parser.add_argument("--build-x86", action="store_true")
    arguments = parser.parse_args()
    build_aicpu(arguments.build_aicpu, build_x86=arguments.build_x86)
