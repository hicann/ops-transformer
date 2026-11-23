#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Targeted A2/A5 gradient checks using the repository's compiled ACLNN example.

Example on an Ascend host:
  python3 test_chunk_gradients.py --arch a2 --runner ./test_aclnn_text_gated_residual_grad

The runner must use the installed operator under review. Each case gets its own
temporary directory. Large cases produce roughly 1 GiB of input and output files
at a time. Set TMPDIR to choose the filesystem. Only the target gradient is checked.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class Case:
    name: str
    dtype: str
    n: int
    c: int
    h: int
    r: int
    gamma: bool
    output: str
    expected: float


CASES = (
    Case("xn-alias", "fp16", 16, 2, 16, 16, False, "dw_gate", 4.0),
    Case("lowp-dx", "fp16", 16384, 4, 2560, 320, True, "dgamma", 4.0),
    Case("chunk-dgamma", "bf16", 15360, 4, 2560, 320, True, "dgamma", 1.0),
    Case("short-dgamma", "fp16", 17, 2, 128, 16, True, "dgamma", 17.0),
    Case("streamk-dgamma", "bf16", 31, 4, 2560, 320, True, "dgamma", 31.0),
)


def encode(value, dtype):
    if dtype == "fp32":
        return struct.pack("<f", value)
    if dtype == "fp16":
        return struct.pack("<e", value)
    # Every BF16 fixture value is exactly representable, including +/-2**27.
    (bits,) = struct.unpack("<I", struct.pack("<f", value))
    if bits & 0xFFFF:
        raise ValueError("inexact BF16 fixture value: " + str(value))
    return struct.pack("<H", bits >> 16)


def write_constant(path, count, dtype, value):
    word = encode(value, dtype)
    with path.open("wb") as out:
        if value == 0:
            out.truncate(count * len(word))
            return
        block_elems = min(count, (1 << 20) // len(word))
        block = word * block_elems
        for _ in range(count // block_elems):
            out.write(block)
        out.write(word * (count % block_elems))


def write_row(path, index, width, dtype, value):
    word = encode(value, dtype)
    with path.open("r+b") as out:
        out.seek(index * width * len(word))
        out.write(word * width)


def prepare(case, directory):
    n, c, h, r = case.n, case.c, case.h, case.r
    d = c * h
    (directory / "meta.txt").write_text(
        f"{case.dtype} {n} {c} {h} {r} {int(case.gamma)} 1\n", encoding="ascii"
    )
    mixed, inject_grad = 0, 0
    if case.name == "xn-alias":
        # xn=1, gpre=1/4. Zero DX must leave every xn row intact for G56.
        inject_grad = 1
    elif case.name == "lowp-dx":
        # direct=-1; dlogits=1; two gate rows give fp32 DX=1+2**-12.
        mixed, inject_grad = -8, 8
    elif case.name in ("short-dgamma", "streamk-dgamma"):
        # direct=1, xhat=1. Empty AIV lanes contribute exactly zero.
        mixed = 2 * c

    # With x=1 and forward eps=0, the saved inv_rms is exactly one.
    inputs = (
        ("grad_mixed", n * h, case.dtype, mixed),
        ("grad_inject", n * c, case.dtype, inject_grad),
        ("x", n * d, case.dtype, 1),
        ("w_down", r * d, case.dtype, 0),
        ("w_up", d * r, case.dtype, 0),
        ("w_gate", c * d, case.dtype, 0),
        ("inv_rms", n * c, "fp32", 1),
        ("inject", n * c, case.dtype, 1),
        ("down_pre_act", n * r, case.dtype, 0),
        ("up", n * d, case.dtype, 0),
    )
    for name, count, dtype, value in inputs:
        write_constant(directory / (name + ".bin"), count, dtype, value)
    if case.gamma:
        # xNorm=0 and gate logits=0, including when w_gate has nonzero rows.
        write_constant(directory / "gamma.bin", d, "fp32", -1)
    if case.name == "lowp-dx":
        for row, value in enumerate((1, 2**-12)):
            write_row(directory / "w_gate.bin", row, d, case.dtype, value)
    if case.name == "chunk-dgamma":
        # direct=grad_mixed/8. Each 5120-row chunk has one nonzero row.
        for row, value in ((0, 2**27), (5120, 8), (10240, -(2**27))):
            write_row(directory / "grad_mixed.bin", row, h, case.dtype, value)


def check_output(case, directory):
    path = directory / (case.output + ".bin")
    d = case.c * case.h
    fmt, count = ("<f", d) if case.output == "dgamma" else ("<e", case.c * d)
    data = path.read_bytes()
    if len(data) != count * struct.calcsize(fmt):
        raise ValueError(
            f"{path.name} has {len(data)} bytes, expected {count * struct.calcsize(fmt)}"
        )
    mistakes = 0
    first = ""
    for index, (value,) in enumerate(struct.iter_unpack(fmt, data)):
        if value != case.expected:
            if mistakes == 0:
                first = f"first mismatch at {index}, got {value!r}"
            mistakes += 1
    if mistakes:
        raise ValueError(
            f"{case.output}: {mistakes}/{count} differ from {case.expected:g}; {first}"
        )
    return f"all {count} {case.output} elements equal {case.expected:g}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arch",
        choices=("a2", "a5"),
        required=True,
        help="select cases for the device running the example",
    )
    parser.add_argument(
        "--runner", type=Path, required=True, help="compiled ACLNN example"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--case",
        choices=[case.name for case in CASES],
        help="run only one case; xn-alias is also usable as an A5 control",
    )
    args = parser.parse_args()
    runner = args.runner.resolve()
    if not runner.is_file():
        parser.error(f"runner does not exist: {runner}")
    cases = (
        [case for case in CASES if case.name == args.case]
        if args.case
        else [case for case in CASES if args.arch == "a2" or case.name != "xn-alias"]
    )
    failures = 0
    for case in cases:
        print(f"RUN {args.arch} {case.name} ({case.dtype}, N={case.n})", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="tgrg-review-") as tmp:
                directory = Path(tmp)
                prepare(case, directory)
                subprocess.run(
                    [str(runner), str(directory), str(args.device)], check=True
                )
                message = check_output(case, directory)
            print(f"PASS {case.name}: {message}", flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            failures += 1
            print(f"FAIL {case.name}: {error}", file=sys.stderr, flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
