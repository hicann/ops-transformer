#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Generate tiling data binary for InplacePartialRotaryMul kernel UT.

Usage: python3 gen_tiling.py <tiling_key> <slice_start> <slice_end>

The tiling data binary contains the RopeRegbaseTilingData struct serialized as int64_t
values. Actual tiling data values need to be filled in for each specific test case.

To obtain tiling data values:
  1. Run the OpHost UT for the same shape/dtype to get the tiling data output string
  2. Convert the space-separated int64 string to a numpy array
  3. Add a new case to the tiling_params dict below

Example tiling data format (from OpHost UT output):
  tilingData: B CosB S D N blockNumB blockFactorB ... sliceStart sliceEnd sliceLength ...

The struct layout (RopeRegbaseTilingData, ~100 int64 fields):
  Field order: B, CosB, S, D, N, blockNumB, blockFactorB, blockNumS, blockFactorS,
               ubLoopNumS, ubFactorS, ubTailFactorS, ubLoopNumB, ubFactorB, ubTailFactorB,
               ubLoopNumN, ubFactorN, ubTailFactorN, rotaryMode, dAlign, dSplitCoef,
               blockNumBS, blockFactorBS, blockTailBS, blockNumN, blockFactorN, blockTailN,
               ubFactorBS, ubFactorN, sliceStart, sliceEnd, sliceLength,
               ... (A3-specific fields) ...
"""

import sys
import numpy as np

# Total size of RopeRegbaseTilingData in int64_t elements
TILING_DATA_SIZE = 100  # Adjust based on actual struct size

# TILING_KEY_NOOP = 1: used when sliceStart == sliceEnd (empty slice, no rope)
TILING_KEY_NOOP = 1

# Pre-calibrated tiling data for specific test cases
# Key: (tiling_key, slice_start, slice_end)
# Value: list of int64_t values
tiling_params = {
    # === No-op (empty slice, sliceStart == sliceEnd) ===
    # Kernel reads sliceLength==0 and returns without computation
    (TILING_KEY_NOOP, 32, 32): None,  # placeholder, tiling data sets sliceLength=0
    # === AAndB: x=[2,4,8,128], cos=[2,4,8,128], FP16, slice=[0,128] → TilingKey 20040 ===
    # TODO: Fill in from OpHost UT run on Ascend dev machine
    (20040, 0, 128): None,
    # AAndB: x=[2,4,8,128], cos=[1,1,1,128], FP16, slice=[0,128] → TilingKey 20041
    (20041, 0, 128): None,
    # BAB: x=[2,4,8,64], cos=[1,4,1,64], FP16, slice=[0,64] → TilingKey 20020
    (20020, 0, 64): None,
    # Fast path: Key 1 (isBrc=true), x=[1,4,1,64], FP16, slice=[0,64]
    (1, 0, 64): None,
    # Fast path: Key 2 (isBrc=false), x=[1,1,1,64], FP16, slice=[0,64]
    (2, 0, 64): None,
    # Generic Split S: x=[1,4,1,256], FP16, slice=[0,256] → TilingKey 2000
    (2000, 0, 256): None,
}


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 gen_tiling.py <tiling_key> [slice_start] [slice_end]",
            file=sys.stderr,
        )
        sys.exit(1)

    tiling_key = int(sys.argv[1])
    slice_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    slice_end = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    key = (tiling_key, slice_start, slice_end)

    if key in tiling_params and tiling_params[key] is not None:
        params = tiling_params[key]
    else:
        # Fallback: zero-filled tiling data
        # Note: this is a placeholder. Actual tiling data must be calibrated
        # from the OpHost UT output for each specific shape/dtype combination.
        print(
            f"Warning: no calibrated tiling data for (key={tiling_key}, "
            f"slice=[{slice_start},{slice_end}]), using zero-filled placeholder",
            file=sys.stderr,
        )
        params = [0] * TILING_DATA_SIZE

    base_params = np.array(params, dtype=np.int64)
    with open("tiling.bin", "wb") as f:
        base_params.tofile(f)

    print(
        f"Written tiling.bin: tiling_key={tiling_key}, "
        f"slice=[{slice_start},{slice_end}], size={len(params)} int64"
    )


if __name__ == "__main__":
    main()
