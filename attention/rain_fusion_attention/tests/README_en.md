# RainFusionAttention Test Suite

English | [简体中文](./README.md)

## Overview

This directory contains the complete test suite of the RainFusionAttention operator.

## Test Cases

### 1. TND Format Test

- **Basic TND format test**: Verifies the correctness of the standard TND [T,N,D] format
- Configuration: batch=1, qSeq=256, kvSeq=512, heads=8

### 2. Tests with Different blockShape Configurations

- **Small block test (32x32)**: Verifies the small block sparse pattern
- **Large block test (128x128)**: Verifies the large block sparse pattern
- **Asymmetric block test (64x128)**: Verifies the asymmetric block configuration

### 3. Boundary Condition Tests

- **Unaligned sequence lengths**: qSeq=250, kvSeq=500 (not divisible by 64)
- **Small sequence lengths**: qSeq=64, kvSeq=128 (extremely small sequences)
- **Single-head configuration**: numHeads=1 (boundary case)
- **Multiple batches**: batch=2 (multi-batch processing)

### 4. Sparsity Tests

- **High sparsity**: sparsity=0.1 (10%, extremely sparse)
- **Low sparsity**: sparsity=0.8 (80%, close to dense)

## Compilation and Running

```bash
# Compile the tests
cd /path/to/ops-transformer-dev
bash build.sh --soc=ascend910b

# Run the tests
./test_rain_fusion_attention
```

## Test Output

The tests output the execution result of each test case:

```
--- Test: Basic TND Format ---
Config: batch=1, qSeq=256, kvSeq=512, heads=8, blockShape=[64,64]
Result: PASS

======================================
  Test Summary
======================================
Total tests:   10
Passed:        10 (100%)
Failed:        0
Skipped:       0
======================================
```

## Adding a New Test

Add a new `TestConfig` to the `tests` vector in `test_rain_fusion_attention.cpp`:

```cpp
tests.push_back({
    batch,          // batch size
    qSeqlen,        // Q sequence length
    kvSeqlen,       // KV sequence length
    numHeads,       // number of heads
    numKvHeads,     // number of KV heads
    headDim,        // head dimension
    blockShapeX,    // block shape X
    blockShapeY,    // block shape Y
    sparsity,       // sparsity ratio
    "Test Name"     // test name
});
```

## Precautions

1. The tests must run in an environment with Ascend devices.
2. Ensure that the CANN environment is correctly configured.
3. The tests use a fixed random seed to ensure reproducible results.
4. Currently only the TND format is supported, and BNSD format tests are skipped.
