/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_ARCH_SELECT_H_
#define QUANT_BLOCK_SPARSE_ATTN_ARCH_SELECT_H_

// DAV_9201 and DAV_9202 use the independent 960 implementation.
#if defined(DAV_9201) || defined(DAV_9202) || defined(__DAV_9201__) || defined(__DAV_9202__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#define QBSA_ARCH_9102R1 1
#else
#define QBSA_ARCH_9102R1 0
#endif

// FP8 uses float descales; MXFP8 uses E8M0 descales. Keep their A6
// implementations in separate translation units to avoid shared-header clashes.
#if QBSA_ARCH_9102R1 && (!defined(ORIG_DTYPE_Q_DESCALE) || ORIG_DTYPE_Q_DESCALE != DT_FLOAT8_E8M0)
#define QBSA_ARCH92_FP8 1
#else
#define QBSA_ARCH92_FP8 0
#endif

#endif // QUANT_BLOCK_SPARSE_ATTN_ARCH_SELECT_H_
