/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deter.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_DETER_H_
#define FLASH_ATTENTION_SCORE_GRAD_DETER_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "flash_attention_score_grad_common.h"

namespace commondef {
constexpr int64_t DETER_PREFIX_NUM = 132;
constexpr int64_t DETER_PREFIX_THRESHOLD = 128;
constexpr int64_t M_BASE_64 = 64;
constexpr int64_t N_BASE_128 = 128;

__aicore__ inline int64_t Square(int64_t num)
{
    return num * num;
}

__aicore__ inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

__aicore__ inline int64_t AbsCeil(int64_t num1, int64_t num2)
{
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (abs(num1) + abs(num2) - 1) / abs(num2);
    return isNegative ? -result : result;
}

struct CoordinateInfo {
    int64_t batchId = 0;
    int64_t n2Idx = 0;
    int64_t gIdx = 0;
    int64_t s1Idx = 0;
    int64_t s2Idx = 0;

    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t sparseMode = NO_MASK;

    int64_t mOffset = 0;
    int64_t nOffset = 0;

    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

template <uint8_t mode = 0>
__aicore__ inline void TransDeterRound(int64_t &r, bool &useEvenCol, int64_t rMax)
{
    if constexpr (mode == 0) {
        return;
    } else if constexpr (mode == 1) {
        useEvenCol = r % NUM_TWO == 1;
        r = (r + 1) / NUM_TWO;
        return;
    } else {
        if (r > rMax) {
            r = r - rMax;
            useEvenCol = true;
        }
        return;
    }
}

template <uint8_t mode = 0>
__aicore__ inline void TransTilingSplitMode(int64_t &m, int64_t &n, int64_t &r, bool &useEvenCol, int64_t rMax)
{
    if constexpr (mode == 0) {
        return;
    } else if constexpr (mode == 1) {
        m = Ceil<int64_t>(m, static_cast<int64_t>(NUM_TWO));
        TransDeterRound<mode>(r, useEvenCol, rMax);
        return;
    } else {
        n = Ceil<int64_t>(n, static_cast<int64_t>(NUM_TWO));
        TransDeterRound<mode>(r, useEvenCol, rMax);
        return;
    }
}

template <uint8_t mode = 0, const bool IS_TND = false>
__aicore__ inline void TransTilingSplitModeBack(CoordinateInfo &coordinateInfo, bool useEvenCol)
{
    if constexpr (mode == 0) {
        return;
    } else if constexpr (mode == 1) {
        coordinateInfo.s1Idx = useEvenCol ? coordinateInfo.s1Idx * NUM_TWO - 1 : coordinateInfo.s1Idx * NUM_TWO;
        if constexpr (IS_TND) {
            coordinateInfo.s1Outer = Ceil<int64_t>(coordinateInfo.actualS1Len, M_BASE_64);
        }
        return;
    } else {
        coordinateInfo.s2Idx = useEvenCol ? NUM_TWO * coordinateInfo.s2Idx : NUM_TWO * coordinateInfo.s2Idx - 1;
        if constexpr (IS_TND) {
            coordinateInfo.s2Outer = Ceil<int64_t>(coordinateInfo.actualS2Len, N_BASE_128);
        }
        return;
    }
}

__aicore__ inline void InitCoordinateInfo(int64_t s1Outer, int64_t s2Outer, int64_t mOffset, int64_t nOffset,
                                          CoordinateInfo &coordinateInfo)
{
    coordinateInfo.s1Outer = s1Outer;
    coordinateInfo.s2Outer = s2Outer;
    coordinateInfo.mOffset = mOffset;
    coordinateInfo.nOffset = nOffset;
}

__aicore__ inline void CalDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                     CoordinateInfo &coordinate)
{
    k = Min(k, b * m);
    if (j > k) {
        coordinate.batchId = -1;
        return;
    }

    int64_t p = (Ceil<int64_t>(r, m) - 1) * k + j;
    // Determine w and y
    int64_t w = p % b;
    w = (w != 0) ? w : b;
    int64_t y = Ceil<int64_t>(p, b);

    int64_t y1 = y % m;
    y1 = (y1 != 0) ? y1 : m;
    int64_t r1 = r % m;
    r1 = (r1 != 0) ? r1 : m;

    // Calculate x
    int64_t x = y1 + r1 - 1;
    if (x > m) {
        x -= m;
    }

    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

__aicore__ inline void CalDenseIndexForSingleN(int64_t k, int64_t m, int64_t b, int64_t j, int64_t r, int64_t R,
                                               CoordinateInfo &coordinate)
{
    int64_t n = 1;
    int64_t ID = (j - 1) * R + r;
    int64_t num = m * n;

    int64_t delta1 = (ID - 1) / num + 1;
    int64_t delta = ID % num;
    delta = delta == 0 ? num : delta;

    int64_t g = Gcd(m, R);
    int64_t t1 = R / g;
    int64_t t2 = m / g;

    int64_t x = ((delta - 1) % m) + 1;
    int64_t y = (delta - 1) / m + 1;

    if (t1 < n) {
        int64_t n1 = (n % t1);
        n1 = n1 == 0 ? t1 : n1;
        if (y <= n - n1) {
            int64_t delta_adj = Ceil<int64_t>(y, t1);
            delta += delta_adj;
            if (delta > delta_adj * t2 * R) {
                delta -= t2 * R;
            }
            x = ((delta - 1) % m) + 1;
            y = (delta - 1) / m + 1;
        }
    }
    coordinate.batchId = delta1;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
}

__aicore__ inline void CalDenseSwizzleIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                            CoordinateInfo &coordinate)
{
    j = j - 1;
    r = r - 1;
    k = Min(k, b * m);
    if (j > k) {
        coordinate.batchId = -1;
        return;
    }
    int64_t p = (r / m) * k + j;
    int64_t w = p / n;
    int64_t y = p % n;
    int64_t x = (y + r) % m;
    if (x >= m) {
        x -= m;
    }
    w += 1;
    x += 1;
    y += 1;
    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

__aicore__ inline void CalGQADenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t core_id, int64_t round_id,
                                        int64_t g, CoordinateInfo &coordinate, int64_t denseRound = 0)
{
    coordinate.batchId = -1;
    k = Min(Min(k, b * g * m), b * n);
    int64_t R = denseRound > 0 ? denseRound : Max(Max(Ceil<int64_t>(b * n * g, k), Ceil<int64_t>(n, m)), g);
    if (core_id < 1 || core_id > k || round_id < 1 || round_id > R * m) {
        return;
    }
    int64_t ID = (core_id - 1) * R + Ceil<int64_t>(round_id, m);
    int64_t local_id = round_id % m;
    local_id = local_id != 0 ? local_id : m;

    int64_t num = g * n;
    if (ID > num * b) {
        return;
    }

    int64_t N = b * g;
    int64_t b_id = ID % N;
    b_id = b_id != 0 ? b_id : N;
    b_id = Ceil<int64_t>(b_id, g);
    int64_t y = Ceil<int64_t>(ID, N);
    int64_t w = ID % g;
    w = w != 0 ? w : g;

    int64_t gcd = Gcd(N, R);
    int64_t t1 = R / gcd;
    // 实际不需要计算t2也可以 因为t2*R就是t1*b
    int64_t t2 = N / gcd;

    int64_t t1_new = t1 * m;
    int64_t y1 = y % t1_new;
    y1 = y1 != 0 ? y1 : t1_new;
    int64_t offset = Ceil<int64_t>(y1, t1);

    if (t1_new < n) {
        int64_t n1 = (n % t1_new);
        n1 = n1 != 0 ? n1 : t1_new;
        if (y <= n - n1) {
            int64_t delta = Ceil<int64_t>(y, t1_new);
            ID += delta;
            if (ID > (delta - 1) * t2 * m * R + offset * t2 * R) {
                ID -= t2 * R;
            }
            b_id = ID % N;
            b_id = b_id != 0 ? b_id : N;
            b_id = Ceil<int64_t>(b_id, g);
            w = ID % g;
            w = w != 0 ? w : g;
            y = Ceil<int64_t>(ID, N);
        }
    }

    int64_t x = local_id + offset - 1;
    if (x > m) {
        x -= m;
    }

    coordinate.batchId = w + (b_id - 1) * g;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
}
__aicore__ inline void CalCausalG2kSingleBatchDeterIndex(int64_t k, int64_t j, int64_t a, int64_t L1, int64_t offset,
                                                         CoordinateInfo &coordinate)
{
    int64_t x, y = 0;
    if (j % NUM_TWO == 1) { // 奇核分支
        if (a <= L1 - j + 1) {
            y = j + offset;
            x = y + a - 1;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + NUM_TWO * L1 - NUM_TWO * k + 1 - a;
        }
    } else { // 偶核分支
        if (a >= L1 - NUM_TWO * k + 1 + j) {
            y = j + offset;
            x = y + NUM_TWO * L1 - NUM_TWO * k + 1 - a;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + a - 1;
        }
    }
    coordinate.batchId = 0;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalNoRecSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                           CoordinateInfo &coordinate)
{
    coordinate.batchId = -1;

    // 核心编号超过 ⌊n/2⌋ + 1 时无任务可处理
    if (j > (n / NUM_TWO) + 1) {
        return;
    }

    int64_t x, y = 0;
    if (j % NUM_TWO == 1) { // 奇核
        if (r + j <= n + 1) {
            x = r + j - 1;
            y = j;
        } else {
            x = NUM_TWO * n + NUM_TWO - j - r;
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    } else { // 偶核
        if (j <= r + 1 - (n % NUM_TWO)) {
            x = n + j - r - 1 + n % NUM_TWO;
            y = j;
        } else {
            x = n + NUM_TWO + r - j - (n % NUM_TWO);
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    }

    // 检查 (x, y) 是否在下三角范围内
    if (y >= 1 && y <= m && y <= x && x <= m) {
        coordinate.batchId = 0;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    }
    return;
}

// 矩形
__aicore__ inline void CalCausalRecSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                         CoordinateInfo &coordinate)
{
    int64_t x, y = 0;
    // Choose method based on core count
    if (NUM_TWO * k < m + 1 && k < n) {
        CalCausalG2kSingleBatchDeterIndex(k, j, r, m, 0, coordinate);
        x = coordinate.s1Idx;
        y = coordinate.s2Idx;
    } else {
        CalCausalNoRecSingleBatchDeterIndex(k, m, m, j, r, coordinate);
        if (coordinate.batchId != -1) {
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
        } else {
            x = m + 1;
            y = m + 1;
        }
    }

    // Check if the computed position is within the valid submatrix
    if (y >= 1 && y <= n && y <= x && x <= m) {
        coordinate.batchId = 0;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

// 计算单batch切分
__aicore__ inline void CalCausalSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                      CoordinateInfo &coordinate)
{
    // 0. special case
    if (k >= (n / NUM_TWO) + 1) {
        CalCausalRecSingleBatchDeterIndex(k, m, n, j, r, coordinate);
        return;
    }

    // 1. compute ell and t1, t2, t3
    int64_t ell = n % k;
    int64_t t1 = n / (NUM_TWO * k);
    int64_t t2 = 0;
    int64_t t3 = (n / k) % NUM_TWO;

    int64_t rm, L1, a, offset;
    // 2. 2k-block groups
    int64_t bound1 = (NUM_TWO * m + 1) * t1 - NUM_TWO * k * t1 * t1;
    if (r <= bound1) {
        // solve for group index i
        int64_t disc = Square((NUM_TWO * m + 1)) - NUM_EIGHT * k * r;
        int64_t i = Ceil<int64_t>(((NUM_TWO * m + 1) - sqrt(disc)), (NUM_FOUR * k));
        rm = (NUM_TWO * m + 1) * (i - 1) - NUM_TWO * k * (i - 1) * (i - 1);
        offset = NUM_TWO * k * (i - 1);
        L1 = m - NUM_TWO * k * (i - 1);
        a = r - rm;
        CalCausalG2kSingleBatchDeterIndex(k, j, a, L1, offset, coordinate);
        return;
    } else {
        rm = bound1;
        offset = NUM_TWO * k * t1;
    }

    // 4. remaining block
    int64_t rem = t3 * k + ell;
    coordinate.batchId = -1;
    if (rem > 0) {
        L1 = m - n + rem;
        a = r - rm;
        if (t3 == 0 && j <= rem) {
            coordinate.batchId = 0;
            coordinate.s2Idx = offset + j;
            coordinate.s1Idx = coordinate.s2Idx + a - 1;
            return;
        } else if (t3 == 1) {
            CalCausalRecSingleBatchDeterIndex(k, L1, L1, j, a, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }

            int64_t shift = NUM_TWO * k * t1 + NUM_THREE * k * t2;
            coordinate.s1Idx += shift;
            coordinate.s2Idx += shift;
            return;
        } else {
            return;
        }
    }
    return;
}

__aicore__ inline void CalCausalIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                      CoordinateInfo &coordinate)
{
    // 1. b1, b2, rm1
    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t delta = m - n;
    int64_t sizeTri = n * (m + delta + 1) / NUM_TWO;
    int64_t rm1 = b1 * sizeTri;

    int64_t w, x, y = 0;
    // --- 第一段：完整矩形 ---
    if (r <= rm1) {
        int64_t a = r % sizeTri;
        a = a != 0 ? a : sizeTri;
        w = Ceil<int64_t>(r, sizeTri) + b1 * (j - 1);
        int64_t n1 = n / NUM_TWO * NUM_TWO;
        int64_t L = NUM_TWO * m - n1 + 1;
        int64_t rmLocal = (n1 * L) / NUM_TWO;
        if (a <= rmLocal) {
            y = Ceil<int64_t>(a, L);
            int64_t r1 = a % L != 0 ? a % L : L;
            x = r1 + y - 1;
            if (x > m) {
                x = NUM_TWO * m + 1 - x;
                y = n1 + 1 - y;
            }
        } else {
            int64_t a1 = a - rmLocal;
            y = n;
            x = a1 - 1 + y;
        }
        w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
        w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    // 2. 行块压平分给 floor(b2/2) 个 batch
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t a2 = r - rm1;
    int64_t half_ = b2 / NUM_TWO;
    int64_t rm2 = (NUM_TWO * m - t * k + 1) * t * half_;
    if (a2 >= 1 && a2 <= rm2) {
        // 调用 dense 子块
        CalDenseIndex(k, NUM_TWO * m - t * k + 1, t * k, half_, j, a2, coordinate);
        int64_t wSub = coordinate.batchId;
        int64_t xSub = coordinate.s1Idx;
        int64_t ySub = coordinate.s2Idx;
        // 新增范围检查
        int64_t max_x = NUM_TWO * m - t * k + 1;
        int64_t max_y = t * k;
        int64_t max_w = half_;
        if (xSub >= 1 && xSub <= max_x && ySub >= 1 && ySub <= max_y && wSub >= 1 && wSub <= max_w) {
            // 计算映射
            if (xSub - ySub <= m - t * k) {
                w = NUM_TWO * wSub - 1 + b1 * k;
                x = m + 1 - xSub;
                y = t * k + 1 - ySub;
            } else {
                w = NUM_TWO * wSub + b1 * k;
                x = xSub - m + t * k - 1;
                y = ySub;
            }
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
        } else {
            coordinate.batchId = -1;
        }
        return;
    }

    // 3. 剩余，分两种：b2 mod2 == 1 或 else
    int64_t a3 = r - rm1 - rm2;
    int64_t rm3 = 0;
    if (b2 % NUM_TWO == 1) {
        int64_t t1 = n / (NUM_TWO * k);
        int64_t t3 = t % NUM_TWO;
        // 奇数 b2
        if (t3 == 1) {
            int64_t m1 = m - t1 * NUM_TWO * k;
            rm3 = (m + m1 + 1) * t1;
            rm3 += ell == 0 ? m1 : Max(m1, NUM_TWO * m1 - NUM_TWO * k + 1);
            if (a3 >= 1 && a3 <= rm3) {
                CalCausalSingleBatchDeterIndex(k, m, n, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = b;
                    return;
                } else {
                    return;
                }
            }
            b -= 1;
            b2 -= 1;
        } else {
            // 偶数 b2
            rm3 = (NUM_TWO * m - t * k + 1) * t / NUM_TWO;
            if (a3 >= 1 && a3 <= rm3) {
                CalCausalSingleBatchDeterIndex(k, m, t * k, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = b;
                    return;
                } else {
                    return;
                }
            }
        }
    }

    // 4. 零碎余数 ℓ 部分
    int64_t a4 = a3 - rm3;
    int64_t p = Ceil<int64_t>(ell, NUM_TWO);
    int64_t ell1 = ell + 1 - (ell % NUM_TWO);
    int64_t block = (b2 * p) / k;
    int64_t res0 = (b2 * p) % k;
    // 情况 A
    if (a4 > block * (ell1 + NUM_TWO * delta) && res0 <= k / NUM_TWO) {
        int64_t offs = a4 - block * (ell1 + NUM_TWO * delta);
        // 前 res0 cores
        if (j >= 1 && j <= res0) {
            int64_t limit = (res0 - j) / b2 + Ceil<int64_t>(ell1, NUM_TWO) + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
                w = w != 0 ? w : b2;
                y = p - (res0 - j) / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    coordinate.batchId = b1 * k + w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }

        // 后 res0 cores
        if (k - res0 + 1 <= j && j <= k) {
            int64_t idx = j - (k - res0 + 1);
            int64_t limit = (ell1 - 1) / NUM_TWO - idx / b2 + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + k + 1 - j) % b2;
                w = w != 0 ? w : b2;
                y = p + 1 + idx / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    coordinate.batchId = b1 * k + w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }
        coordinate.batchId = -1;
        return;
    } else {
        // 情况 B
        w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
        w = w != 0 ? w : b2;
        int64_t g = Ceil<int64_t>((k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j), b2);
        if (g >= 1 && g <= p) {
            int64_t a5 = a4 % (ell1 + NUM_TWO * delta);
            a5 = a5 != 0 ? a5 : (ell1 + NUM_TWO * delta);
            int64_t x0, y0 = 0;
            if (g % NUM_TWO == 1) {
                if (a5 <= ell - g + 1 + delta) {
                    x0 = g + a5 - 1;
                    y0 = g;
                } else {
                    x0 = NUM_TWO * ell + NUM_TWO * delta + NUM_TWO - g - a5;
                    y0 = ell + 1 + (ell % NUM_TWO) - g;
                }
            } else {
                if (a5 >= (g + 1 + delta - (ell % NUM_TWO))) {
                    x0 = g + ell + NUM_TWO * delta + 1 - (ell % NUM_TWO) - a5;
                    y0 = g;
                } else {
                    x0 = a5 + ell - g + (ell % NUM_TWO);
                    y0 = ell + 1 + (ell % NUM_TWO) - g;
                }
            }

            if (y0 >= 1 && y0 <= ell && y0 <= x0 && x0 <= ell + delta) {
                coordinate.batchId = b1 * k + w;
                coordinate.s1Idx = x0 + t * k;
                coordinate.s2Idx = y0 + t * k;
                return;
            }
        }
    }
    coordinate.batchId = -1;
    return;
}

__aicore__ inline void CalCausalSwizzleIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                             CoordinateInfo &coordinate)
{
    // 按照相邻B或者N拼接成一个完整S1S2
    int64_t nNew = n + 1;
    if (m != n) {
        nNew = (n - m + NUM_TWO) + (n + 1);
    }
    int64_t bNew = b >> 1;
    CalDenseSwizzleIndex(k, m, nNew, bNew, j, r, coordinate);
    if (coordinate.batchId == -1) {
        return;
    }

    int64_t w = coordinate.batchId;
    int64_t x = coordinate.s1Idx;
    int64_t y = coordinate.s2Idx;
    if (m == n) {
        if (y >= x + 1) {
            y = (n << 1) - m - y + NUM_TWO;
            x = m + 1 - x;
            w = (w << 1);
        } else {
            w = (w << 1) - 1;
        }
    } else {
        if (y >= x + (n + 1) - m + 1) {
            y = ((n + 1) << 1) - m - y + NUM_TWO;
            x = m + 1 - x;
            w = (w << 1);
        } else {
            w = (w << 1) - 1;
        }
    }

    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
}

__aicore__ inline int64_t CalLeftUpCausalSwizzleMaxRound(int64_t k, int64_t m, int64_t n, int64_t b)
{
    int64_t pairCount = b >> 1;
    if (k <= 0 || m <= 0 || n <= 0 || pairCount <= 0) {
        return 0;
    }

    int64_t virtualM = m;
    int64_t virtualN = m + 1;
    int64_t activeK = Min(k, m * pairCount);
    if (m > n) {
        virtualM = NUM_TWO * m - n + 1;
        virtualN = n;
        activeK = Min(k, n * pairCount);
    }
    return virtualM * Ceil<int64_t>(virtualN * pairCount, activeK);
}

__aicore__ inline void CalLeftUpCausalSwizzleIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                                   CoordinateInfo &coordinate)
{
    coordinate.batchId = -1;
    int64_t pairCount = b >> 1;
    if (k <= 0 || m <= 0 || n <= 0 || pairCount <= 0 || j < 1 || r < 1) {
        return;
    }

    if (m <= n) {
        int64_t activeK = Min(k, m * pairCount);
        if (j > activeK) {
            return;
        }
        CalCausalSwizzleIndex(activeK, m, m, b, j, r, coordinate);
        return;
    }

    int64_t virtualM = NUM_TWO * m - n + 1;
    int64_t activeK = Min(k, n * pairCount);
    if (j > activeK) {
        return;
    }
    int64_t columnId = (r - 1) / virtualM * activeK + j - 1;
    if (columnId >= n * pairCount) {
        return;
    }

    int64_t pairId = columnId / n + 1;
    int64_t virtualS2 = columnId % n + 1;
    int64_t virtualS1 = (r - 1) % virtualM + 1;
    int64_t oddBatchLen = m - virtualS2 + 1;
    if (virtualS1 <= oddBatchLen) {
        coordinate.batchId = NUM_TWO * pairId - 1;
        coordinate.s1Idx = virtualS2 + virtualS1 - 1;
        coordinate.s2Idx = virtualS2;
        return;
    }

    int64_t evenBatchOffset = virtualS1 - oddBatchLen;
    coordinate.batchId = NUM_TWO * pairId;
    coordinate.s1Idx = m - evenBatchOffset + 1;
    coordinate.s2Idx = n - virtualS2 + 1;
}

__aicore__ inline void CalGQACausalIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, int64_t g,
                                         CoordinateInfo &coordinate)
{
    // 1. b1, b2, rm1
    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t delta = m - n;
    int64_t sizeTri = n * (m + delta + 1) / NUM_TWO;
    int64_t sizeTriGroup = sizeTri * g;
    int64_t rm1 = b1 * sizeTriGroup;

    int64_t w, x, y = 0;
    // --- 第一段：完整矩形 ---
    if (r <= rm1) {
        int64_t a = r % sizeTriGroup;
        a = a != 0 ? a : sizeTriGroup;
        int64_t N1_id = Ceil<int64_t>(a, sizeTri);
        a = r % sizeTri;
        a = a != 0 ? a : sizeTri;
        int64_t b_id = k * (Ceil<int64_t>(r, sizeTriGroup) - 1) + j;

        int64_t n1 = n / NUM_TWO * NUM_TWO;
        int64_t L = NUM_TWO * m - n1 + 1;
        int64_t rmLocal = (n1 * L) / NUM_TWO;
        if (a <= rmLocal) {
            y = Ceil<int64_t>(a, L);
            int64_t r1 = a % L != 0 ? a % L : L;
            x = r1 + y - 1;
            if (x > m) {
                x = NUM_TWO * m + 1 - x;
                y = n1 + 1 - y;
            }
        } else {
            int64_t a1 = a - rmLocal;
            y = n;
            x = a1 - 1 + y;
        }

        b_id = ((b_id - 1) / k) * k + ((y - 1 + (b_id - 1)) % k) + 1;
        w = (b_id - 1) * g + N1_id;

        // w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
        // w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    // 2. 行块压平分给 floor(b2/2) 个 batch
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t a2 = r - rm1;
    int64_t half_ = b2 / NUM_TWO;
    int64_t rm2 = (NUM_TWO * m - t * k + 1) * t * half_ * g;
    if (a2 >= 1 && a2 <= rm2) {
        // 调用 dense 子块
        CalGQADenseIndex(k, NUM_TWO * m - t * k + 1, t * k, half_, j, a2, g, coordinate);
        int64_t wSub = coordinate.batchId;
        int64_t xSub = coordinate.s1Idx;
        int64_t ySub = coordinate.s2Idx;
        // 新增范围检查
        int64_t max_x = NUM_TWO * m - t * k + 1;
        int64_t max_y = t * k;
        int64_t max_w = half_ * g;
        if (xSub >= 1 && xSub <= max_x && ySub >= 1 && ySub <= max_y && wSub >= 1 && wSub <= max_w) {
            // 计算映射
            int64_t b_id = Ceil<int64_t>(wSub, g);
            int64_t N1_id = wSub % g;
            N1_id = N1_id != 0 ? N1_id : g;
            if (xSub - ySub <= m - t * k) {
                b_id = NUM_TWO * b_id - 1;
                x = m + 1 - xSub;
                y = t * k + 1 - ySub;
            } else {
                b_id = NUM_TWO * b_id;
                x = xSub - m + t * k - 1;
                y = ySub;
            }
            w = b1 * k * g + (b_id - 1) * g + N1_id;
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
        } else {
            coordinate.batchId = -1;
        }
        return;
    }

    // 3. 剩余，分两种：b2 mod2 == 1 或 else
    int64_t a3 = r - rm1 - rm2;
    int64_t rm3 = 0;
    if (b2 % NUM_TWO == 1) {
        int64_t t1 = n / (NUM_TWO * k);
        int64_t t3 = t % NUM_TWO;
        // 奇数 b2
        if (t3 == 1) {
            int64_t m1 = m - t1 * NUM_TWO * k;
            rm3 = (m + m1 + 1) * t1;
            rm3 += ell == 0 ? m1 : Max(m1, NUM_TWO * m1 - NUM_TWO * k + 1);
            if (a3 >= 1 && a3 <= rm3 * g) {
                int64_t N1_id = Ceil<int64_t>(a3, rm3);
                a3 = a3 % rm3;
                a3 = a3 != 0 ? a3 : rm3;
                CalCausalSingleBatchDeterIndex(k, m, n, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = (b - 1) * g + N1_id;
                    return;
                } else {
                    return;
                }
            }
            b -= 1;
            b2 -= 1;
        } else {
            // 偶数 b2
            rm3 = (NUM_TWO * m - t * k + 1) * t / NUM_TWO;
            if (a3 >= 1 && a3 <= rm3 * g) {
                int64_t N1_id = Ceil<int64_t>(a3, rm3);
                a3 = a3 % rm3;
                a3 = a3 != 0 ? a3 : rm3;
                CalCausalSingleBatchDeterIndex(k, m, t * k, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = (b - 1) * g + N1_id;
                    return;
                } else {
                    return;
                }
            }
        }
    }

    // 4. 零碎余数 ℓ 部分
    int64_t a4 = a3 - rm3 * g;
    int64_t N1_id = a4 % g;
    N1_id = N1_id != 0 ? N1_id : g;
    a4 = Ceil<int64_t>(a4, g);
    int64_t p = Ceil<int64_t>(ell, NUM_TWO);
    int64_t ell1 = ell + 1 - (ell % NUM_TWO);
    int64_t block = (b2 * p) / k;
    int64_t res0 = (b2 * p) % k;
    // 情况 A
    if (a4 > block * (ell1 + NUM_TWO * delta) && res0 <= k / NUM_TWO) {
        int64_t offs = a4 - block * (ell1 + NUM_TWO * delta);
        // 前 res0 cores
        if (j >= 1 && j <= res0) {
            int64_t limit = (res0 - j) / b2 + Ceil<int64_t>(ell1, NUM_TWO) + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
                w = w != 0 ? w : b2;
                y = p - (res0 - j) / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    w = (w - 1) * g + N1_id + b1 * k * g;
                    coordinate.batchId = w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }

        // 后 res0 cores
        if (k - res0 + 1 <= j && j <= k) {
            int64_t idx = j - (k - res0 + 1);
            int64_t limit = (ell1 - 1) / NUM_TWO - idx / b2 + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + k + 1 - j) % b2;
                w = w != 0 ? w : b2;
                y = p + 1 + idx / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    w = (w - 1) * g + N1_id + b1 * k * g;
                    coordinate.batchId = w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }
        coordinate.batchId = -1;
        return;
    } else {
        // 情况 B
        w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
        w = w != 0 ? w : b2;
        int64_t gTail = Ceil<int64_t>((k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j), b2);
        if (gTail >= 1 && gTail <= p) {
            int64_t a5 = a4 % (ell1 + NUM_TWO * delta);
            a5 = a5 != 0 ? a5 : (ell1 + NUM_TWO * delta);
            int64_t x0, y0 = 0;
            if (gTail % NUM_TWO == 1) {
                if (a5 <= ell - gTail + 1 + delta) {
                    x0 = gTail + a5 - 1;
                    y0 = gTail;
                } else {
                    x0 = NUM_TWO * ell + NUM_TWO * delta + NUM_TWO - gTail - a5;
                    y0 = ell + 1 + (ell % NUM_TWO) - gTail;
                }
            } else {
                if (a5 >= (gTail + 1 + delta - (ell % NUM_TWO))) {
                    x0 = gTail + ell + NUM_TWO * delta + 1 - (ell % NUM_TWO) - a5;
                    y0 = gTail;
                } else {
                    x0 = a5 + ell - gTail + (ell % NUM_TWO);
                    y0 = ell + 1 + (ell % NUM_TWO) - gTail;
                }
            }

            if (y0 >= 1 && y0 <= ell && y0 <= x0 && x0 <= ell + delta) {
                w = (w - 1) * g + N1_id + b1 * k * g;
                coordinate.batchId = w;
                coordinate.s1Idx = x0 + t * k;
                coordinate.s2Idx = y0 + t * k;
                return;
            }
        }
    }
    coordinate.batchId = -1;
    return;
}

struct BandInfo {
    int64_t k = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
    int64_t b = 0;
    int64_t b1 = 0;
    int64_t b2 = 0;
    int64_t L1 = 0;
    int64_t L2 = 0;
    int64_t L3 = 0;
    int64_t nSeg = 0;
    int64_t R1 = 0;
    int64_t R2 = 0;
    int64_t R3 = 0;
    int64_t Rm = 0;
    int64_t rm = 0;
    int64_t rm2 = 0;
    int64_t hybridPairCount = 0;
};

__aicore__ inline void GenBandHybridInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                         BandInfo &bandInfo)
{
    p = Min(p, m);
    q = Min(q, n);
    m = Min(m, n + p - 1);
    n = Min(n, m + q - 1);

    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;
    if (p + q <= m) {
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
    } else {
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(n - q, m - 1);
    }
    bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
    bandInfo.hybridPairCount = Max(0, Min(bandInfo.L1, bandInfo.L3 - p + 1));
}

__aicore__ inline bool SetBandHybridCoordinate(const BandInfo &bandInfo, int64_t w, int64_t y, int64_t localRound,
                                               CoordinateInfo &coordinate)
{
    int64_t x = y + localRound - bandInfo.q;
    int64_t xMin = Max(1, y - bandInfo.q + 1);
    int64_t xMax = Min(bandInfo.m, y + bandInfo.p - 1);
    if (x < xMin || x > xMax) {
        return false;
    }
    coordinate.batchId = w;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
    return true;
}

__aicore__ inline void CalBandHybridIndex(const BandInfo &bandInfo, int64_t j, int64_t r, CoordinateInfo &coordinate)
{
    coordinate.batchId = -1;
    int64_t slot = bandInfo.p + bandInfo.q - 1;
    int64_t colsPerBatch = bandInfo.nSeg - bandInfo.hybridPairCount;
    if (j < 1 || j > bandInfo.k || r < 1 || slot <= 0 || colsPerBatch <= 0) {
        return;
    }

    int64_t layer = (r - 1) / slot;
    int64_t localRound = (r - 1) % slot + 1;
    int64_t globalColumn = layer * bandInfo.k + j;
    if (globalColumn > bandInfo.b * colsPerBatch) {
        return;
    }
    int64_t w = (globalColumn - 1) / colsPerBatch + 1;
    int64_t localColumn = (globalColumn - 1) % colsPerBatch + 1;

    if (localColumn <= bandInfo.hybridPairCount) {
        int64_t y = localColumn;
        if (SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate)) {
            return;
        }
        int64_t seg3Index = bandInfo.p + localColumn - 1;
        y = bandInfo.L1 + bandInfo.L2 + seg3Index;
        SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
        return;
    }

    if (localColumn <= bandInfo.hybridPairCount + bandInfo.L2) {
        int64_t y = bandInfo.L1 + localColumn - bandInfo.hybridPairCount;
        SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
        return;
    }

    int64_t singleIndex = localColumn - bandInfo.hybridPairCount - bandInfo.L2;
    int64_t unpairedSeg1Count = bandInfo.L1 - bandInfo.hybridPairCount;
    if (singleIndex <= unpairedSeg1Count) {
        int64_t y = bandInfo.hybridPairCount + singleIndex;
        SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
        return;
    }

    int64_t unpairedSeg3Index = singleIndex - unpairedSeg1Count;
    int64_t unpairedSeg3Prefix = Min(bandInfo.p - 1, bandInfo.L3);
    int64_t seg3Index =
        unpairedSeg3Index <= unpairedSeg3Prefix ? unpairedSeg3Index : unpairedSeg3Index + bandInfo.hybridPairCount;
    int64_t y = bandInfo.L1 + bandInfo.L2 + seg3Index;
    SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
}

__aicore__ inline void GenBandInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, BandInfo &bandInfo)
{
    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;

    // 1. initial b1, b2
    bandInfo.b1 = b / k;
    bandInfo.b2 = b % k;
    // p+q>m的分支
    if (p + q > m) {
        // 2. lengths
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(m - 1, n - q);
        // 3. redefine nSeg
        bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        // 4. segment sizes
        bandInfo.R1 = (p + m - 1) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = m * bandInfo.L2;
        bandInfo.R3 = (NUM_TWO * m - 1 - bandInfo.L3) * bandInfo.L3 / NUM_TWO;
        bandInfo.Rm = bandInfo.R1 + bandInfo.R2 + bandInfo.R3;
        // 5. rm = b1 * Rm
        bandInfo.rm = bandInfo.b1 * bandInfo.Rm;
        bandInfo.rm2 = m * Ceil<int64_t>(n * b, Min(k, b * m));
    } else {
        // p+q<=m的分支
        // 2. lengths
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        // 3. redefine nSeg
        bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        // 4. segment sizes
        bandInfo.R1 = (NUM_TWO * p - NUM_TWO + q) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = (p + q - 1) * bandInfo.L2;
        bandInfo.R3 = (p + q - NUM_TWO) * bandInfo.L3 - (bandInfo.L3 * (bandInfo.L3 - 1)) / NUM_TWO;
        bandInfo.Rm = bandInfo.R1 + bandInfo.R2 + bandInfo.R3;
        // 5. rm = b1 * Rm
        bandInfo.rm = bandInfo.b1 * bandInfo.Rm;
        bandInfo.rm2 = Ceil<int64_t>(n * b, k) * (p + q - 1);
    }
}

__aicore__ inline void CalGQADenseIndexNoTune(int64_t k, int64_t m, int64_t n, int64_t b, int64_t core_id,
                                              int64_t round_id, int64_t g, CoordinateInfo &coordinate)
{
    coordinate.batchId = -1;
    k = Min(Min(k, b * g * m), b * n);
    int64_t R = Max(Max(Ceil<int64_t>(b * n * g, k), Ceil<int64_t>(n, m)), g);
    if (core_id < 1 || core_id > k || round_id < 1 || round_id > R * m) {
        return;
    }

    int64_t ID = (core_id - 1) * R + Ceil<int64_t>(round_id, m);
    int64_t local_id = round_id % m;
    local_id = local_id != 0 ? local_id : m;
    if (ID > g * n * b) {
        return;
    }

    int64_t N = b * g;
    int64_t b_id = ID % N;
    b_id = b_id != 0 ? b_id : N;
    b_id = Ceil<int64_t>(b_id, g);
    coordinate.s1Idx = local_id;
    coordinate.s2Idx = Ceil<int64_t>(ID, N);
    int64_t w = ID % g;
    w = w != 0 ? w : g;
    coordinate.batchId = w + (b_id - 1) * g;
}

__aicore__ inline void GenGQABandInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g,
                                      BandInfo &bandInfo)
{
    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;

    // 1. initial b1, b2
    bandInfo.b1 = b / k;
    bandInfo.b2 = b % k;
    if (p + q > m) {
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(m - 1, n - q);
        // redefine n
        n = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        bandInfo.R1 = (p + m - 1) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = m * bandInfo.L2;
    } else {
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        // redefine m, n
        if (bandInfo.L3 == 0) {
            m = p + q + bandInfo.L2 - NUM_TWO;
        }
        n = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
    }
    bandInfo.n = n;
    // Rm_group内有列的轮次约束
    bandInfo.Rm = m * n - (m - p) * (m - p + 1) / NUM_TWO - (n - q) * (n - q + 1) / NUM_TWO;
    int64_t Rm_group = bandInfo.Rm * g;
    bandInfo.rm = bandInfo.b1 * Rm_group;

    if (p + q > m) {
        if (p + q <= n) {
            int64_t n1 = p + q - 1;
            bandInfo.rm2 = m * Max(Max(Ceil<int64_t>(bandInfo.b2 * n1 * g, k), Ceil<int64_t>(n1, m)), g);
        } else {
            bandInfo.rm2 = m * Max(Max(Ceil<int64_t>(bandInfo.b2 * n * g, k), Ceil<int64_t>(n, m)), g);
        }
    } else {
        if (n > m) {
            bandInfo.R1 =
                (p + q - 1) * Max(Max(Ceil<int64_t>(bandInfo.b2 * n * g, k), Ceil<int64_t>(n, (p + q - 1))), g);
            bandInfo.R2 =
                m * Max(Max(Ceil<int64_t>(bandInfo.b2 * (p + q - 1) * g, k), Ceil<int64_t>((p + q - 1), m)), g);
            bandInfo.rm2 = Min(bandInfo.R1, bandInfo.R2);
        } else {
            int64_t m1 = p + q - 1;
            bandInfo.rm2 = m1 * Max(Max(Ceil<int64_t>(bandInfo.b2 * n * g, k), Ceil<int64_t>(n, m1)), g);
        }
    }
}

__aicore__ inline void CalBandIndex(const BandInfo &bandInfo, int64_t j, int64_t r, CoordinateInfo &coordinate)
{
    int64_t w, x, y = 0;
    coordinate.batchId = -1;

    int64_t k = bandInfo.k;
    int64_t m = bandInfo.m;
    int64_t n = bandInfo.n;
    int64_t p = bandInfo.p;
    int64_t q = bandInfo.q;
    int64_t b = bandInfo.b;
    int64_t b1 = bandInfo.b1;
    int64_t b2 = bandInfo.b2;
    int64_t L1 = bandInfo.L1;
    int64_t L2 = bandInfo.L2;
    int64_t L3 = bandInfo.L3;
    int64_t nSeg = bandInfo.nSeg;
    int64_t R1 = bandInfo.R1;
    int64_t R2 = bandInfo.R2;
    int64_t R3 = bandInfo.R3;
    int64_t Rm = bandInfo.Rm;
    int64_t rm = bandInfo.rm;
    int64_t a;
    // p+q>m的分支
    if (p + q > m) {
        if (r <= rm) {
            a = r % Rm;
            w = Ceil<int64_t>(r, Rm) + b1 * (j - 1);
            if (a == 0) {
                a = Rm;
            }

            // subsegment selection
            if (a <= R1) {
                int64_t L11 = L1 / NUM_TWO * NUM_TWO;
                int64_t L = NUM_TWO * p + L11 - 1;
                int64_t local_round = L11 * L / NUM_TWO;
                if (a <= local_round) {
                    y = Ceil<int64_t>(a, L);
                    int64_t r1 = a % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = p + y - r1;
                    if (x < 1) {
                        y = L11 + 1 - y;
                        x = 1 - x;
                    }
                } else {
                    x = a - local_round;
                    y = L1;
                }
            } else if (a <= R1 + R2) {
                int64_t a2 = a - R1;
                y = Ceil<int64_t>(a2, m);
                x = a2 % m;
                if (x == 0) {
                    x = m;
                }
                y = y + L1;
            } else {
                int64_t a3 = a - R1 - R2;

                int64_t L31 = L3 / NUM_TWO * NUM_TWO;
                int64_t L = NUM_TWO * m - L31 - 1;
                int64_t local_round = L31 * L / NUM_TWO;
                if (a3 <= local_round) {
                    y = Ceil<int64_t>(a3, L);
                    int64_t r1 = a3 % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = y + r1;
                    if (x > m) {
                        y = L31 + 1 - y;
                        x = NUM_TWO * m + 1 - x;
                    }
                } else {
                    x = m - a3 + local_round + 1;
                    y = L3;
                }
                y = y + L1 + L2;
            }
            w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
            w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }

        // --- second phase: overflow rows ---
        a = r - rm;

        CalDenseIndex(k, m, n, b2, j, a, coordinate);
        w = coordinate.batchId;
        if (w != -1) {
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
            if (y <= m - p && x >= p + y) {
                return;
            }
            if (y > L1 + L2 && x <= y - L1 - L2) {
                return;
            }
            if (w > 0 && w < b2 + 1 && x > 0 && x < m + 1 && y > 0 && y < n + 1) {
                coordinate.batchId = b1 * k + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            }
        }
        return;
    }

    // p+q<=m的分支
    if (L3 == 0) {
        m = p + q + L2 - NUM_TWO;
    }
    if (r <= rm) {
        a = r % Rm;
        w = Ceil<int64_t>(r, Rm) + b1 * (j - 1);
        if (a == 0) {
            a = Rm;
        }

        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, (p + q - 1));
            x = a2 % (p + q - 1) + (y - 1);
            if (x == y - 1) {
                x = (p + q - 1) + y - 1;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * (p + q) - L31 - NUM_THREE;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1 + 1 + m - (p + q);
                if (x > m) {
                    y = L31 + 1 - y;
                    x = NUM_TWO * m + 1 - x;
                }
            } else {
                x = m - a3 + local_round + 1;
                y = L3;
            }
            y = y + L1 + L2;
        }

        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    if (b2 == 0) {
        coordinate.batchId = -1;
        return;
    }

    // --- second phase: overflow rows ---
    a = r - rm;
    int64_t seg = p + q - 1;
    int64_t a1 = Ceil<int64_t>(a, seg);
    int64_t a2 = a % seg;
    a2 = a2 == 0 ? seg : a2;

    // if no L3 or nSeg - m < 1
    if (L3 == 0 || nSeg - m < 1) {
        int64_t idx = (a1 - 1) * k + j;
        w = Ceil(idx, nSeg);
        y = idx % nSeg;
        if (y == 0) {
            y = nSeg;
        }

        x = y + a2 - q;
        if (x >= 1 && x <= m) {
            coordinate.batchId = w + b1 * k;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }
        return;
    }

    // else L3>0 and nSeg-m >=1
    if (L3 > 0 && nSeg - m >= 1) {
        y = (a1 - 1) * k + j;
        x = y + a2 - q;
        if (x < 1) {
            coordinate.batchId = b;
            coordinate.s1Idx = x + m;
            coordinate.s2Idx = y + m;
            return;
        }

        w = Ceil<int64_t>(x, m);
        x = x % m;
        if (x == 0) {
            x = m;
        }
        y = x + q - a2;
        if (w == b2 && y > m) {
            return;
        }
        if (y >= 1 && y <= n) {
            coordinate.batchId = w + b1 * k;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }
        return;
    }
    return;
}

template <bool isAlign = false>
__aicore__ inline int64_t GetPrefixByBidx(const __gm__ uint8_t *actualSeqQlenAddr,
                                          const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix,
                                          int64_t bIdx, int64_t step)
{
    int64_t w = bIdx / step;

    int64_t prefixNum = prefix[w];
    if (bIdx % step > 0) {
        w = w * step;
        int64_t lastSeqQLen = w == 0 ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[w - 1];
        int64_t lastSeqKvLen = w == 0 ? 0 : ((__gm__ int64_t *)actualSeqKvlenAddr)[w - 1];
        int64_t currentSeqQLen, currentSeqKvLen;
        while (w < bIdx) {
            currentSeqQLen = ((__gm__ int64_t *)actualSeqQlenAddr)[w];
            currentSeqKvLen = ((__gm__ int64_t *)actualSeqKvlenAddr)[w];
            prefixNum += (currentSeqQLen - lastSeqQLen) *
                         (isAlign ? AlignTo16(currentSeqKvLen - lastSeqKvLen) : (currentSeqKvLen - lastSeqKvLen));
            w += 1;
            lastSeqQLen = currentSeqQLen;
            lastSeqKvLen = currentSeqKvLen;
        }
    }
    return prefixNum;
}

__aicore__ inline void GetSeqQlenKvlenByBidx(const __gm__ uint8_t *actualSeqQlenAddr,
                                             const __gm__ uint8_t *actualSeqKvlenAddr, int64_t bIdx,
                                             int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx - 1];
    }
}

__aicore__ inline void CalVirtualIndex(int64_t flag, int64_t &m, int64_t &n)
{
    if (m < n) {
        n = m;
    }
    if (flag == 0) {
        m = NUM_TWO * m - n + 1;
    } else if (flag == 1) {
        m = m - (n + 1) / NUM_TWO + 1;
        n = n / NUM_TWO;
    } else {
        m = m - n / NUM_TWO;
        n = (n + 1) / NUM_TWO;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool isDense, const bool IS_N_EQUAL>
__aicore__ inline void UpdateMNPQ(int64_t actualCalcS1Token, int64_t actualCalcS2Token, CoordinateInfo &coordinateInfo,
                                  int64_t &actualM, int64_t &actualN)
{
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    int64_t actualS1Len = coordinateInfo.actualS1Len;
    int64_t actualS2Len = coordinateInfo.actualS2Len;
    if (coordinateInfo.sparseMode == BAND || coordinateInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }

    int64_t m = coordinateInfo.s1Outer;
    int64_t n = coordinateInfo.s2Outer;
    int64_t p = Ceil<int64_t>(actualCalcS1Token, CUBE_BASEM) + 1;
    int64_t q = Ceil<int64_t>(actualCalcS2Token, CUBE_BASEN) + 1;
    p = p > m ? m : p;
    q = q > n ? n : q;

    // 负数场景变换
    if (p < 0) {
        coordinateInfo.mOffset = 0;
        coordinateInfo.nOffset = -p;
        n = n + p;
        q = p + q;
        p = 1;
    } else if (q < 0) {
        coordinateInfo.mOffset = -q;
        coordinateInfo.nOffset = 0;
        m = m + q;
        p = p + q;
        q = 1;
    } else {
        coordinateInfo.mOffset = 0;
        coordinateInfo.nOffset = 0;
    }

    if (p + q <= m) {
        int64_t L1 = q - 1;
        int64_t L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        int64_t L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));

        if (L3 == 0) {
            actualM = p + q + L2 - NUM_TWO;
        } else {
            actualM = m;
        }
        actualN = L1 + L2 + L3;
    } else {
        actualM = m;
        actualN = Min(m - 1 + q, n);
    }

    coordinateInfo.m = actualM;
    coordinateInfo.n = actualN;
    coordinateInfo.p = p;
    coordinateInfo.q = q;

    if constexpr (!isDense) {
        return;
    }

    if (p + q <= actualM) {
        if (actualN > actualM || (IS_N_EQUAL && actualN == actualM)) {
            actualN = p + q - 1;
        } else {
            actualM = p + q - 1;
        }
    } else {
        if (p + q <= actualN) {
            actualN = p + q - 1;
        }
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDDenseSwizzleIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                               const __gm__ uint8_t *actualSeqKvlenAddr,
                                               const __gm__ uint64_t *tndS2BlockPrefixSum, int64_t b, int64_t n2,
                                               int64_t g, int64_t j, int64_t k, int64_t r,
                                               CoordinateInfo &coordinateInfo)
{
    j -= 1;
    r -= 1;
    coordinateInfo.batchId = -1;
    int64_t n1 = n2 * g;
    for (int64_t bIdx = 0; bIdx < b; bIdx++) {
        if (r < tndS2BlockPrefixSum[bIdx + 1]) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            int64_t batchPrefix = static_cast<int64_t>(tndS2BlockPrefixSum[bIdx]);
            int64_t delta = r - batchPrefix;
            int64_t linearIdx = delta / s1OuterTmp * k + j;
            if (linearIdx >= s2OuterTmp * n1) {
                break;
            }
            // Keep this coordinate order identical to Python mode 2: head-major S2, then rotate S1.
            int64_t n1Idx = linearIdx / s2OuterTmp;
            int64_t s2Idx = linearIdx % s2OuterTmp;
            int64_t s1Idx = (s2Idx + delta) % s1OuterTmp;
            coordinateInfo.actualS1Len = actualS1Len;
            coordinateInfo.actualS2Len = actualS2Len;
            coordinateInfo.s1Outer = s1OuterTmp;
            coordinateInfo.s2Outer = s2OuterTmp;
            coordinateInfo.batchId = bIdx + 1;
            coordinateInfo.n2Idx = n1Idx / g + 1;
            coordinateInfo.gIdx = 1;
            coordinateInfo.s1Idx = s1Idx + 1;
            coordinateInfo.s2Idx = s2Idx + 1;
            break;
        }
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const uint8_t DETER_SPARSE_TYPE = DETER_DENSE,
          const bool IS_N_EQUAL = true>
__aicore__ inline void CalTNDDenseIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                        const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix,
                                        int64_t deterMaxRound, int64_t b, int64_t n2, int64_t g, int64_t j, int64_t r,
                                        uint8_t flag, int64_t step, CoordinateInfo &coordinateInfo)
{
    coordinateInfo.batchId = -1;
    if (r > deterMaxRound) {
        return;
    }
    int64_t n1 = n2 * g;
    int64_t ID = (j - 1) * deterMaxRound + r;
    int64_t w = 0;
    int64_t batchId = 0;

    int64_t prefixMaxIndex = Ceil<int64_t>(b + 1, step);

    // GQA 需要将 b n2 g 转换为 b*n2 g
    if constexpr (!IS_N_EQUAL) {
        while (w * step < b && prefixMaxIndex > w + 1 && ID > prefix[w + 1] * n1) {
            w += 1;
        }
        batchId = w;
    } else {
        while ((w + 1) * step < b && ID > prefix[w + 1] * n1) {
            w += 1;
        }
        batchId = w;
    }

    int64_t delta = ID - prefix[batchId] * n1;

    w = w * step;
    if (w >= b) {
        return;
    }
    int64_t m, n, p, q;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, w, actualS1Len, actualS2Len);

    m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
    n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
    coordinateInfo.actualS1Len = actualS1Len;
    coordinateInfo.actualS2Len = actualS2Len;
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
    int64_t actualCalcS1Token = coordinateInfo.p;
    int64_t actualCalcS2Token = coordinateInfo.q;

    if constexpr (DETER_SPARSE_TYPE == DETER_BAND) {
        UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, true, IS_N_EQUAL>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m,
                                                             n);
        p = coordinateInfo.p;
        q = coordinateInfo.q;
    }

    if constexpr (DETER_SPARSE_TYPE == DETER_CAUSAL) {
        CalVirtualIndex(flag, m, n);
    }

    if (unlikely(step > 1)) {
        int64_t batchBaseNum = m * n * n1;
        while (delta > batchBaseNum && w < b) {
            delta = delta - batchBaseNum;
            w += 1;
            if (w >= b) {
                return;
            }
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, w, actualS1Len, actualS2Len);
            m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            coordinateInfo.s1Outer = m;
            coordinateInfo.s2Outer = n;
            if constexpr (DETER_SPARSE_TYPE == DETER_CAUSAL) {
                CalVirtualIndex(flag, m, n);
            } else if constexpr (DETER_SPARSE_TYPE == DETER_BAND) {
                coordinateInfo.actualS1Len = actualS1Len;
                coordinateInfo.actualS2Len = actualS2Len;
                UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, true, IS_N_EQUAL>(actualCalcS1Token, actualCalcS2Token,
                                                                     coordinateInfo, m, n);
                p = coordinateInfo.p;
                q = coordinateInfo.q;
            }

            batchBaseNum = m * n * n1;
        }
        coordinateInfo.actualS1Len = actualS1Len;
        coordinateInfo.actualS2Len = actualS2Len;
    }
    int64_t currentBaseNum = m * n;
    batchId = w + 1;

    int64_t x, y;
    if constexpr (IS_N_EQUAL) {
        int64_t deltaN = (delta - 1) / currentBaseNum + 1;
        delta = delta % currentBaseNum;
        delta = delta != 0 ? delta : currentBaseNum;
        int64_t gcd = Gcd(m, deterMaxRound);
        int64_t t1 = deterMaxRound / gcd;
        int64_t t2 = m / gcd;

        x = ((delta - 1) % m) + 1;
        y = (delta - 1) / m + 1;
        if (t1 < n) {
            int64_t nTail = n % t1;
            nTail = nTail == 0 ? t1 : nTail;
            if (y <= n - nTail) {
                int64_t delta_adj = Ceil<int64_t>(y, t1);
                delta += delta_adj;
                if (delta > delta_adj * t2 * deterMaxRound) {
                    delta -= t2 * deterMaxRound;
                }
                x = ((delta - 1) % m) + 1;
                y = (delta - 1) / m + 1;
            }
        }
        coordinateInfo.batchId = (batchId - 1) * n1 + deltaN;
    } else {
        currentBaseNum = currentBaseNum * g;
        int64_t deltaN = (delta - 1) / currentBaseNum + 1;
        delta = delta % currentBaseNum;
        delta = delta != 0 ? delta : currentBaseNum;

        int64_t m_new = m * g;
        int64_t gcd = Gcd(m_new, deterMaxRound);
        int64_t t1 = deterMaxRound / gcd;
        int64_t t2 = m_new / gcd;

        x = delta % m_new;
        x = x != 0 ? x : m_new;
        y = Ceil<int64_t>(delta, m_new);

        if (t1 < n) {
            int64_t nTail = (n % t1);
            nTail = nTail != 0 ? nTail : t1;
            if (y <= n - nTail) {
                int64_t delta_adj = Ceil<int64_t>(y, t1);
                delta += delta_adj;
                if (delta > delta_adj * t2 * deterMaxRound) {
                    delta -= t2 * deterMaxRound;
                }
                x = delta % m_new;
                x = x != 0 ? x : m_new;
                y = Ceil<int64_t>(delta, m_new);
            }
        }

        int64_t N1_id = Ceil<int64_t>(x, m);
        x = x % m;
        x = x != 0 ? x : m;
        coordinateInfo.batchId = batchId;
        coordinateInfo.n2Idx = deltaN;
        coordinateInfo.gIdx = N1_id;
    }
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalPosWholeBatch(int64_t m, int64_t n, int64_t a, CoordinateInfo &coordinateInfo)
{
    int64_t n1 = n / NUM_TWO * NUM_TWO;
    int64_t L = NUM_TWO * m - n1 + 1;
    int64_t rm_local = (n1 * L) / NUM_TWO;
    int64_t x, y;
    if (a <= rm_local) {
        y = Ceil<int64_t>(a, L);
        int64_t r1 = a % L;
        r1 = r1 != 0 ? r1 : L;
        x = r1 + y - 1;
        if (x > m) {
            x = NUM_TWO * m + 1 - x;
            y = n1 + 1 - y;
        }
    } else {
        int64_t a1 = a - rm_local;
        y = n;
        x = a1 - 1 + y;
    }
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
    return;
}

__aicore__ inline void UpdataCausalMN(CoordinateInfo &coordinateInfo, int64_t &m, int64_t &n)
{
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
    n = m < n ? m : n;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDCausalIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                         const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix0,
                                         const __gm__ int64_t *prefix1, const __gm__ int64_t *prefix2, int64_t b,
                                         int64_t N1, int64_t k, int64_t j, int64_t r, int64_t step,
                                         CoordinateInfo &coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t N10 = N1 / k;
    int64_t N11 = N1 % k / NUM_TWO;
    int64_t R01 = prefix0[maxRoundIndex];
    int64_t R02 = prefix0[maxRoundIndex + 1];
    int64_t R0 = R01 + R02;
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t R2 = prefix2[maxRoundIndex];
    coordinateInfo.batchId = -1;
    if (r <= R01) {
        int64_t a_judge = Ceil<int64_t>(r * NUM_TWO, N10);
        int64_t w = 0;
        while ((w + 1) * step < b && a_judge > prefix0[w + 1]) {
            w += 1;
        }
        int64_t batch_id = w * step;
        int64_t a = r - prefix0[w] * N10 / NUM_TWO;

        if (batch_id >= b) {
            return;
        }

        int64_t actualS1Len, actualS2Len, m, n;
        int64_t round_batch;

        GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, actualS1Len, actualS2Len);
        m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        UpdataCausalMN(coordinateInfo, m, n);
        round_batch = (NUM_TWO * m - n + 1) * n / NUM_TWO;

        while (a > round_batch * N10) {
            batch_id += 1;
            if (batch_id >= b) {
                return;
            }
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, actualS1Len, actualS2Len);
            m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            UpdataCausalMN(coordinateInfo, m, n);
            a = a - round_batch * N10;
            round_batch = (NUM_TWO * m - n + 1) * n / NUM_TWO;
        }

        int64_t a0 = a % round_batch;
        a0 = a0 != 0 ? a0 : round_batch;
        CalCausalPosWholeBatch(m, n, a0, coordinateInfo);
        int64_t x = coordinateInfo.s1Idx;
        int64_t y = coordinateInfo.s2Idx;

        w = (a - 1) / round_batch * k + j;
        // b1组，每组k个batch行交错
        w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        batch_id = batch_id * N1 + w;
        coordinateInfo.batchId = batch_id;
        coordinateInfo.actualS1Len = actualS1Len;
        coordinateInfo.actualS2Len = actualS2Len;
        return;
    } else if (r > R01 && r <= R0) {
        int64_t a = r - R01;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, R02, b,
                                                               N11, 1, j, a, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        int64_t batch_id1 = Ceil<int64_t>(coordinateInfo.batchId, N11);
        int64_t batch_id2 = coordinateInfo.batchId % N11;
        batch_id2 = batch_id2 != 0 ? batch_id2 : N11;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
            batch_id2 = NUM_TWO * batch_id2 - 1;
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
            batch_id2 = NUM_TWO * batch_id2;
        }
        coordinateInfo.batchId = (batch_id1 - 1) * N1 + N10 * k + batch_id2;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (R0 < r && r <= R0 + R1) {
        int64_t a = r - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1, b, 1,
                                                               1, j, a, 1, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
        }
        coordinateInfo.batchId = coordinateInfo.batchId * N1;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else {
        int64_t a = r - R1 - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix2, R2, b, 1,
                                                               1, j, a, NUM_TWO, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer};
        n = m < n ? m : n;
        coordinateInfo.s1Idx += n / NUM_TWO;

        coordinateInfo.batchId = coordinateInfo.batchId * N1;
        coordinateInfo.s1Idx = coordinateInfo.s1Idx;
        return;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDGQACausalIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                            const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix0,
                                            const __gm__ int64_t *prefix1, const __gm__ int64_t *prefix2, int64_t b,
                                            int64_t g, int64_t N2, int64_t k, int64_t j, int64_t r, int64_t step,
                                            bool coreDivide, CoordinateInfo &coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t N2_1 = N2 / NUM_TWO;
    int64_t R0 = prefix0[maxRoundIndex];
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t R2 = prefix2[maxRoundIndex];

    coordinateInfo.batchId = -1;
    bool isFlag0{false}, isFlag1{false}, isFlag2{false};
    int64_t k2 = Ceil<int64_t>(k, N2);
    int64_t k1 = k - k2;
    int64_t curRoundId{0}, curCoreId{j};
    if ((!coreDivide || (coreDivide && (1 <= j && j <= k1))) && r <= R0) {
        isFlag0 = true;
        curRoundId = r;
    } else if (((!coreDivide && R0 < r && r <= R0 + R1) || (coreDivide && k1 < j && j <= k && r <= R1))) {
        isFlag1 = true;
        curRoundId = coreDivide ? r : r - R0;
        curCoreId = coreDivide ? j - k1 : j;
    } else if (!coreDivide || (coreDivide && k1 < j && j <= k)) {
        isFlag2 = true;
        curRoundId = coreDivide ? r - R1 : r - R0 - R1;
        curCoreId = coreDivide ? j - k1 : j;
    }

    if (isFlag0) {
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL, false>(
            actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, R0, b, N2_1, g, j, curRoundId, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
            coordinateInfo.n2Idx = NUM_TWO * coordinateInfo.n2Idx - 1;
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
            coordinateInfo.n2Idx = NUM_TWO * coordinateInfo.n2Idx;
        }
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (isFlag1) {
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1,
                                                                      R1, b, 1, g, curCoreId, curRoundId, 1, step,
                                                                      coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t x_new;
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
        }
        coordinateInfo.n2Idx = N2;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (isFlag2) {
        int64_t a = r - R1 - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix2,
                                                                      R2, b, 1, g, curCoreId, curRoundId, NUM_TWO, step,
                                                                      coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer};
        n = m < n ? m : n;
        coordinateInfo.s1Idx += n / NUM_TWO;

        coordinateInfo.n2Idx = N2;
        coordinateInfo.s1Idx = coordinateInfo.s1Idx;
        return;
    } else {
        coordinateInfo.batchId = -1;
    }
}

__aicore__ inline int64_t BinarySearch(const __gm__ int64_t *prefix, int64_t b, int64_t rThreshold, int64_t step)
{
    int64_t w = 0;
    while ((w + 1) * step < b && rThreshold > prefix[w + 1]) {
        w += 1;
    }
    return w;
}

__aicore__ inline void CalPosWholeBatch(int64_t m, int64_t n, int64_t p, int64_t q, int64_t a,
                                        CoordinateInfo &coordinateInfo)
{
    int64_t L1, L2, L3, R1, R2, R3, Rm;
    int64_t x, y;
    if (p + q > m) {
        L1 = m - p;
        L2 = p + q - m;
        L3 = Min(m - 1, n - q);
        R1 = (p + m - 1) * L1 / NUM_TWO;
        R2 = m * L2;
        R3 = (NUM_TWO * m - 1 - L3) * L3 / NUM_TWO;
        Rm = R1 + R2 + R3;

        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2 && m != 0) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, m);
            x = a2 % m;
            if (x == 0) {
                x = m;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * m - L31 - 1;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1;
                if (x > m) {
                    y = L31 + 1 - y;
                    x = x - m + y;
                }
            } else {
                x = a3 - local_round + L3;
                y = L3;
            }
            y = y + L1 + L2;
        }
        coordinateInfo.s1Idx = x;
        coordinateInfo.s2Idx = y;
        return;
    } else {
        L1 = q - 1;
        L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        R1 = (NUM_TWO * p - NUM_TWO + q) * L1 / NUM_TWO;
        R2 = (p + q - 1) * L2;
        R3 = (p + q - NUM_TWO) * L3 - (L3 * (L3 - 1)) / NUM_TWO;
        Rm = R1 + R2 + R3;
        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                } else {
                    x = p + y - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, (p + q - 1));
            x = a2 % (p + q - 1) + (y - 1);
            if (x == y - 1) {
                x = (p + q - 1) + y - 1;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * (p + q) - L31 - NUM_THREE;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1 + 1 + m - (p + q);
                if (x > m) {
                    y = L31 + 1 - y;
                    x = x - (p + q - 1) + y;
                }
            } else {
                x = a3 - local_round + L3 + m - (p + q - 1);
                y = L3;
            }
            y = y + L1 + L2;
        }
        coordinateInfo.s1Idx = x;
        coordinateInfo.s2Idx = y;
        return;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDBandIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                       const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix0,
                                       const __gm__ int64_t *prefix1, int64_t b, int64_t N1, int64_t k, int64_t j,
                                       int64_t r, int64_t step, CoordinateInfo &coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t R0 = prefix0[maxRoundIndex];
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t N11 = N1 / k;
    int64_t N12 = N1 % k;

    int64_t m, n, p, q, x, y;
    if (r <= R0) {
        int64_t batch_id = BinarySearch(prefix0, b, r, step);
        int64_t a = r - prefix0[batch_id];
        int64_t round_batch = 0;
        batch_id = batch_id * step - 1;
        int64_t actualCalcS1Token = coordinateInfo.p;
        int64_t actualCalcS2Token = coordinateInfo.q;
        do {
            batch_id++;
            a = a - round_batch * N11;
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, coordinateInfo.actualS1Len,
                                  coordinateInfo.actualS2Len);
            coordinateInfo.s1Outer = (coordinateInfo.actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            coordinateInfo.s2Outer = (coordinateInfo.actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            // 重新计算m、n、p、q
            UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, false, true>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m, n);
            p = coordinateInfo.p;
            q = coordinateInfo.q;
            round_batch = m * n - (m - p) * (m - p + 1) / NUM_TWO - (n - q) * (n - q + 1) / NUM_TWO;
        } while (batch_id < b && a > round_batch * N11);

        int64_t a0 = a % round_batch;
        a0 = a0 != 0 ? a0 : round_batch;
        CalPosWholeBatch(m, n, p, q, a0, coordinateInfo);
        y = coordinateInfo.s2Idx;
        // 错位
        int64_t w = (a - 1) / round_batch * k + j;
        // b1组，每组k个batch行交错
        coordinateInfo.batchId = batch_id * N1 + w;
        return;
    } else if (R0 < r && r <= R0 + R1) {
        int64_t a = r - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_BAND>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1, b, N12,
                                                             1, j, a, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        m = coordinateInfo.m;
        n = coordinateInfo.n;
        p = coordinateInfo.p;
        q = coordinateInfo.q;
        x = coordinateInfo.s1Idx;
        y = coordinateInfo.s2Idx;
        int64_t batch_id = coordinateInfo.batchId;

        int64_t batch_id2 = batch_id % N12;
        batch_id2 = batch_id2 != 0 ? batch_id2 : N12;
        batch_id = (Ceil<int64_t>(batch_id, N12) - 1) * N1 + N11 * k + batch_id2;

        if (p + q <= m) {
            if (n >= m) {
                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinateInfo.batchId = batch_id;
                } else {
                    y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;

                } else {
                    x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                }
                return;
            }
        } else {
            if (p + q <= n) {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                } else if (y < x - p + 1 && y + p + q - 1 <= n) {
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s2Idx = y + p + q - 1;
                } else {
                    coordinateInfo.batchId = -1;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    return;
                }
            }
        }
    }
    coordinateInfo.batchId = -1;
    return;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDGQABandIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                          const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix0,
                                          const __gm__ int64_t *prefix1, int64_t b, int64_t g, int64_t N2, int64_t k,
                                          int64_t j, int64_t r, int64_t step, CoordinateInfo &coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t R1 = prefix1[maxRoundIndex];

    int64_t m, n, p, q, x, y;
    if (r <= R1) {
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_BAND, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1,
                                                                    b, N2, g, j, r, 0, step, coordinateInfo);

        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t batch_id = coordinateInfo.batchId;
        m = coordinateInfo.m;
        n = coordinateInfo.n;
        p = coordinateInfo.p;
        q = coordinateInfo.q;
        x = coordinateInfo.s1Idx;
        y = coordinateInfo.s2Idx;

        if (p + q <= m) {
            if (n > m) {
                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinateInfo.batchId = batch_id;
                } else {
                    y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    coordinateInfo.batchId = batch_id;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;

                } else {
                    x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    coordinateInfo.batchId = batch_id;
                }
                return;
            }
        } else {
            if (p + q <= n) {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                } else if (y < x - p + 1 && y + p + q - 1 <= n) {
                    coordinateInfo.s2Idx = y + p + q - 1;
                    coordinateInfo.batchId = batch_id;
                } else {
                    coordinateInfo.batchId = -1;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    coordinateInfo.batchId = batch_id;
                    return;
                }
            }
        }
    }
    coordinateInfo.batchId = -1;
    return;
}

__aicore__ inline void CalGQABandIndex(const BandInfo &bandInfo, int64_t j, int64_t r, int64_t N1,
                                       CoordinateInfo &coordinate)
{
    int64_t w, x, y = 0;
    coordinate.batchId = -1;

    int64_t k = bandInfo.k;
    int64_t m = bandInfo.m;
    int64_t n = bandInfo.n;
    int64_t p = bandInfo.p;
    int64_t q = bandInfo.q;
    int64_t b = bandInfo.b;
    int64_t L1 = bandInfo.L1;
    int64_t L2 = bandInfo.L2;
    int64_t L3 = bandInfo.L3;
    int64_t Rm = bandInfo.Rm;
    int64_t rm1 = bandInfo.rm;
    int64_t Rm_group = Rm * N1;
    int64_t a, b_id, N1_id;

    int64_t b1 = b / k;
    int64_t b2 = b % k;
    if (r <= rm1) {
        a = r % Rm_group;
        a = a != 0 ? a : Rm_group;
        N1_id = Ceil<int64_t>(a, Rm);
        a = a % Rm;
        a = a != 0 ? a : Rm;
        b_id = k * (Ceil<int64_t>(r, Rm_group) - 1) + j;
        // 计算最终坐标
        CalPosWholeBatch(m, n, p, q, a, coordinate);
        x = coordinate.s1Idx;
        y = coordinate.s2Idx;
        b_id = ((b_id - 1) / k) * k + ((y - 1 + (b_id - 1)) % k) + 1;
        w = (b_id - 1) * N1 + N1_id;
        coordinate.batchId = w;
        return;
    }

    // 第二部分
    a = r - rm1;
    if (b2 == 0) {
        return;
    }
    if (p + q > m) {
        int64_t R1 = bandInfo.R1;
        int64_t R2 = bandInfo.R2;
        // 对称性好的情况特殊处理
        if (NUM_TWO * b2 == k && L1 == L3 && m > 1 && L1 > 0) {
            b_id = (j + 1) / NUM_TWO;
            if (L2 % NUM_TWO == 1) {
                int64_t R0 = Max(NUM_TWO * p, m);
                R0 = m == 1 ? 1 : R0;
                // 此处1-R0的轮次优先处理
                if (a <= R0 * N1) {
                    int64_t N1_id = Ceil<int64_t>(a, R0);
                    a = a % R0;
                    a = a != 0 ? a : R0;
                    if (j % NUM_TWO == 1) {
                        if (n >= NUM_THREE) {
                            if (a <= p) {
                                coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                                coordinate.s1Idx = a;
                                coordinate.s2Idx = 1;
                                return;
                            } else if (p < a && a <= NUM_TWO * p) {
                                coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                                coordinate.s1Idx = L3 + a - p;
                                coordinate.s2Idx = n;
                                return;
                            }
                            return;
                        }
                    } else {
                        y = (n + 1) / NUM_TWO;
                        if (a <= p) {
                            coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                            coordinate.s1Idx = a + m - p;
                            coordinate.s2Idx = y;
                            return;
                        } else {
                            x = a - p;
                            if (x > m - p) {
                                coordinate.batchId = -1;
                                return;
                            } else {
                                coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                                coordinate.s1Idx = x;
                                coordinate.s2Idx = y;
                                return;
                            }
                        }
                    }
                }
                a -= (R0 - p) * N1;
                R2 -= m;
            }

            if (a <= R1 * N1) {
                if (L2 % NUM_TWO == 1) {
                    N1_id = Ceil<int64_t>((a - p * N1), (R1 - p));
                    a = (a - p * N1) % (R1 - p);
                    a = a != 0 ? a : (R1 - p);
                    a += p;
                } else {
                    N1_id = Ceil<int64_t>(a, R1);
                    a = a % R1;
                    a = a != 0 ? a : R1;
                }

                int64_t L11 = L1 / 2 * 2;
                int64_t L = 2 * p + L11 - 1;
                int64_t local_round = L11 * L / 2;
                if (a <= local_round) {
                    y = Ceil<int64_t>(a, L);
                    int64_t r1 = a % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = p + y - r1;
                    if (x < 1) {
                        y = L11 + 1 - y;
                        x = 1 - x;
                    } else {
                        x = p + y - x;
                    }
                } else {
                    x = a - local_round;
                    y = L1;
                }
            } else if (a <= (R1 + (R2 / NUM_TWO)) * N1) {
                a -= R1 * N1;
                int64_t R_local = R2 / 2;
                N1_id = Ceil<int64_t>(a, R_local);
                a = a % R_local;
                a = a != 0 ? a : R_local;

                y = Ceil<int64_t>(a, m);
                x = a % m;
                if (x == 0) {
                    x = m;
                }
                y = y + L1;
            } else {
                return;
            }

            if (j % NUM_TWO == 0) {
                y = n + 1 - y;
                if (y <= q) {
                    x = x + 1;
                    if (x > m) {
                        x = x - m;
                    }
                } else {
                    x = L3 + y + x - n;
                }
            }
            coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }

        // 拼接
        if (p + q <= n) {
            CalGQADenseIndex(k, m, p + q - 1, b2, j, a, N1, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            w = coordinate.batchId;
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
            if (x - p + 1 <= y && y <= x + q - 1) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            } else if (y < x - p + 1 && 1 <= y + p + q - 1 && y + p + q - 1 <= n) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y + p + q - 1;
                return;
            } else {
                coordinate.batchId = -1;
                return;
            }
        } else {
            CalGQADenseIndex(k, m, n, b2, j, a, N1, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            w = coordinate.batchId;
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;

            if (x - p + 1 <= y && y <= x + q - 1) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            } else {
                coordinate.batchId = -1;
                return;
            }
        }
    } else {
        if (n > m) {
            int64_t r_case1 = bandInfo.R1;
            int64_t r_case2 = bandInfo.R2;
            if (r_case2 <= r_case1) {
                CalGQADenseIndex(k, m, p + q - 1, b2, j, a, N1, coordinate);
                if (coordinate.batchId == -1) {
                    return;
                }
                w = coordinate.batchId;
                x = coordinate.s1Idx;
                y = coordinate.s2Idx;

                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinate.batchId = b1 * k * N1 + w;
                    coordinate.s1Idx = x;
                    coordinate.s2Idx = y;
                    return;
                } else {
                    y = Ceil<int64_t>((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    if (1 <= y && y <= n) {
                        coordinate.batchId = b1 * k * N1 + w;
                        coordinate.s1Idx = x;
                        coordinate.s2Idx = y;
                        return;
                    } else {
                        coordinate.batchId = -1;
                        return;
                    }
                }
            } else {
                CalGQADenseIndexNoTune(k, p + q - 1, n, b2, j, a, N1, coordinate);
                if (coordinate.batchId == -1) {
                    return;
                }
                w = coordinate.batchId;
                x = coordinate.s1Idx;
                y = coordinate.s2Idx;

                x = x + y - q;
                if (1 <= x && x <= m) {
                    coordinate.batchId = b1 * k * N1 + w;
                    coordinate.s1Idx = x;
                    coordinate.s2Idx = y;
                    return;
                } else {
                    coordinate.batchId = -1;
                    return;
                }
            }
        } else {
            CalGQADenseIndex(k, p + q - 1, n, b2, j, a, N1, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            w = coordinate.batchId;
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;

            if (x - p + 1 <= y && y <= x + q - 1) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            } else {
                x = Ceil<int64_t>((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                if (1 <= x && x <= m) {
                    coordinate.batchId = b1 * k * N1 + w;
                    coordinate.s1Idx = x;
                    coordinate.s2Idx = y;
                    return;
                } else {
                    coordinate.batchId = -1;
                    return;
                }
            }
        }
    }
    coordinate.batchId = -1;
    return;
}
__aicore__ inline bool CalTNDDenseLine(int64_t k, int64_t m, int64_t n, int64_t copies, int64_t j, int64_t r,
                                       int64_t &w, int64_t &x, int64_t &y)
{
    if (j < 1 || j > k || r < 1 || copies <= 0 || m <= 0 || n <= 0) {
        return false;
    }
    const int64_t delta = r - 1;
    const int64_t y1 = (delta / m) * k + j;
    if (y1 > n * copies) {
        return false;
    }
    w = (y1 - 1) / n + 1;
    y = (y1 - 1) % n + 1;
    x = (y + delta - 1) % m + 1;
    return true;
}

__aicore__ inline void CalTNDG2k(int64_t k, int64_t j, int64_t a, int64_t l1, int64_t offset, int64_t &x, int64_t &y)
{
    if (j % NUM_TWO == 1) {
        if (a <= l1 - j + 1) {
            y = j + offset;
            x = y + a - 1;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + NUM_TWO * l1 - NUM_TWO * k + 1 - a;
        }
    } else {
        if (a >= l1 - NUM_TWO * k + 1 + j) {
            y = j + offset;
            x = y + NUM_TWO * l1 - NUM_TWO * k + 1 - a;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + a - 1;
        }
    }
}

__aicore__ inline bool CalTNDCase0(int64_t m, int64_t n, int64_t j, int64_t r, int64_t &x, int64_t &y)
{
    if (j > (n / NUM_TWO) + 1 || j < 1 || r < 1) {
        return false;
    }
    if (j % NUM_TWO == 1) {
        if (r + j <= n + 1) {
            x = r + j - 1;
            y = j;
        } else {
            x = NUM_TWO * n + NUM_TWO - j - r;
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    } else {
        if (j <= r + 1 - (n % NUM_TWO)) {
            x = n + j - r - 1 + n % NUM_TWO;
            y = j;
        } else {
            x = n + NUM_TWO + r - j - (n % NUM_TWO);
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    }
    return y >= 1 && y <= m && y <= x && x <= m;
}

__aicore__ inline bool CalTNDCase0Rec(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r, int64_t &x, int64_t &y)
{
    if (NUM_TWO * k < m + 1 && k < n) {
        CalTNDG2k(k, j, r, m, 0, x, y);
    } else {
        if (!CalTNDCase0(m, m, j, r, x, y)) {
            return false;
        }
    }
    return y >= 1 && y <= n && y <= x && x <= m;
}

__aicore__ inline int64_t CalTNDCase0RecRounds(int64_t k, int64_t m, int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    if (NUM_TWO * k < m + 1 && k < n) {
        return Max(static_cast<int64_t>(0), NUM_TWO * m - NUM_TWO * k + 1);
    }
    return m;
}

__aicore__ inline int64_t CalTNDSingleCausalRounds(int64_t k, int64_t m, int64_t n)
{
    const int64_t t = n / (NUM_TWO * k);
    const int64_t ell = n % (NUM_TWO * k);
    const int64_t rounds2k = (NUM_TWO * m + 1) * t - NUM_TWO * k * t * t;
    if (ell == 0) {
        return rounds2k;
    }
    if (t == 0) {
        return CalTNDCase0RecRounds(k, m, n);
    }
    const int64_t l = m - NUM_TWO * k * t;
    const int64_t phaseCount = Ceil<int64_t>(ell, k);
    int64_t roundsTail = 0;
    for (int64_t s = 0; s < phaseCount; ++s) {
        roundsTail += Max(static_cast<int64_t>(0), l - s * k);
    }
    return rounds2k + roundsTail;
}

__aicore__ inline bool CalTNDSingleCausalPos(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r, int64_t &x,
                                             int64_t &y)
{
    if (j < 1 || j > k || r < 1) {
        return false;
    }
    const int64_t t = n / (NUM_TWO * k);
    const int64_t ell = n % (NUM_TWO * k);
    const int64_t bound2k = (NUM_TWO * m + 1) * t - NUM_TWO * k * t * t;
    if (r <= bound2k) {
        int64_t a = r;
        int64_t offset = 0;
        for (int64_t i = 0; i < t; ++i) {
            const int64_t l = m - offset;
            const int64_t blockRounds = NUM_TWO * l - NUM_TWO * k + 1;
            if (a <= blockRounds) {
                CalTNDG2k(k, j, a, l, offset, x, y);
                return y >= 1 && y <= n && y <= x && x <= m;
            }
            a -= blockRounds;
            offset += NUM_TWO * k;
        }
        return false;
    }
    int64_t a = r - bound2k;
    if (ell == 0) {
        return false;
    }
    const int64_t offset = NUM_TWO * k * t;
    const int64_t l = m - offset;
    if (t == 0) {
        return CalTNDCase0Rec(k, m, n, j, a, x, y);
    }
    const int64_t phaseCount = Ceil<int64_t>(ell, k);
    for (int64_t s = 0; s < phaseCount; ++s) {
        const int64_t phaseRounds = Max(static_cast<int64_t>(0), l - s * k);
        if (a <= phaseRounds) {
            const int64_t yLocal = s * k + j;
            const int64_t xLocal = yLocal + a - 1;
            if (yLocal >= 1 && yLocal <= ell && yLocal <= xLocal && xLocal <= l) {
                x = offset + xLocal;
                y = offset + yLocal;
                return true;
            }
            return false;
        }
        a -= phaseRounds;
    }
    return false;
}

__aicore__ inline bool IsTNDRightDownValid(int64_t m, int64_t n, int64_t x, int64_t y)
{
    return x >= 1 && x <= m && y >= 1 && y <= n && y <= x + n - m + 1;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void LoadTNDOuterMN(const __gm__ uint8_t *actualSeqQlenAddr, const __gm__ uint8_t *actualSeqKvlenAddr,
                                      int64_t batchType, int64_t &m, int64_t &n, CoordinateInfo &coordinateInfo)
{
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batchType, actualS1Len, actualS2Len);
    if (actualS1Len <= 0 || actualS2Len <= 0) {
        m = 0;
        n = 0;
    } else {
        m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        if (m < 1) {
            m = 1;
        }
        if (n < 1) {
            n = 1;
        }
    }
    coordinateInfo.actualS1Len = actualS1Len;
    coordinateInfo.actualS2Len = actualS2Len;
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
}

__aicore__ inline int64_t GetTNDRightDownRowShift(int64_t m, int64_t n)
{
    return Max(static_cast<int64_t>(0), m - n - 1);
}

__aicore__ inline void TrimTNDCausalMN(int64_t &m, int64_t &n, bool leftUp)
{
    if (leftUp) {
        if (m < n) {
            n = m;
        }
    } else {
        m -= GetTNDRightDownRowShift(m, n);
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void ExpandTNDCausalRun(const __gm__ uint8_t *actualSeqQlenAddr,
                                          const __gm__ uint8_t *actualSeqKvlenAddr, int64_t b, int64_t bIdx,
                                          bool leftUp, int64_t &runHead, int64_t &runEnd, int64_t &m, int64_t &n,
                                          CoordinateInfo &coordinateInfo)
{
    LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, m, n, coordinateInfo);
    TrimTNDCausalMN(m, n, leftUp);
    runHead = bIdx;
    while (runHead > 0) {
        int64_t prevM = 0;
        int64_t prevN = 0;
        CoordinateInfo prevInfo;
        LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead - 1, prevM, prevN,
                                               prevInfo);
        TrimTNDCausalMN(prevM, prevN, leftUp);
        if (prevM != m || prevN != n) {
            break;
        }
        runHead -= 1;
    }
    runEnd = bIdx;
    while (runEnd + 1 < b) {
        int64_t nextM = 0;
        int64_t nextN = 0;
        CoordinateInfo nextInfo;
        LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runEnd + 1, nextM, nextN,
                                               nextInfo);
        TrimTNDCausalMN(nextM, nextN, leftUp);
        if (nextM != m || nextN != n) {
            break;
        }
        runEnd += 1;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void LoadTNDBandMN(const __gm__ uint8_t *actualSeqQlenAddr, const __gm__ uint8_t *actualSeqKvlenAddr,
                                     int64_t bIdx, int64_t &m, int64_t &n, int64_t &p, int64_t &q,
                                     CoordinateInfo &coordinateInfo)
{
    LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, m, n, coordinateInfo);
    if (m < 1 || n < 1) {
        p = 0;
        q = 0;
        return;
    }
    int64_t actualP = coordinateInfo.p;
    int64_t actualQ = coordinateInfo.q;
    int64_t dummyM = 0;
    int64_t dummyN = 0;
    UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, false, true>(actualP, actualQ, coordinateInfo, dummyM, dummyN);
    p = coordinateInfo.p;
    q = coordinateInfo.q;
    m = coordinateInfo.m > 0 ? coordinateInfo.m : m;
    n = coordinateInfo.n > 0 ? coordinateInfo.n : n;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void ExpandTNDBandRun(const __gm__ uint8_t *actualSeqQlenAddr,
                                        const __gm__ uint8_t *actualSeqKvlenAddr, int64_t b, int64_t bIdx,
                                        int64_t &runHead, int64_t &runEnd, int64_t &m, int64_t &n, int64_t &p,
                                        int64_t &q, CoordinateInfo &coordinateInfo)
{
    const int64_t rawP = coordinateInfo.p;
    const int64_t rawQ = coordinateInfo.q;
    const int64_t rawSparseMode = coordinateInfo.sparseMode;
    LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, m, n, p, q, coordinateInfo);
    runHead = bIdx;
    while (runHead > 0) {
        int64_t prevM = 0;
        int64_t prevN = 0;
        int64_t prevP = 0;
        int64_t prevQ = 0;
        CoordinateInfo prevInfo;
        prevInfo.sparseMode = rawSparseMode;
        prevInfo.p = rawP;
        prevInfo.q = rawQ;
        LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead - 1, prevM, prevN, prevP,
                                              prevQ, prevInfo);
        if (prevM != m || prevN != n || prevP != p || prevQ != q) {
            break;
        }
        runHead -= 1;
    }
    runEnd = bIdx;
    while (runEnd + 1 < b) {
        int64_t nextM = 0;
        int64_t nextN = 0;
        int64_t nextP = 0;
        int64_t nextQ = 0;
        CoordinateInfo nextInfo;
        nextInfo.sparseMode = rawSparseMode;
        nextInfo.p = rawP;
        nextInfo.q = rawQ;
        LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runEnd + 1, nextM, nextN, nextP,
                                              nextQ, nextInfo);
        if (nextM != m || nextN != n || nextP != p || nextQ != q) {
            break;
        }
        runEnd += 1;
    }
}

__aicore__ inline int64_t GetTNDPrefixTailIndex(int64_t b, int64_t step)
{
    return b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
}

__aicore__ inline void GetTNDLeftUpVirt(int64_t m, int64_t n, int64_t &virtM, int64_t &virtN)
{
    if (m < n) {
        n = m;
    }
    virtM = NUM_TWO * m - n + 1;
    virtN = n;
}

__aicore__ inline void GetTNDRightDownVirt(int64_t m, int64_t n, int64_t &virtM, int64_t &virtN)
{
    m -= GetTNDRightDownRowShift(m, n);
    if (m < 1 || n < 1) {
        virtM = 0;
        virtN = 0;
        return;
    }
    virtM = m;
    virtN = NUM_TWO * n - m + NUM_THREE;
}

constexpr int64_t TND_LINE_TINY_AREA = 8;

__aicore__ inline bool TNDRunIsLineWorthy(int64_t k, int64_t virtM, int64_t virtN, int64_t pairCount)
{
    if (virtM <= 0 || virtN <= 0 || virtM * virtN <= TND_LINE_TINY_AREA) {
        return false;
    }
    return pairCount > 0 && virtN * pairCount >= k && virtM >= Min(k, virtN);
}

__aicore__ inline int64_t CalTNDBandWideCols(int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t nNew = Min(m - 1 + q, n);
    const int64_t l1 = m - p;
    const int64_t l2 = p + q - m;
    const int64_t l3 = nNew - l1 - l2;
    const int64_t foldCols = Max(static_cast<int64_t>(0), l3 - p + 1);
    return nNew - foldCols;
}

__aicore__ inline int64_t CalTNDBandNarrowRounds(int64_t k, int64_t n1, int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t l1 = q - 1;
    const int64_t l2 = Min(n - q + 1, m + NUM_TWO - p - q);
    const int64_t l3 = Max(static_cast<int64_t>(0), Min(p + n - m - 1, p + q - NUM_TWO));
    int64_t overlap = 0;
    int64_t nNew = l1 + l2 + l3;
    if (l3 != 0 && p <= l3) {
        overlap = Min(l3 - p + 1, l1);
    }
    const int64_t seg = p + q - 1;
    return seg * Ceil<int64_t>((nNew - overlap) * n1 + overlap, k);
}

__aicore__ inline int64_t CalTNDBandLineRounds(int64_t k, int64_t n1, int64_t m, int64_t n, int64_t p, int64_t q)
{
    if (p + q < NUM_TWO) {
        return 0;
    }
    if (p >= m) {
        return Ceil<int64_t>(n * n1, k) * m;
    }
    if (p + q <= m) {
        return CalTNDBandNarrowRounds(k, n1, m, n, p, q);
    }
    return Ceil<int64_t>(CalTNDBandWideCols(m, n, p, q) * n1, k) * m;
}

__aicore__ inline void MapTNDTriToBatch(int64_t tri, int64_t n1, int64_t runHead, int64_t &batchId)
{
    batchId = (runHead + tri / n1) * n1 + (tri % n1) + 1;
}

__aicore__ inline void MapTNDCopyToBatch(int64_t copyId, int64_t n1, int64_t runHead, int64_t &batchId)
{
    MapTNDTriToBatch(copyId - 1, n1, runHead, batchId);
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool LEFT_UP>
__aicore__ inline void BindTNDCausalMappedBatch(const __gm__ uint8_t *actualSeqQlenAddr,
                                                const __gm__ uint8_t *actualSeqKvlenAddr, int64_t runHead, int64_t n1,
                                                int64_t n1Id, int64_t x, int64_t y, CoordinateInfo &coordinateInfo)
{
    MapTNDTriToBatch(n1Id - 1, n1, runHead, coordinateInfo.batchId);
    int64_t realM = 0;
    int64_t realN = 0;
    LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead + (n1Id - 1) / n1, realM,
                                           realN, coordinateInfo);
    coordinateInfo.s1Idx = LEFT_UP ? x : (x + GetTNDRightDownRowShift(realM, realN));
    coordinateInfo.s2Idx = y;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void BindTNDBandMappedBatch(const __gm__ uint8_t *actualSeqQlenAddr,
                                              const __gm__ uint8_t *actualSeqKvlenAddr, int64_t runHead, int64_t n1,
                                              int64_t copyId, int64_t x, int64_t y, int64_t rawP, int64_t rawQ,
                                              CoordinateInfo &coordinateInfo)
{
    MapTNDCopyToBatch(copyId, n1, runHead, coordinateInfo.batchId);
    coordinateInfo.p = rawP;
    coordinateInfo.q = rawQ;
    int64_t dummyM = 0;
    int64_t dummyN = 0;
    int64_t dummyP = 0;
    int64_t dummyQ = 0;
    LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead + (copyId - 1) / n1, dummyM,
                                          dummyN, dummyP, dummyQ, coordinateInfo);
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
}

__aicore__ inline bool MapTNDLeftUp(int64_t m, int64_t n, int64_t pairId, int64_t pairBatchCount, int64_t &x,
                                    int64_t &y, int64_t &n1Id)
{
    if (x > m) {
        y = n + 1 - y;
        x = NUM_TWO * m + 1 - x;
        n1Id = NUM_TWO * pairId;
    } else if (x < y) {
        y = n + 1 - y;
        x = n - x;
        n1Id = NUM_TWO * pairId;
    } else {
        n1Id = NUM_TWO * pairId - 1;
    }
    return n1Id >= 1 && n1Id <= pairBatchCount && y >= 1 && y <= n && y <= x && x <= m;
}

__aicore__ inline bool MapTNDRightDown(int64_t m, int64_t n, int64_t pairId, int64_t pairBatchCount, int64_t &x,
                                       int64_t &y, int64_t &n1Id)
{
    const int64_t nExt = n + 1;
    if (y >= x + nExt - m + 1) {
        y = NUM_TWO * nExt - m - y + NUM_TWO;
        x = m + 1 - x;
        n1Id = NUM_TWO * pairId;
    } else {
        n1Id = NUM_TWO * pairId - 1;
    }
    return n1Id >= 1 && n1Id <= pairBatchCount && IsTNDRightDownValid(m, n, x, y);
}

__aicore__ inline bool MapTNDBandWide(int64_t m, int64_t n, int64_t p, int64_t q, int64_t &x, int64_t &y)
{
    const int64_t l1 = m - p;
    const int64_t l2 = p + q - m;
    const int64_t l3 = n - l1 - l2;
    const int64_t foldCols = Max(static_cast<int64_t>(0), l3 - p + 1);
    if (foldCols > 0 && y <= foldCols && x >= p + y) {
        y = n - foldCols + y;
    } else if (y <= l1 && x >= p + y) {
        return false;
    }
    if (y > l1 + l2 && x <= y - l1 - l2) {
        return false;
    }
    return x >= 1 && x <= m && y >= 1 && y <= n;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool LEFT_UP>
__aicore__ inline void CalTNDCausalLineRunIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                                const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix,
                                                int64_t b, int64_t n1, int64_t k, int64_t j, int64_t r, int64_t step,
                                                CoordinateInfo &coordinateInfo)
{
    coordinateInfo.batchId = -1;
    const int64_t batchTypeBase = BinarySearch(prefix, b, r, step);
    int64_t a = r - prefix[batchTypeBase];
    int64_t bIdx = batchTypeBase * step;
    if (bIdx >= b || a < 1) {
        return;
    }
    while (bIdx < b) {
        int64_t runHead = bIdx;
        int64_t runEnd = bIdx;
        int64_t m = 0;
        int64_t n = 0;
        ExpandTNDCausalRun<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, b, bIdx, LEFT_UP, runHead,
                                                   runEnd, m, n, coordinateInfo);
        const int64_t groupSize = runEnd - runHead + 1;
        if (bIdx > runHead) {
            bIdx = runEnd + 1;
            continue;
        }
        int64_t virtM = 0;
        int64_t virtN = 0;
        if (LEFT_UP) {
            GetTNDLeftUpVirt(m, n, virtM, virtN);
        } else {
            GetTNDRightDownVirt(m, n, virtM, virtN);
        }
        const int64_t total = n1 * groupSize;
        const int64_t runPairs = total / NUM_TWO;
        const int64_t pairSpace = runPairs * NUM_TWO;
        const int64_t pairRounds = (runPairs > 0 && virtN > 0) ? (Ceil<int64_t>(virtN * runPairs, k) * virtM) : 0;
        const int64_t singleRounds =
            (total % NUM_TWO == 1) ? (LEFT_UP ? CalTNDSingleCausalRounds(k, m, n) : (Ceil<int64_t>(n, k) * m)) : 0;
        const int64_t runRounds = pairRounds + singleRounds;
        if (a <= runRounds) {
            if (a <= pairRounds && runPairs > 0) {
                int64_t pairId = 0;
                int64_t x = 0;
                int64_t y = 0;
                const bool ok = CalTNDDenseLine(k, virtM, virtN, runPairs, j, a, pairId, x, y);
                int64_t n1Id = 0;
                const bool mapped = ok && (LEFT_UP ? MapTNDLeftUp(m, n, pairId, pairSpace, x, y, n1Id) :
                                                     MapTNDRightDown(m, n, pairId, pairSpace, x, y, n1Id));
                if (mapped) {
                    BindTNDCausalMappedBatch<CUBE_BASEM, CUBE_BASEN, LEFT_UP>(actualSeqQlenAddr, actualSeqKvlenAddr,
                                                                              runHead, n1, n1Id, x, y, coordinateInfo);
                }
                return;
            }
            const int64_t singleRound = a - pairRounds;
            int64_t x = 0;
            int64_t y = 0;
            bool ok = false;
            if (LEFT_UP) {
                ok = CalTNDSingleCausalPos(k, m, n, j, singleRound, x, y);
            } else {
                int64_t w = 0;
                ok = CalTNDDenseLine(k, m, n, 1, j, singleRound, w, x, y) && IsTNDRightDownValid(m, n, x, y);
            }
            if (ok) {
                BindTNDCausalMappedBatch<CUBE_BASEM, CUBE_BASEN, LEFT_UP>(actualSeqQlenAddr, actualSeqKvlenAddr,
                                                                          runHead, n1, total, x, y, coordinateInfo);
            }
            return;
        }
        a -= runRounds;
        bIdx = runEnd + 1;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDCausalLineIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                             const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix0,
                                             int64_t b, int64_t N1, int64_t k, int64_t j, int64_t r, int64_t step,
                                             int64_t sparseMode, CoordinateInfo &coordinateInfo)
{
    if (sparseMode == RIGHT_DOWN_CAUSAL) {
        CalTNDCausalLineRunIndex<CUBE_BASEM, CUBE_BASEN, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, b, N1,
                                                                k, j, r, step, coordinateInfo);
    } else {
        CalTNDCausalLineRunIndex<CUBE_BASEM, CUBE_BASEN, true>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, b, N1, k,
                                                               j, r, step, coordinateInfo);
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDBandLineIndex(const __gm__ uint8_t *actualSeqQlenAddr,
                                           const __gm__ uint8_t *actualSeqKvlenAddr, const __gm__ int64_t *prefix,
                                           int64_t b, int64_t n1, int64_t k, int64_t j, int64_t r, int64_t step,
                                           CoordinateInfo &coordinateInfo)
{
    coordinateInfo.batchId = -1;
    const int64_t rawP = coordinateInfo.p;
    const int64_t rawQ = coordinateInfo.q;
    const int64_t batchTypeBase = BinarySearch(prefix, b, r, step);
    int64_t a = r - prefix[batchTypeBase];
    int64_t bIdx = batchTypeBase * step;
    if (bIdx >= b || a < 1) {
        return;
    }

    while (bIdx < b) {
        int64_t runHead = bIdx;
        int64_t runEnd = bIdx;
        int64_t m = 0;
        int64_t n = 0;
        int64_t p = 0;
        int64_t q = 0;
        ExpandTNDBandRun<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, b, bIdx, runHead, runEnd, m, n,
                                                 p, q, coordinateInfo);
        const int64_t groupSize = runEnd - runHead + 1;
        if (bIdx > runHead) {
            bIdx = runEnd + 1;
            continue;
        }
        const int64_t copies = n1 * groupSize;
        const int64_t soloRounds = CalTNDBandLineRounds(k, copies, m, n, p, q);
        if (a > soloRounds) {
            a -= soloRounds;
            bIdx = runEnd + 1;
            continue;
        }

        if (p >= m) {
            int64_t w = 0;
            int64_t x = 0;
            int64_t y = 0;
            if (CalTNDDenseLine(k, m, n, copies, j, a, w, x, y) && w >= 1 && w <= copies && x >= 1 && x <= m &&
                y >= 1 && y <= n && y <= x + q - 1) {
                BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x,
                                                               y, rawP, rawQ, coordinateInfo);
            }
            return;
        }

        if (p + q > m) {
            const int64_t denseN = CalTNDBandWideCols(m, n, p, q);
            int64_t w = 0;
            int64_t x = 0;
            int64_t y = 0;
            if (CalTNDDenseLine(k, m, denseN, copies, j, a, w, x, y) && MapTNDBandWide(m, n, p, q, x, y) && w >= 1 &&
                w <= copies) {
                BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x,
                                                               y, rawP, rawQ, coordinateInfo);
            }
            return;
        }

        const int64_t l3 = Max(static_cast<int64_t>(0), Min(p + n - m - 1, p + q - NUM_TWO));
        const int64_t seg = p + q - 1;
        const int64_t a1 = Ceil<int64_t>(a, seg);
        int64_t a2 = a % seg;
        a2 = a2 != 0 ? a2 : seg;
        if (l3 == 0 || n - m < 1) {
            const int64_t idx = (a1 - 1) * k + j;
            const int64_t w = Ceil<int64_t>(idx, n);
            int64_t y = idx % n;
            y = y != 0 ? y : n;
            const int64_t x = y + a2 - q;
            if (w >= 1 && w <= copies && x >= 1 && x <= m && y >= 1 && y <= n) {
                BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x,
                                                               y, rawP, rawQ, coordinateInfo);
            }
            return;
        }
        const int64_t yAbs = (a1 - 1) * k + j;
        const int64_t xAbs = yAbs + a2 - q;
        if (xAbs < 1) {
            return;
        }
        const int64_t w = Ceil<int64_t>(xAbs, m);
        int64_t x = xAbs % m;
        x = x != 0 ? x : m;
        const int64_t y = x + q - a2;
        if (w >= 1 && w <= copies && x >= 1 && x <= m && y >= 1 && y <= n) {
            BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x, y,
                                                           rawP, rawQ, coordinateInfo);
        }
        return;
    }
}

} // namespace commondef
#endif // _FLASH_ATTENTION_SCORE_GRAD_DETER_H_
