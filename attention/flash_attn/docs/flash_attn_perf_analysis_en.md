# Flash Attention D=256 Performance Modeling: Selecting the Basic Block (M, N)

> A purely theoretical analysis: based on Ascend hardware specifications and the blocked computation model of Flash Attention, this document derives the basis for selecting the basic block (M, N) when D=256 (Q/K/V head dimension of 256, non-quantized fp16/bf16, dV = D).

## 1. Modeling Objective: Cube Bound

Flash Attention on Ascend consists of multiple engine pipelines:

- **Cube (Mmad)**: the two matrix multiplications bmm1 (Q·Kᵀ) and bmm2 (P·V);
- **MTE**: transferring Q/K/V from GM into L1, and the fractal transfers from L1 to L0A/L0B;
- **Fixpipe**: moving bmm results from L0C out to UB;
- **Vector**: softmax (row max/exp/sum) and online accumulation of the output.

Among these engines, **only the Cube computation is incompressible** — the FMA count of a matrix multiplication is uniquely determined by the problem size, while transfer-in, transfer-out, and softmax can all overlap with other stages. The performance upper bound of the kernel is therefore a fully loaded Cube pipeline, and the modeling objective is **Cube Bound**:

$$ T_{cube} \ge T_{mte2}, \qquad T_{cube} \ge T_{fixpipe}, \qquad T_{cube} \ge T_{vec} $$

That is, the Cube always has data to compute, and the time of all other stages is completely shadowed by the Cube time. The Vector-side softmax/accumulation runs on the Vector cores paired with the Cube; its workload is proportional to M(N+D) and is far smaller than $T_{cube}$ for practical block sizes, so it imposes no additional constraint. This document focuses on the two tightest conditions: **MTE2 (transfer-in)** and **Fixpipe (transfer-out)**.

## 2. Model Setup

### 2.1 Basic Block and Iteration Structure

- **M**: the number of block rows handled by each Cube core per iteration along the Q sequence direction (the M dimension of the bmm);
- **N**: the block size along the KV sequence direction (the N dimension of the bmm); one iteration processes N rows of KV;
- **D**: the head dimension (the K dimension of the bmm or the output column count of bmm2; dV = D when non-quantized).

One outer iteration (processing one KV block):

```
BMM1:   S_Q[M,D] × S_K[D,N] → scores[M,N]
softmax (online, maintaining row max/sum across iterations)
BMM2:   P'[M,N] × V[N,D] → partial output [M,D], accumulated online with previous values
```

Architecturally, one Cube core pairs with two Vector cores (1:2). The Cube computes M rows at a time, and each of the two Vector cores handles the softmax and accumulation for M/2 rows. **Vector-side buffers are counted by M/2 rows**.

### 2.2 Hardware Specifications (Single Core)

| Parameter | Value | Description |
| --- | --- | --- |
| CoreNum | 32 | Number of cores |
| Frequency | 1.6 GHz | Clock frequency |
| L1 | 512 KB | Residency of Q/P'/KV fragments |
| L0A / L0B | 64 KB | Fractal buffers for the left/right bmm matrices (2×32K sub-buffers each) |
| L0C | 256 KB | Accumulation/output buffer for bmm results (multiple rotating copies) |
| UB | 248 KB (usable) | Landing buffer for bmm results and all Vector-side buffers |
| L2 | 128 MB / 5.2 TB/s | Equivalent to ~101 B/cycle per core |
| HBM | 128 GB / 1.6 TB/s | Equivalent to ~31 B/cycle per core |

> Per-core bandwidth conversion: total bandwidth ÷ number of cores ÷ frequency. L2: 5.2×10¹² ÷ (32 × 1.6×10⁹) ≈ 101 B/cycle; HBM: 1.6×10¹² ÷ (32 × 1.6×10⁹) ≈ 31 B/cycle.

The Cube completes one `16×16×16` mma per cycle, for a throughput of **FMA = 4096 FMA/cycle** (model assumption).

## 3. Conditions for Achieving Cube Bound (Performance-Side Constraints)

### 3.1 Time Model

Counted by the **total work per iteration**. The two matmuls together require `2·M·N·D` FMAs:

$$ T_{cube} = \frac{2 M N D}{FMA} $$

The Q block (`M·D`) is transferred in only once during the first iteration, and the cost is amortized when S2 is large enough. K/V are transferred in as `2·N·D` per iteration:

$$ T_{mte2} \approx \frac{2 N D \cdot \text{sizeof}(kv)}{Mix_{bw}}, \qquad Mix_{bw} = h \cdot L2_{bw} + (1-h) \cdot HBM_{bw} $$

Here $h$ is the L2 hit rate, and $Mix_{bw} \in [31,\ 101]$ B/cycle. The bmm1/bmm2 results (fp32) are moved from L0C to UB by Fixpipe:

$$ T_{fixpipe} = \frac{(M N + M D) \cdot \text{sizeof}(fp32)}{UB_{bw}} $$

### 3.2 Condition 1: Lower Bound of M (Shadowing MTE2)

Canceling N and D on both sides of $T_{cube} > T_{mte2}$ (**independent of D and N**):

$$ M > \frac{2 \cdot FMA}{Mix_{bw}} = \frac{8192}{Mix_{bw}} $$

| Mix_bw | Lower bound of M |
| --- | --- |
| 101 (all L2 hits) | M > 81.1, that is **M ≥ 82** |
| 31 (all HBM misses) | M > 264.3, that is **M ≥ 265** |

That is: **when M < 82, the kernel is necessarily mte2 bound no matter how good the L2 hit rate is**. When M ∈ [82, 265), the L2 hit rate must satisfy

$$ h \ge \frac{8192/M - 31}{101 - 31} $$

The larger M is, the lower the requirement on h (M=96 requires 78%, M=128 requires 47%, and no requirement applies from M ≥ 265 onward).

### 3.3 Condition 2: Lower Bound of N (Shadowing Fixpipe)

Canceling M on both sides of $T_{cube} > T_{fixpipe}$ (**independent of M**):

$$ UB_{bw} > 2 \cdot FMA \cdot \frac{N+D}{N D} = 8192 \cdot \left(\frac{1}{D} + \frac{1}{N}\right) \quad\Longleftrightarrow\quad N > \frac{8192}{UB_{bw} - 8192/D} $$

At D=256, the condition becomes `UB_bw > 32 + 8192/N`: N=128 requires UB_bw > 96, and N=256 requires UB_bw > 64. The larger N is, the more KV rows dilute the bmm2 output (`M×D`, independent of N) and the smaller the fixpipe pressure becomes — **this condition directionally favors large N**. (UB_bw is not listed in the hardware table; see §8 for the pending measurement. If UB_bw is on the same level as the L2 bandwidth (~101), then N ≥ 119 is required, and N=128 satisfies it.)

### 3.4 Numerical Example (M=128, N=128, D=256)

| Item | Value | Cycles |
| --- | --- | --- |
| cube | 2×128×128×256 = 8.39M FMA | **2048** |
| KV transfer-in | 131072 B | h=100%: 1298 / h=47%: 2048 (critical) / h=0: 4228 |
| fixpipe transfer-out | (128²+128×256)×4 = 196608 B | UB_bw=96: 2048 (critical) / =128: 1536 |

Note that **at D=256, the fixpipe transfer-out volume (196.6K per iteration) exceeds the KV transfer-in volume (131.1K per iteration)**; fixpipe is the second-tightest constraint after MTE2. For comparison, at (M=64, N=256): `T_cube` is also 2048 (the same M·N), but KV transfer-in is 262144 B, which requires 2596 cycles > 2048 even at h=100% — a direct illustration that **M=64 < 82 is necessarily mte2 bound**.

## 4. Space Constraints (Capacity-Side Constraints)

Each level of the memory hierarchy must hold the working set of the basic block, and multi-buffer rotation must maintain pipeline overlap. The following derivation assumes the conventional design in which the bmm1 result buffer and the bmm2 result buffer are **allocated independently** (§7 relaxes this assumption).

### 4.1 L0C: bmm Result Buffer

L0C (256K) holds the bmm1 result (`M·N·4`) and the bmm2 result (`M·D·4`) in 4 rotating buffers of 64K each. The bmm2 result can be **chunked along the dV direction** to fit the buffer capacity, with the number of chunks $t = \lceil M D \cdot 4 / 64K \rceil$. To preserve Cube→Fixpipe overlap, the number of buffers consumed per iteration (1 + t) must be smaller than the total buffer count of 4. Therefore:

$$ M N \cdot 4 \le 64K \;\Rightarrow\; \mathbf{M N \le 16384}; \qquad t \le 2 \;\Rightarrow\; M D \le 32K \;\Rightarrow\; \mathbf{M \le 128}\ (D{=}256) $$

dV chunking does not change the total Mmad volume (the matrix multiplication accumulates by chunk). The cost is that the L0C rotation depth drops from 2.0 (t=1) to 4/3 (t=2), and the number of fixpipe calls doubles while the total volume stays unchanged.

### 4.2 L1: Input Fragment Residency

| Fragment | Buffering | Size |
| --- | --- | --- |
| Q block | Double buffering | `2 × M·D·2 = 4MD` |
| P' (softmax result, written back for bmm2) | Triple buffering | `3 × M·N·2 = 6MN` |
| K/V (time-multiplexed over the same buffer set) | At least double buffers | `≥ 2 × 2 × N·D·2 = 4ND` (more buffer copies mean deeper pipelining) |

$$ 4MD + 6MN + 4ND \le 512K $$

### 4.3 UB: Result Buffers and the Vector-Side Working Set

| Buffer | Size |
| --- | --- |
| bmm1 result (fp32, double buffering) | `2 × (M/2)·N·4 = 4MN` |
| bmm2 result (fp32, double buffering) | `2 × (M/2)·D·4 = 4MD` |
| P' (bf16, double buffering, plus 1 row of fractal-alignment padding) | `≈ 2 × (M/2)·N·2 = 2MN` |
| Output accumulation staging (fp32, single buffer) | `(M/2)·D·4 = 2MD` |
| Mask (bool, double buffering) | `2 × (M/2)·N·1 = MN` |
| softmax scalars (triple buffering for each of max/sum/exp) + lse (double buffering) + miscellaneous | `≈ 54M + 512` |

$$ 7MN + 6MD + 54M + 512 \le 248K\ (253952\,B) $$

### 4.4 L0A/L0B (Qualitative)

Each is 64K (2×32K sub-buffers). bmm inputs reside as 16×16 fractals. After the K direction (D=256) is chunked by 128, when M and N are both ≤ 128, each input fragment is exactly 32K and fills one sub-buffer, so the requirement is naturally satisfied. An overly large M or N would force the input side to degrade (single buffering or repeated loads). Therefore L0A/L0B does not get its own inequality, but it supports the upper bound of M, N ≤ 128.

## 5. Substituting D=256: The Feasible Region of (M, N)

Substituting D=256 into the constraints above (separate-buffer design):

| Constraint | Inequality (D=256) | N=128 | N=256 |
| --- | --- | --- | --- |
| Condition 1 (MTE2) | M > 8192/Mix_bw | M ≥ 82 (h=100%) to 265 (h=0) | Same as left |
| Condition 2 (Fixpipe) | N > 8192/(UB_bw−32) | Requires UB_bw > 96 | Requires UB_bw > 64 |
| L0C | MN ≤ 16384, M ≤ 128 | M ≤ 128 | **M ≤ 64** |
| L1 | 4MD+6MN+4ND ≤ 512K | M ≤ 219 | M ≤ 102 |
| UB (separate buffers) | 7MN+6MD+54M+512 ≤ 248K | **M ≤ 102** | M ≤ 74 |
| **Combined** | | **82 < M ≤ 102** | **M ≤ 64, no cube bound window** |

Two observations:

1. **N=256 is locked down by L0C**: `MN ≤ 16384` forces M ≤ 64 < 82, so no matter how the other constraints are relaxed, the kernel cannot reach the cube bound threshold.
2. **The window for N=128 is extremely narrow**: the feasible region (82, 102] hugs the threshold of 82. Even at the upper bound M=102, the L2 hit rate must be ≥ 70%, leaving minimal margin. Moreover, block sizes must align to the 16-element fractal, and practical choices are powers of two (M ∈ {64, 128}): M=128 exceeds the UB upper bound of 102 (311K > 248K), and M=64 falls below the threshold of 82. **Under the separate-buffer design, D=256 has no feasible cube bound solution at the conventional block sizes**.

The fallback is (64, 256): all space constraints are satisfied (UB 212K, L1 416K), but M=64 < 82, so the kernel is **doomed to be mte2 bound** — this is the theoretical root cause of why D=256 performs far worse than D=128.

## 6. Performance Impact of M and N Within the Feasible Region

The previous section gives inequalities (ranges rather than fixed values). How to pick values within these ranges depends on the individual performance impact of M and N.

### 6.1 M: The Larger the Better Within the Feasible Region

- **Arithmetic intensity**: the number of FMAs per KV byte is `M/2`, so M is the only arithmetic intensity knob. The larger M is, the looser Condition 1 becomes, the lower the required L2 hit rate (M=102 requires 70%, M=128 requires 47%), and the more stable the cube bound;
- **Q amortization**: the amortized burden of the Q transfer relative to the cube time is independent of M (both grow linearly with M), so it creates no preference;
- **Cost**: tail-block waste (the last block idles when the total number of Q-side rows is not divisible by M) and coarser task granularity. When Q-side rows are plentiful (prefill or long sequences), the cost is negligible. **When the total number of Q-side rows is smaller than M, large blocks idle extensively, and M must be reduced**. In that case M is below 82 anyway, cube bound is unreachable, and a different selection logic applies (see §8).

Conclusion: **when Q-side rows are plentiful, take M at the upper bound of the feasible region**.

### 6.2 N: Bidirectional Impact, Coupled with M on L0C

- **Benefits of large N**: Condition 2 (fixpipe) becomes looser; the number of KV iterations `S2/N` decreases, amortizing loop and synchronization overhead;
- **Benefits of small N**: the MN terms in UB/L0C and the N·D term in L1 all shrink with N; each K/V L1 fragment becomes smaller (64K per fragment at N=128, allowing deep rotation with 4 buffers; 128K per fragment at N=256, with only 2 buffers), deepening the overlap between MTE2 and Cube; tail-block waste is smaller when S2 is not divisible by N.

Key coupling: **the L0C constraint `MN ≤ 16384` binds M and N to the same budget** — each step up in N halves the upper limit of M. Condition 1 (M ≥ 82) is the only entry point to cube bound, while Condition 2 only needs to be met (N=128 already satisfies it when UB_bw ≥ 96). The selection order is therefore: **first secure M above the threshold and as large as possible, then choose N within the remaining budget such that Condition 2 holds**. At D=256, this means N=128 with M at the upper bound.

## 7. Discovery of Buffer Redundancy: Sharing the bmm1/bmm2 Results Expands the Feasible Region

### 7.1 Redundancy Argument

In §4.3, the bmm1 result buffer (4MN) and the bmm2 result buffer (4MD) each use independent double buffering, but their **lifecycles do not overlap**:

```
bmm1(L) result → softmax(L) finishes consuming → this buffer dies
bmm2(L) result → accumulate(L) finishes consuming → this buffer dies
```

The dataflow is a circular chain `bmm1 → softmax → bmm2 → accumulation → bmm1 (next round)`. Within the same iteration, softmax finishes reading the bmm1 result before bmm2 starts producing; the bmm2 result is not fully read by the accumulation until just before the next round of softmax. Therefore **the two pairs of double buffers can share the same pair of rotating buffers**, with the buffer capacity set to the larger of the two, `(M/2)·max(N,D)·4`. When D=256 ≥ N, the bmm1 result (`(M/2)·N·4`) fits entirely inside the bmm2 buffer — the bmm1 double buffers (4MN) disappear entirely.

### 7.2 New UB Space Formula

$$ \underbrace{4M \cdot \max(N,D)}_{\text{shared result buffer}} + 2MN + 2MD + MN + 54M + 512 \le 248K $$

When D=256 and N ≤ 256, $\max(N,D)=D$, and the formula simplifies to:

$$ \mathbf{3MN + 6MD + 54M + 512 \le 248K} \qquad (\text{original } 7MN + 6MD + \dots) $$

The saving equals 4MN (when N ≤ D). For (M=128, N=128): the saving is 64K, and the UB usage drops from 311K (over the limit) to **247.25K ≤ 248K (768B to spare)**.

### 7.3 The Expanded Feasible Region and a Better Choice

| Constraint | N=128 | N=256 |
| --- | --- | --- |
| UB (shared buffers) | **M ≤ 128** | M ≤ 107 |
| L0C / L1 | M ≤ 128 / M ≤ 219 | **M ≤ 64** / M ≤ 102 |
| **Combined** | **82 < M ≤ 128** | M ≤ 64 (still no window) |

This echoes the conclusions of §6:

- The window for N=128 expands from (82, 102] to **(82, 128]**, which happens to include the M=128 choice: the required L2 hit rate drops from 70% to 47%, and cube bound changes from barely meeting the line to having margin — **this is the quantitative demonstration that buffer sharing enables a better (M, N) choice**;
- The N=256 side is unchanged (locked at M ≤ 64 by the L0C constraint MN ≤ 16K), which further confirms the judgment in §6.2: under L0C coupling, N must yield to M;
- Sharing requires **allocating buffers by result order rather than by operator order**, and a circular wait must guarantee that the bmm1 write in round L occurs later than the completion of the bmm2 read in round L−2. In other words, the space gain needs the support of synchronization and does not come for free.

## 8. Conclusions

| Scenario | Choice | Rationale |
| --- | --- | --- |
| D=256, plentiful Q-side rows (prefill) | **M=128, N=128** | Falls at the upper bound of the feasible region (82, 128] after buffer sharing; an L2 hit rate ≥ ~47% suffices for cube bound; the bmm2 result is chunked by dV/128 to fit the L0C buffer |
| D=256, total Q-side rows < 128 (decode or small batch) | **M=64, N=256** | The workload forces M down, so cube bound is unreachable anyway (M < 82); large N halves the number of KV iterations, and (64,256) sits exactly against the `MN = 16K` L0C/mask boundary, wasting no space |

Summary of the modeling methodology:

1. **Performance conditions give lower bounds**: cube bound is equivalent to `M > 8192/Mix_bw` (82 to 265, independent of D/N) and `N > 8192/(UB_bw − 8192/D)`;
2. **Space constraints give upper bounds**: L0C `MN ≤ 16384` and `M ≤ 128` (after dV chunking); L1 `4MD+6MN+4ND ≤ 512K`; UB is determined by the result buffers and the Vector working set;
3. **Within the feasible region, larger M is better** (the only arithmetic intensity knob); N shares the L0C budget with M, and M takes priority;
4. **Buffer reuse expands the feasible region**: the bmm1/bmm2 result lifecycles are staggered and can share buffers, raising the UB upper bound from M ≤ 102 to M ≤ 128 and turning cube bound from marginal to achievable — space optimization ultimately translates into freed performance options.

**To be added**: the measured value of the UB bandwidth UB_bw, needed to close Condition 2; at D=256, the fixpipe transfer-out volume exceeds the KV transfer-in volume, which is worth verifying with dedicated profiling.
