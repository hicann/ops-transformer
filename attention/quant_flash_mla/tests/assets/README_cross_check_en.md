# QuantFlashMla: Three-Way cross_check Comparison

Under full FP8 quantization, the P quantization paths of the CPU golden and the kernel differ (the kernel independently takes the running max and quantizes P for each s2 segment produced by core partitioning). If the two-way criterion "NPU against golden" is used directly, the quantization noise of the golden itself is treated as error, producing false failures.

The three-way comparison instead uses a high-precision bench as the common baseline:

| Role | Source | Parameter Position in `CrossCheckComparison` |
| --- | --- | --- |
| target | Output of the quant_flash_mla operator under test | `output` |
| bench | New fp64 CPU reference (global running max) | `golden` (common baseline) |
| third_party | Sectioned quantization CPU reference | `third_party` |

The criterion is "the error of the NPU relative to the bench / the error of the CPU reference relative to the bench", that is, the error of the NPU does not exceed a multiple of the reference noise.

## Call Chain

```text
customize_inputs  →  inputs.generate_qfmla_inputs()      generates operator inputs with a fixed seed
        │
        ├─ Operator execution                → outputs = [atten(bf16), lse(fp32)?]
        │
golden  →  impl/golden.cpu_qfmla()
             ├─ generate_data()  replays with the same seed, obtaining q/k/v/descale bitwise identical to the operator
             ├─ fp8_fullquant_mla_golden()            → two-way golden (global quantization)
             └─ _cache_bench()
                  ├─ fp8_fullquant_mla_bench()        → _BENCH_CACHE[case]
                  └─ fp8_fullquant_mla_sectioned_golden() → _REF_CACHE[case]
        │
compare →  impl/compare.compare()  ← spec takes the two caches by compare_context.testcase_name
```

`spec.compare` passes `bench_outputs` / `third_party_outputs` only when `cross_check_enabled()` is true; otherwise, it follows the original two-way comparison and generates no additional references.

## bench Specification

- Computation precision: fp64 accumulation and fp64 softmax; the fp32 accumulation error is not reproduced
- Inputs: dequantized q/k/v (only multiplied by descale); **no requantization**
- Retains the operator-semantic fixed-scale cast of P: for FP8, `p*448 → e4m3 → fp32`
- The running max is **global** (once for the entire KV) and **does not reproduce the metadata sectioning**
- Output: fp32, converted according to the output layout of the case (BSND/BNSD/TND) and then cached

## third_party (Sectioned Reference) Specification

`fp8_fullquant_mla_sectioned_golden` replicates the sectioned quantization behavior of the kernel:

- Sectioning: `m_base = 64` and `s2_base = 256`; **each s2 basic block forms a separate section**
- Within a section: independent running max, P cast with the fixed scale above, and accumulation of `(o, sum, m)`
- Between sections: merged according to FD semantics, `o = o*exp(m_old-m_new) + o_seg*exp(m_seg-m_new)`
- Output: converted to bf16 with the same shape as the operator output

**It deliberately does not depend on the metadata core partitioning result.** The real sectioning of the kernel is "the union of several contiguous basic blocks after core partitioning". Sectioning by single blocks is its finest-grained upper bound: the number of sections can only be larger, and the quantization noise of the reference can only be no smaller than that of the target, so the ratio is not inflated by an overly clean reference. The cost is that it is not element-wise identical to the kernel.

The metadata-partitioned version (`get_qfmla_section_info`) is retained in `qfmla_golden.py`, controlled by `ENABLE_SECTION_GOLDEN`, and disabled by default.

## Decision Method

`compare._run_cross_check` processes output channels one by one, usually atten (bf16) + lse (fp32):

1. The three arrays are uniformly upcast to fp32, and their shapes must be consistent;
2. Large value range `|bench| >= small_value`:
   - `mare = max|target-bench| / max|third-bench|`
   - `mere = mean|target-bench| / mean|third-bench|`
   - `rmse = sqrt(mean((target-bench)^2)) / sqrt(mean((third-bench)^2))`
3. Small value range `|bench| < small_value`: count the points where `|x-bench| > small_value_atol`, and require `count_target / count_third <= small_value_ratio`;
4. NaN/Inf uses the built-in three-way consistency decision of the framework;
5. Division denominator protection and numerator/denominator lower bounds are implemented inside the TTK `CrossCheckComparison`.

PASS is returned when all three ratios do not exceed the thresholds in the table above.

## Thresholds

Standard `cross_check`, level `L0`, `mare/mere/rmse = 10/2/2`:

| Output dtype | small_value | small_value_atol | small_value_ratio |
| --- | --- | --- | --- |
| float16 | `2^-10` | `2^-16` | 2 |
| bfloat16 | `2^-10` | `2^-16` | 2 |
| float32 (lse) | `2^-14` | `2^-30` | 2 |


## CPU Offline Generation and NPU Replay

Using the same version of assets, generate data on the CPU side with `--no-prof --dump in,golden
--compare cross_check --manual-data-dirs DATA_ROOT`. The generation stage saves the bench and fixed-block reference of each case in
`DATA_ROOT/_qfmla_cross_check/` (NPZ, without using pickle); the inputs and ordinary golden are still saved by TTK in the original bin/npy/pt format.

Copy the **entire DATA_ROOT** to the NPU machine, including the `_qfmla_cross_check` subdirectory;
the NPU side continues to use `--manual-data-dirs DATA_ROOT --compare cross_check --plugin ASSETS`.
The plugin loads the offline reference when no bench exists in memory, so there is no need to recompute the CPU reference on the NPU machine.
The newly added files are auxiliary data of the plugin and are placed outside the TTK case directory to avoid being cleaned up during TTK writes or rejected during reads.

The reference files validate the case name, protocol version, the shape/byte content of the nine actual data inputs, and the main operator attributes.
The metadata is generated by the target machine and does not participate in the input digest. After attributes such as inputs, layouts, masks, or scales change,
the data must be regenerated; missing, mismatched, or corrupted references cause the comparison to fail, without degrading to two-way.
When data with the same name is prepared again, the reference of that case is overwritten.

Data generated by the old version does not have these offline references; update the assets and regenerate the data on the CPU side.
When cross_check is not enabled by default or the offline golden dump is not executed, these auxiliary files are not written.
