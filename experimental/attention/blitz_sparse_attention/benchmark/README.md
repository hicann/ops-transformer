# How to use the benchmarking/correctness program benchmark.py

# Installation

To install the sparse kernel go in the `ops-transformer` project home folder and run:

```shell
bash build.sh --make_clean --experimental -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention
./build/cann-ops-transformer-custom_linux-"$(uname -i)".run
(cd experimental/attention/blitz_sparse_attention/torch_interface && bash build.sh custom)
```

Fast recompilation only of bf16 kernel variant

```shell
bash build.sh --make_clean --experimental -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention --op-variant=2
./build/cann-ops-transformer-custom_linux-"$(uname -i)".run
```

Oneliner smoketest and benchmark

```shell
(cd experimental/attention/blitz_sparse_attention/benchmark && python test_attn.py && python benchmark.py)
```

Benchmark + plot pipeline (writes `benchmark.png` summarising all
`(BLOCK_SIZE_Q, BLOCK_SIZE_KV, sparsity)` combinations):

```shell
(cd experimental/attention/blitz_sparse_attention/benchmark && python benchmark.py | tee bench.log /dev/tty | python plot.py)
```

## Parameters

Set the `ALL_CAPS` constants at the top of the file to control what gets run. Wherever a constant is a list, the sweep covers **every combination** of all listed values.

* `B_VALS`: batch sizes to test. For block-sparsity only `B_VALS = [1]` works for now.
* `H_VALS`: number of heads
* `S_VALS`: sequence lengths
* `D_VALS`: head dimensions
* `N_REPEATS`: how many runs to do to estimate time
* `N_WARMUP`: how many warmup runs before running `N_REPEATS` calls
* `SPARSITY_VALS`: how many blocks to activate: randomly chosen, but each Q-block row will have the same number of blocks. The block-sparse pattern is the only kernel mode the benchmark exercises.
* `BLOCK_SHAPES`: List of `(BLOCK_SIZE_Q, BLOCK_SIZE_KV)` pairs to sweep. The kernel accepts every combination of `BLOCK_SIZE_Q ∈ {128, 256, 512, 1024}` and `BLOCK_SIZE_KV ∈ {128, 256, 512, 1024}` at runtime via the `block_shape` op-attr — no rebuild needed to switch granularities. The default sweep is the full 4 × 4 = 16-combination grid. The first column of the benchmark output labels every row with its active block shape. `BLOCK_SIZE_Q` and `BLOCK_SIZE_KV` aliases (set to `BLOCK_SHAPES[0]` for backward compatibility) are still readable from the top of `benchmark.py`. 
* `BLOCK_MASK_SEED`: for reproducibility of random sampling. Can set to any value to change random sampling
* `FRAMES_BY_BLOCK_SHAPE`: 2-D dict keyed by `(BLOCK_SIZE_Q, BLOCK_SIZE_KV)` returning a `FrameWidths(left_cols, right_cols, top_rows, bottom_rows)` namedtuple — extra blocks forced active on top of the sparsity budget. Models a typical sink/streaming pattern. Per-pair entries scale to keep the forced-token footprint comparable across granularities; see Example 2 for the derivation.
* `PRINT_OUTPUTS`: For manual visualization of output tensors
* `PRINT_MASK`: To visually check if the mask is as expected
* `PRINT_BLOCK_EQUALITY`: if the correctness test fails, you can check which blocks in the outputs match with the reference and which not. The block granularity to visualize is given by `PRINT_HEIGHT` and `PRINT_WIDTH`
* `RUN_REFERENCE`: Besides measuring time, also test correctness. This only is possible for small enough masks, otherwise you will get OOM in the reference run. For this reason, you can disable this for very long contexts, and enable it for short enough sequences.
* `TORCH_REFERENCE`: If `True`, correctness will be tested against the torch attention implementation. If `False`, for block-sparse attention the reference is the TorchNPU dense with the corresponding attention mask (like in `sparse_block` mode). Please only use `H=1`, as torch_npu doesn't support different masks for different heads.

## Examples

### 1 - Test correctness and speed with multiple values (and shorter sequence lengths)

#### Input

Adjust the following values in the `===== Begin of Parameter Sweep Definitions ====` section

```python
DTYPE = torch.bfloat16
INPUT_LAYOUT = "BNSD"  # [batch_size, num_hea
B_VALS = [1]
H_VALS = [16, 14]
S_VALS = [4096, 10_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 20
N_WARMUP = 2

SPARSITY_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BLOCK_SHAPES = [(128, 512)]
SABI_SORTED = True
FRAMES_BY_BLOCK_SHAPE = {(128, 512):None} # no frame forced to be selected

PRINT_OUTPUTS = False
PRINT_MASK = False
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8
RUN_REFERENCE = True
TORCH_REFERENCE = "npu_fusion_attention"
```

#### Output

```shell
========================================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'  SABI_SORTED=True  TORCH_REFERENCE='npu_fusion_attention'
========================================================================================================================
  block_shape   H   B    s_q   s_kv    D  frame(L,R,T,B)  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------------------------------------
      128x512  16   1   4096   4096  128               -      0.00             yes             859.30            1619.89
      128x512  16   1   4096   4096  128               -      0.05             yes            1000.11            1617.22
      128x512  16   1   4096   4096  128               -      0.10             yes             997.50            1593.49
      128x512  16   1   4096   4096  128               -      0.20             yes             997.37            1457.39
      128x512  16   1   4096   4096  128               -      0.30             yes             997.62            1359.71
      128x512  16   1   4096   4096  128               -      0.40             yes             997.41            1221.68
      128x512  16   1   4096   4096  128               -      0.50             yes             997.88            1114.68
      128x512  16   1   4096   4096  128               -      0.60             yes             996.95            1281.74
      128x512  16   1   4096   4096  128               -      0.70             yes             997.40             986.38
      128x512  16   1   4096   4096  128               -      0.80             yes             997.62             875.81
      128x512  16   1   4096   4096  128               -      0.90             yes             997.56             752.91
      128x512  16   1  10000  10000  128               -      0.00             yes            5696.26            6649.24
      128x512  16   1  10000  10000  128               -      0.05             yes            7533.60            6319.64
      128x512  16   1  10000  10000  128               -      0.10             yes            7498.81            6310.39
      128x512  16   1  10000  10000  128               -      0.20             yes            7574.27            5369.41
      128x512  16   1  10000  10000  128               -      0.30             yes            7520.82            4952.99
      128x512  16   1  10000  10000  128               -      0.40             yes            7572.14            4159.39
      128x512  16   1  10000  10000  128               -      0.50             yes            7538.75            3560.27
      128x512  16   1  10000  10000  128               -      0.60             yes            7561.45            3079.28
      128x512  16   1  10000  10000  128               -      0.70             yes            7482.04            2406.52
      128x512  16   1  10000  10000  128               -      0.80             yes            7565.93            1857.96
      128x512  16   1  10000  10000  128               -      0.90             yes            7533.11            1266.50
      128x512  14   1   4096   4096  128               -      0.00             yes             784.19            1534.82
      128x512  14   1   4096   4096  128               -      0.05             yes             906.90            1693.06
      128x512  14   1   4096   4096  128               -      0.10             yes             908.12            1515.25
      128x512  14   1   4096   4096  128               -      0.20             yes             905.51            1398.79
      128x512  14   1   4096   4096  128               -      0.30             yes             908.47            1287.53
      128x512  14   1   4096   4096  128               -      0.40             yes             906.55            1173.67
      128x512  14   1   4096   4096  128               -      0.50             yes             907.90            1064.68
      128x512  14   1   4096   4096  128               -      0.60             yes             906.96            1065.85
      128x512  14   1   4096   4096  128               -      0.70             yes             908.62             946.82
      128x512  14   1   4096   4096  128               -      0.80             yes             907.40             869.84
      128x512  14   1   4096   4096  128               -      0.90             yes             908.16             723.44
      128x512  14   1  10000  10000  128               -      0.00             yes            4857.91            5930.11
      128x512  14   1  10000  10000  128               -      0.05             yes            6695.51            5614.66
      128x512  14   1  10000  10000  128               -      0.10             yes            6680.30            5312.90
      128x512  14   1  10000  10000  128               -      0.20             yes            6621.15            4772.31
      128x512  14   1  10000  10000  128               -      0.30             yes            6690.78            5062.18
      128x512  14   1  10000  10000  128               -      0.40             yes            6622.58            3757.85
      128x512  14   1  10000  10000  128               -      0.50             yes            6661.94            3208.64
      128x512  14   1  10000  10000  128               -      0.60             yes            6630.15            2696.30
      128x512  14   1  10000  10000  128               -      0.70             yes            6641.87            2193.39
      128x512  14   1  10000  10000  128               -      0.80             yes            6672.09            1715.01
      128x512  14   1  10000  10000  128               -      0.90             yes            6685.34            1186.83
========================================================================================================================
```

Reference goes OOM for 10_000 sequence length and 24 heads (not shown).

On the left you see the inputs, then the "yes" line checks correctness, after that you see the reference runtime and our runtime. After that you see the memory bandwidth usage, which is much better in our kernel.

Note that for short sequence lengths, the speedups are negligible because overheads are higher than actual computation.

### 2 - block_shape granularity sweep (S=118806, H=3, D=128, BF16)

The runtime `block_shape` attr lets the same kernel run at every combination of
`(BLOCK_SIZE_Q, BLOCK_SIZE_KV)` with each dimension in `{128, 256, 512, 1024}`
—**16 granularities** sharing one kernel binary. Finer sabi (smaller block
sizes) buys more sparsity-pattern resolution per Q-row at the cost of more
sabi-tensor HBM traffic + more sub-block plumbing per cube tile; coarser sabi
buys cheaper per-row metadata at the cost of resolution. The sweep below
measures the resulting latency/throughput trade-off at S = 118 806.

#### Why `(left_cols, right_cols, top_rows, bottom_rows)` scales as `(29,15,8,4)` / `(29,15,8,4)`

The benchmark forces a frame in the attention matrix that models a typical
generative-model sink/streaming pattern: **the first ~3600 attention rows and
columns plus the last ~6 rows and columns are always selected**. 6 tokens is
less than a single block in every dimension, so it always rounds up to one
full block-row / block-column at the bottom/right.

`FrameWidths.left_cols` / `right_cols` are in `BLOCK_SIZE_KV`-token units, and
`top_rows` / `bottom_rows` are in `BLOCK_SIZE_Q`-token units, so the per-pair
values fall out of rounding the constant 3600-token target up to whole blocks
in each dimension independently:

| size  | `ceil(3600 / size)` |
|------:|:-------------------:|
|  128  | 29                  |
|  256  | 15                  |
|  512  |  8                  |
| 1024  |  4                  |

That gives a 2-D `FRAMES_BY_BLOCK_SHAPE[(BLOCK_SIZE_Q, BLOCK_SIZE_KV)]` table
where `left_cols ∈ {29, 15, 8, 4}` (indexed by `BLOCK_SIZE_KV`), `top_rows ∈
{29, 15, 8, 4}` (indexed by `BLOCK_SIZE_Q`), and `right_cols = bottom_rows = 1`
everywhere. e.g. `(BLOCK_SIZE_Q=1024, BLOCK_SIZE_KV=1024)` ⇒
`FrameWidths(left_cols=4, right_cols=1, top_rows=4, bottom_rows=1)`.

The test files apply the same scaling rule with a smaller target (~256 tokens)
for their `SPARSE_FRAME`.

#### Input

```python
B_VALS = [1]; H_VALS = [3]; S_VALS = [118_806]; D_VALS = [128]
N_REPEATS = 10; N_WARMUP = 2
SPARSITY_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BLOCK_SHAPES = [(bsq, bskv)
                for bsq  in (128, 256, 512, 1024)
                for bskv in (128, 256, 512, 1024)]                 # 4 × 4 = 16 pairs
# Per-pair frame: left_cols/top_rows are scaled by 128 / size in each dim so
# the forced-token footprint stays comparable across block shapes.
FRAMES_BY_BLOCK_SHAPE = {
    (bsq, bskv): FrameWidths(
        left_cols   = {128: 29, 256: 15, 512: 8, 1024: 4}[bskv],
        right_cols  = 1,
        top_rows    = {128: 29, 256: 15, 512: 8, 1024: 4}[bsq],
        bottom_rows = 1,
    )
    for bsq  in (128, 256, 512, 1024)
    for bskv in (128, 256, 512, 1024)
}
TORCH_REFERENCE = "npu_fusion_attention"
```

#### Output

Run `python benchmark.py | tee bench.log /dev/tty | python plot.py` to also generate
a `speedup-vs-sparsity` figure with one curve per `(sequence_length, block_shape)`
pair. The full raw table below was captured at S=118 806, H=3, D=128, BF16:

<details>
<summary>Full raw benchmark table (16 block shapes × 11 sparsities — click to expand)</summary>

```shell
========================================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'  SABI_SORTED=True  TORCH_REFERENCE='npu_fusion_attention'
========================================================================================================================
  block_shape   H   B    s_q   s_kv    D  frame(L,R,T,B)  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------------------------------------
      128x128   3   1 118806 118806  128               -      0.00             yes          157210.33          200781.80
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.05             N/A                N/A          161036.06
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.10             N/A                N/A          158057.97
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.20             N/A                N/A          152368.05
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.30             N/A                N/A          144510.33
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.40             N/A                N/A          133058.51
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.50             N/A                N/A          117734.07
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.60             N/A                N/A           99123.58
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.70             N/A                N/A           78038.51
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.80             N/A                N/A           55330.05
      128x128   3   1 118806 118806  128     (29,1,29,1)      0.90             N/A                N/A           24609.61
      128x256   3   1 118806 118806  128               -      0.00             yes          159207.32          199820.03
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.05             N/A                N/A          158132.10
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.10             N/A                N/A          151330.90
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.20             N/A                N/A          137840.45
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.30             N/A                N/A          123943.16
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.40             N/A                N/A          107830.05
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.50             N/A                N/A           91377.56
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.60             N/A                N/A           74124.60
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.70             N/A                N/A           56262.56
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.80             N/A                N/A           37968.08
      128x256   3   1 118806 118806  128     (15,1,29,1)      0.90             N/A                N/A           19270.95
      128x512   3   1 118806 118806  128               -      0.00             yes          161129.31          198749.26
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.05             N/A                N/A          173208.98
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.10             N/A                N/A          160075.51
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.20             N/A                N/A          137263.96
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.30             N/A                N/A          119119.91
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.40             N/A                N/A          102260.27
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.50             N/A                N/A           85403.20
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.60             N/A                N/A           68388.53
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.70             N/A                N/A           51450.43
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.80             N/A                N/A           34669.06
      128x512   3   1 118806 118806  128      (8,1,29,1)      0.90             N/A                N/A           17906.81
     128x1024   3   1 118806 118806  128               -      0.00             yes          162207.12          202698.22
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.05             N/A                N/A          174917.09
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.10             N/A                N/A          165089.04
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.20             N/A                N/A          145535.10
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.30             N/A                N/A          125322.22
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.40             N/A                N/A          105297.35
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.50             N/A                N/A           87966.00
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.60             N/A                N/A           69510.69
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.70             N/A                N/A           52431.44
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.80             N/A                N/A           35000.96
     128x1024   3   1 118806 118806  128      (4,1,29,1)      0.90             N/A                N/A           18146.70
      256x128   3   1 118806 118806  128               -      0.00             yes          163067.77          200198.46
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.05             N/A                N/A          165123.96
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.10             N/A                N/A          162855.57
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.20             N/A                N/A          153739.60
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.30             N/A                N/A          145058.70
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.40             N/A                N/A          133083.78
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.50             N/A                N/A          117827.78
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.60             N/A                N/A           99380.31
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.70             N/A                N/A           78359.30
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.80             N/A                N/A           55515.84
      256x128   3   1 118806 118806  128     (29,1,15,1)      0.90             N/A                N/A           24680.32
      256x256   3   1 118806 118806  128               -      0.00             yes          163126.33          200421.86
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.05             N/A                N/A          162244.29
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.10             N/A                N/A          154796.35
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.20             N/A                N/A          141090.41
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.30             N/A                N/A          125764.73
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.40             N/A                N/A          109879.25
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.50             N/A                N/A           93020.79
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.60             N/A                N/A           75177.50
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.70             N/A                N/A           56512.46
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.80             N/A                N/A           37968.77
      256x256   3   1 118806 118806  128     (15,1,15,1)      0.90             N/A                N/A           19722.16
      256x512   3   1 118806 118806  128               -      0.00             yes          164016.82          202325.94
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.05             N/A                N/A          175537.23
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.10             N/A                N/A          161875.20
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.20             N/A                N/A          144422.60
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.30             N/A                N/A          123083.90
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.40             N/A                N/A          105284.77
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.50             N/A                N/A           88914.20
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.60             N/A                N/A           71715.88
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.70             N/A                N/A           53809.10
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.80             N/A                N/A           34646.38
      256x512   3   1 118806 118806  128      (8,1,15,1)      0.90             N/A                N/A           17254.83
     256x1024   3   1 118806 118806  128               -      0.00             yes          164625.84          203211.69
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.05             N/A                N/A          176233.89
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.10             N/A                N/A          166313.94
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.20             N/A                N/A          148188.48
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.30             N/A                N/A          128503.65
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.40             N/A                N/A          109221.57
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.50             N/A                N/A           91500.32
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.60             N/A                N/A           73028.33
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.70             N/A                N/A           54234.94
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.80             N/A                N/A           35525.95
     256x1024   3   1 118806 118806  128      (4,1,15,1)      0.90             N/A                N/A           16849.22
      512x128   3   1 118806 118806  128               -      0.00             yes          165319.87          201165.34
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.05             N/A                N/A          167500.90
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.10             N/A                N/A          163980.60
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.20             N/A                N/A          155223.03
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.30             N/A                N/A          145277.70
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.40             N/A                N/A          133691.13
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.50             N/A                N/A          118228.98
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.60             N/A                N/A           99471.91
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.70             N/A                N/A           78237.73
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.80             N/A                N/A           55266.32
      512x128   3   1 118806 118806  128      (29,1,8,1)      0.90             N/A                N/A           24597.66
      512x256   3   1 118806 118806  128               -      0.00             yes          165073.32          201308.03
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.05             N/A                N/A          165942.18
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.10             N/A                N/A          157005.85
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.20             N/A                N/A          142381.26
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.30             N/A                N/A          126352.48
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.40             N/A                N/A          110357.71
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.50             N/A                N/A           93363.94
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.60             N/A                N/A           75421.60
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.70             N/A                N/A           57303.78
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.80             N/A                N/A           39200.84
      512x256   3   1 118806 118806  128      (15,1,8,1)      0.90             N/A                N/A           19915.82
      512x512   3   1 118806 118806  128               -      0.00             yes          165493.05          202528.27
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.05             N/A                N/A          178655.80
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.10             N/A                N/A          167690.99
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.20             N/A                N/A          148863.20
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.30             N/A                N/A          131446.69
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.40             N/A                N/A          114359.11
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.50             N/A                N/A           95379.45
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.60             N/A                N/A           75719.90
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.70             N/A                N/A           55887.92
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.80             N/A                N/A           36444.14
      512x512   3   1 118806 118806  128       (8,1,8,1)      0.90             N/A                N/A           17844.47
     512x1024   3   1 118806 118806  128               -      0.00             yes          165981.21          202457.74
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.05             N/A                N/A          179652.09
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.10             N/A                N/A          169494.82
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.20             N/A                N/A          150344.48
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.30             N/A                N/A          133085.55
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.40             N/A                N/A          115268.55
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.50             N/A                N/A           95859.71
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.60             N/A                N/A           74771.77
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.70             N/A                N/A           55201.18
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.80             N/A                N/A           36086.59
     512x1024   3   1 118806 118806  128       (4,1,8,1)      0.90             N/A                N/A           18220.92
     1024x128   3   1 118806 118806  128               -      0.00             yes          166277.64          199614.44
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.05             N/A                N/A          168760.72
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.10             N/A                N/A          164258.64
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.20             N/A                N/A          155011.16
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.30             N/A                N/A          145592.77
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.40             N/A                N/A          133456.46
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.50             N/A                N/A          117487.83
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.60             N/A                N/A           98545.56
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.70             N/A                N/A           77536.25
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.80             N/A                N/A           54579.08
     1024x128   3   1 118806 118806  128      (29,1,4,1)      0.90             N/A                N/A           23215.96
     1024x256   3   1 118806 118806  128               -      0.00             yes          165954.82          201113.94
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.05             N/A                N/A          169928.98
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.10             N/A                N/A          159145.01
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.20             N/A                N/A          142113.40
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.30             N/A                N/A          127223.95
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.40             N/A                N/A          109795.79
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.50             N/A                N/A           92943.02
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.60             N/A                N/A           75015.89
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.70             N/A                N/A           56391.08
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.80             N/A                N/A           38051.85
     1024x256   3   1 118806 118806  128      (15,1,4,1)      0.90             N/A                N/A           18906.36
     1024x512   3   1 118806 118806  128               -      0.00             yes          166466.77          202403.93
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.05             N/A                N/A          184474.29
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.10             N/A                N/A          175285.35
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.20             N/A                N/A          156707.23
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.30             N/A                N/A          136099.63
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.40             N/A                N/A          116428.54
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.50             N/A                N/A           96237.19
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.60             N/A                N/A           75197.93
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.70             N/A                N/A           55883.06
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.80             N/A                N/A           36185.81
     1024x512   3   1 118806 118806  128       (8,1,4,1)      0.90             N/A                N/A           17193.75
    1024x1024   3   1 118806 118806  128               -      0.00             yes          166565.60          203121.20
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.05             N/A                N/A          183960.52
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.10             N/A                N/A          174192.35
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.20             N/A                N/A          156145.59
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.30             N/A                N/A          136221.70
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.40             N/A                N/A          115564.39
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.50             N/A                N/A           94306.43
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.60             N/A                N/A           74998.33
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.70             N/A                N/A           55769.76
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.80             N/A                N/A           36448.80
    1024x1024   3   1 118806 118806  128       (4,1,4,1)      0.90             N/A                N/A           17044.48
========================================================================================================================
```

</details>

Only sparsities `0.0 / 0.5 / 0.9` are kept above for brevity; the live run
emits every sparsity in `SPARSITY_VALS` for every pair in `BLOCK_SHAPES`. The
`frame(L,R,T,B)` column is `-` at sparsity 0 because the mask is dense, so
the frame has no effect on the kept blocks.

#### Speedup vs `npu_fusion_attention` (dense)— full 4 × 4 grid

Each cell is `Ref_Latency(sparsity=0) / Our_Latency(sparsity=p)`. Values
**< 1.0** mean BSA is slower than PFA dense; **> 1.0** mean BSA is faster.

| Q\KV | sparsity | 128   | 256   | 512   | 1024  |
|------|----------|------:|------:|------:|------:|
| 128  | 0.3      | 1.09× | 1.28× | 1.35× | 1.29× |
|      | 0.5      | 1.34× | 1.74× | 1.89× | 1.84× |
|      | 0.8      | 2.84× | 4.19× | **4.65×** | 4.63× |
| 256  | 0.3      | 1.12× | 1.30× | 1.33× | 1.28× |
|      | 0.5      | 1.38× | 1.75× | 1.84× | 1.80× |
|      | 0.8      | 2.94× | 4.30× | **4.73×** | 4.63× |
| 512  | 0.3      | 1.14× | 1.31× | 1.26× | 1.25× |
|      | 0.5      | 1.40× | 1.77× | 1.73× | 1.73× |
|      | 0.8      | 2.99× | 4.21× | 4.54× | **4.60×** |
| 1024 | 0.3      | 1.14× | 1.30× | 1.22× | 1.22× |
|      | 0.5      | 1.42× | 1.79× | 1.73× | 1.77× |
|      | 0.8      | 3.05× | 4.36× | **4.60×** | 4.57× |

* Speedup grows monotonically with sparsity in every cell — BSA never regresses
  *worse* than ~0.78× even at sparsity 0, and reaches **6.4×–9.8× at sparsity
  0.9** across the grid.
* **BLOCK_SIZE_KV is the dominant lever.** At sparsity 0.9, moving KV from
  128 → 1024 buys 1.4×–1.5× on top of the existing speedup at every Q. The
  same trend holds for sparsity 0.8 (≈ 1.5×–1.6× from KV=128 to KV=1024).
  This is because a wider sabi entry means less sabi-tensor HBM traffic per
  kept block. Note however that KV=1024 is not always the strict winner at
  sparsity 0.9: for Q=128 and Q=512, KV=512 narrowly beats KV=1024 (9.00× vs
  8.94×, 9.27× vs 9.11×)— at the coarsest KV the forced-frame footprint
  starts to eat into the kept-block budget.
* **BLOCK_SIZE_Q has a much smaller effect.** Q-tile re-parameterisation does
  not change the cube tile width (`basicSInnerSize = 512` for D=128 BF16) or
  the per-Q-row workload; the only effect is collapsing several 128-token Q
  groups onto a single sabi row, which is mostly free. At sparsity 0.9 the
  spread between Q=128 and Q=1024 (same KV) is ≤ 12 %.
* **The dense-mode regression (sparsity 0) is essentially flat across KV.**
  The overhead vs PFA sits at ≈ 17–22 % across the whole grid (0.78×–0.83×)
  with no strong KV trend — the wide-KV packer no longer buys additional
  bandwidth at sparsity 0 in this configuration.
* **128×128 remains the most expensive baseline.** Finest sabi resolution but
  the most metadata-per-kept-block.
* **The fastest cell at high sparsity is KV=1024 at Q∈{256, 1024}**— both
  reach 9.77× over PFA dense at sparsity 0.9. KV=512 also lands within ~5 %
  of the top (9.00×–9.68×). Picking a granularity is therefore an
  application-level trade-off between pattern-fidelity and end-to-end
  latency; the kernel does not force one.

The PFA reference (`npu_fusion_attention` dense) is shape-invariant — the
small differences in the `Ref_Latency_[usec]` column across block shapes are
timing noise from re-running the same workload 16 times.

### 3 - softmax_lse correctness tests (test_lse.py)

`test_lse.py` verifies the `softmax_lse` (log-sum-exp) second output of the kernel.

```shell
pytest test_lse.py -v
```

Every test case is exercised at all **16 default block shapes**
(`{128, 256, 512, 1024} × {128, 256, 512, 1024}`) via a parametrized
`block_shape` fixture, so the live test count is ~16× the per-shape count
below (~1600 instances total). A handful of cases at small S or high sparsity
are auto-skipped because the test parameterisation becomes inherently
unsatisfiable (forced-frame density exceeds `1 − sparsity`, or the row's
expected kept-KV-block count drops below 1)— see the `pytest.skip` rules in
the file.

#### What is tested per block shape (B=1 only)

| Test | Reference | Shapes | Dtypes | Notes |
|---|---|---|---|---|
| `test_bsa_lse_vs_fias` | FIAS dense | 14 shapes, S up to 24 000 | bfloat16, float16 | Dense mode (no sabi) |
| `test_bsa_lse_vs_fias_sparse` | FIAS sparse_mode=1 | 8 shapes, S up to 24 000 | bfloat16 | H=1, no border frame |
| `test_bsa_lse_vs_fias_sparse_framed` | FIAS sparse_mode=1 | 2 shapes, S ≥ 16 384 | bfloat16 | H=1, FrameWidths(2,1,3,1) |
| `test_bsa_lse_sparse_multihead` | Python float32 reference | 4 shapes, S ≤ 1 024 | bfloat16 | H=2,4; Python loops over sabi |
| `test_bsa_lse_zero_input` | Analytical: log(S) | 5 shapes | bfloat16, float16 | Q=K=V=0 ⟹ uniform softmax |
| `test_bsa_lse_shape_and_dtype` | — | 3 shapes | bfloat16 | Checks shape=[B,H,S], dtype=float32 |
| `test_bsa_lse_disabled_returns_empty` | — | 3 shapes | bfloat16 | flag=False ⟹ numel=0 |

#### Why FIAS is the LSE reference

FIAS (`npu_fused_infer_attention_score`) already exposes a `softmax_lse` output via its `softmaxLseFlag` bool attr (output index 1, shape `[B, N, S, 1]`, float32).  BSA computes the identical quantity — `log(Σ exp(q·kᵀ/√d))` over attended tokens — so FIAS with a matching token-level attention mask serves as a drop-in reference for H=1 shapes.

For H>1 shapes FIAS cannot be used directly (its `sparse_mode=1` requires a `[B,1,S,S]` broadcast mask; passing `[B,H,S,S]` hangs the NPU), so a Python float32 reference that loops over the sabi indices is used instead.

#### Tolerances

ATOL = 0.001
RTOL = 0.01

float16 is excluded from sparse tests: FIAS processes `-inf`-masked tiles through its flash-attention tiling while BSA skips them entirely; the resulting LSE divergence exceeds any reasonable tolerance at high sparsity.

## Test setup

| Component      | Version                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| Host CPU       | aarch64                                                                                                       |
| Device         | Ascend 910B2                                                                                                  |
| Device Driver  | 25.3.rc1                                                                                                      |
| Docker image   | `docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops` |
| OS             | Ubuntu 22.04                                                                                                  |
| CANN           | 8.5.0-beta.1                                                                                                  |
| Python         | 3.11.10                                                                                                       |
| torch          | 2.8.0+cpu                                                                                                     |
| TorchNPU      | 2.8.0                                                                                                         |
