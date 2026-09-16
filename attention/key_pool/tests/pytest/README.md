# KeyPool Tests

This directory contains the maintained KeyPool precision validation entry
point. The test uses the public `torch_npu.key_pool` interface and the
independent CPU Golden implementation. Legacy parameter-combination,
equivalence, profiling, and ad-hoc regression scripts are intentionally not
part of the release tree.

## Maintained Files

| File | Purpose |
| --- | --- |
| `test_key_pool_precision_standalone.py` | Direct NPU-versus-Golden precision check |
| `key_pool_golden.py` | CPU Golden and independent Torch Oracle used by the precision check |
| `key_pool_public_loader.py` | Loads only the installed KeyPool public module |

## Coverage

The standalone check covers representative:

```text
BSH [B, S, H]
TH  [T, H] with cu_seqlens
K/Gate projection
optional LayerNorm
state-cache update and Block Table addressing
Gate + APE pooling
all supported compression ratios
current-input output capacity with undefined unused tail
non-empty RoPE rejection through the public interface contract
```

RoPE remains an interface placeholder. Non-empty `cos`/`sin` inputs must be
rejected explicitly and must not be silently ignored.

The NPU scripts are intended to run after the CANN, custom OPP, and wheel
environments have been loaded. They do not require `torchair`.

Run the maintained validation directly:

```bash
python3 test_key_pool_precision_standalone.py
```

The script has no command-line parameters. It generates four representative
BSH/TH cases, with and without LayerNorm, historical cache data, non-trivial
start positions, shuffled block tables, zero-length TH batches, and multiple
compression ratios. It validates both `pooled_key` and the in-place
`state_cache` update against the CPU Golden and exits with a non-zero status
when any case fails.

BSH output is `[B, ceil(S/r), D]`; TH output is `[min(T, T//r+B), D]`.
The operator does not initialize unused output rows. For each batch, only
`(start_pos + seq_len) // cmp_ratio - start_pos // cmp_ratio` rows are valid.
TH concatenates valid batch results without gaps; unused rows are at the end
of the whole output. BSH has an unused tail per batch.
The precision comparison zeros unused rows in CPU comparison copies of both
outputs; it does not change the operator output or the state cache. Valid
output elements are also checked separately so unused capacity cannot dilute
the precision metric. Performance measurements do not perform this masking.
