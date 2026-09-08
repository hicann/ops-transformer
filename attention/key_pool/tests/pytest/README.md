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
fixed-capacity pooled output and zero tail
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
