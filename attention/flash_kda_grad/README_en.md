# FlashKDA Grad — CANNBotDSL 0.8
English | [简体中文](./README.md)

This directory independently maintains the FlashKDA training backward operator, using **CANNBotDSL 0.8**.
Host parameter checking, workspace preparation, the compilation cache, and the kernel main entry remain in the same main file;
they are not rewritten as PyPTO, and they are not integrated into the repository's ACLNN/CMake build.

## Directory Structure

```text
flash_kda_grad/
├── op_kernel/
│   ├── flash_kda_grad.py             # Host and kernel main entry, shared computation
│   ├── flash_kda_grad_metadata.py    # Metadata generation and core splitting
│   ├── flash_kda_grad_persistent.py  # Persistent scheduling
│   ├── flash_kda_grad_staged.py      # Staged scheduling
│   └── flash_kda_grad_recompute.py   # Recomputation
└── tests/
    ├── _kdag_paths.py                # Source loading for this directory, without depending on the original repository's test tools
    ├── pytest.ini                   # Only declares markers
    ├── ut/test_flash_kda_grad_ut.py  # UT, exception contracts, performance/comparison CLI
    └── ttk/
        ├── flash_kda_grad_ttk/       # api, golden, spec
        ├── cases/
        │   ├── flash_kda_grad_stc.csv       # 130 entries
        │   └── flash_kda_grad_exception.csv # 60 entries
        ├── README.md
        └── run_one.sh
```

The five Python modules of the operator continue to use the verified 0.8 implementation; saving the cache and recomputation share the public interfaces.
The `op_kernel` directory contains Host logic; this directory name does not mean that it contains only device-side code.
The golden, peer, and CSV content of the tests and the native `cross_check L2` criterion remain unchanged.

## Environment and Import

Use an already activated CANN/Conda environment; `cannbotdsl==0.8.0`, PyTorch,
`torch_npu`, and pytest of matching versions are required. TTK acceptance also requires ops-test-kit. The scripts do not switch environments or write fixed server paths.

You can import directly from the repository root directory:

```python
from attention.flash_kda_grad.op_kernel.flash_kda_grad import (
    flash_kda_grad_npu,
    flash_kda_grad_from_forward,
)
from attention.flash_kda_grad.op_kernel.flash_kda_grad_metadata import (
    flash_kda_grad_metadata,
)
```

`flash_kda_grad_npu` receives q/k/v, gate, beta, the upstream gradients do/dht, Aqk/Akk,
the backward metadata, and the corresponding forward caches. The output order is
`dq, dk, dv, dh0, dg, dbeta, dA, dBias`.
`disable_recompute=True` consumes the saved cache; `False` rebuilds the recomputable cache inside the backward pass.
BNSD, BSND, and TND are supported; for TND, the total number of tokens is split into multiple sequences by `cu_seqlens`.

## UT and Exception Interception

Run the following commands in the repository root directory. Select explicitly by marker; by default, no cases are filtered:

```bash
python -m pytest -q attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py \
  -m 'not npu and not ascendc_toolchain and not whitebox'

python -m pytest -q -s attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py \
  -k 'exception_csv_host_contract or exception_csv_npu_contract or exception_csv_special_contract'
```

The mutations of the exception CSV are executed by the preceding contract tests; ordinary TTK E2E does not interpret the mutation of the exception CSV,
and an ordinary execution failure cannot be regarded as successful interception. The 60 exception entries correspond to 117 contract checks.

When run independently, three "real forward metadata compatibility" UTs and one "original repository Native packaging" UT
are explicitly skipped. These checks depend on components of the original repository that are not included in this directory; to execute them, set
`KDA_DSL_ROOT=/path/to/compatible/cannbot-dsl_8`. The other UTs and the default STC do not depend on the original repository.

## TTK Accuracy and Performance

The scripts locate the operator and assets from their own location and can be invoked from any directory.
If TTK is not yet installed, you can set `TTK_REPO=/path/to/ops-test-kit`.
It is recommended that inputs/golden and outputs be placed in newly created directories of your own, to avoid writing into code directories:

```bash
export KDA_ASSETS="$PWD/attention/flash_kda_grad/tests/ttk"
export KDA_DEVICES=0
export KDA_MANUAL_DATA=/path/to/input_and_golden
export KDA_OUT=/path/to/results.csv

bash "$KDA_ASSETS/run_one.sh" validate
bash "$KDA_ASSETS/run_one.sh" -t kdag_stc_109
bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109
KDA_PROF=true bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109
```

Without `-t`, all 130 STC cases are selected by default and executed separately according to the `disable_recompute` attribute of the CSV.
`dump` mode generates new inputs/golden, and `replay` reads existing data; generated data does not mean that NPU accuracy passes.
The four replaced cases 022/065/066/067 must use newly defined data; you cannot reuse old golden files with the same numbers.
By default, `cache_source=ref` constructs the forward cache with the test reference implementation; the optional `cache_source=fwd`
requires a compatible real forward operator provided through `KDA_DSL_ROOT`.

For detailed inputs, comparison logs, and run modes, see [TTK Description](tests/ttk/README_en.md).
