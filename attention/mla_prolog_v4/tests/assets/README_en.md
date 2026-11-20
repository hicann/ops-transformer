# mla_prolog_v4 TTK

The assets directory contains the TTK end-to-end test resources for this operator: test cases, API adapters, input preparation, CPU Golden references, precision comparison, and graph execution support.

Use the E2E mode of [ops-test-kit](https://gitcode.com/cann/ops-test-kit) to test the public `mla_prolog_v2` operator in this repository. Supports unquantized weights, with `qkv_quant_mode` values 0, 1, 2, and 3.
The TTK output slots are `q`, `descale_q`, `qr`, and `kv_cache`. The operator updates `kv_cache` in place.

Supported hardware: Ascend 950. The build target is `ascend950`.

## Source location

`mla_prolog_v4_ttk_ops.py` resolves and loads `attention/mla_prolog_v4/op_dsl/mla_prolog_v4.py` relative to its own directory, then calls the public `mla_prolog_v2` API.
Loading the module registers the Torch operator. The same process reuses the loaded module.

The test entry loads the implementation directly from the repository. Depending on the runtime environment, it can match installed Native artifacts.
It does not use the `torch_extension` entry from an installed wheel.

After setting `ASSETS` and `PYTHONPATH` as described below, check the resolved source path:

```bash
python3 -c "import mla_prolog_v4_ttk_ops as adapter; print(adapter._resolve_sample_file())"
```

## Directory structure

The tree below lists documentation and implementation files. CSV files in the same directory provide the test cases.

```text
attention/mla_prolog_v4/tests/assets/
├── mla_prolog_v4_ttk_ops.py  # Public API adapter and source loader
├── spec.py  # TTK plugin and hook registration
├── impl/
│   ├── audit.py  # Contract checks and input restoration before timing
│   ├── audit_reference.py  # Contract check reference helpers
│   ├── compare.py  # Precision comparison, acceptance, and reporting
│   ├── golden.py  # CPU Golden input and output adapter
│   ├── golden_reference.py  # Access to ordinary and high-precision references
│   ├── graph.py  # ACLGraph execution entry
│   ├── inputs.py  # Input preparation after generation
│   ├── npu_preprocess.py  # Device preprocessing before timing and capture
│   ├── pytest_reference.py  # Independent numeric references and acceptance rules
│   └── transport.py  # Data carriers and storage-layout transport
├── README.md
└── README_en.md
```

## Environment setup

Prepare Python, PyTorch, torch_npu, and cannbotdsl versions compatible with the device and CANN installation.
PyTorch must support the floating-point data types used by the test cases. If using a virtual environment, activate your own environment first.

Replace the placeholder paths with paths on your machine:

```bash
export CANN_ROOT="/path/to/cann"
export DSL_REPO="/path/to/ops-transformer"
export TTK_REPO="/path/to/ops-test-kit"
source "$CANN_ROOT/set_env.sh"

export ASSETS="$DSL_REPO/attention/mla_prolog_v4/tests/assets"
export PYTHONPATH="$ASSETS:$TTK_REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$TTK_REPO"
python3 -m pip install -r requirements.txt
python3 -m ttk info
```

Set `CANNBOTDSL_CACHE_DIR` to reuse the compilation cache:

```bash
export CANNBOTDSL_CACHE_DIR="$HOME/.cache/cannbotdsl_compile"
```

Execution can include compilation when neither an installed Native artifact nor a compilation cache entry matches.

## Running tests

Run the commands below from the ops-test-kit repository root, using the environment variables defined above.
CPU Golden generation and plugin precision comparison are enabled by default. An explicit `--compare` option is usually unnecessary.

### Validate cases

Check CSV and API declarations without device computation:

```bash
python3 -m ttk e2e -i "$ASSETS/mla_prolog_v4.csv" \
    --plugin "$ASSETS" --validate
```

### Eager mode

```bash
python3 -m ttk e2e -i "$ASSETS/mla_prolog_v4.csv" \
    --plugin "$ASSETS" --devices 0 --run 1 --warmup false \
    -o mla_prolog_v4_result.csv
```

### ACLGraph mode

Add `--aclgraph` to execute `impl/graph.py::MlaPrologV4AclGraph` through the `npugraph_ex` backend of `torch.compile`.
By default, both eager and graph branches run, and each is compared with the CPU Golden.

```bash
python3 -m ttk e2e -i "$ASSETS/mla_prolog_v4.csv" \
    --plugin "$ASSETS" --devices 0 --aclgraph --run 1 --warmup false \
    -o mla_prolog_v4_aclgraph_result.csv
```

To run only the graph branch, append `--eager=false` to the command above.

### Disable Golden

`--golden-mode Disable` skips Golden generation and precision comparison and can be used to observe device execution.
Successful execution does not establish precision correctness; the precision status is `SKIP`.

```bash
python3 -m ttk e2e -i "$ASSETS/mla_prolog_v4.csv" \
    --plugin "$ASSETS" --devices 0 --golden-mode Disable --run 1 --warmup false \
    -o mla_prolog_v4_execution_result.csv
```

This option can also be combined with `--aclgraph`.

### Select devices

Use `--devices` to specify the TTK device allowlist. For example, `--devices 1` selects device 1.
`--device` specifies the number of devices, not a device ID.

Other useful options:

- `-t <case_name>`: filter cases by name.
- `--seed 42`: fix the random input seed.
- `--dump-on-fail`: save inputs, outputs, and Golden data on failure.

## Reuse saved inputs and Golden data

Generate and save data first, then load it with the same CSV to avoid regenerating inputs and recomputing the CPU Golden.
`MANUAL_DATA` below must refer to a directory dedicated to the current case set.

```bash
export MANUAL_DATA="$PWD/manual_data/mla_prolog_v4"

# Prepare data without device execution
python3 -m ttk e2e -i "$ASSETS/mla_prolog_v4.csv" \
    --plugin "$ASSETS" --no-prof --manual-data-dirs "$MANUAL_DATA" \
    --dump in,golden --dump-format pt

# Load data and execute on the device
python3 -m ttk e2e -i "$ASSETS/mla_prolog_v4.csv" \
    --plugin "$ASSETS" --manual-data-dirs "$MANUAL_DATA" --devices 0 \
    -o mla_prolog_v4_manual_result.csv
```

Keep the default `--golden-mode Enable` when loading data. Do not combine this workflow with `--validate` or `--cpu`.
Specify a single output directory for data preparation. When loading, the CSV shapes and data types are checked against the saved files.
Loading saved data does not run the input customization hook; random seeds and generation parameters do not change the saved inputs.
Use a new data directory after changing the case input definitions or reference computation.

## Precision criteria

`impl/compare.py` calls `compare_numeric_outputs` in `impl/pytest_reference.py`.
When two-way comparison fails, `fallback_segments` runs according to the reference rules.
`accept_output_comparisons` then combines whole-output and segment acceptance decisions.

The numeric acceptance log reports the acceptance status of each output.
`accepted_by` can be `TWO_WAY`, `THREE_WAY`, `SPLIT_FALLBACK`, `FAIL`, or `UNVERIFIED`.

| Output | Criterion |
|---|---|
| Q | Modes 0 and 1: BF16 mixed-error criteria; mode 2: E4M3FN code distance; mode 3: HiFloat8 reference before quantization |
| descale_q | FP32 scale mixed-error criteria; verify an empty output when disabled |
| QR | BF16 criteria; verify an empty output when disabled |
| kv_cache | Compare updated regions using valid indices; mode 1 checks KV, KR, and descale separately; compare untouched logical regions byte for byte |

Shapes and data types must satisfy the output contract. Missing comparison context, comparison errors, and unverified results do not establish a precision pass.
The match rate displayed by TTK and the final acceptance status are recorded separately. A two-way match rate is not a three-way acceptance result.
`--compare close` does not replace the acceptance rules implemented by the plugin.
