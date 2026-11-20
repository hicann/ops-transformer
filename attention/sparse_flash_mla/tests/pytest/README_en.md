# sparse_flash_mla pytest tests

English | [简体中文](./README.md)

Activate CANN and any required custom packages, and prepare compatible PyTorch, `torch_npu`, and
`torchair` versions. Run the commands below from this directory; use `bash test_run.sh --help`
for all options.

## Common Usage

### Single-case testing

Configure the cases in `sparse_flash_mla_paramset.py`:

```bash
bash test_run.sh single
```

### Generate PT files in batch

Read cases from an Excel sheet and generate CPU golden data and PT files:

```bash
bash test_run.sh save \
  -E ./excel/example.xlsx -S CSA -P ./data
```

`-E`, `-S`, and `-P` select the Excel file, sheet, and PT output directory. The script defaults
are used when they are omitted.

### Execute PT files in batch

`load` runs eager tests and `load_graph` runs ACLGraph tests. Both commands read existing PT files:

```bash
bash test_run.sh load \
  -P ./data -R ./result/smla_eager.xlsx
bash test_run.sh load_graph \
  -P ./data -R ./result/smla_graph.xlsx
```

Without `-E`, all PT files in the directory are selected. Add `-E` and `-S` to filter them by the
Excel `testcase_name` column without regenerating PT files:

```bash
bash test_run.sh load_graph \
  -P ./data -E ./excel/example.xlsx -S CSA
```

### SuperKernel

Append `--super-kernel` to a `single`, `load`, or `load_graph` command:

```bash
bash test_run.sh single --super-kernel
bash test_run.sh load_graph -P ./data --super-kernel
```

The option is disabled by default. When enabled, ACLGraph SuperKernel is used; metadata is still
generated outside the graph. `debug_per_op_max_core_num=1` enables single-operator full-core
debugging; it does not restrict execution to one core.

### Batch Consistency

The pytest suites for `mixed_quant_sparse_flash_mla` (MQSMLA) and `quant_sparse_flash_mla` (QSMLA) both depend on SMLA’s [batch_consistency module](./batch_consistency/).
When testing MQSMLA or QSMLA, keep its operator directory alongside `sparse_flash_mla/`, or copy the entire `batch_consistency/` directory from here into that operator’s `tests/pytest/` directory.

Control batch consistency with `--batch-consistency`:

```bash
bash test_run.sh load_graph -P ./data --batch-consistency on
```

Supported values are `auto` (default), `on`, and `off`. In `auto` mode, the case or PT settings
determine whether batch consistency is enabled.
