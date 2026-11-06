# lightning_indexer Excel Case Conversion Tool

English | [简体中文](./README.md)

This directory converts the parameter Excel tables of this operator into TTK E2E or ACLNN Excel case tables.
`convert.py` handles workbook read/write and append operations, and `case_schema.py` describes the tensors, attributes, and outputs of this operator.
The entire directory can be copied and used independently; no import or path dependencies exist between operators.

## Running Conditions

Python 3.9+ and `openpyxl>=3.1` are required. The conversion depends only on the Python standard library and openpyxl. PyTorch, pytest, CANN, and an NPU are not required.
If an existing Python environment lacks the dependency, run `python3 -m pip install "openpyxl>=3.1"`.

The following commands run from the current directory. You can also replace `convert.py` with its full path and call it from any directory.
The input supports `.xlsx/.xlsm`, and the output is also Excel.

## Three Output Modes

1. Output to a directory, keeping the input file name and the selected sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. Output to a specified Excel file, creating or appending sheets named after the input sheets:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. Output to a specified sheet of a specified Excel file, merging and appending in the input selection order:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` accepts multiple names and can be repeated. Enclose names that contain spaces in quotes.
If not specified, the first sheet is read. Duplicate or nonexistent names cause errors. Names are not split by commas.
`--mode` defaults to `e2e`. `--output-dir` and `-o/--output` are mutually exclusive.

When directory output or specified-file output encounters a sheet with the same name, the tool first asks whether to append.
Only `yes`, `y`, and `Y` are accepted. Any other answer or end of input cancels the entire write, including the sheets that have already been confirmed.
Sheet name comparison is case-insensitive, and the existing spelling in the target file is retained.
Explicitly providing `--output-sheet` means the append target is already specified, so no repeated confirmation occurs.

## Input Table Format

The first non-empty row is the header, and each row represents one case. Fully empty rows are ignored.
Two source types are supported: the pytest parameter table of this operator, and an E2E description table that already contains `tensor_view_shapes/tensor_dtypes/attributes`.
Feeding an ACLNN description table causes an error, preventing duplicated output tensors from being appended.
Lists, tuples, and dictionaries use Python literals, such as `[0, 16, 32]`, `(1, 16, 8, 128)`, and `{'layout_q': 'TND'}`.
Expressions are parsed as literals and cannot call Python functions.

Required parameters are `batch_size/q_seq/k_seq/q_head_num/k_head_num/head_dim`, `qk_dtype`, and the layouts of query and key. Case names support `testcase_name/Testcase_Name/case_name/Testcase_Prefix`, and aliases must be consistent.
query supports `BSND/TND`. The V1 key supports `BSND/TND/PA_BSND`, and the V2 key supports `BSND/TND/PA_BBND`.

- V1 uses `layout_query/layout_key` and `sparse_count/sparse_mode`. `act_seq_q/act_seq_k` must be provided.
  TND uses cumulative lengths without a leading 0; BSND and PA use the valid length of each batch.
  `weight_dtype`, `actual_seq_dtype`, and the `*_datarange` of each input can be explicitly specified.

- For PA, `block_size` defaults to 128, and `block_num` defaults to the value computed from the capacity. If the explicitly specified value is insufficient, the conversion is rejected.
- Data range columns: `query_datarange/key_datarange/weights_datarange`. Quantization operators additionally have `q_scale_datarange/k_scale_datarange`.

`is_enabled` defaults to True. An explicit False stays False; the enabled state is not modified based on historical test results.
`precision_tolerances/absolute_precision` can explicitly override the conversion defaults.
`remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are kept as the values in the table.
`attributes1`..`attributes9` are merged into `attributes`; later columns override earlier attributes with the same name.
Some fields of the output format are used only by the corresponding TTK mode.

In the BSND layout, parameter tables allow omitting the sequence lengths of query or non-PA KV/key. Input data is generated with the full length of the corresponding S axis,
and the corresponding optional tensor slots in the output remain None. The TND prefix and the KV/key lengths of PA must be explicitly provided.

### Original Table Sample Before Conversion

| testcase_name | batch_size | q_seq | k_seq | q_head_num | k_head_num | head_dim | qk_dtype | weight_dtype | actual_seq_dtype | layout_query | layout_key | act_seq_q | act_seq_k | sparse_count | sparse_mode | return_value | query_datarange | key_datarange | weights_datarange |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| li_tnd_example | 2 | 1 | 4096 | 64 | 1 | 128 | bfloat16 | bfloat16 | int32 | TND | TND | [1, 2] | [4096, 8192] | 2048 | 3 | False | [-1, 1] | [-1, 1] | [-1, 1] |

## Output and Append Contract

- E2E API: `torch_npu.npu_lightning_indexer`, which contains 6 input tensor slots; optional tensors remain None.
- ACLNN API: `aclnnLightningIndexer`, which adds the real output shape and dtype after the inputs and fills in `output_tensor_indexes`.
- The input and output must not be the same file. Symlink/hardlink aliases of the same file are rejected, and the target file must not be a symlink.
- Appending aligns by target header names and keeps old columns and old rows. Missing new columns are appended to the end of the header. One sheet must not mix E2E and ACLNN.
- All selected sheets and the target table are checked before writing. On failure, the target file is not modified. Writing uses a temporary file in the same directory, followed by an atomic replacement.
- The existing target must be a plain TTK case table. Appending is not performed for protected sheets, sheets with merged cells, or Excel structured tables.
  The existing content, formulas, and cell styles of other sheets are preserved with the workbook.
- Input formulas must have calculation caches saved by Excel; the tool does not calculate formulas. If no cache exists, recalculate in Excel and save first.
- Directory output to `.xlsm` keeps the source VBA, and appending to an existing `.xlsm` keeps the target VBA. Creating a new `.xlsm` requires an `.xlsm` source.
  Converting to `.xlsx` outputs only the case table.
- The exit code is 0 on success and 1 on input errors, rejected appends, or write failures.

## Relationship with TTK Execution

When `return_value=False`, ACLNN writes the values output slot as `None`; when it is enabled, the real shape and dtype are kept.

The conversion generates parameter descriptions. Actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator.
Parameter table output contains the attributes required by the existing input adapters. Execution still requires the matching TTK version, operator package, and device environment.
