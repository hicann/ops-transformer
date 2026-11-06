# kv_quant_sparse_flash_attention_v2 Excel Case Conversion Tool

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

Required parameters are `B/S1/S2/N1/N2/D/K`, `layout_query/layout_kv`, `q_type`, `scale_value`, and
`actual_seq_q/actual_seq_kv`. Case names support `Testcase_Prefix/testcase_name/Testcase_Name/case_name`, and aliases must be consistent.

- query supports `BSND/TND`; KV supports `BSND/TND/PA_BSND`. Non-PA parameter tables require the query and KV layouts to be identical.
- `actual_seq_q/actual_seq_kv` must contain B values. For TND, they are cumulative lengths without a leading 0; for other layouts, they are valid lengths.
  `T1/T2` (if specified) must match the TND tail.
- `sparse_block_size` defaults to 1, and K must be divisible by it. `rope_head_dim` defaults to 64.
- For PA, `block_size` defaults to 256, and `block_num` defaults to the value computed from the valid length.
- Data ranges use `range_query/range_key/range_query_rope/range_key_rope/range_sinks`.

- `key_quant_mode/value_quant_mode` must be provided. `tile_size` defaults to 128, and `quant_scale_repo_mode` defaults to 1.
  Parameter tables construct packed KV. For the standalone dequant scale contract, use the full tensor description table.
- `kv_type=hifloat8` uses byte storage and retains the dtype semantics. The query shape includes RoPE, and the output shape excludes RoPE.
- V2 uses `enable_sinks` and `return_softmax_lse`. The current parameter conversion does not include TQ4.


`is_enabled` defaults to True. An explicit False stays False; the enabled state is not modified based on historical test results.
`precision_tolerances/absolute_precision` can explicitly override the conversion defaults.
`remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are kept as the values in the table.
`attributes1`..`attributes9` are merged into `attributes`; later columns override earlier attributes with the same name.
Some fields of the output format are used only by the corresponding TTK mode.

In the BSND layout, parameter tables allow omitting the sequence lengths of query or non-PA KV/key. Input data is generated with the full length of the corresponding S axis,
and the corresponding optional tensor slots in the output remain None. The TND prefix and the KV/key lengths of PA must be explicitly provided.

### Original Table Sample Before Conversion

| Testcase_Prefix | B | S1 | S2 | N1 | N2 | D | K | layout_query | layout_kv | q_type | kv_type | scale_value | actual_seq_q | actual_seq_kv | block_size | rope_head_dim | sparse_block_size | sparse_mode | attention_mode | key_quant_mode | value_quant_mode | tile_size | quant_scale_repo_mode | enable_sinks | return_softmax_lse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qsfa_v2_decode_example | 1 | 1 | 4096 | 64 | 1 | 512 | 2048 | TND | PA_BSND | bfloat16 | hifloat8 | 0.041666666666666664 | [1] | [4096] | 128 | 64 | 1 | 3 | 2 | 2 | 2 | 128 | 1 | True | True |

## Output and Append Contract

- E2E API: `qsfa_v2_ttk_ops.kv_quant_sparse_flash_attention_v2_ttk`, which contains 10 input tensor slots; optional tensors remain None.
- ACLNN API: `aclnnKvQuantSparseFlashAttentionV2`, which adds the real output shape and dtype after the inputs and fills in `output_tensor_indexes`.
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

The conversion generates parameter descriptions. Actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator.
Parameter table output contains the attributes required by the existing input adapters. Execution still requires the matching TTK version, operator package, and device environment.

The current assets do not register an ACLNN TestSpec for this operator, so the ACLNN output serves as an interface description for review first.

E2E execution requires adding the assets of this operator to PYTHONPATH to load the dtype adapter.
