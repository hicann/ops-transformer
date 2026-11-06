# sparse_flash_attention Excel Test Case Conversion Tool

English | [简体中文](./README.md)

This directory converts the parameter Excel table of this operator into TTK E2E or ACLNN Excel test case tables. `convert.py` handles workbook read/write and appending, and `case_schema.py` describes the tensors, attributes, and outputs of this operator. The entire directory can be copied and used independently, without import or path dependencies between operators.

## Prerequisites

Python 3.9+ and `openpyxl>=3.1`. Conversion depends only on the Python standard library and openpyxl; PyTorch, pytest, CANN, and an NPU are not required. If the existing Python environment lacks the dependency, run `python3 -m pip install "openpyxl>=3.1"`.

Run the following commands from the current directory. You can also replace `convert.py` with its full path and call it from any directory. The input supports `.xlsx/.xlsm`, and the output is also Excel.

## Three Output Modes

1. Output to a directory, keeping the input file name and the selected sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. Output to a specified Excel file, creating or appending sheets based on the input sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. Output to a specified sheet in a specified Excel file, merging and appending in the input selection order:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` accepts multiple names and can be repeated. Quote names that contain spaces. If it is not specified, the first sheet is read. Duplicate or nonexistent names raise errors. Names are not split by commas. `--mode` defaults to `e2e`, and `--output-dir` and `-o/--output` are mutually exclusive.

When directory output or specified-file output encounters a sheet with the same name, the tool first asks whether to append. Only `yes`, `y`, and `Y` are accepted. Any other answer or end of input cancels the entire write, including the sheets that have already been confirmed. Sheet name comparison is case-insensitive, and the existing spelling in the target is preserved. Explicitly providing `--output-sheet` means that the append target is already specified, so confirmation is not repeated.

## Input Table Format

The first non-empty row is the header, and each row represents one test case; fully empty rows are ignored. Two source types are supported: the pytest parameter table of this operator, and an E2E description table that already contains `tensor_view_shapes/tensor_dtypes/attributes`. Feeding in an ACLNN description table raises an error to avoid appending duplicate output tensors. Lists, tuples, and dictionaries use Python literals, for example `[0, 16, 32]`, `(1, 16, 8, 128)`, and `{'layout_q': 'TND'}`. Expressions are parsed as literals and cannot call Python functions.

Required parameters are `B/S1/S2/N1/N2/D/K`, `layout_query/layout_kv`, `q_type`, `scale_value`, and `actual_seq_q/actual_seq_kv`. Test case names support `Testcase_Prefix/testcase_name/Testcase_Name/case_name`, and aliases must be consistent.

- Query supports `BSND/TND`; KV supports `BSND/TND/PA_BSND`. Non-PA parameter tables require the query and KV layouts to be identical.
- `actual_seq_q/actual_seq_kv` must have B values. For TND, they are cumulative lengths without a leading 0; for other layouts, they are valid lengths. `T1/T2` (if specified) must match the end of the TND values.
- `sparse_block_size` defaults to 1, and K must be divisible by it; `rope_head_dim` defaults to 64.
- For PA, `block_size` defaults to 256, and `block_num` is calculated from valid lengths by default.
- Data ranges use `range_query/range_key/range_query_rope/range_key_rope/range_sinks`.

- `q_type/kv_type` supports the FP16/BF16 aliases. Use `use_sinks=True` to enable sinks.


`is_enabled` defaults to True; an explicit False remains False, and the enabled state is not modified based on historical test results. `precision_tolerances/absolute_precision` can explicitly override the conversion defaults. `remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are preserved with the values in the table. `attributes1`..`attributes9` are merged into `attributes`; later columns overwrite earlier attributes with the same name. Some fields of the output format are used only by the corresponding TTK mode.

In the BSND layout, the parameter table allows omitting the sequence lengths of query or non-PA KV/key. Input data is generated with the full length of the corresponding S axis, and the corresponding optional tensor slots in the output remain None. TND prefix and PA KV/key lengths must be provided explicitly.

### Original Table Sample Before Conversion

| Testcase_Prefix | B | S1 | S2 | N1 | N2 | D | K | layout_query | layout_kv | q_type | kv_type | scale_value | actual_seq_q | actual_seq_kv | block_size | rope_head_dim | sparse_block_size | sparse_mode | attention_mode | use_sinks | return_softmax_lse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sfa_decode_example | 1 | 1 | 4096 | 64 | 1 | 512 | 2048 | TND | PA_BSND | bfloat16 | bfloat16 | 0.041666666666666664 | [1] | [4096] | 128 | 64 | 1 | 3 | 2 | False | False |

When a data range needs to be specified, columns such as `range_query` and `range_key` can be added, for example, fill a cell with `[-1, 1]`.

## Output and Append Contract

- E2E API: `torch.ops.cann_ops_transformer.sparse_flash_attention`, which contains 10 input tensor slots; optional tensors remain None.
- ACLNN API: `aclnnSparseFlashAttentionV2`, which adds the actual output shapes and dtypes after the inputs and fills in `output_tensor_indexes`.
- When `return_softmax_lse=False`, ACLNN writes the two softmax output slots as `None` and does not create empty tensors; when it is enabled, the real shape and dtype are kept.
- The input and output must not be the same file. Symlink/hardlink aliases of the same file are rejected, and the target file must not be a symlink.
- Appending aligns by target header names and keeps existing columns and rows; missing new columns are appended to the end of the header. A sheet cannot contain both E2E and ACLNN cases.
- All selected sheets and the target table are checked before writing. On failure, the target file is not modified. Writing uses a temporary file in the same directory, followed by an atomic replacement.
- The existing target must be a plain TTK test case table. Appending is not performed on case sheets that are protected, contain merged cells, or are Excel structured tables. The existing content, formulas, and cell styles of other sheets are preserved with the workbook.
- Input formulas must have calculation caches saved by Excel; the tool does not compute formulas. If no cache exists, recalculate and save the file in Excel first.
- Directory output for `.xlsm` preserves the source VBA, and appending to an existing `.xlsm` preserves the target VBA. Creating a new `.xlsm` requires an `.xlsm` source. Converting to `.xlsx` outputs only the case table.
- The exit code is 0 on success and 1 on input errors, rejected appends, or write failures.

## Relationship with TTK Execution

Conversion generates parameter descriptions. Actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator. The parameter table output contains the attributes required by the existing input adapters. Execution still requires a matching TTK version, the operator package, and a device environment.
