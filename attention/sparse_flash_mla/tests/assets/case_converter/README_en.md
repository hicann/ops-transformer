# sparse_flash_mla Excel Case Conversion Tool

English | [简体中文](./README.md)

This directory converts the parameter Excel table of this operator into a TTK E2E or ACLNN Excel case table.
`convert.py` handles workbook reading, writing, and appending, and `case_schema.py` describes the tensors, attributes, and outputs of this operator.
The entire directory can be copied and used independently; the operators have no import or path dependencies on each other.

## Prerequisites

Python 3.9+ and `openpyxl>=3.1`. The conversion depends only on the Python standard library and openpyxl; PyTorch, pytest, CANN, and an NPU are not required.
If the existing Python environment lacks the dependency, run `python3 -m pip install "openpyxl>=3.1"`.

Run the following commands from the current directory. You can also replace `convert.py` with its full path and call it from any directory.
The input supports `.xlsx/.xlsm`, and the output is also Excel.

## Three Output Modes

1. Output to a directory, keeping the input file name and the selected sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. Output to a specified Excel file, creating or appending sheets by input sheet name:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. Output to a specified sheet in a specified Excel file, merging and appending in the order of the input selection:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` accepts multiple names and can be repeated; quote names that contain spaces.
If the option is not specified, the tool reads the first sheet. Duplicate or nonexistent names cause an error; names are not split by commas.
`--mode` defaults to `e2e`, and `--output-dir` and `-o/--output` are mutually exclusive.

For directory output and specified-file output, the tool first asks whether to append when it encounters a sheet with the same name.
Only `yes`, `y`, and `Y` are accepted. Any other answer or an end of input cancels the entire write, including the sheets that were already confirmed.
Sheet name comparison is case-insensitive and keeps the existing spelling in the target file.
Explicitly providing `--output-sheet` means the append target is already specified, so no repeated confirmation is needed.

## Input Table Format

The first non-empty row is the header, and each row represents a case; fully empty rows are ignored.
Two sources are supported: the pytest parameter table of this operator, and an E2E description table that already contains `tensor_view_shapes/tensor_dtypes/attributes`.
An ACLNN description table as input causes an error, which avoids duplicating the appended output tensors.
Lists, tuples, and dictionaries use Python literals, for example `[0, 16, 32]`, `(1, 16, 8, 128)`, and `{'layout_q': 'TND'}`.
Expressions are parsed as literals and cannot call Python functions.

The main parameters are `B/S1/S2/N1/N2/D`, `layout_q/layout_kv`, and `q_type/ori_kv_type`. For dtype defaults and optional API attributes, see `case_schema.py` in this directory.
`testcase_name`, `Testcase_Name`, `Testcase_Prefix`, or `case_name` can serve as the case name; the name aliases in the same row must be consistent.

- The query layout supports `BSND/TND`; the KV layout supports `BSND/TND/PA_BBND`.
- `template_run_mode` (also accepts `template_mode`) supports `SWA/HCA/CSA/ORI_SPARSE/ORI_CMP_SPARSE`.
  The ORI sparse branch requires `K1`, and the CMP sparse branch requires `K`; when the template field is absent, the value is derived from `K/cmp_ratio`.
- `seqused_q`, `seqused_ori_kv`, `seqused_cmp_kv`, and their `cu_seqlens_*` values can be provided.
  A cumulative sequence starts from 0 and contains `B+1` values; the default is the full capacity of each batch.
  When non-uniform TND capacity is specified, the cumulative sequence must be provided, and `T1/T2/T3` (if specified) must match the last element.
- `block_size1/block_size2` default to 128; `block_num1/block_num2` can be provided, and the conversion is rejected when they are insufficient.
- `ori_mask_mode/cmp_mask_mode`, `ori_win_left/ori_win_right`, `topk_value_mode`, and
  `return_softmax_lse` are kept as API attributes. `ori_topk_length/cmp_topk_length` use separate `*_values` data attributes to avoid name conflicts with the tensor inputs; a single value is expanded to all query/KV heads, multiple values must match the tensor size, and the range is [0, K].
- `q_datarange/ori_kv_datarange/cmp_kv_datarange` use two endpoints and support `inf/-inf/nan`.
  When deterministic values for the full or random sequence are missing, the tool refuses to guess random results.

`is_enabled` defaults to True; an explicit False stays False, and the enabled state is not modified based on historical test results.
`precision_tolerances/absolute_precision` can explicitly override the conversion defaults.
`remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are kept as the values in the table.
`attributes1`..`attributes9` are merged into `attributes`; later columns override earlier attributes with the same name.
Some fields of the output format are used only by the corresponding TTK mode.

### Original Table Sample Before Conversion

| Testcase_Name | B | S1 | S2 | N1 | N2 | D | K | layout_q | layout_kv | q_type | ori_kv_type | cmp_kv_type | template_run_mode | cmp_ratio | softmax_scale | block_size1 | block_size2 | cu_seqlens_q | seqused_q | seqused_ori_kv | seqused_cmp_kv | ori_mask_mode | cmp_mask_mode | ori_win_left | ori_win_right | use_sinks | return_softmax_lse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smla_csa_decode_example | 1 | 1 | 8192 | 64 | 1 | 512 | 512 | TND | PA_BBND | bfloat16 | bfloat16 | bfloat16 | CSA | 4 | 0.04419417 | 128 | 128 | [0, 1] | [1] | [8192] | [2048] | 4 | 3 | 127 | 0 | True | False |

When a data range needs to be specified, `q_datarange/ori_kv_datarange/cmp_kv_datarange` columns can be added, for example `[-1, 1]`.

## Output and Append Contract

- E2E API: `torch.ops.cann_ops_transformer.sparse_flash_mla`, which contains 18 input tensor slots; optional tensors stay None.
- ACLNN API: `aclnnSparseFlashMla`, which adds the real output shapes and dtypes after the inputs and fills in `output_tensor_indexes`.
- The input and output cannot be the same file; symlink/hardlink aliases within the same file are rejected, and the target file cannot be a symlink.
- Appending aligns by the target header names and keeps old columns and old rows; missing new columns are appended to the end of the header. One sheet cannot contain both E2E and ACLNN cases.
- The tool checks all selected sheets and the target table before writing; a failure does not modify the target file. The write uses a temporary file in the same directory, followed by an atomic replacement.
- The existing target must be a normal TTK case table; the tool does not append to case sheets that are protected, contain merged cells, or are Excel structured tables.
  Existing content, formulas, and cell styles of other sheets are kept with the workbook.
- Input formulas must have calculation caches saved by Excel; the tool does not compute formulas. If no cache exists, recalculate and save in Excel first.
- Directory output to `.xlsm` keeps the source VBA; appending to an existing `.xlsm` keeps the target VBA. Creating a new `.xlsm` requires an `.xlsm` source.
  Conversion to `.xlsx` outputs only the case table.
- The exit code is 0 on success, and 1 on input errors, rejected appends, or write failures.

## Relation to TTK Execution

When `return_softmax_lse=False`, ACLNN writes the optional LSE output slot as `None`; when it is enabled, the real shape and dtype are kept.

The conversion generates the parameter description; actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator.
The parameter table output contains the attributes required by the existing input adapter. Execution still requires the corresponding version of TTK, the operator package, and the device environment.

The metadata input uses `(-1,)` for deferred allocation and is generated by the existing `npu_preprocess` at execution time;
no separate metadata sheet is created, and the metadata operator is not called on the conversion machine. Existing static metadata descriptions in the input are kept.
