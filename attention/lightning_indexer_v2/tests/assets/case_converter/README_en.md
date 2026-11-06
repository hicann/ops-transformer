# lightning_indexer_v2 Excel Case Conversion Tool

English | [简体中文](./README.md)

This directory converts the parameter Excel table of this operator into TTK E2E or ACLNN Excel case tables.
`convert.py` handles workbook read/write and appending, and `case_schema.py` handles the tensor, attribute, and output descriptions of this operator.
The entire directory can be copied and used independently; there are no import or path dependencies between operators.

## Running Requirements

Python 3.9+ and `openpyxl>=3.1`. Conversion depends only on the Python standard library and openpyxl; PyTorch, pytest, CANN, or NPU is not required.
If an existing Python environment lacks the dependency, run `python3 -m pip install "openpyxl>=3.1"`.

The following commands are executed from the current directory. You can also replace `convert.py` with its full path and call it from any directory.
Input supports `.xlsx/.xlsm`, and the output is also Excel.

## Three Output Methods

1. Output to a directory, keeping the input file name and the selected sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. Output to a specified Excel file, creating or appending by input sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. Output to a specified sheet of a specified Excel file, merging and appending in the input selection order:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` can specify multiple names and can be repeated; quote names that contain spaces.
If it is not specified, the first sheet is read. Duplicate or nonexistent names cause an error; names are not split by commas.
`--mode` defaults to `e2e`; `--output-dir` and `-o/--output` are mutually exclusive.

When directory output or specified-file output encounters a sheet with the same name, the tool first asks whether to append.
Only `yes`, `y`, and `Y` are accepted; any other answer or end of input cancels the entire write, and other confirmed sheets are also not written.
Sheet name comparison is case-insensitive, and the existing spelling in the target is retained.
Explicitly providing `--output-sheet` indicates that the append target has been specified, and confirmation is not repeated.

## Input Table Format

The first non-empty row is the header, and one row represents one case; fully empty rows are ignored.
Two sources are supported: the pytest parameter table of this operator, and E2E description tables that already contain `tensor_view_shapes/tensor_dtypes/attributes`.
Feeding an ACLNN description table causes an error to avoid duplicating output tensors.
Lists, tuples, and dictionaries use Python literals, for example `[0, 16, 32]`, `(1, 16, 8, 128)`, and `{'layout_q': 'TND'}`.
Expressions are parsed as literals, and Python functions cannot be called.

Required parameters are `batch_size/q_seq/k_seq/q_head_num/k_head_num/head_dim`, `qk_dtype`, and the layouts of the query and key; case names support `testcase_name/Testcase_Name/case_name/Testcase_Prefix`, and aliases must be consistent.
The query supports `BSND/TND`; the V1 key supports `BSND/TND/PA_BSND`, and the V2 key supports `BSND/TND/PA_BBND`.

- V2 uses `layout_q/layout_k` and `topk/mask_mode`, and also accepts `layout_query/layout_key` and `sparse_count/sparse_mode`.
- `cu_seqlens_q/cu_seqlens_k`, `seqused_q/seqused_k`, and `cmp_residual_k` can be provided.
  TND must have `B+1` cumulative lengths, and `q_t_size/k_t_size` match the ends; PA must provide `B` `seqused_k` entries.
- `max_seqlen_q` defaults to -1; when BSND has no `seqused_q`, the full Q sequence length is used, and it can remain -1.
  `output_idx_offset` uses the two end points; `return_value` determines whether the values output is needed.

- For PA, `block_size` defaults to 128, and `block_num` defaults to being calculated by capacity; conversion is rejected when the explicit value is insufficient.
- Data range columns: `query_datarange/key_datarange/weights_datarange`; quantization operators additionally have `q_scale_datarange/k_scale_datarange`.

`is_enabled` defaults to True; an explicit False stays False, and the enabled state is not modified based on historical test results.
`precision_tolerances/absolute_precision` can explicitly override the conversion default values.
`remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are kept as the values in the table.
`attributes1`..`attributes9` are merged into `attributes`; later columns override earlier attributes with the same name.
Some fields of the output format are used only by the corresponding TTK mode.

### Original Table Sample Before Conversion

| testcase_name | batch_size | q_seq | k_seq | q_head_num | k_head_num | head_dim | qk_dtype | weight_dtype | layout_q | layout_k | cu_seqlens_q | cu_seqlens_k | seqused_q | seqused_k | topk | mask_mode | cmp_ratio | max_seqlen_q | return_value | query_datarange | key_datarange | weights_datarange |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liv2_tnd_example | 2 | 1 | 4096 | 64 | 1 | 128 | bfloat16 | bfloat16 | TND | TND | [0, 1, 2] | [0, 4096, 8192] | [1, 1] | [4096, 4096] | 2048 | 3 | 1 | 1 | 0 | [-1, 1] | [-1, 1] | [-1, 1] |

## Output and Append Contract

- E2E API: `torch.ops.cann_ops_transformer.lightning_indexer`, which contains 11 input tensor slots; optional tensors remain None.
- ACLNN API: `aclnnLightningIndexerV2`, which adds real output shapes and dtypes after the inputs and fills in `output_tensor_indexes`.
- The input and output must not be the same file; symlink/hardlink aliases of the same file are rejected, and the target file cannot be a symlink.
- Appending aligns by target header names and keeps old columns and rows; missing new columns are appended to the end of the header. One sheet cannot mix E2E and ACLNN.
- All selected sheets and the target table are checked before writing; on failure, the target file is not modified. Writing uses a temporary file in the same directory and then replaces it atomically.
- The existing target must be a normal TTK case table; appending is not performed to case sheets that are protected, contain merged cells, or are Excel structured tables.
  Existing content, formulas, and cell styles of other sheets are kept with the workbook.
- Input formulas must have calculation caches saved by Excel; the tool does not calculate formulas. If there is no cache, recalculate and save in Excel first.
- `.xlsm` directory output keeps the source VBA; appending to an existing `.xlsm` keeps the target VBA. Creating a new `.xlsm` requires an `.xlsm` source.
  Converting to `.xlsx` outputs only the case table.
- The exit code is 0 on success, and 1 on input errors, rejected appends, or write failures.

## Relationship with TTK Execution

Conversion generates parameter descriptions; actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator.
The parameter table output contains the attributes required by the existing input adapter. Execution still requires the corresponding TTK version, operator package, and device environment.

The metadata input uses `(-1,)` for delayed allocation and is generated by the existing `npu_preprocess` at execution time;
no separate metadata sheet is created, and the metadata operator is not called on the conversion machine. Existing static metadata descriptions in the input are kept.
When `return_softmax_lse/return_value` is disabled, the corresponding ACLNN optional outputs are `(0,)`.
