# quant_lightning_indexer Excel Case Conversion Tool

English | [简体中文](./README.md)

This directory converts the parameter Excel table of this operator into a TTK E2E or ACLNN Excel case table. `convert.py` handles workbook read/write and append operations, and `case_schema.py` handles the tensor, attribute, and output descriptions of this operator. The entire directory can be copied and used independently, and there are no import or path dependencies between operators.

## Running Requirements

Python 3.9+ and `openpyxl>=3.1` are required. The conversion depends only on the Python standard library and openpyxl; PyTorch, pytest, CANN, and an NPU are not needed. If an existing Python environment lacks the dependency, run `python3 -m pip install "openpyxl>=3.1"`.

The following commands are executed from the current directory. You can also replace `convert.py` with its full path and call it from any directory. The input supports `.xlsx/.xlsm`, and the output is also Excel.

## Three Output Modes

1. Output to a directory, keeping the input file name and the selected sheet names:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. Output to a specified Excel file, creating or appending by input sheet name:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. Output to a specified sheet of a specified Excel file, merging and appending in the input selection order:

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` can specify multiple names and can also be used repeatedly; quote names that contain spaces. When it is not specified, the first sheet is read; duplicate or nonexistent names report errors; names are not split by commas. `--mode` defaults to `e2e`, and `--output-dir` and `-o/--output` are mutually exclusive.

When directory output or specified-file output encounters a sheet with the same name, the tool first asks whether to append. Only `yes`, `y`, and `Y` are accepted; any other answer or the end of input cancels the entire write, including sheets that were already confirmed. Sheet name comparison is case-insensitive, and the existing spelling in the target is retained. Explicitly providing `--output-sheet` means the append target is already specified, so no repeated confirmation is performed.

## Input Table Format

The first non-empty row is the header, and one row represents one case; fully empty rows are ignored. Two sources are supported: the pytest paramset of this operator, and an E2E description table that already contains `tensor_view_shapes/tensor_dtypes/attributes`. Inputting an ACLNN description table reports an error to avoid appending output tensors repeatedly. Lists, tuples, and dictionaries use Python literals, for example `[0, 16, 32]`, `(1, 16, 8, 128)`, and `{'layout_q': 'TND'}`. Expressions are parsed as literals, and Python functions cannot be called.

Required parameters are `batch_size/q_seq/k_seq/q_head_num/k_head_num/head_dim`, `qk_dtype`, the layouts of query and key, and `dequant_dtype`. Case names support `testcase_name/Testcase_Name/case_name/Testcase_Prefix`, and aliases must be consistent. query supports `BSND/TND`; V1 key supports `BSND/TND/PA_BSND`, and V2 key supports `BSND/TND/PA_BBND`.

- V1 uses `layout_query/layout_key` and `sparse_count/sparse_mode`; `act_seq_q/act_seq_k` must be provided.
  TND uses cumulative lengths without a leading 0; BSND and PA use the valid length of each batch.
  `weight_dtype`, `actual_seq_dtype`, and the `*_datarange` of each input can be specified explicitly.

- `query_quant_mode/key_quant_mode` must be provided; HiFloat8 semantics can be declared through `qk_dtype=hifloat8`.

- For PA, `block_size` defaults to 128, and `block_num` defaults to being computed by capacity; the conversion is refused when the explicit value is insufficient.
- Data range columns: `query_datarange/key_datarange/weights_datarange`; quantized operators additionally have `q_scale_datarange/k_scale_datarange`.

`is_enabled` defaults to True; an explicit False stays False, and the enabled state is not modified based on historical test results. `precision_tolerances/absolute_precision` can explicitly override the conversion defaults. `remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are kept as the values in the table. `attributes1` to `attributes9` are merged into `attributes`; later columns override earlier attributes with the same name. Some fields of the output format are used only by the corresponding TTK mode.

Under the BSND layout, the paramset allows omitting the query or non-PA KV/key sequence lengths, generates input data with the full length of the corresponding S axis, and keeps the corresponding optional tensor slots as None in the output; the TND prefix and the PA KV/key lengths must be provided explicitly.

### Original Table Sample Before Conversion

| testcase_name | batch_size | q_seq | k_seq | q_head_num | k_head_num | head_dim | qk_dtype | weight_dtype | dequant_dtype | actual_seq_dtype | layout_query | layout_key | act_seq_q | act_seq_k | sparse_count | sparse_mode | query_quant_mode | key_quant_mode | block_size | query_datarange | key_datarange | weights_datarange | q_scale_datarange | k_scale_datarange |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qli_paged_example | 1 | 1 | 4096 | 64 | 1 | 128 | float8_e4m3fn | bfloat16 | float32 | int32 | BSND | PA_BSND | [1] | [4096] | 2048 | 3 | 0 | 0 | 128 | [-1, 1] | [-1, 1] | [-1, 1] | [0.01, 1] | [0.01, 1] |
## Output and Append Contract

- E2E API: `torch_npu.npu_quant_lightning_indexer`, containing 8 input tensor slots; optional tensors stay None.
- ACLNN API: `aclnnQuantLightningIndexer`, which adds the real output shape and dtype after the inputs and fills in `output_tensor_indexes`.
- The input and output must not be the same file; symlink/hardlink aliases of the same file are refused, and the target file cannot be a symlink.
- Appends are aligned by target header names, retaining old columns and old rows; missing new columns are appended to the end of the header. One sheet cannot mix E2E and ACLNN.
- All selected sheets and the target table are checked before writing; on failure, the target file is not modified. Writing uses a temporary file in the same directory, followed by an atomic replacement.
- The existing target must be a plain TTK case table; appends are not performed to case sheets that are protected, contain merged cells, or are Excel structured tables.
  The existing content, formulas, and cell styles of other sheets are retained with the workbook.
- Input formulas must have the calculation cache saved by Excel; the tool does not compute formulas. If there is no cache, recalculate and save in Excel first.
- `.xlsm` directory output retains the source VBA; appending to an existing `.xlsm` retains the target VBA. Creating a new `.xlsm` requires an `.xlsm` source.
  Converting to `.xlsx` outputs only the case table.
- The exit code is 0 on success and 1 on input errors, refused appends, or write failures.

## Relationship with TTK Execution

The conversion generates parameter descriptions; actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator. The paramset output contains the attributes required by the existing input adapters. Execution still requires the corresponding version of TTK, the operator package, and the device environment.

The current assets do not register an ACLNN TestSpec for this operator, so the ACLNN output first serves as an interface description for review.
