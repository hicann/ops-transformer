# mixed_quant_sparse_flash_mla Excel Test Case Conversion Tool

English | [简体中文](./README.md)

This directory converts the parameter Excel table of this operator into TTK E2E or ACLNN Excel test case tables.
`convert.py` handles workbook read/write and appending, and `case_schema.py` describes the tensors, attributes, and outputs of this operator.
The entire directory can be copied and used independently, with no import or path dependencies between operators.

## Prerequisites

Python 3.9+ and `openpyxl>=3.1`. The conversion depends only on the Python standard library and openpyxl, and does not require PyTorch, pytest, CANN, or an NPU.
When the existing Python environment lacks the dependency, run `python3 -m pip install "openpyxl>=3.1"`.

The following commands run in the current directory. You can also replace `convert.py` with its full path and call it from any directory.
The input supports `.xlsx/.xlsm`, and the output is also Excel.

## Three Output Methods

1. Output to a directory, keeping the input file name and the selected sheet name:

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

`-s/--sheet` can specify multiple names or be used repeatedly. Quote names that contain spaces.
If not specified, the first sheet is read. Duplicate or non-existent names cause errors, and names are not split by commas.
`--mode` defaults to `e2e`, and `--output-dir` and `-o/--output` are mutually exclusive.

When directory output or specified-file output encounters a sheet with the same name, the tool first asks whether to append.
Only `yes`, `y`, and `Y` are accepted. Any other answer or end of input cancels the entire write, including sheets that have already been confirmed.
Sheet name comparison is case-insensitive, and the existing spelling in the target is preserved.
Explicitly providing `--output-sheet` means that the append target has been specified, so no repeated confirmation is needed.

## Input Table Format

The first non-empty row is the header, and each row represents a test case. Fully empty rows are ignored.
Two source types are supported: the pytest parameter table of this operator, and an E2E description table that already contains `tensor_view_shapes/tensor_dtypes/attributes`.
An input ACLNN description table causes an error, preventing duplicate appended output tensors.
Lists, tuples, and dictionaries use Python literals, for example `[0, 16, 32]`, `(1, 16, 8, 128)`, and `{'layout_q': 'TND'}`.
Expressions are parsed as literals, and calling Python functions is not allowed.

The main parameters are `B/S1/S2/N1/N2/D`, `layout_q/layout_kv`, `q_type/ori_kv_type`, `quant_mode/softmax_scale`, and `rope_head_dim`. For default dtypes and optional API attributes, see `case_schema.py` in this directory.
`testcase_name`, `Testcase_Name`, `Testcase_Prefix`, or `case_name` can be used as the test case name. Name aliases in the same row must be consistent.

- The query layout supports `BSND/TND`, and the KV layout supports `BSND/TND/PA_BBND`.
- `template_run_mode` (which also accepts `template_mode`) supports `SWA/HCA/CSA/ORI_SPARSE/ORI_CMP_SPARSE`.
  The ORI sparse branch requires `K1`, and the CMP sparse branch requires `K`. When no template field exists, the value is derived from `K/cmp_ratio`.
- `seqused_q`, `seqused_ori_kv`, `seqused_cmp_kv`, and their `cu_seqlens_*` can be provided.
  Cumulative sequences start from 0 and contain `B+1` values. The default is the full capacity of each batch.
  When non-uniform TND capacity is specified, cumulative sequences must be provided, and `T1/T2/T3` (if specified) must match the end values.
- `block_size1/block_size2` default to 128. `block_num1/block_num2` can be provided, and the conversion is rejected when they are insufficient.
- `ori_mask_mode/cmp_mask_mode`, `ori_win_left/ori_win_right`, `topk_value_mode`,
  and `return_softmax_lse` are kept as API attributes. `ori_topk_length/cmp_topk_length` use separate `*_values` data attributes to avoid name conflicts with tensor inputs. One value expands to all query/KV heads, multiple values must match the tensor size, and the range is [0, K].
- `q_datarange/ori_kv_datarange/cmp_kv_datarange` use two endpoints and support `inf/-inf/nan`.
  When deterministic values of a complete or random sequence are missing, the tool refuses to guess random results.

`is_enabled` defaults to True. An explicit False stays False, and the enabled state is not modified based on historical test results.
`precision_tolerances/absolute_precision` can explicitly override the conversion defaults.
`remark/network_name/priority/soc_series/golden_api/device_ids` and existing batch fields are kept as the values in the table.
`attributes1`..`attributes9` are merged into `attributes`, and later columns overwrite earlier attributes with the same name.
Some fields of the output format are used only by the corresponding TTK mode.

### Original Table Sample Before Conversion

| Testcase_Name | B | S1 | S2 | N1 | N2 | D | K | layout_q | layout_kv | q_type | ori_kv_type | cmp_kv_type | quant_mode | rope_head_dim | tile_size | template_run_mode | cmp_ratio | softmax_scale | block_size1 | block_size2 | cu_seqlens_q | seqused_q | seqused_ori_kv | seqused_cmp_kv | ori_mask_mode | cmp_mask_mode | ori_win_left | ori_win_right | use_sinks | return_softmax_lse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mqsmla_csa_decode_example | 1 | 1 | 8192 | 64 | 1 | 512 | 512 | TND | PA_BBND | bfloat16 | float8_e4m3fn | float8_e4m3fn | 1 | 64 | 64 | CSA | 4 | 0.04419417 | 128 | 128 | [0, 1] | [1] | [8192] | [2048] | 4 | 3 | 127 | 0 | True | False |

## Output and Append Contract

- E2E API: `torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla`, containing 18 input tensor slots. Optional tensors remain None.
- ACLNN API: `aclnnMixedQuantSparseFlashMla`, which adds the real output shapes and dtypes after the inputs and fills `output_tensor_indexes`.
- The input and output must not be the same file. Symlink/hardlink aliases of the same file are rejected, and the target file cannot be a symlink.
- Appending aligns by target header names and keeps old columns and old rows. Missing new columns are appended to the end of the header. One sheet cannot mix E2E and ACLNN.
- All selected sheets and the target table are checked before writing. A failure does not modify the target file. Writing uses a temporary file in the same directory and then performs an atomic replacement.
- The existing target must be a plain TTK test case table. The tool does not append to case sheets that are protected, contain merged cells, or are Excel structured tables.
  Existing content, formulas, and cell styles of other sheets are preserved with the workbook.
- Input formulas must have the calculation cache saved by Excel, and the tool does not calculate formulas. When no cache exists, recalculate and save in Excel first.
- `.xlsm` directory output keeps the source VBA, and appending to an existing `.xlsm` keeps the target VBA. Creating a new `.xlsm` requires an `.xlsm` source.
  Converting to `.xlsx` outputs only the test case table.
- The exit code is 0 on success, and 1 on input errors, refused appends, or write failures.

## Relationship with TTK Execution

When `return_softmax_lse=False`, ACLNN writes the optional LSE output slot as `None`; when it is enabled, the real shape and dtype are kept.

The conversion generates parameter descriptions, and actual execution reuses the inputs, Golden, and TestSpec in the assets of this operator.
The parameter table output contains the attributes required by the existing input adapters. Execution still requires the matching TTK version, operator package, and device environment.

The metadata input uses `(-1,)` for deferred allocation and is generated at execution time by the existing `npu_preprocess`.
No separate metadata sheet is created, and the metadata operator is not called on the conversion machine. The existing static metadata description in the input is preserved.

The parameter table supports quant_mode 1/2/3. Mode 3 converts according to the A2/A3 TurboQuant CSA constraints:
TND/PA_BBND, D=512, N2=1, and N1 a multiple of 4 from 4 to 128. q/ori_kv use the same FP16/BF16,
cmp_kv uses UINT8 with a last dimension of 258, K is 512/1024, cmp_ratio is 4/128, and the block_size of both paths is a multiple of 16 from 16 to 1024.
cu_seqlens_q and seqused_ori_kv are constructed by default from B/S1/S2. Non-uniform sequences can be explicitly provided, and T1 supports 0.
Mode 3 rejects seqused_q, seqused_cmp_kv, the prefixes of both KV paths, residual, and topk_length.
mask is fixed to 4/3, ori_win_right is fixed to 0, topk_value_mode is fixed to 1, and ori_win_left is non-negative.
The optional sinks input is controlled by use_sinks, which defaults to True.

The independent CPU input branch in the assets of this operator generates valid TQ4 nibbles, FP16 scales, PA page tables, and causal CMP indexes,
enabling the TTK input generator to construct data on the CPU. The complete CLI of Test-kit 0920 still forbids manual-data prepare for dynamic metadata,
so the input generation capability is not equivalent to a runnable full CLI prepare, and the related framework constraints need to be lifted separately.
The pytest general generator and CPU Golden still support only modes 1/2.
Complete Golden/device precision verification for mode 3 is not part of the input preparation function, and incorrectly calling Golden produces a clear diagnosis.
