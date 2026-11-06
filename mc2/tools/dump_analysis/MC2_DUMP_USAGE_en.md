# Usage Guide for the MC2 Dump Analysis Scripts

This document describes how to use the MC2 DFX Dump analysis scripts in `mc2/tools/dump_analysis`. For routine testing, always use `parse_mc2_dump.py`. When you pass a Dump file, the script parses only that file and generates the corresponding TXT file. When you pass an operator name and the Dump root directory, the script parses the directory in batch and generates an independent, directly viewable TXT file for each source file.

The unified entry point does not parse `exception_info.*`, does not generate NPY files, and does not generate the merged `analysis.txt`.

## 1. Supported Operators and Paths

| Operator | Path | Recommended Alias |
| --- | --- | --- |
| `AllGatherMatmulV2` | CCU | `agmmv2` |
| `AllGatherMatmulV3` | AIV + URMA | `agmmv3` |
| `AlltoAllMatmul` | Mainline CCU | `a2amm` |
| `AlltoAllMatmul` | Apace CCU | `a2amm` |
| `AlltoAllMatmulV2` | AIV + URMA | `a2ammv2` |
| `MatmulReduceScatterV2` | CCU | `mmrs` |

The mainline CCU and Apace CCU paths of `AlltoAllMatmul` use the same command. The script automatically identifies the path based on TilingData.

## 2. Environment Requirements

- Python 3.8 or later.
- The unified entry point uses only the Python standard library; NumPy is not required.
- Run the command in the `mc2/tools/dump_analysis` directory, or use the absolute path of the script.

```bash
cd ops-transformer_dfx/mc2/tools/dump_analysis
```

## 3. Minimal Usage

```bash
python3 parse_mc2_dump.py <operator> [dump_dir]
```

- `<operator>`: Operator name or alias, case-insensitive.
- `[dump_dir]`: Dump root directory; if omitted, the current directory is used by default.

Common commands:

```bash
# AllGatherMatmulV2, CCU
python3 parse_mc2_dump.py agmmv2 /path/to/dump

# AllGatherMatmulV3, AIV + URMA
python3 parse_mc2_dump.py agmmv3 /path/to/dump

# AlltoAllMatmul, automatically identifies mainline CCU or Apace CCU
python3 parse_mc2_dump.py a2amm /path/to/dump

# AlltoAllMatmulV2, AIV + URMA
python3 parse_mc2_dump.py a2ammv2 /path/to/dump

# MatmulReduceScatterV2, CCU
python3 parse_mc2_dump.py mmrs /path/to/dump
```

If the current directory is the Dump root directory, the directory parameter can be omitted:

```bash
python3 /path/to/ops-transformer_dfx/mc2/tools/dump_analysis/parse_mc2_dump.py mmrs
```

## 4. Automatically Identified Files

### 4.1 Core Files for All MC2 DFX Paths

| File | Expected | Parsed Content |
| --- | --- | --- |
| `tiling_data_*` | Required | Tiling parameters, communication parameters, WorkspaceLayout, and the offset and length of each Segment |
| `args_info_*` | Required | GM addresses corresponding to Kernel parameters |
| `workspace_segN_typeM_*` | Required | Stores the actually collected Workspace Segments; `N` starts from 0 and the order is consistent with the WorkspaceLayout in the current TilingData |

"Required" here means that the operator has correctly enabled MC2 DFX Dump. If any type of file is missing, first check the DFX switch, the exception callback registration, the WorkspaceLayout in TilingData, and whether the Dump directory is correct.

The updated code has two TilingData prefixes. The parsing script selects the prefix based on the operator and no longer assumes that WorkspaceLayout is always located at offset 0:

| Operator | TilingData Prefix | WorkspaceLayout Location |
| --- | --- | ---: |
| AllGatherMatmulV2, AlltoAllMatmul mainline/Apace, MatmulReduceScatterV2 | `Mc2InitTiling → Mc2CcTiling → DfxDumpInfo` | Current default offset 576 |
| AllGatherMatmulV3, AlltoAllMatmulV2 | `DfxDumpInfo` is the first member | offset 0 |

The current source code always declares the `dumpInfo` member in each TilingData. Whether DFX content is included is controlled by `MC2_DFX_ENABLE` inside `DfxDumpInfo`. When DFX is enabled, WorkspaceLayout is at offset 0 relative to `DfxDumpInfo`, the total Peermem size is at relative offset 208, PeermemLayout is at relative offset 216, and the total size of `DfxDumpInfo` is 424 bytes. The parsing script reads the Dump layout with DFX enabled.

### 4.2 CCU Path Files

The following files belong only to the CCU path, which includes AllGatherMatmulV2, AlltoAllMatmul mainline/Apace, and MatmulReduceScatterV2:

| File | Expected | Parsed Content |
| --- | --- | --- |
| `xn_addr_info_*` | Expected | XN address table, parsed as little-endian `uint64` with 64 addresses per row |
| `cke_addr_info_*` | Expected | CKE address table, parsed as little-endian `uint64` with 64 addresses per row |

The AIV + URMA path has no XN/CKE files.

### 4.3 AIV + URMA Path Files

The following files are generated only for the URMA path, that is, AllGatherMatmulV3 and AlltoAllMatmulV2:

| File | Expected | Parsed Content |
| --- | --- | --- |
| `comm_context_*` | Expected | URMA communication context, including rank, channel, and communication buffer addresses; the result also provides the command for using `parse_matrix.py` to view raw bytes |

The CCU path has no `comm_context_*` files.

### 4.4 Files Not Parsed by the Unified Entry Point

- `exception_info.*`: The unified entry point no longer reads or parses this file.
- `peermem_window_*` and `peermem_segN_typeM_*`: The unified entry point does not read or parse these files.
- Operator `.o`, `*_host.o`, and `.json` files: These files are not MC2 Tiling/Workspace parsing inputs of this script and are ignored.

## 5. TXT Output Directory and Content for Batch Parsing

When you use `python3 parse_mc2_dump.py <operator> [dump_dir]` for batch parsing, the results are output to by default:

```text
<dump_dir>/mc2_dump_analysis/<standard_operator_name>/
```

If the original Dump contains rank subdirectories, the output retains the same directory hierarchy. For example:

```text
<dump_dir>/
└── mc2_dump_analysis/
    └── MatmulReduceScatterV2/
        ├── 0/
        │   ├── tiling_data_MatmulReduceScatterV2.xxx.txt
        │   ├── args_info_MatmulReduceScatterV2.xxx.txt
        │   ├── workspace_seg0_type8_MatmulReduceScatterV2.xxx.txt
        │   └── workspace_seg1_type12_MatmulReduceScatterV2.xxx.txt
        └── 1/
            └── ...
```

Single-file mode also generates one-to-one corresponding TXT files. However, because the command does not separately provide the Dump root directory, the output location starts from the directory containing the original file. For details, see Section 6.

Output rules:

- Each identified source file generates a `.txt` file with the same name. Files from different invocations and different ranks are not mixed.
- No NPY files are generated.
- The summary `analysis.txt` is not generated. When you run the script again, the legacy file left by the old version is deleted.
- The parsing results are also displayed in the terminal, making it easy to directly observe the execution.
- The unified entry point parses only the supported files actually scanned. It does not report an error separately because a type of file is missing. Manually check against Section 4 whether the files expected for the current path are complete.

Main content of each type of TXT file:

| Type | TXT Content |
| --- | --- |
| TilingData | Field-based results supported by the script, path information, WorkspaceLayout, and Segment offsets and sizes |
| ArgsInfo | Kernel parameter names and corresponding GM addresses |
| Common Workspace Segment | `segIdx`, `segType`, meaning, file size, and a decimal preview of the first 64 bytes |
| `type12/STATE_DUMP` | Displays the core ID, whether the core has been written, the current execution position, the communication phase, the number of commits, and the number of waits for each core. When the number of AIC cores is read from TilingData, it also indicates whether the core is a C core or a V core. |
| XN/CKE | The number of rows and columns of the address table, the number of non-zero addresses, and the hexadecimal addresses row by row |
| URMA CommContext | Communication context fields and a command hint for using `parse_matrix.py` to view raw bytes |

A common data Workspace can be very large. Therefore, the unified entry point does not expand the entire binary into text. Use `parse_matrix.py` to view specified rows and columns.

## 6. Parsing a Single File

Single-file parsing also uses the unified entry point. Pass only one Dump file to parse:

```bash
python3 parse_mc2_dump.py <dump_file>
```

`<dump_file>` is the actual Dump file path. The file usually has no `.bin` suffix.

The parsed content is still fully displayed in the terminal, and a corresponding TXT file is generated:

```text
<dump_file directory>/mc2_dump_analysis/<standard_operator_name>/<full original file name>.txt
```

For example, the input:

```text
/data/0/workspace_seg2_type12_MatmulReduceScatterV2.xxx
```

generates:

```text
/data/0/mc2_dump_analysis/MatmulReduceScatterV2/
└── workspace_seg2_type12_MatmulReduceScatterV2.xxx.txt
```

The parsed fields in the TXT file are consistent with the terminal results, and the operator name, file type, and source file name are added at the beginning of the file. The script reads only the original Dump and does not overwrite or modify the original file.

The script automatically performs two-layer identification based on the file name:

| Identified Content | Identification Method |
| --- | --- |
| Operator | Automatically identified from `AllGatherMatmulV2`, `AllGatherMatmulV3`, `AlltoAllMatmul`, `AlltoAllMatmulV2`, or `MatmulReduceScatterV2` in the file name. `MatmulReduceScatter` in MMRS file names is also identified. |
| File type | The parsing method is automatically selected based on the `tiling_data_`, `args_info_`, `workspace_segN_typeM_`, `xn_addr_info_`, `cke_addr_info_`, or `comm_context_` prefix. |

### 6.1 Examples

```bash
# Parse TilingData
python3 parse_mc2_dump.py \
  /path/to/tiling_data_MatmulReduceScatterV2.xxx

# Parse ArgsInfo
python3 parse_mc2_dump.py \
  /path/to/args_info_MatmulReduceScatterV2.xxx

# Parse type12 StateDump
python3 parse_mc2_dump.py \
  /path/to/workspace_seg2_type12_MatmulReduceScatterV2.xxx

# Parse URMA CommContext
python3 parse_mc2_dump.py \
  /path/to/comm_context_AlltoAllMatmulV2.xxx

# Parse the CCU XN address table
python3 parse_mc2_dump.py \
  /path/to/xn_addr_info_AlltoAllMatmul.xxx
```

`seg2` in the examples is not a fixed value. Use the actually generated file names.

### 6.2 Processing Rules in Single-File Mode

- One source file generates only one `.txt` file with the same name, and the results are also displayed in the terminal. No NPY files are generated, and the merged `analysis.txt` is not generated.
- The single-file command always passes only one `<dump_file>`. The script automatically identifies the operator and file type internally, and no additional output path needs to be specified when generating the TXT file.
- When parsing `tiling_data_*`, the complete TilingData is displayed, which already contains WorkspaceLayout. No additional mode parameter is required.
- When parsing a common `workspace_segN_typeM_*` file, the Segment meaning, file size, and a decimal preview of the first 64 bytes are displayed. Use `parse_matrix.py` in Section 7 to view the actual matrix values.
- When parsing `type12/STATE_DUMP`, the script automatically searches the directory containing the current file for the `tiling_data_*` file of the same operator with the closest timestamp, reads the number of AIC cores, and identifies the AlltoAllMatmul mainline/Apace label. This is an automatic association inside the script; the command line still needs to pass only the StateDump file. If the corresponding TilingData does not exist in the same directory, the StateDump values can still be parsed, but the core type may be displayed as "unknown".
- After parsing `comm_context_*`, the script provides the complete command for using `parse_matrix.py` to view raw bytes.
- If the file name does not contain the operator name, parsing continues only when the `tiling_data_*` file in the same directory can uniquely determine the operator; otherwise, an explicit error is reported.

## 7. Viewing Workspace Data or Raw Bytes by Row and Column

`parse_matrix.py` reads Workspace data by data type, matrix shape, and row/column range, and can also view the raw bytes of other files as `uint8`.

Basic format:

```bash
python3 parse_matrix.py <bin_file> --dtype <type> --rows <M> --cols <N> [display range parameters]
```

Required parameters:

| Parameter | Description |
| --- | --- |
| `<bin_file>` | Path of the Workspace Segment file. This script can also be used to view the raw bytes of `comm_context_*` files. |
| `--dtype <type>` | The actual data type of the elements in the file. Do not guess the type only from the Segment type; confirm it based on TilingData and the Kernel data flow. |
| `--rows <M>` | The total number of rows of the matrix corresponding to the original file, not the number of rows to print. For example, `--rows 8192` indicates that the complete matrix has 8192 rows. |
| `--cols <N>` | The number of elements in each row of the matrix corresponding to the original file, not the number of columns to print. For example, `--cols 12288` indicates that each row has 12288 elements. |

`rows` and `cols` together describe the shape of the complete matrix. For example, `--rows 8192 --cols 12288` interprets the file as an `8192 × 12288` matrix with a total of `8192 × 12288` elements. Row-major order is used by default. The element index of row `row` and column `col` in the file is:

```text
element index = row × cols + col
```

Therefore, `rows` and `cols` must be filled in based on TilingData, the shape of the operator output matrix, and the Kernel Workspace data layout. To control which rows and columns are actually printed, use `--row`, `--row-range`, `--col`, or `--col-range`.

Display range and output parameters:

| Parameter | Description |
| --- | --- |
| `--row-range START END` | Displays a continuous row range `[START, END)`. |
| `--col-range START END` | Displays a continuous column range `[START, END)`. |
| `--row R [R ...]` | Displays one or more discrete row numbers. |
| `--col C [C ...]` | Displays one or more discrete column numbers. |
| `--transpose` | Used when the file is in column-major layout. Values are fetched by the logical coordinate `col × rows + row`. |

Do not use `--row` and `--row-range` together, and do not use `--col` and `--col-range` together.

If `--row` or `--row-range` is not specified, the script always displays the first 16 rows, and no additional preview row count parameter is required. To view the last rows, use `--row-range START END` or `--row R [R ...]` directly.

To view the first 10 rows and columns 0 to 63 of each row:

```bash
python3 parse_matrix.py \
  /path/to/workspace_seg1_type8_MatmulReduceScatterV2.xxx \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row-range 0 10 --col-range 0 64
```

Ranges follow the left-closed and right-open rule: `--row-range 0 10` indicates rows 0 to 9, and `--col-range 0 64` indicates columns 0 to 63.

To view discrete rows and columns:

```bash
python3 parse_matrix.py workspace.bin \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row 0 8 16 --col 0 1 63 64
```

Supported `--dtype` values: `bf16`, `fp16`, `fp32`, `int8`, `uint8`, `fp8_e4m3`, `fp8_e5m2`, `e8m0`, and `fp4_e2m1`.

To view the raw bytes of `comm_context_*` files, use `--dtype uint8 --rows 1 --cols <file size in bytes>`. This is only a byte view; the structured fields are still subject to the parsing result of `parse_mc2_dump.py <comm_context file>`.

## 8. Workspace Segment Meanings

`segN` in the file name is the Segment sequence number in the current WorkspaceLayout, and `typeM` carries the data semantics. The Segment sequence number may change with the path and conditions. Do not determine the content only by `segN`.

Files are generated only for the Segments actually collected. System-reserved Segments, empty Segments, or uncollected Segments may have no corresponding files. Therefore, the `segN` you see may not start from 0 and may not be consecutive.

| type | Meaning |
| ---: | --- |
| 0 | `LIB_API`, Workspace reserved for the system library |
| 1 | `ND2NZ`, format conversion buffer |
| 2 | `GATHER`, Gather data buffer |
| 3 | `GATHER_SCALE1` |
| 4 | `GATHER_SCALE` |
| 5 | `COMM_OUT`, communication output buffer |
| 6 | `PERMUTE_OUT`, transpose output buffer |
| 7 | `BIAS` |
| 8 | `MM_RESULT`, Matmul result/communication send area of some paths |
| 9 | `RECV_BUF`, communication receive area |
| 10 | `COMM_INT8` |
| 11 | `DYNAMIC_QUANT` |
| 12 | `STATE_DUMP`, per-core running and communication status |
| 13 | `MM_WORKSPACE`, temporary area inside Matmul, not the result matrix |
| 255 | `RESERVED`, alignment or reserved area |

Based on the Workspace construction code of each operator in the current `ops-transformer`, the Segment order is as follows. Segments marked with "optional" appear only when the length is greater than 0. Therefore, the actual `segN` may change:

| Operator/Path | Current Workspace Segment Order |
| --- | --- |
| AllGatherMatmulV2 | `LIB_API → GATHER_SCALE1 (optional) → GATHER (optional) → BIAS (optional) → STATE_DUMP` |
| AllGatherMatmulV3 | `LIB_API → STATE_DUMP` |
| AlltoAllMatmul mainline CCU | `DYNAMIC_QUANT/commX1Scale (optional) → COMM_OUT (optional) → DYNAMIC_QUANT/transX1Scale (optional) → PERMUTE_OUT (optional) → STATE_DUMP` |
| AlltoAllMatmul Apace CCU | `DYNAMIC_QUANT/commX1Scale (optional) → COMM_OUT (optional) → STATE_DUMP` |
| AlltoAllMatmulV2 | `LIB_API → STATE_DUMP` |
| MatmulReduceScatterV2 Direct ReduceScatter | `MM_RESULT → MM_WORKSPACE (optional) → RESERVED/padding (optional) → STATE_DUMP` |
| MatmulReduceScatterV2 A2A + ReduceSum | `MM_RESULT → RECV_BUF → MM_WORKSPACE (optional) → RESERVED/padding (optional) → STATE_DUMP` |

The current Segment tables of AlltoAllMatmul and MMRS start from the user Workspace received by the Kernel and do not additionally register the `LIB_API` Segment. The other three types of operators still register `LIB_API` in the Segment table. During parsing, the actual `segCount/type/offset/size` in the current TilingData prevails.

Cross-check the final meaning against all of the following:

1. The `WorkspaceSegType` enumeration in the current branch.
2. The Segment order, offsets, and sizes in `BuildWorkspaceLayout()` on the Tiling side.
3. The Workspace address offsets and data types of GlobalTensor on the Kernel side.

## 9. How to Read StateDump Results

`type12/STATE_DUMP` is a running status area divided by core, not matrix data. It cannot be interpreted by the data type and row/column count of a common matrix. Currently, each core occupies 512 bytes, and the script refers to these 512 bytes as a `slot`.

All operator backends parse the following basic fields:

- `magic` / `magicValid`: Whether the slot has been initialized by the Kernel. The valid value is `0x5A5A5A5A`.
- `coreId` / `coreType`: Core ID and core type. If the number of AIC cores for the current run is not obtained, the core type is displayed as "unknown".
- `turn`: Current execution round.
- `position`: Execution position recorded by the Kernel. The meaning of the label depends on the operator and path.
- `phase`: Current communication phase.
- `commit`: Number of communications committed by the core.
- `wait`: Number of communications that the core has waited to complete.

Most backends expand only the slots with `magicValid=true`. If `activeSlots=0`, no initialized StateDump slot is found in the Dump. Check the start address of the StateDump Segment, whether Kernel initialization is executed, and the collection time of the exception Dump.

The MMRS backend also displays the following auxiliary parsing information:

- `expectedSlots`: Expected number of core records to be used. In the current MMRS, if the number of AIC cores is `N`, the corresponding number of AIV cores is `2N`, so `3N` slots are expected to be used. If no matching TilingData exists in the same directory, the script cannot obtain `N` and can only display based on the maximum number of slots in the file.
- `validExpectedSlots`: Number of slots that pass the magic check within the expected range.

The preceding two items are used to control and describe the display range and are not fields in the `StateDumpPerCore` structure. The script no longer generates derived fields such as `pending` and `state`. The communication status is directly subject to `commCommitCount` and `commWaitCount` in the source code.

## 10. Recommended Troubleshooting Process

```text
Execute the operator and generate the MC2 DFX Dump
        ↓
Use parse_mc2_dump.py to parse the directory by operator name
        ↓
View the corresponding TilingData/ArgsInfo/Workspace TXT files
        ↓
Check the offsets and sizes in WorkspaceLayout against the TilingData TXT
        ↓
To view specific values, use parse_matrix.py to view the specified rows and columns of the target Segment
        ↓
Confirm the problem phase based on the Kernel Workspace address offsets and data flow
```

Common checks:

- "No supported Dump file found" is displayed: Confirm that the Dump root directory and operator name are correct, and check whether the file names contain the corresponding operator name.
- No XN/CKE in the Dump directory: First confirm that the current path is really the CCU path.
- No CommContext in the Dump directory: First confirm that the current path is the AIV + URMA path.
- The Workspace file size is inconsistent with the Segment size in TilingData: Check the Segment offsets, alignment/padding, and Dump timing.
- All StateDump slots are invalid: Check Kernel initialization, the Workspace start address, and the phase where the exception occurred.
- The core type in StateDump is displayed as "unknown": Confirm that the `tiling_data_*` file of the same invocation and the StateDump file are in the same directory and that the file names contain the same operator name. The single-file entry point automatically reads `aicCoreNum` or `usedCoreNum` from the TilingData file.
