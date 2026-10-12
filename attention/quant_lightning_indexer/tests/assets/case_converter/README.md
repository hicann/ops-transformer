# quant_lightning_indexer Excel 用例转换工具

本目录将本算子的参数 Excel 表转换为 TTK E2E 或 ACLNN Excel 用例表。
`convert.py` 负责工作簿读写与追加，`case_schema.py` 负责本算子的 tensor、属性和输出描述。
整个目录可独立复制使用，各算子之间没有导入或路径依赖。

## 运行条件

Python 3.9+ 和 `openpyxl>=3.1`。转换只依赖 Python 标准库和 openpyxl，不需要 PyTorch、pytest、CANN 或 NPU。
已有 Python 环境缺少依赖时可执行 `python3 -m pip install "openpyxl>=3.1"`。

下列命令从当前目录执行，也可以把 `convert.py` 换成其完整路径，从任意目录调用。
输入支持 `.xlsx/.xlsm`，输出也是 Excel。

## 三种输出方式

1. 输出到目录，保持输入文件名和选中的 sheet 名：

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 --output-dir output --mode e2e
```

2. 输出到指定 Excel 文件，按输入 sheet 名创建或追加：

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --mode aclnn
```

3. 输出到指定 Excel 文件的指定 sheet，按输入选择顺序合并追加：

```bash
python3 convert.py -i input.xlsx -s Sheet1 Sheet2 -o cases.xlsx --output-sheet Cases --mode e2e
```

`-s/--sheet` 可指定多个名称，也可重复使用；带空格的名称请加引号。
未指定时读取第一张 sheet，重复或不存在的名称报错；名称不按逗号拆分。
`--mode` 缺省为 `e2e`，`--output-dir` 与 `-o/--output` 互斥。

目录输出和指定文件输出遇到同名 sheet 时，工具会先询问是否追加。
只接受 `yes`、`y`、`Y`；其他回答或输入结束取消整次写入，已经确认的其他 sheet 也不写入。
sheet 名比较不区分大小写，保留目标现有拼写。
显式提供 `--output-sheet` 表示已指定追加目标，不重复确认。

## 输入表格式

第一行非空行为表头，一行代表一个用例；忽略全空行。
支持两种来源：本算子的 pytest 参数表，以及已包含 `tensor_view_shapes/tensor_dtypes/attributes` 的 E2E 描述表。
输入 ACLNN 描述表会报错，避免重复附加输出 tensor。
list、tuple、字典使用 Python 字面量，例如 `[0, 16, 32]`、`(1, 16, 8, 128)`、`{'layout_q': 'TND'}`。
表达式使用字面量解析，不能调用 Python 函数。

必需参数为 `batch_size/q_seq/k_seq/q_head_num/k_head_num/head_dim`、`qk_dtype`、查询和 key 的布局、`dequant_dtype`；用例名称支持 `testcase_name/Testcase_Name/case_name/Testcase_Prefix`，别名须一致。
查询支持 `BSND/TND`；V1 key 支持 `BSND/TND/PA_BSND`，V2 key 支持 `BSND/TND/PA_BBND`。

- V1 使用 `layout_query/layout_key`、`sparse_count/sparse_mode`；必须提供 `act_seq_q/act_seq_k`。
  TND 使用不含开头 0 的累积长度；BSND 和 PA 使用每个 batch 的有效长度。
  `weight_dtype`、`actual_seq_dtype` 和各输入的 `*_datarange` 可显式指定。

- 必须提供 `query_quant_mode/key_quant_mode`；可通过 `qk_dtype=hifloat8` 声明 HiFloat8 语义。

- PA 的 `block_size` 缺省 128，`block_num` 缺省按容量计算，显式不足时拒绝转换。
- 数据范围列：`query_datarange/key_datarange/weights_datarange`，量化算子另有 `q_scale_datarange/k_scale_datarange`。

`is_enabled` 缺省 True；显式 False 保持 False，不根据历史测试结果修改启用状态。
`precision_tolerances/absolute_precision` 可显式覆盖转换默认值。
`remark/network_name/priority/soc_series/golden_api/device_ids` 及已有 batch 字段按表中值保留。
`attributes1`..`attributes9` 合并进 `attributes`；后续列覆盖之前的同名属性。
输出格式的部分字段只由对应 TTK 模式使用。

参数表在 BSND 布局下允许省略 query 或非 PA 的 KV/key 序列长度，按相应 S 轴完整长度生成输入数据，
输出对应可选 tensor 槽位保持 None；TND prefix 及 PA 的 KV/key 长度必须显式提供。

## 输出与追加合同

- E2E API：`torch_npu.npu_quant_lightning_indexer`，包含 8 个输入 tensor 槽位；可选 tensor 保持 None。
- ACLNN API：`aclnnQuantLightningIndexer`，在输入后添加真实输出形状和 dtype，并填写 `output_tensor_indexes`。
- 输入和输出不得为同一个文件；拒绝同文件的 symlink/hardlink 别名，目标文件不能是 symlink。
- 追加按目标表头名称对齐，保留旧列和旧行；缺少的新列追加到表头末尾。一个 sheet 不能混放 E2E 和 ACLNN。
- 写出前检查所有选中 sheet 和目标表；失败不修改目标文件。写入使用同目录临时文件，再原子替换。
- 现有目标必须是普通 TTK 用例表；不向受保护、合并单元格或 Excel structured table 的用例页追加。
  其他 sheet 的现有内容、公式和单元格样式随工作簿保留。
- 输入公式须有 Excel 保存的计算缓存；工具不计算公式。无缓存时请先在 Excel 中重新计算并保存。
- `.xlsm` 目录输出保留源 VBA；追加已有 `.xlsm` 保留目标 VBA。新建 `.xlsm` 需要 `.xlsm` 来源。
  转成 `.xlsx` 只输出用例表。
- 成功退出码为 0，输入错误、拒绝追加或写入失败为 1。

## 与 TTK 执行的关系

转换生成参数描述；实际执行沿用本算子 assets 中的输入、Golden 和 TestSpec。
参数表输出会包含现有输入适配器需要的属性。执行仍需对应版本 TTK、算子包和设备环境。

当前 assets 没有注册此算子的 ACLNN TestSpec，ACLNN 输出先作为接口描述供检视。
