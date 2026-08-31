# kv_quant_sparse_flash_attention pytest测试框架

## 功能说明

基于pytest测试框架，实现kv_quant_sparse_flash_attention算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

支持三条主流程：

- `single`：基于`kv_quant_sparse_flash_attention_paramset.py`的固定参数直接构造输入并拉起NPU单算子执行。
- `batch_save`：从Excel读取参数，生成包含CPU golden的`.pt`用例文件。
- `gen_excel_from_paramset`：从paramset生成Excel文件。
- `batch_exec`：从已有`.pt`文件批量回放执行NPU算子并对比精度。

## 当前实现范围

### 参数说明

以下参数约束已经下沉到框架校验和本文档中，便于统一维护：

- `layout_query`: 支持`BSND`、`TND`
- `layout_kv`: 支持`BSND`、`TND`、`PA_BSND`
- 非PA场景要求`layout_query == layout_kv`
- `q_type`: 仅支持`torch.float16`、`torch.bfloat16`
- `kv_dtype`: 支持`hifloat8`、`float8_e4m3fn`，也兼容`None`作为`float8_e4m3fn`默认生成路径
- `N1`: 仅支持`1/2/4/8/16/32/48/64`
- `N2`: 仅支持`1`
- `sparse_mode`: 仅支持`0`和`3`
- `sparse_block_size`: 当前仅支持`1`
- `key_quant_mode` / `value_quant_mode`: 仅支持`2`
- `tile_size`: 当前仅支持`128`
- `quant_scale_repo_mode`: 当前仅支持`1`
- `attention_mode`: 支持`0`、`2`；当取`2`时，`rope_head_dim`必须为`64`
- `block_size`: 仅在`PA_BSND`生效，且要求为16的倍数
- `actual_seq_q` / `actual_seq_kv`: 如果传入，长度必须等于`B`

更完整的算子定义和输入约束，请同步参考：

- `attention/kv_quant_sparse_flash_attention/README.md`

### 环境配置

#### 前置要求

1. 完成环境安装和环境变量配置，具体操作请参考：[环境部署](../../../../docs/zh/install/quick_install.md)
2. TorchNPU安装包下载路径（需及时更换为最新版本）：[Attention融合算子Experimental使用说明](../../../../experimental/attention/Attention融合算子Experimental使用说明.md)

#### custom包调用

支持custom包调用

## 文件结构

```text
pytest/
├── README.md
├── pytest.ini   # 创建测试标记
├── test_run.sh   # 执行脚本
├── check_valid_param.py   # 参数约束拦截
├── kv_quant_sparse_flash_attention_golden.py  # tensor转换/cpu侧算子golden实现
├── kv_quant_sparse_flash_attention_paramset.py  # 单用例入参配置
├── result_compare_method.py  # 输出精度对比
├── utils.py   # 参数解析/cpu npu执行入口
├── test_kv_quant_sparse_flash_attention_single.py # 单用例运行主程序
├── test_kv_quant_sparse_flash_attention_batch.py # 从pt文件批量执行NPU测试
└── batch/
    ├── kv_quant_sparse_flash_attention_process.py # npu接口
    ├── test_kv_quant_sparse_flash_attention_pt_save.py  # 从Excel批量生成pt文件
    ├── gen_excel_from_paramset.py # 从paramset生成Excel文件
    └── excel/
        ├── example.xlsx  # 示例Excel用例文件
        └── gen_example_xlsx.py  # 生成示例Excel的脚本
```

## 使用方法

在`attention/kv_quant_sparse_flash_attention/test/pytest`目录下执行：

### 命令格式

```bash
bash test_run.sh <模式> [-E excel_path] [-S sheet] [-P path] [-O output_path]
```

### 参数选项

| 选项 | 说明 | 适用模式 |
| --- | --- | --- |
| `-E excel_path` | 指定Excel文件路径，默认`./excel/example.xlsx` | batch_save |
| `-S sheet` | 指定Excel Sheet页名，默认`Sheet1` | batch_save/gen_excel_from_paramset |
| `-P path` | 指定路径（不同模式含义不同，详见下表） | single/batch_save/batch_exec/gen_excel_from_paramset |

| 模式 | `-P`参数含义 | 默认值 |
| --- | --- | --- |
| single | paramset文件名 | `kv_quant_sparse_flash_attention_paramset` |
| batch_save | pt文件保存路径 | `./pt_files/` |
| batch_exec | pt文件执行路径（目录或单个文件） | `./pt_files/` |
| gen_excel_from_paramset | paramset文件名 | `kv_quant_sparse_flash_attention_paramset` |

**gen_excel_from_paramset模式额外参数：**

| 选项 | 说明 | 默认值 |
| --- | --- | --- |
| `-E excel_output` | 输出Excel文件路径 | `./excel/example.xlsx` |
| `-S sheet` | Excel Sheet页名 | `Sheet1` |

### single

手动配置`kv_quant_sparse_flash_attention_paramset.py`的参数，或使用`-P`指定其他paramset文件。

```bash
bash test_run.sh single                              # 使用默认paramset
bash test_run.sh single -P my_paramset                # 使用指定的paramset文件
```

### batch_save

从Excel读取参数，生成包含CPU golden的`.pt`用例文件。

```bash
bash test_run.sh batch_save                           # 使用默认Excel和Sheet
bash test_run.sh batch_save -E ./test.xlsx            # 指定Excel文件
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1  # 指定Excel和Sheet
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1 -P ./output_pt/  # 指定全部参数
bash test_run.sh batch_save -S Sheet1 -E ./test.xlsx  # 参数顺序可任意
```

### gen_excel_from_paramset

从paramset生成Excel文件。

```bash
bash test_run.sh gen_excel_from_paramset                           # 使用默认paramset
bash test_run.sh gen_excel_from_paramset -P my_paramset            # 指定paramset文件
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx  # 指定输出路径
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx -S decode  # 指定Sheet名
```

### batch_exec

从`.pt`文件批量回放执行NPU算子并对比精度。

```bash
bash test_run.sh batch_exec                           # 执行默认目录下所有pt文件
bash test_run.sh batch_exec -P ./pt_files/test.pt     # 执行单个pt文件
bash test_run.sh batch_exec -P ./custom_pt_dir/       # 执行指定目录下所有pt文件
```

下面给一个可直接参考的Excel用例，列名需与batch框架读取字段保持一致：

| Testcase_Prefix | Testcase_Number | layout_query | layout_kv | q_type | kv_dtype | B | S1 | S2 | N1 | N2 | D | K | scale_value | key_quant_mode | value_quant_mode | sparse_block_size | tile_size | rope_head_dim | sparse_mode | attention_mode | quant_scale_repo_mode | block_size | block_num | actual_seq_q | actual_seq_kv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tnd_sample | 1 | TND | TND | torch.bfloat16 | hifloat8 | 2 | 8 | 8 | 16 | 1 | 512 | 4 | 0.04166666666666666 | 2 | 2 | 1 | 128 | 64 | 3 | 2 | 1 | 256 |  | [5,8] | [6,8] |

## 结果文件

- `result.xlsx`：记录每个用例的关键信息、执行状态与`fulfill_percent`
- `./pt_files/*.pt`：batch流程生成的中间测试用例
