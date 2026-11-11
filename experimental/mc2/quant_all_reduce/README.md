## QuantAllReduce <<<>>>直调示例

# 文件目录

```
├── kernel                                  # 算子核函数目录
│    ├── quant_all_reduce_mte_one_shot.h
│    ├── quant_all_reduce_tiling_data.h
│    ├── mte_comm.h
│    ├── vec_comp.h
│    ├── utils.h
├── scripts                                 # 运行脚本工具
│   ├── data_generate.py                         # 用例数据生成脚本
│   ├── data_compare.py                     # 精度对比脚本
│   └── QuantAllReduce.csv                  # 算子用例输入相关信息
├── quant_all_reduce.cpp                    # <<<>>>核函数入口
├── quant_all_reduce.h                      # <<<>>>核函数入口声明头文件
├── test_quant_all_reduce.cpp               # <<<>>>直调示例代码
├── start.sh
├── run.sh                                  # 工程运行脚本
├── result.csv                              # 算子运行结果汇总
├── README.md
└── CMakeLists.txt
```

# 文件改动

新增接口：
quant_all_reduce_mte_one_shot.h
外部创建mc2Context与tilingData，传入GM地址mc2Context，tilingGM
__aicore__ inline void InitWithMc2Context(GM_ADDR x, GM_ADDR scales, GM_ADDR output, GM_ADDR mc2Context, GM_ADDR tilingGM, TPipe *pipe);
mte_comm.h
__aicore__ inline void InitHcclContextByAddr(GM_ADDR mc2Context);

# 工程运行

1. 设置环境变量

``` shell
source /usr/local/Ascend/cann/set_env.sh
```

2. run.sh运行指令

``` shell
bash run.sh -v ascend950pr_9599 -b Release -c ON -r OFF -n 2
```

3. 选项简介
-c cmake-rebuild --- 编译开关
    ON-执行编译，OFF-不执行编译。

-v soc-version --- 芯片版本
    ascend950pr_9599，芯片版本，根据环境不同芯片型号选择正确版本。

-i install-path --- cann包安装路径（source环境后不用指定）
    cann包路径，安装cann包后，source /usr/local/Ascend/cann/set_env.sh

-b build-type --- 编译选项Release or Debug

-p install-prefix --- 编译文件输出目录
    编译完成后，编译产物路径，默认路径为运行目录下的out/文件夹中。

-r run-test --- 运行开关
    ON-运行用例，OFF-不运行用例
    编译后才可执行，可复用编译产物，默认执行第一条用例

-n target_line --- 执行用例行号
   指定执行用例的行号，用例csv文件在scripts/QuantAllReduce.csv中。

-f profing-type --- profing开关
    ON-采集性能数据，OFF-不采集性能数据

4. 指令常用组合

``` shell
bash run.sh -v ascend950pr_9599 -b Release -c ON -r OFF --- 仅编译
bash run.sh -v ascend950pr_9599 -b Release -c ON -r ON -n 2 --- 编译运行并指定行号
bash run.sh -v ascend950pr_9599 -b Release -c OFF -r ON -n 2 --- 不编译直接运行
```
