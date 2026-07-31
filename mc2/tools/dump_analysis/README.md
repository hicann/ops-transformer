# win区dump数据解析脚本

## 背景说明

- exception dump：算子在出现aic error后，会触发exception dump功能，将算子输入输出tensor dump成bin文件，其中dispatchv2&combinev2算子还额外注册异常处理回调，会将通信中保存状态位的windows内存(或win区内存)dump成bin文件，其中该bin文件中的最后1M的数据称为win区数据。里面包含卡住、aic error等常见问题定位所需数据。

- profiling数据：在整网运行中，通常会调用多个算子进行联跑，通常会出现部分卡耗时过长的情况，该现象通常是由于卡中某个算子出现异常耗时导致的快慢卡现象，需要开发profiling数据分析工具排查可能出现异常耗时的算子以及对应的卡号，完成分析后需输出csv文件,并打屏输出对应的异常点。

- graph图文件：当调用cache_compile接口时(调用方式可查看<https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00099.html)，会在指定路径下生成graph图文件，该文件中包含每一张卡在每一轮执行的算子次数以及名称，可用于定位因执行序不一致导致的卡死现象。>

- plog日志：当设置环境变量ASCEND_SLOG_PRINT_TO_STDOUT=1时，会在/root/ascend/log下生成对应的plog日志，可以通过plog日志查看对应的报错信息以及错误码。

- 在dispatch&combine问题定位中，需要根据多卡数据进行整体分析判断，分析难度大，因此需要开发多卡数据分析工具来快速分析，要求能够根据已有数据，一键式解析数据，按照既定规则分析出可能异常的点，直接打印异常情况描述，部分数据解析完成后需输出csv文件，工具需打印当前正在进行哪一项分析。

## 功能说明

- dump数据解析: 对指定的dump数据进行解析，获得该dump数据的win区数据，获取出每张卡对应的moe专家数、使用核数、执行次数、epworldsize以及0/1标识位，并判断异常卡的每个核上的这些参数是否一致，并将分析出有异常的点输出，可以用于定位moe专家数、epworldsize输入异常的问题。

- 输入异常分析: 可以用于分析当算子卡死时，是否是因为参数输入异常导致的卡死

- 执行序分析: 可以用于定位因执行次数不匹配导致的卡死问题。

- 状态位分析: 获取每个卡中每个核的执行位置信息，判断哪些核没有等到状态位，并根据对应的dispatchv2&combinev2的0/1标识位，到对应的0/1状态区找出没等到状态的核里面具体是第几个状态位没有等到，用于展示算子卡死后的具体现象。

- graph图文件分析：可以用于定位因各卡算子执行次数或执行顺序不一致导致的卡死问题

- plog日志分析：可以用于初步定位一些常见问题导致的异常问题

- profiling数据分析：对profiling数据中的Duration、aiv_vec_time, aiv_scalar_time, aiv_mte2_time, aiv_mte3_time数据进行分析，计算每个算子的平均抖动,并将异常项输出,可以用于定位性能异常的原因
Duration：算子执行的耗时，单位us
aiv_vec_time：向量类运算指令在AI Vector Core上的耗时，单位us
aiv_scalar_time：标量类运算指令在AI Vector Core上的耗时，单位us
aiv_mte2_time：片上内存->AI CORE搬运类指令在AI Vector Core上的耗时，单位us
aiv_mte3_time：AI CORE->片上内存搬运类指令在AI Vector Core上的耗时，单位us

## 脚本输入输出说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>TARGET_DIR</td>
   <td>可选输入</td>
   <td>dump数据所在的文件路径，调用pytorch接口的日志落盘位置通过ASCEND_WORK_PATH环境变量控制，通过查看日志可以得到dump数据的落盘位置，未设置该环境变量时，dump数据落盘在当前目录下extra-info/data-dump/路径,该参数分析dump数据时必填。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>TOOL_PATH</td>
   <td>可选输入</td>
   <td>装包路径中cann仓所在的文件路径，如:TOOL_PATH=install_pkg/cann-x.x.x/,该参数分析dump数据时必填。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SOC_VERSION</td>
   <td>可选输入</td>
   <td>构造dump数据时对应的版本输入，当前仅支持如:SOC_VERSION=910_93 or 950,该参数分析dump数据时必填。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SHARE_EXPERT_CARD_COUNT</td>
   <td>可选输入</td>
   <td>构造dump数据时的输入共享专家卡数，未输入时默认值为0, SHARE_EXPERT_CARD_COUNT <= 总卡数，如:SHARE_EXPERT_CARD_COUNT=1。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>SP_MOE_NUM</td>
   <td>可选输入</td>
   <td>构造dump数据时的输入特殊专家数，未输入时默认值为0, SP_MOE_NUM >0，如:SHARE_EXPERT_CARD_COUNT=1。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>TP_WORLDSIZE</td>
   <td>可选输入</td>
   <td>构造dump数据时的TP_WORLDSIZE,未输入时默认值为1, TP_WORLDSIZE >= 1，如:TP_WORLDSIZE=1。</td>
   <td>INT</td>
  </tr>
  <tr>
    <td>SHARE_EXPERT_NUM</td>
    <td>可选输入</td>
    <td>构造dump数据时的输入共享专家数，未输入时默认值为0, SHARE_EXPERT_NUM > 0，如:SHARE_EXPERT_NUM=1</td>
    <td>INT</td>
  </tr>
  <tr>
    <td>START_A2_CLUSTER</td>
    <td>可选输入</td>
    <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：解析dump数据时，区分是否开始分析多机数据，未输入时默认值为0，输入为1时开启多机分析，需先得到各机解析结果再执行，如:START_A2_CLUSTER=1</td>
    <td>INT</td>
  </tr>
  <tr>
    <td>PROFILING_PATH</td>
    <td>可选输入</td>
    <td>指定profiling数据的存放路径，未输入时默认在TARGET_DIR下查找，如:PROFILING_PATH=xxx</td>
    <td>INT</td>
  </tr>
    <tr>
    <td>PLOG_PATH</td>
    <td>可选输入</td>
    <td>指定plog日志的存放路径，如:PROFILING_PATH=xxx,该参数分析plog日志时必填</td>
    <td>INT</td>
  </tr>
    <tr>
    <td>GRAPH_PATH</td>
    <td>可选输入</td>
    <td>指定graph图文件的存放路径，如:GRAPH_PATH=xxx,该参数分析graph图文件时必填，该参数需要指定到对应模型的graph图文件路径</td>
    <td>INT</td>
  </tr>
  <tr>
   <td>打屏日志</td>
   <td>输出</td>
   <td>Info:分析出的win区数据信息，如:该卡dispatch使用核数:72,combine使用核数:72。该卡dispatch moe专家数:62,combine moe专家数:62<br> Warning:分析出的异常点。如:执行次数dispatch = combine + 1，因此认为该卡算子执行流程卡死在dispatch算子上。</td>
   <td>str</td>
  </tr>
  <tr>
   <td>win_status_list</td>
   <td>输出</td>
   <td>存放win区中0/1标识区指定的dispatchv2/combinev2的0/1状态区的所有状态位数据，用于查看没等到状态的核里面具体是第几个状态位没有等到。<br>
       如:解析出该卡dispatchv2的状态区存储在0区，则将win区dump数据中的dispatch 0区状态区[0:64 * 1024]的数据存入win_status_list.csv</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_all_card_data</td>
   <td>输出</td>
   <td>存放每张卡dispatchv2&combinev2的执行次数、moe专家数以及globalbs,并判断各卡上的这些参数是否一致。<br> 如:d0 dispatch: 1 (指第一张卡dispatchv2算子运行了1次)</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_analysis_error</td>
   <td>输出</td>
   <td>存放状态位分析出的异常点及异常位置，即把打屏日志中未显示的具体的状态位异常的Warning信息存储在该csv文件中。<br>如:d0卡dispatchv2算子第35个核的第2个状态位在dispatch 1区状态区上显示未等到</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data</td>
   <td>输出</td>
   <td>存放解析出的各卡的moe专家数、使用核数、执行次数、epworldsize、rankid、globalbs、建立hccl通信链路时的输入rankid和globalbs、计算得出的bs、k、0/1标识位数据。<br>如:d0_dispatch_moe专家数:65</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_data_list</td>
   <td>输出</td>
   <td>存放解析出的各卡中每个核的执行位置情况以及0/1标识位数据，<br>如:d0卡dispatchv2的使用核数为72，则将72个核的win区的moe专家数记录为一个长度为72的列表并储存至win_data_list，d0_dispatch_0/1标识区数据:[0:71]。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>win_all_card_expandidx</td>
   <td>输出</td>
   <td>存放解析出的各卡中输入expertids、dump数据中读取的expandidx(仅挂在combine算子上的卡，否则为空)、dump数据中读取的epsendcnt(仅挂在combine算子上的卡，否则为空)、本卡专家数、该卡挂在哪个算子上，<br>如:0卡本卡专家数:8。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>profiling_all_data</td>
   <td>输出</td>
   <td>以0卡的数据列表为基准，存放解析出的每张卡中每个type的Duration(us)列的数据以及他们对应的最大最小值和抖动,并将这些数据按调用次数存储,如对应单元格中为非法字符或对应卡的profiling数据丢失,则用NA代替，<br>如:0卡| type:dispatch| 第n次调用:0| max:xx| min:xx| 抖动:xx| 卡0:xx| ……| 卡15: xx 。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>profiling_avg_data</td>
   <td>输出</td>
   <td>根据输出profiling_all_data.csv表中的数据，计算得到每个type的平均抖动,总调用次数以及有效调用次数(去除第0次和NA项),并将结果存储至profiling_avg_data.csv中，<br>如:type:dispatch| 调用次数:60 |有效调用次数(去除第0次和NA项): 52| 平均抖动(us): xx。</td>
   <td>csv表</td>
  </tr>
  <tr>
   <td>error_plog</td>
   <td>输出</td>
   <td>输出PLOG_PATH下所有匹配已知报错原因的ERROR,并根据种类进行分类并输出至该csv表中<br>如:错误类型错误信息。</td>
   <td>csv表</td>
  </tr>
 </tbody>
</table>

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| dump_analysis.sh脚本直调 | bash dump_analysis.sh TARGET_DIR=xxx TOOL_PATH=xxx PROFILING_PATH=xxx SP_MOE_NUM=0 TP_WORLDSIZE=1 SOC_VERSION=950 SHARE_EXPERT_CARD_COUNT=1 SHARE_EXPERT_NUM=1 GRAPH_PATH=xxx PLOG_PATH=xxx| 通过对sh脚本进行入参对指定的TARGET_DIR路径下的dump数据进行dump数据解析、对PROFILING_PATH下的profiling数据进行分析、对GRAPH_PATH下的图文件数据进行分析以及对PLOG_PATH下的plog日志进行分析。 |
| dump_analysis.sh脚本直调 | bash dump_analysis.sh SOC_VERSION=910B START_A2_CLUSTER=1 | 将START_A2_CLUSTER=0时解析的结果放在同一台设备相同路径后执行，通过sh脚本对各机进行解析多机数据, A2 Dispatch和Combine Fullmesh方案建议使用 |

## 样例代码结果说明

[INFO] 开始分析卡1数据<br>
[INFO] 解析文件:/xxx/xxx/xxx/data-dump/mc2_exception_info.xxx<br>
<b>上述结果为当前正在分析的卡的数据信息，会输出当前分析卡的序列号以及对应的dump文件</b><br>

[INFO] 1. 开始进行输入异常分析

[INFO] 1. 该卡dispatch&combine使用核数为dispatch:72,combine:72<br>
[INFO] 1.1 dispatch各核的rankid:[1,...,1]<br>
[INFO] 1.1 dispatch各核的epworldsize:[2,...,2]<br>
[INFO] 1.1 dispatch各核建立hccl通信链路时的输入rankid:[0,...,0]<br>
[INFO] 1.1 dispatch各核建立hccl通信链路时的输入epworldsize:[0,...,0]<br>
[INFO] 1.1 dispatch各核的moe专家数:[1,...,1]<br>
[INFO] 1.1 dispatch各核的globalbs:[1,...,1]<br>
[INFO] 1.1 dispatch_rankid:1 dispatch_hccl_rankid:0, dispatch epworldsize:2, dispatch_hccl epworldsize:0, dispatch moe专家数:16, dispatch globalbs:2,根据dispatch输入计算的bs:1<br>
[ERROR] 1.1 dispatch win区数据中的rankid:1与建立hccl通信链路时的rankid输入:0不同<br>
[ERROR] 1.1 dispatch win区数据中的epworldsize:1与建立hccl通信链路时的epworldsize输入:0不同<br>

[INFO] 1.1 combine各核的rankid:[1,...,1]<br>
[INFO] 1.1 combine各核的epworldsize:[2,...,2]<br>
[INFO] 1.1 combine各核建立hccl通信链路时的rankid输入:[0,...,0]<br>
[INFO] 1.1 combine各核建立hccl通信链路时的epworldsize输入:[0,...,0]<br>
[INFO] 1.1 combine各核的moe专家数:[1,...,1]<br>
[INFO] 1.1 combine各核的globalbs:[1,...,1]<br>
[INFO] 1.1 combine_rankid:1 combine_hccl_rankid:0, combine epworldsize:2, combine_hccl epworldsize:0, combine moe专家数:16, combine globalbs:2,根据combine输入计算的bs:1<br>
[ERROR] 1.1 combine win区数据中的rankid:1与建立hccl通信链路时的rankid输入:0不同<br>
[ERROR] 1.1 combine win区数据中的epworldsize:1与建立hccl通信链路时的epworldsize输入:0不同<br>
<b>上述结果为对dump数据中的各项数据进行取值以及对比rankid和epworldsize的输入与建立hccl通信链路时的输入是否相同，用于各项参数输入异常导致的卡死问题，并将分析出的异常用ERROR的形式输出。</b><br><br>

[INFO] 1.2该卡不为共享专家卡<br>
[INFO] 1.2本卡专家数为:8<br>
[INFO] 1.2根据计算得出该卡的bs=1<br>
[INFO] 1.2根据dump数据文件名MoeDistributeDispatchV2......hote.o，判断该算子执行流程卡死在dispatch<br>
[INFO] 1.3该卡的输入expertids为[[ 2 5 ...... 11]]<br>

[INFO] 1.3未检测到该卡的expertids有输入异常<br>
[INFO] 1.3根据计算得出该卡的k为:8<br>
[INFO] 1.5输入异常分析完成<br>
<b>上述结果为对dump数据中的输入expertids、epsendcnt进行取值以及校验expertids是否有非法输入与校验epworldsize、rankid是否hccl中的输入相同，用于这两个参数输入异常导致的卡死问题，并将分析出的异常用ERROR的形式输出。</b><br><br>

[INFO] 2. 开始执行序分析<br>
[INFO] 2. dispatch各核执行次数:[7,7,7,7,7,7,.....,7]<br>
[INFO] 2. combine各核执行次数:[6,6,6,6,6,6,6,....,6]<br>
[WARNING] 2. dispatch执行次数:7 = combine执行次数:6 + 1，该卡算子执行流程卡死在dispatch上<br>
[INFO] 2. 执行序分析完成<br>
<b>上述结果为执行序分析，会输出当前分析卡的对应文件、当前卡中的dispatch&combine算子的使用核数、各个核的dispatch&combine算子执行次数以及对比执行次数用于定位因执行次数不匹配导致的卡死问题，并将分析出的异常用warning的形式输出。</b><br><br>

[INFO] 3. 开始dispatch状态位分析<br>
[INFO] 3.1 dispatch_epworldsize:2, dispatch moe专家数:65<br>
[INFO] 3.1 dispatch各核执行位置情况:[1,1,1,1,1,...,1]<br>
[INFO] 3.1该卡不为共享专家卡<br>
[INFO] 3.1 dispatch总状态位:130<br>
[INFO] 3.2 dispatch 1区状态区数据:int32<br>
[INFO] 3.2 dispatch 1区状态区数据shape:16384<br>
[INFO] 3.2 dispatch 1区状态区数据:[0,3,0,.......,]<br>
[INFO] 3.2 dispatch中各核分配到的状态位数量:[2,2,2,2,2,....,1,1,1,...,1]<br>
[ERROR] 3.2 dispatch中有如下下标的核未等到状态[0,1,2,3,...,71]（共72个核）<br>
[INFO] 3.3 dispatch状态位分析完成<br>
<b>上述结果为dispatch的状态位分析，会输出当前卡dispatch所使用的epworldsize、moe专家数以及总状态位数量，以及会将各核的执行位置情况、对应状态区的数据信息、各核分配到的状态位数量情况以及没有等到状态的核的下标打印出来，并将分析出的异常用ERROR的形式输出。</b><br><br>

[INFO] 4. 开始combine状态位分析<br>
[INFO] 4.1 combine_epworldsize:2, combine moe专家数:65<br>
[INFO] 4.1 combine各核执行位置情况:[2,2,2,2,2,2,2,1,...,1]<br>
[INFO] 4.1 combine总状态位:272<br>
[INFO] 4.2 combine 1区状态区数据:int32<br>
[INFO] 4.2 combine 1区状态区数据shape:81920<br>
[INFO] 4.2 combine 1区状态区数据:[1,3,0,0,0,0,.......,0]<br>
[INFO] 4.2 combine中各核分配到的状态位数量:[17,17,...0,.......,0]<br>
[ERROR] 4.2 combine中有如下下标的核未等到状态[7,8,9,...,71]（共65个核）<br>
[INFO] 4.3 combine状态位分析完成<br>
<b>上述结果为combine的状态位分析，会输出当前卡combine所使用的epworldsize、moe专家数以及总状态位数量，以及会将各核的执行位置情况、对应状态区的数据信息、各核分配到的状态位数量情况以及没有等到状态的核的下标打印出来，并将分析出的异常用ERROR的形式输出。</b><br><br>

[INFO] 5. 数据归档<br>
[INFO] 5. 该卡的dispatch&combine的使用核数、epworldsize、moe专家数、0/1标识区数据已归档至win_data.csv<br>
[INFO] 5. 该卡中各核的dispatch&combine的使用核数、执行位置、0/1标识区数据已归档至win_data_list.csv<br>
[INFO] 5. 该卡所使用的dispatch&combine的状态区数据已归档至win_status_list.csv<br>
[INFO] 5. 分析出的错误详细信息已归档至win_analysis_error.csv<br>
<b>上述结果为分析数据归档，将分析时用到的win区dump数据归档至对应的csv文件中，详细信息可参考脚本输出说明</b><br><br>

[INFO] 6. 开始进行多卡的dispatch&combine数据对比<br>
[INFO] 6. 多卡的dispatch&combine的执行次数完全相同<br>
[INFO] 6. 多卡的dispatch&combine的moe专家数完全相同<br>
[INFO] 6. 多卡的dispatch&combine的globalbs完全相同<br>
[INFO] 6. 各卡的dispatch执行次数:[7,7]<br>
[INFO] 6. 各卡的combine执行次数:[6,6]<br>
[INFO] 6. 各卡的dispatch的moe专家数:[1616]<br>
[INFO] 6. 各卡的combine的moe专家数:[16,16]<br>
[INFO] 6. 各卡的dispatch的globalbs:[2,2]<br>
[INFO] 6. 各卡的combine的globalbs:[2,2]<br>
[INFO] 6. 各卡的dispatch&combine执行次数、moe专家数、globalbs数据已归档至win_all_card_data.csv<br>
<b>上述结果为多卡间的dispatch&combine的执行次数、moe专家数、globalbs对比，当多卡间的dispatch/combine算子的上述数据不同时，将该异常以warning的形式输出，其中moe专家数的异常以ERROR的形式输出，并将各卡的dispatch&combine执行次数存储至win_all_card_data.csv。</b>

[INFO] 7. 开始所有卡的expandidx、epsendcnt对比<br>
[INFO] 7. 所有卡计算得到的epsendcnt如下:{0: [1, 1, 2, 2, 4, 5],…… 2: [2, 2, 3, 4, 5, 7]}<br>
[INFO] 7. 卡0的epsendcnt没有异常<br>
[INFO] 7. 卡1算子执行流程未卡死在combine算子，不进行epsendcnt对比<br>
[INFO] 7. 所有专家计算得到的expertidx如下:{0: [(0, 0, 6)],……, 14: [(0, 0, 1), (1, 0, 2)]}<br>
[INFO] 7. 卡0的expandidx没有异常<br>
[INFO] 7. 卡1的算子执行流程并未卡死在combine算子上，不进行异常分析<br>
[INFO] 7. 各卡的expertids、dump数据中读取的expandidx、本卡专家数、该卡算子执行流程卡死在哪个算子上已归档至win_all_card_expandidx.csv<br>
[INFO] 7. 所有卡的expandidx对比完成<br>
<b>上述结果为多卡间的dispatch&combine的expandidx、epsendcnt输入校验，该功能会计算出所有专家对应的expertidx以及每张卡的epsendcnt，当该卡挂在combine算子上时，对该卡的输入expertidx、epsendcnt与计算的expertidx、epsendcnt进行对比，如果不同，将该异常以ERROR的形式输出，并将各卡的输入expertids、dump数据中读取的expandidx、epsendcnt(仅挂在combine算子上的卡，否则为空)、本卡专家数、该卡挂在哪个算子上，五项数据存储至win_all_card_expandidx.csv。</b>

-----------------------<br>
在指定的路径: xxxxxx/xxx/xxx/xx下找到16张卡的profiling数据<br>
[INFO] 开始进行profiling数据分析，指定profiling数据路径xxx/xxx/xxx/xxx<br>
[INFO] 0卡profiling数据: xxx/xxx/xxx/xxx.csv<br>
[INFO] 0卡profiling数据: NONE
[INFO] 16卡profiling数据: xxx/xxx/xxx/xxx.csv<br>
<b>上述结果为开始分析profiling数据前，先将解析到有多少张卡的profiling数据,以及将每张卡对应的profiling数据以INFO的形式输出,如果没有找到对应卡的profiling数据则输出NONE,便于后续查找对应卡的数据</b>

[ERROR] 12卡的profiling数据文件不存在，对应数据用NA替代,跳过该卡的分析<br>
[INFO] profiling详细数据已存储至profiling_all_data.csv<br>
[INFO] 开始计算所有算子的平均抖动<br>
[INFO] 计算算子平均抖动时，不计算算子第0次调用时出现的抖动值<br>
[INFO] 所有算子的平均抖动已存储至profiling_avg_data.csv<br>

<b>上述结果为以0卡的type为基准，将每张的profiling数据中每个type对应的Duration(us)列数据按照对应type的调用次数全部存储至profiling_all_data.csv,其中非法字符以及未找到对应卡的profiling数据均以NA项代替,并根据profiling_all_data.csv的数据计算所有算子的平均抖动,其中不计算第0次调用的抖动值以及NA项,并将结果存储至profiling_avg_data.csv,用于查找数据中抖动过大的算子,以及异常项</b>

[INFO] 开始对每张卡的profiling数据进行aiv_vec_time, aiv_scalar_time, aiv_mte2_time, aiv_mte3_time数据分析<br>
[INFO] 默认对["MoeDistributeDispatch", "MoeDistributeCombine"] 算子进行上述四项数据分析<br>
[INFO] 跳过第0层进行检测<br>
[ERROR] 0卡| Type=MoeDistributeDispatch |第1次出现 | 列名=aiv_vec_time(us)| 值=4.71| 平均值=805.05| 超出范围402.53 - 1207.58(平均值的±50%)<br>
[ERROR] 12卡profiling数据不存在，跳过分析该卡数据<br>
[INFO] profiling数据分析完成<br>
<b>上述结果为对每张卡的profiling数据进行aiv_vec_time, aiv_scalar_time, aiv_mte2_time, aiv_mte3_time四项数据分析，且仅对指定的算子进行分析,该分析会检测是否有非法字符项、值是否大于0、是否超出平均值的±50%</b>

[INFO] 开始进行错误日志分析<br>
[INFO] 检测到(总线错误:0x800000错误，请进一步查看1520日志):51个<br>
[INFO] 检测到GE错误:92个，请前往该网址,搜索GE Errors查询对应报错信息及解决方法: <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/index/index.html><br>
[INFO] 检测到Aicore kernel execute failed类报错:23个，下列为前10条报错的具体信息<br>
[INFO] 检测到Error happened, origin_op_name类报错:42个，下列为前10条报错的具体信息<br>
[INFO] 开始检测是否所有卡都在dispatch or combine算子上执行失败<br>
[ERROR] 卡1没有在dispatch or combine算子上执行失败，在了xxx算子上执行失败<br>
[INFO] 所有检测到的plog日志中的错误类型均写入errors_plog.csv<br>
[INFO] 错误日志分析完成<br>
<b>上述结果为对PLOG_PATH下的所有log日志进行分析，并按照已知的几种规则对检测到的常见报错给出解决方法或关键报错信息,并将所有检测到的ERROR信息输出至error_plog.csv中</b>

[INFO] 开始进行graph图文件分析<br>
[INFO] 卡0 graph文件:xxxxxxxx.json<br>
[INFO] 卡1 graph文件:xxxxxxxx.json<br>
[INFO] 开始对比卡0 <-> 卡1的算子执行顺序、次数<br>
[INFO] 卡0 <-> 卡1完全一致(顺序+内容+次数相同)<br>
[INFO] graph图文件分析完成<br>
<b>上述结果为对GRAPH_PATH下对应卡的graph图文件分析，并根据文件中的信息,判断每张卡的算子执行顺序、次数、内容是否完全一致,不一致则视为异常</b>
