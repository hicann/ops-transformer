# Win-Area Dump Data Analysis Script

English | [简体中文](./README.md)

## Background

- exception dump: After an aic error occurs in an operator, the exception dump function is triggered, and the input and output tensors of the operator are dumped into bin files. The dispatchv2&combinev2 operators additionally register an exception handling callback, which dumps the Windows memory (or win-area memory) that stores the status bits during communication into a bin file. The last 1M of data in this bin file is called the win-area data. It contains the data required for locating common problems such as hangs and aic errors.

- profiling data: In an end-to-end network run, multiple operators are usually invoked together, and some devices may take too long. This symptom is usually the fast-slow device problem caused by abnormal time consumption of an operator on a device. A profiling data analysis tool is required to identify the operators with possible abnormal time consumption and the corresponding device IDs. After the analysis is complete, the tool needs to output a CSV file and print the corresponding abnormal points to the screen.

- graph files: When the cache_compile API is called (for the calling method, see <https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00099.html), graph files are generated in the specified path. Each graph file contains the number of executions and names of operators on each device in each round, which can be used to locate hang symptoms caused by inconsistent execution sequences.>

- plog logs: When the environment variable ASCEND_SLOG_PRINT_TO_STDOUT=1 is set, the corresponding plog logs are generated in /root/ascend/log. The plog logs can be used to view the corresponding error information and error codes.

- In dispatch&combine problem locating, overall analysis and judgment need to be performed based on multi-device data, which is difficult to analyze. Therefore, a multi-device data analysis tool is required for quick analysis. The tool must be able to parse data with one click based on the existing data, analyze the possibly abnormal points according to predefined rules, and directly print the description of the abnormal symptoms. A CSV file needs to be output after some data is parsed, and the tool must print the analysis currently being performed.

## Function Description

- dump data parsing: Parses the specified dump data to obtain the win-area data of the dump data, obtains the moe expert count, number of used cores, execution count, epworldsize, and 0/1 flag bits of each device, and checks whether these parameters are consistent on each core of the abnormal device. The analyzed abnormal points are output, which can be used to locate problems caused by abnormal moe expert count or epworldsize input.

- Input abnormality analysis: Used to analyze whether a hang of an operator is caused by abnormal parameter input

- Execution sequence analysis: Used to locate hang problems caused by mismatched execution counts.

- Status bit analysis: Obtains the execution position information of each core on each device, determines which cores have not received the status bits, and finds the specific status bit that a core has not received in the corresponding 0/1 status area based on the 0/1 flag bits of dispatchv2&combinev2. This analysis is used to display the specific symptoms after an operator hangs.

- graph file analysis: Used to locate hang problems caused by inconsistent operator execution counts or execution orders across devices

- plog log analysis: Used to preliminarily locate abnormal problems caused by some common issues

- profiling data analysis: Analyzes the Duration, aiv_vec_time, aiv_scalar_time, aiv_mte2_time, and aiv_mte3_time data in profiling data, calculates the average jitter of each operator, and outputs the abnormal items, which can be used to locate the cause of performance abnormalities
Duration: Operator execution time, in us
aiv_vec_time: Time consumed by vector operation instructions on the AI Vector Core, in us
aiv_scalar_time: Time consumed by scalar operation instructions on the AI Vector Core, in us
aiv_mte2_time: Time consumed by on-chip memory -> AI CORE data movement instructions on the AI Vector Core, in us
aiv_mte3_time: Time consumed by AI CORE -> on-chip memory data movement instructions on the AI Vector Core, in us

## Script Input and Output Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>TARGET_DIR</td>
   <td>Optional input</td>
   <td>File path where the dump data is located. The storage location of logs written by the PyTorch API is controlled by the ASCEND_WORK_PATH environment variable. The storage location of the dump data can be obtained from the logs. If this environment variable is not set, the dump data is stored in the extra-info/data-dump/ path in the current directory. This parameter is required for dump data analysis.</td>
   <td>str</td>
  </tr>
  <tr>
   <td>TOOL_PATH</td>
   <td>Optional input</td>
   <td>File path of the cann repository in the package installation path, for example, TOOL_PATH=install_pkg/cann-x.x.x/. This parameter is required for dump data analysis.</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SOC_VERSION</td>
   <td>Optional input</td>
   <td>Version input corresponding to the constructed dump data. Currently only values such as SOC_VERSION=910_93 or 950 are supported. This parameter is required for dump data analysis.</td>
   <td>str</td>
  </tr>
  <tr>
   <td>SHARE_EXPERT_CARD_COUNT</td>
   <td>Optional input</td>
   <td>Number of shared expert devices input when the dump data is constructed. The default value is 0 if not input. SHARE_EXPERT_CARD_COUNT <= total number of devices, for example, SHARE_EXPERT_CARD_COUNT=1.</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>SP_MOE_NUM</td>
   <td>Optional input</td>
   <td>Number of special experts input when the dump data is constructed. The default value is 0 if not input. SP_MOE_NUM >0, for example, SHARE_EXPERT_CARD_COUNT=1.</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>TP_WORLDSIZE</td>
   <td>Optional input</td>
   <td>TP_WORLDSIZE when the dump data is constructed. The default value is 1 if not input. TP_WORLDSIZE >= 1, for example, TP_WORLDSIZE=1.</td>
   <td>INT</td>
  </tr>
  <tr>
    <td>SHARE_EXPERT_NUM</td>
    <td>Optional input</td>
    <td>Number of shared experts input when the dump data is constructed. The default value is 0 if not input. SHARE_EXPERT_NUM > 0, for example, SHARE_EXPERT_NUM=1</td>
    <td>INT</td>
  </tr>
  <tr>
    <td>START_A2_CLUSTER</td>
    <td>Optional input</td>
    <td><term>Atlas A2 products</term>: When parsing dump data, distinguishes whether to start analyzing multi-machine data. The default value is 0 if not input. Multi-machine analysis is enabled when the value is 1. The parsing results of each machine must be obtained before execution, for example, START_A2_CLUSTER=1</td>
    <td>INT</td>
  </tr>
  <tr>
    <td>PROFILING_PATH</td>
    <td>Optional input</td>
    <td>Specifies the storage path of profiling data. By default, the data is searched in TARGET_DIR if not input, for example, PROFILING_PATH=xxx</td>
    <td>INT</td>
  </tr>
    <tr>
    <td>PLOG_PATH</td>
    <td>Optional input</td>
    <td>Specifies the storage path of plog logs, for example, PROFILING_PATH=xxx. This parameter is required for plog log analysis</td>
    <td>INT</td>
  </tr>
    <tr>
    <td>GRAPH_PATH</td>
    <td>Optional input</td>
    <td>Specifies the storage path of graph files, for example, GRAPH_PATH=xxx. This parameter is required for graph file analysis. This parameter must point to the graph file path of the corresponding model</td>
    <td>INT</td>
  </tr>
  <tr>
   <td>Screen log</td>
   <td>Output</td>
   <td>Info: Analyzed win-area data information, for example, number of cores used by dispatch on this device: 72, number of cores used by combine: 72. Number of moe experts of dispatch on this device: 62, number of moe experts of combine: 62<br> Warning: Analyzed abnormal points. For example, execution count dispatch = combine + 1. Therefore, the operator execution flow of this device is considered to hang on the dispatch operator.</td>
   <td>str</td>
  </tr>
  <tr>
   <td>win_status_list</td>
   <td>Output</td>
   <td>Stores all status bit data in the 0/1 status area of dispatchv2/combinev2 specified by the 0/1 flag area in the win area, and is used to view the specific status bit that a core has not received.<br>
       For example, if the status area of dispatchv2 of this device is parsed to be stored in area 0, the data of the dispatch area 0 status area [0:64 * 1024] in the win-area dump data is stored in win_status_list.csv</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>win_all_card_data</td>
   <td>Output</td>
   <td>Stores the execution counts, moe expert counts, and globalbs of dispatchv2&combinev2 on each device, and checks whether these parameters are consistent across devices.<br> For example, d0 dispatch: 1 (indicating that the dispatchv2 operator of the first device ran once)</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>win_analysis_error</td>
   <td>Output</td>
   <td>Stores the abnormal points and abnormal positions found by status bit analysis, that is, the specific Warning information about status bit abnormalities that is not displayed in screen logs is stored in this CSV file.<br>For example, the second status bit of core 35 of the dispatchv2 operator on device d0 is displayed as not received in the dispatch area 1 status area</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>win_data</td>
   <td>Output</td>
   <td>Stores the parsed moe expert count, number of used cores, execution count, epworldsize, rankid, globalbs, input rankid and globalbs when the HCCL communication link is established, computed bs, k, and 0/1 flag bit data of each device.<br>For example, d0_dispatch_moe expert count: 65</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>win_data_list</td>
   <td>Output</td>
   <td>Stores the parsed execution position status and 0/1 flag bit data of each core on each device,<br>For example, if the number of cores used by dispatchv2 on device d0 is 72, the moe expert counts of the 72 cores in the win area are recorded as a list of length 72 and stored in win_data_list, and the d0_dispatch 0/1 flag area data is [0:71].</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>win_all_card_expandidx</td>
   <td>Output</td>
   <td>Stores the parsed input expertids of each device, expandidx read from the dump data (only for devices running the combine operator; otherwise, empty), epsendcnt read from the dump data (only for devices running the combine operator; otherwise, empty), local expert count, and the operator on which the device runs,<br>For example, local expert count of device 0: 8.</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>profiling_all_data</td>
   <td>Output</td>
   <td>Based on the data list of device 0, stores the parsed Duration (us) column data of each type on each device and the corresponding maximum, minimum, and jitter values, and stores these data by invocation count. If a cell contains an invalid character or the profiling data of the corresponding device is lost, NA is used instead,<br>For example: device 0| type:dispatch| nth invocation: 0| max:xx| min:xx| jitter:xx| device 0:xx| ...| device 15: xx.</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>profiling_avg_data</td>
   <td>Output</td>
   <td>Calculates the average jitter, total invocation count, and valid invocation count (excluding the 0th invocation and NA items) of each type based on the data in the profiling_all_data.csv output table, and stores the results in profiling_avg_data.csv,<br>For example: type:dispatch| invocation count: 60 |valid invocation count (excluding the 0th invocation and NA items): 52| average jitter (us): xx.</td>
   <td>CSV table</td>
  </tr>
  <tr>
   <td>error_plog</td>
   <td>Output</td>
   <td>Outputs all ERROR entries under PLOG_PATH that match known error causes, classifies them by type, and outputs them to this CSV file<br>For example: error type error message.</td>
   <td>CSV table</td>
  </tr>
 </tbody>
</table>

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| Direct invocation of the dump_analysis.sh script | bash dump_analysis.sh TARGET_DIR=xxx TOOL_PATH=xxx PROFILING_PATH=xxx SP_MOE_NUM=0 TP_WORLDSIZE=1 SOC_VERSION=950 SHARE_EXPERT_CARD_COUNT=1 SHARE_EXPERT_NUM=1 GRAPH_PATH=xxx PLOG_PATH=xxx| Passes parameters to the sh script to parse the dump data under the specified TARGET_DIR path, analyze the profiling data under PROFILING_PATH, analyze the graph file data under GRAPH_PATH, and analyze the plog logs under PLOG_PATH. |
| Direct invocation of the dump_analysis.sh script | bash dump_analysis.sh SOC_VERSION=910B START_A2_CLUSTER=1 | Places the parsing results obtained when START_A2_CLUSTER=0 in the same path on the same device and then executes the command. The sh script parses the multi-machine data of each machine. Recommended for the A2 Dispatch and Combine Fullmesh solution |

## Sample Code Result Description

[INFO] Start analyzing data of device 1<br>
[INFO] Parsing file: /xxx/xxx/xxx/data-dump/mc2_exception_info.xxx<br>
<b>The preceding result is the data information of the device currently being analyzed, and outputs the sequence number of the device currently being analyzed and the corresponding dump file</b><br>

[INFO] 1. Start input abnormality analysis

[INFO] 1. The number of cores used by dispatch&combine on this device is dispatch: 72, combine: 72<br>
[INFO] 1.1 rankid of each dispatch core: [1,...,1]<br>
[INFO] 1.1 epworldsize of each dispatch core: [2,...,2]<br>
[INFO] 1.1 input rankid of each dispatch core when the HCCL communication link is established: [0,...,0]<br>
[INFO] 1.1 input epworldsize of each dispatch core when the HCCL communication link is established: [0,...,0]<br>
[INFO] 1.1 moe expert count of each dispatch core: [1,...,1]<br>
[INFO] 1.1 globalbs of each dispatch core: [1,...,1]<br>
[INFO] 1.1 dispatch_rankid:1 dispatch_hccl_rankid:0, dispatch epworldsize:2, dispatch_hccl epworldsize:0, dispatch moe expert count: 16, dispatch globalbs: 2, bs calculated from the dispatch input: 1<br>
[ERROR] 1.1 The rankid:1 in the win-area data of dispatch is different from the input rankid:0 when the HCCL communication link is established<br>
[ERROR] 1.1 The epworldsize:1 in the win-area data of dispatch is different from the input epworldsize:0 when the HCCL communication link is established<br>

[INFO] 1.1 rankid of each combine core: [1,...,1]<br>
[INFO] 1.1 epworldsize of each combine core: [2,...,2]<br>
[INFO] 1.1 input rankid of each combine core when the HCCL communication link is established: [0,...,0]<br>
[INFO] 1.1 input epworldsize of each combine core when the HCCL communication link is established: [0,...,0]<br>
[INFO] 1.1 moe expert count of each combine core: [1,...,1]<br>
[INFO] 1.1 globalbs of each combine core: [1,...,1]<br>
[INFO] 1.1 combine_rankid:1 combine_hccl_rankid:0, combine epworldsize:2, combine_hccl epworldsize:0, combine moe expert count: 16, combine globalbs: 2, bs calculated from the combine input: 1<br>
[ERROR] 1.1 The rankid:1 in the win-area data of combine is different from the input rankid:0 when the HCCL communication link is established<br>
[ERROR] 1.1 The epworldsize:1 in the win-area data of combine is different from the input epworldsize:0 when the HCCL communication link is established<br>
<b>The preceding result obtains the values of each item in the dump data and compares whether the inputs of rankid and epworldsize are the same as the inputs when the HCCL communication link is established. It is used for hang problems caused by abnormal input of each parameter, and outputs the analyzed abnormalities in ERROR form.</b><br><br>

[INFO] 1.2 This device is not a shared expert device<br>
[INFO] 1.2 The local expert count is: 8<br>
[INFO] 1.2 The bs of this device calculated is 1<br>
[INFO] 1.2 Based on the dump data file name MoeDistributeDispatchV2......hote.o, the operator execution flow is determined to hang on dispatch<br>
[INFO] 1.3 The input expertids of this device is [[ 2 5 ...... 11]]<br>

[INFO] 1.3 No input abnormality is detected in the expertids of this device<br>
[INFO] 1.3 The k of this device calculated is: 8<br>
[INFO] 1.5 Input abnormality analysis is complete<br>
<b>The preceding result obtains the values of the input expertids and epsendcnt in the dump data, checks whether the expertids contain invalid input, and checks whether epworldsize and rankid are the same as the inputs in HCCL. It is used for hang problems caused by abnormal input of these two parameters, and outputs the analyzed abnormalities in ERROR form.</b><br><br>

[INFO] 2. Start execution sequence analysis<br>
[INFO] 2. Execution count of each dispatch core: [7,7,7,7,7,7,.....,7]<br>
[INFO] 2. Execution count of each combine core: [6,6,6,6,6,6,6,....,6]<br>
[WARNING] 2. dispatch execution count: 7 = combine execution count: 6 + 1. The operator execution flow of this device hangs on dispatch<br>
[INFO] 2. Execution sequence analysis is complete<br>
<b>The preceding result is the execution sequence analysis. It outputs the corresponding file of the device currently being analyzed, the number of cores used by the dispatch&combine operators on the current device, the execution count of the dispatch&combine operators of each core, and the comparison of execution counts, which is used to locate hang problems caused by mismatched execution counts, and outputs the analyzed abnormalities in warning form.</b><br><br>

[INFO] 3. Start dispatch status bit analysis<br>
[INFO] 3.1 dispatch_epworldsize:2, dispatch moe expert count: 65<br>
[INFO] 3.1 Execution position status of each dispatch core: [1,1,1,1,1,...,1]<br>
[INFO] 3.1 This device is not a shared expert device<br>
[INFO] 3.1 Total dispatch status bits: 130<br>
[INFO] 3.2 Data of the dispatch area 1 status area: int32<br>
[INFO] 3.2 Shape of the data in the dispatch area 1 status area: 16384<br>
[INFO] 3.2 Data of the dispatch area 1 status area: [0,3,0,.......,]<br>
[INFO] 3.2 Number of status bits allocated to each core in dispatch: [2,2,2,2,2,....,1,1,1,...,1]<br>
[ERROR] 3.2 The cores with the following subscripts in dispatch have not received the status: [0,1,2,3,...,71] (72 cores in total)<br>
[INFO] 3.3 Dispatch status bit analysis is complete<br>
<b>The preceding result is the status bit analysis of dispatch. It outputs the epworldsize, moe expert count, and total number of status bits used by dispatch on the current device, prints the execution position status of each core, the data information of the corresponding status area, the number of status bits allocated to each core, and the subscripts of the cores that have not received the status, and outputs the analyzed abnormalities in ERROR form.</b><br><br>

[INFO] 4. Start combine status bit analysis<br>
[INFO] 4.1 combine_epworldsize:2, combine moe expert count: 65<br>
[INFO] 4.1 Execution position status of each combine core: [2,2,2,2,2,2,2,1,...,1]<br>
[INFO] 4.1 Total combine status bits: 272<br>
[INFO] 4.2 Data of the combine area 1 status area: int32<br>
[INFO] 4.2 Shape of the data in the combine area 1 status area: 81920<br>
[INFO] 4.2 Data of the combine area 1 status area: [1,3,0,0,0,0,.......,0]<br>
[INFO] 4.2 Number of status bits allocated to each core in combine: [17,17,...0,.......,0]<br>
[ERROR] 4.2 The cores with the following subscripts in combine have not received the status: [7,8,9,...,71] (65 cores in total)<br>
[INFO] 4.3 Combine status bit analysis is complete<br>
<b>The preceding result is the status bit analysis of combine. It outputs the epworldsize, moe expert count, and total number of status bits used by combine on the current device, prints the execution position status of each core, the data information of the corresponding status area, the number of status bits allocated to each core, and the subscripts of the cores that have not received the status, and outputs the analyzed abnormalities in ERROR form.</b><br><br>

[INFO] 5. Data archiving<br>
[INFO] 5. The number of used cores, epworldsize, moe expert count, and 0/1 flag area data of dispatch&combine on this device have been archived to win_data.csv<br>
[INFO] 5. The number of used cores, execution positions, and 0/1 flag area data of dispatch&combine of each core on this device have been archived to win_data_list.csv<br>
[INFO] 5. The status area data of dispatch&combine used by this device has been archived to win_status_list.csv<br>
[INFO] 5. The detailed analyzed error information has been archived to win_analysis_error.csv<br>
<b>The preceding result is the analysis data archiving. The win-area dump data used in the analysis is archived to the corresponding CSV files. For details, refer to the script output description</b><br><br>

[INFO] 6. Start comparing the dispatch&combine data of multiple devices<br>
[INFO] 6. The execution counts of dispatch&combine are completely the same across devices<br>
[INFO] 6. The moe expert counts of dispatch&combine are completely the same across devices<br>
[INFO] 6. The globalbs of dispatch&combine is completely the same across devices<br>
[INFO] 6. dispatch execution count of each device: [7,7]<br>
[INFO] 6. combine execution count of each device: [6,6]<br>
[INFO] 6. moe expert count of dispatch on each device: [1616]<br>
[INFO] 6. moe expert count of combine on each device: [16,16]<br>
[INFO] 6. globalbs of dispatch on each device: [2,2]<br>
[INFO] 6. globalbs of combine on each device: [2,2]<br>
[INFO] 6. The dispatch&combine execution count, moe expert count, and globalbs data of each device have been archived to win_all_card_data.csv<br>
<b>The preceding result is the comparison of the execution count, moe expert count, and globalbs of dispatch&combine across devices. When the preceding data of the dispatch/combine operators differs across devices, the abnormality is output in warning form, and the abnormality of the moe expert count is output in ERROR form. The dispatch&combine execution counts of each device are stored in win_all_card_data.csv.</b>

[INFO] 7. Start comparing the expandidx and epsendcnt of all devices<br>
[INFO] 7. The epsendcnt calculated for all devices is as follows: {0: [1, 1, 2, 2, 4, 5],... 2: [2, 2, 3, 4, 5, 7]}<br>
[INFO] 7. The epsendcnt of device 0 has no abnormality<br>
[INFO] 7. The operator execution flow of device 1 does not hang on the combine operator, so epsendcnt comparison is not performed<br>
[INFO] 7. The expertidx calculated for all experts is as follows: {0: [(0, 0, 6)],..., 14: [(0, 0, 1), (1, 0, 2)]}<br>
[INFO] 7. The expandidx of device 0 has no abnormality<br>
[INFO] 7. The operator execution flow of device 1 does not hang on the combine operator, so abnormality analysis is not performed<br>
[INFO] 7. The expertids, expandidx read from the dump data, local expert count, and the operator on which the execution flow of each device hangs have been archived to win_all_card_expandidx.csv<br>
[INFO] 7. The expandidx comparison of all devices is complete<br>
<b>The preceding result is the input check of expandidx and epsendcnt of dispatch&combine across devices. This function calculates the expertidx of all experts and the epsendcnt of each device. When the device runs the combine operator, the input expertidx and epsendcnt of the device are compared with the calculated expertidx and epsendcnt. If they are different, the abnormality is output in ERROR form. The input expertids, expandidx read from the dump data, epsendcnt (only for devices running the combine operator; otherwise, empty), local expert count, and the operator on which the device runs of each device are stored in win_all_card_expandidx.csv.</b>

-----------------------<br>
The profiling data of 16 devices is found in the specified path: xxxxxx/xxx/xxx/xx<br>
[INFO] Start profiling data analysis. Specified profiling data path: xxx/xxx/xxx/xxx<br>
[INFO] Profiling data of device 0: xxx/xxx/xxx/xxx.csv<br>
[INFO] Profiling data of device 0: NONE
[INFO] Profiling data of device 16: xxx/xxx/xxx/xxx.csv<br>
<b>The preceding result shows that before profiling data analysis starts, the number of devices whose profiling data is parsed is output first, and the profiling data corresponding to each device is output in INFO form. If the profiling data of a device is not found, NONE is output, making it easy to find the data of the corresponding device later</b>

[ERROR] The profiling data file of device 12 does not exist. The corresponding data is replaced with NA, and the analysis of this device is skipped<br>
[INFO] The detailed profiling data has been stored in profiling_all_data.csv<br>
[INFO] Start calculating the average jitter of all operators<br>
[INFO] When calculating the average jitter of operators, the jitter value of the 0th invocation of an operator is not calculated<br>
[INFO] The average jitter of all operators has been stored in profiling_avg_data.csv<br>

<b>The preceding result is based on the types of device 0. The Duration (us) column data corresponding to each type in the profiling data of each device is all stored in profiling_all_data.csv by the invocation count of the corresponding type. Invalid characters and missing profiling data of the corresponding device are replaced with NA items. The average jitter of all operators is calculated based on the data in profiling_all_data.csv, excluding the jitter value of the 0th invocation and NA items, and the results are stored in profiling_avg_data.csv. This is used to find operators with excessive jitter and abnormal items in the data</b>

[INFO] Start the aiv_vec_time, aiv_scalar_time, aiv_mte2_time, and aiv_mte3_time data analysis on the profiling data of each device<br>
[INFO] By default, the preceding four data analysis items are performed on the ["MoeDistributeDispatch", "MoeDistributeCombine"] operators<br>
[INFO] Layer 0 is skipped during detection<br>
[ERROR] Device 0| Type=MoeDistributeDispatch |1st occurrence | column name=aiv_vec_time(us)| value=4.71| average=805.05| out of the range 402.53 - 1207.58 (±50% of the average)<br>
[ERROR] The profiling data of device 12 does not exist. The analysis of the data of this device is skipped<br>
[INFO] Profiling data analysis is complete<br>
<b>The preceding result performs the four data analysis items of aiv_vec_time, aiv_scalar_time, aiv_mte2_time, and aiv_mte3_time on the profiling data of each device, and analyzes only the specified operators. This analysis detects whether there are invalid character items, whether the value is greater than 0, and whether the value exceeds ±50% of the average</b>

[INFO] Start error log analysis<br>
[INFO] Detected (bus error: 0x800000 error, check the 1520 log for details): 51 occurrences<br>
[INFO] Detected 92 GE errors. Go to the website and search for GE Errors to view the corresponding error information and solutions: <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/index/index.html><br>
[INFO] Detected 23 errors of the Aicore kernel execute failed type. The details of the first 10 errors are as follows<br>
[INFO] Detected 42 errors of the Error happened, origin_op_name type. The details of the first 10 errors are as follows<br>
[INFO] Start detecting whether all devices fail to execute on the dispatch or combine operator<br>
[ERROR] Device 1 does not fail to execute on the dispatch or combine operator, but fails to execute on the xxx operator<br>
[INFO] All detected error types in the plog logs are written to errors_plog.csv<br>
[INFO] Error log analysis is complete<br>
<b>The preceding result analyzes all log files under PLOG_PATH, provides solutions or key error information for the detected common errors according to several known rules, and outputs all detected ERROR information to error_plog.csv</b>

[INFO] Start graph file analysis<br>
[INFO] Graph file of device 0: xxxxxxxx.json<br>
[INFO] Graph file of device 1: xxxxxxxx.json<br>
[INFO] Start comparing the operator execution order and count between device 0 <-> device 1<br>
[INFO] Device 0 <-> device 1 are completely consistent (the same order, content, and count)<br>
[INFO] Graph file analysis is complete<br>
<b>The preceding result analyzes the graph files of the corresponding devices under GRAPH_PATH, and determines based on the information in the files whether the operator execution order, count, and content of each device are completely consistent. Any inconsistency is regarded as an abnormality</b>

## moe_ep exception dump analysis (IS_MOE_EP=1)

### Background

The MoeEpDispatch/MoeEpCombine operators of the A5(950) chip use a structured win-area layout, which is different from the fixed 1MB raw dump of the original V2 operators. The new `IS_MOE_EP` parameter is added. When it is set to 1, the `dump_analysis_moe_ep.py` script is called for structured parsing. This parameter takes effect only when `SOC_VERSION=950`.

### dump_analysis.sh Parameter Description

| Parameter | Required | Description |
|------|------|------|
| `TARGET_DIR` | Yes | Dump data path. For a single device, it is the directory containing exception_info. For multiple devices, it is the parent directory containing numbered subdirectories (such as 0/, 1/, 3/, and 4/) |
| `TOOL_PATH` | Yes | Path of the tools directory under the package installation path, for example, `xxx/xxx/pkg/cann-x.x.x/` |
| `SOC_VERSION` | Yes | Chip version. Currently only `950` is supported |
| `IS_MOE_EP` | No | 0: uses the original `dump_analysis.py`; 1: uses `dump_analysis_moe_ep.py`. This parameter takes effect only when `SOC_VERSION=950`. Default value: 0 |
| `SP_MOE_NUM` | No | Number of special experts. The default value is 0 if not specified |
| `TP_WORLDSIZE` | No | TP_WORLDSIZE. The default value is 1 if not specified |
| `SHARE_EXPERT_CARD_COUNT` | No | Number of shared expert devices. The default value is 0 if not specified |
| `SHARE_EXPERT_NUM` | No | Number of shared experts. The default value is 0 if not specified |

### Invocation Method

Invoked through `dump_analysis.sh` (the commands for single-device and multi-device are the same and are distinguished by the TARGET_DIR directory structure):

```bash
# Basic invocation
bash dump_analysis.sh TARGET_DIR=/path/to/dump TOOL_PATH=/path/to/cann SOC_VERSION=950 IS_MOE_EP=1

# Full parameter example
bash dump_analysis.sh \
    TARGET_DIR=/path/to/dump \
    TOOL_PATH=/path/to/cann \
    SOC_VERSION=950 \
    IS_MOE_EP=1 \
    SP_MOE_NUM=0 \
    TP_WORLDSIZE=1 \
    SHARE_EXPERT_CARD_COUNT=0 \
    SHARE_EXPERT_NUM=0
```

Single device: TARGET_DIR directly contains the exception_info file. Multiple devices: TARGET_DIR contains subdirectories named by numbers (such as 0/, 1/, 3/, and 4/). Device IDs do not need to be consecutive.

### dump File Description

- `exception_info.*` — CANN framework dump. After being parsed by the CANN parsing script, input and output files such as `input.0*bin` (context) and `input.2*bin` (topk_idx) are generated
- `*.metadata.bin` — Metadata area, containing the `MoeEpDumpMetadata` structure (10 uint32 fields + the offset/size of 6 Regions, where offset/size is uint64)
- `*.per_core_diag.bin` — Per-core diagnosis area, 100 cores x 512B. The 512B slot of each core contains 4 MoeEpCoreDiagRecord entries (64B each), with a fixed mapping by byte offset:

  | byte offset | op_index | Operator |
  |-------------|----------|------|
  | 0           | 0        | dispatch |
  | 64          | 1        | dispatch_epilogue |
  | 128         | 2        | combine |
  | 192         | 3        | combine_epilogue |

  record layout: uint64 opCnt + uint32 runPosition + uint32 epRankId + uint32 aivId
- `*.count_notify.bin` — Count Notify area, epWorldSize 512B blocks, with the first uint32 of each block being the token count
- `*.expert_count.bin` — Expert Count area, epWorldSize 512B blocks, with localExpertNum uint32 values taken from each block
- `*.combine_token_state.bin` — Combine Token State area, nmt*topK 512B blocks, with the first uint32 of each block being the state

### dump File Data Structure Description

#### Metadata structure (`metadata.bin`)

The Metadata area is fixed at 64 KiB. The valid structure (<=512B) is located at the beginning of the area, and the rest is reserved.

```cpp
struct MoeEpDumpRegion {
    uint64_t offset;
    uint64_t size;
};

struct MoeEpDumpMetadata {
    uint32_t layoutVersion;        // Layout version
    uint32_t nmt;                  // numMaxTokensPerRank
    uint32_t topK;                 // topk value
    uint32_t hidden;               // hidden dimension
    uint32_t epWorldSize;          // Total number of EP devices
    uint32_t localExpertNum;       // Number of local experts
    uint32_t aivNum;               // Number of AIV cores
    uint32_t networkMode;          // 0=direct, 1=hybrid
    uint32_t serverNum;            // Number of servers
    uint32_t rankNumPerServer;     // Number of devices per server
    MoeEpDumpRegion regions[6];   // 6 fixed Regions (offset: uint64, size: uint64)
};
```

Region index definition:

| Index | Region Name | Description |
|------|------------|------|
| 0 | PER_CORE_DIAG | Per-core diagnosis information, 100x512B |
| 1 | COUNT_NOTIFY | Dispatch token count of each source rank |
| 2 | EXPERT_COUNT | Token count of each expert on each device |
| 3 | DISPATCH_SLOT_STATE | Dispatch communication completion status |
| 4 | COMBINE_TOKEN_STATE | Combine status of each (token, topk) |
| 5 | HYBRID_SCALEOUT_STATUS | Hybrid scaleout status ({0,0} is filled in direct mode) |

#### Per-core diagnosis record (`per_core_diag.bin`)

The per-core diagnosis area is fixed at 100 cores x 512B, and each AIV uses one 512B slot. In each slot, the first 4 x 64B are allocated to the four operators, and the remaining 256B is reserved.

`MoeEpCoreDiagRecord` is fixed at 64B, and the valid fields occupy the first 20B:

```cpp
struct MoeEpCoreDiagRecord {
    uint64_t opCnt;        // Execution round, incremented by 1 for each Init
    uint32_t runPosition;  // Last completed checkpoint, 0=Init not completed
    int32_t  epRankId;     // Local EP rank
    uint32_t aivId;        // AIV ID
    uint8_t  reserved[44]; // Reserved
};
```

Field offsets:

| Field | Offset | Size |
|------|--------|------|
| opCnt | 0 | 8B |
| runPosition | 8 | 4B |
| epRankId | 12 | 4B |
| aivId | 16 | 4B |
| reserved | 20 | 44B |

`runPosition` is numbered independently by operator, and the definitions differ among operators:

| Value | dispatch | dispatch_epilogue | combine | combine_epilogue |
|----|----------|-------------------|---------|------------------|
| 0 | Init not completed | Init not completed | Init not completed | Init not completed |
| 1 | Init completed | Init completed | Init completed | Init completed |
| 2 | Count wait and receive completed | Waits until all Dispatch statuses arrive | URMA initiation completed | Waits until all Combine statuses arrive |
| 3 | URMA initiation completed | Output processing completed | - | Output processing completed |

#### Count Notify (`count_notify.bin`)

Layout: `epWorldSize` 512B blocks. The first uint32 of each block is the token count sent by the source rank.

| Offset | Size | Field |
|------|------|------|
| rank_id * 512B | 4B | Token count sent by the rank (uint32) |
| rank_id * 512B + 4B | 508B | Reserved |

#### Expert Count (`expert_count.bin`)

Layout: `epWorldSize` 512B blocks. Each block contains `localExpertNum` uint32 values, which are the token counts received by each expert of the rank.

| Offset | Size | Field |
|------|------|------|
| rank_id * 512B + expert_id * 4B | 4B | Token count received by the expert (uint32) |
| rank_id * 512B + localExpertNum * 4B | 512 - localExpertNum*4 B | Reserved |

#### Dispatch Slot State (`dispatch_slot_state.bin`)

Layout: `dispatchNotifyCount * epWorldSize` 512B blocks. The first uint32 of each block is the communication completion status of the (source rank, notify batch).

#### Combine Token State (`combine_token_state.bin`)

Layout: `nmt * topK` 512B blocks. The first uint32 of each block is the return completion status of the (token, topk).

| Offset | Size | Field |
|------|------|------|
| (token_id * topK + topk_id) * 512B | 4B | Return completion status (uint32) |
| ... + 4B | 508B | Reserved |

#### Hybrid Scaleout Status (`hybrid_scaleout_status.bin`)

Layout: `serverNum * nmt` 512B blocks. Each block is the scaleout status in hybrid mode. In direct mode, the Region is `{0, 0}` and no data is output.

### MoeCommContext structure (context / input.0)

```cpp
struct MoeCommContext {
    uint32_t epRankId;              // Device ID
    uint32_t rankSizePerServer;    // Number of ranks per server (epWorldSize = rankSizePerServer * serverNum)
    uint64_t epHcclBuffer[1024];   // Win-area address, print the first rankSize entries
    uint64_t hcommHandle[1024];    // Handle address, print the first rankSize entries
    uint32_t channelsPerRank;       // Number of channels per device
};
```

### Analysis Flow

1. **Parses parameters and builds the device directory list** — Receives `target_dir`, `file_num`, and `is_multi`. For a single device, target_dir is used directly. For multiple devices, directories are built as `target_dir/0` to `target_dir/(N-1)`
2. **Collects the Metadata of all devices** — Traverses each device to read `.metadata.bin`, parses 10 uint32 fields + 6 Regions (offset/size is uint64), and calculates the global expert count = sum of localExpertNum of all devices
3. **Performs single-device analysis device by device**:
   - Prints the Metadata of the device (including start/end prompts)
   - Parses the context (MoeCommContext structure): epRankId, rankSizePerServer, win-area address, Handle address, and channelsPerRank
   - Parses topk_idx (input.2): reshapes the data to (bs, k) based on topK in the Metadata, checks that the value range is 0 <= val < global expert count, and prints all data
   - Analyzes the per-core diagnosis area: reads 4 op records (dispatch/dispatch_epilogue/combine/combine_epilogue), determines the core count (stops when all 4 records are 0), and prints opCnt/runPosition/epRankId/aivId core by core and op by op
   - Analyzes Count Notify/Expert Count/Combine Token State: parses the data of each Region and prints the data with shape descriptions
4. **Writes to Excel** — Outputs `moe_ep_dump_analysis_result.xlsx` with 7 sheets in total

### Excel Output Sheet Description

| Sheet | Columns | Description |
|-------|----|----|
| `metadata` | card_num + 10 fields + 6 Regions (offset/size alternating) + bs + core_count | One row per device |
| `context` | card_num, epRankId, rankSizePerServer, win-area address_0~N, Handle address_0~N, channelsPerRank | Flattened based on the MoeCommContext structure, one row per device, and N is the maximum rankSizePerServer of all devices |
| `topk_idx` | card_num, token_id, topk_id, expert_id | One row per (token, topk) |
| `per_card_per_core` | card_num, core_count, core_id, op_index, op_name, op_count, run_pos, ep_rank_id, aiv_id | One row per core per op |
| `count_notify` | card_num, epWorldSize, rank_id, count | One row per device per rank |
| `expert_count` | card_num, localExpertNum, rank_id, expert_id, count | One row per device per rank per expert |
| `combine_token_state` | card_num, idx, token_state | One row per device per (token, topk) |

### Boundary Protection

- Region data reading: breaks with a warning when `(i+1)*512 > raw_data.shape[0]`
- `collect_all_metadata` failed device: appends default values (field 0, Region [0, 0]) to ensure that indexes are not misaligned
- `get_topk_idx`: uses `//` for integer division, truncates with a warning when the division is not exact
- `read_region_file`: returns the empty array `np.array([], dtype=np.uint8)` when the file is not found, avoiding IndexError
- input file fuzzy matching: `"input.0" in filename and filename.endswith("bin")`, compatible with the `exception_info.*.input.0.undefined.bin` format
