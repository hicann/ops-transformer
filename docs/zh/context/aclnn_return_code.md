# aclnn返回码

调用aclnn API时，常见的接口返回码如[表1](#table1)所示。
对于异常状态码值，可以通过aclGetRecentErrMsg接口（[《Runtime运行时 API》](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)）获取异常信息，可根据报错提示排查问题或联系技术支持。

**表1**  返回状态码 <a id="table1"></a>

| 状态码名称 | 状态码值 | 状态码说明 |
| ----- | ----- | ------ |
| ACLNN_SUCCESS | 0 | 成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 参数校验错误，参数中存在非法的nullptr。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数校验错误，如输入的两个数据类型不满足输入类型推导关系。 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API内部调用npu runtime的接口异常。 |
| ACLNN_ERR_INNER_XXX | 561xxx | API内部发生异常。 |

更多关于ACLNN_ERR_INNER_XXX类状态码的说明如[表2](#table2)所示。

**表2**  异常状态码 <a id="table2"></a>

| 状态码名称 | 状态码值 | 状态码说明 |
| ----- | ----- | ------ |
| ACLNN_ERR_INNER |   561000     |    内部异常：API发生内部异常。     |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR     |  561001      |  内部异常：API内部进行输出shape推导发生错误。       |
| ACLNN_ERR_INNER_TILING_ERROR     |   561002     |   内部异常：API内部做npu kernel的tiling时发生异常。     |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR   |   561003  |  内部异常：API内部做查找npu kernel异常（可能因为算子二进制包未安装）。    |
| ACLNN_ERR_INNER_CREATE_EXECUTOR     |    561101    |   内部异常：API内部创建aclOpExecutor失败（可能因为操作系统异常）。      |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR     |  561102      |   内部异常：API内部未调用uniqueExecutor ReleaseTo。      |
| ACLNN_ERR_INNER_NULLPTR            |   561103     |    内部异常：aclnn API内部发生异常，出现了nullptr的异常。     |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE     |   561104     |   内部异常：aclnn API内部发生异常，算子的属性个数异常。      |
| ACLNN_ERR_INNER_KEY_CONFILICT（废弃）     |   561105     |    **已废弃，请使用最新ACLNN_ERR_INNER_KEY_CONFLICT。**     |
| ACLNN_ERR_INNER_KEY_CONFLICT |   561105     |   内部异常：aclnn API内部发生异常，算子的kernel匹配的hash key发生冲突。 |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE     |   561106     |   内部异常：aclnn API内部发生异常，算子的实现模式参数错误。    |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND     |  561107     |  内部异常：aclnn API内部发生异常，没有检测到需要配置的环境变量ASCEND_OPP_PATH。       |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED     |  561108      |   内部异常：aclnn API内部发生异常，加载算子kernel库中算子信息json文件失败。      |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND     |   561109      |   内部异常：aclnn API内部发生异常，加载算子kernel库中算子信息json文件的某个字段失败。      |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID     |  561110     |     内部异常：aclnn API内部发生异常，算子kernel库中算子信息json文件的format填写为非法值。    |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID     |    561111      |     内部异常：aclnn API内部发生异常，算子kernel库中算子信息json文件的dtype填写为非法值。    |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND     |   561112    |    内部异常：aclnn API内部发生异常，没有加载到算子的二进制kernel库。     |
| ACLNN_ERR_INNER_OP_FILE_INVALID     |  561113     |   内部异常：aclnn API内部发生异常，加载算子json文件字段时，发生异常。   |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND     |  561114      |  内部异常：aclnn API内部发生异常，算子的属性个数与算子信息json中不一致，超过了json中指定的attr个数。       |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH     |   561115      |   内部异常：aclnn API内部发生异常，算子的属性个数与算子信息json中不一致，少于json中指定的attr个数。      |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE     |   561116     |   内部异常：aclnn API内部发生异常，算子的输入个数超出32的限制。      |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL     |   561117     |  内部异常：aclnn API内部发生异常，算子信息json文件信息描述有缺失。       |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID     |  561118     |    内部异常：aclnn API内部发生异常，解析静态二进制json文件中的workspace信息时，发生异常。     |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID     |  561119      |    内部异常：aclnn API内部发生异常，解析静态二进制json文件中的核数使用信息时，发生异常。  |
