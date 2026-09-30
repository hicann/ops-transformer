# Data Formats

A data format is used to describe the service semantics of the axes of a multi-dimensional tensor, indicating the physical layout format of data, such as 1D, 2D, 3D, 4D, and 5D. Generally, the specific formats need to be described in the convolutional neural network (CNN) APIs.

For details about all data formats supported by aclTensor, see "Data Types and Operations > aclFormat" in [AscendCL APIs (C)](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/appdevgapi/aclcppdevg_03_0004.html).

For details about the data format layout principles, see "Concepts, Terms, and Glossary > Neural Networks and Operators > Data Layout Formats" in [Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html).

## Usage Description

Currently, most operator APIs support the ND data format. For example, **aclnnAdd** supports the ND format (that is, N-dimensional tensor, with contiguous layout in low dimensions first). For **aclnnConvolution**, which is a CNN API, the input aclTensor must be set to a format with service semantics, instead of the ND format. Operators of this type can perform computation only when the service semantics of the tensor are known. For example, in 2D convolution, the mapping between NCHW and tensor dimensions should be known.

>**Note:**
>
>- In a two-phase API, **ACL\_FORMAT\__XXXX_** is abbreviated as **_XXXX_** for simplified description.
>- Dimensions in a data format are described as follows: N indicates the batch size, H indicates the height of the feature map, W indicates the width of the feature map, C indicates the channels of the feature map, D indicates the depth of the feature map, and L indicates the length of the feature map.

## Common Data Formats

When creating an aclTensor by calling the **aclCreateTensor** API, you need to set the data format based on the API service requirements. Currently, the supported data formats are as follows:

ACL\_FORMAT\_ND, ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, ACL\_FORMAT\_NDHWC, ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NC, ACL\_FORMAT\_NCL

For a non-ND tensor, the dimension requirements of the tensor are the same as those of format. See the following examples:

- 5D tensor: Must be ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NDHWC, or ACL\_FORMAT\_ND. (If the API parameter description does not specify that the ND format is supported, an error will be reported during API verification if ND is set.)
- 4D tensor: Must be ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, or ACL\_FORMAT\_ND.
- 3D tensor: Must be ACL\_FORMAT\_NCL or ACL\_FORMAT\_ND.
- 2D tensor: Must be ACL\_FORMAT\_NC or ACL\_FORMAT\_ND.
- Other tensors: Must be ACL\_FORMAT\_ND.

## Private Data Formats

Besides the previously mentioned common data formats, additional formats are available, such as ACL\_FORMAT\_NC1HWC0, ACL\_FORMAT\_FRACTAL\_Z, ACL\_FORMAT\_NC1HWC0\_C04, ACL\_FORMAT\_FRACTAL\_NZ, ACL\_FORMAT\_NDC1HWC0, and ACL\_FORMAT\_FRACTAL\_Z\_3D.

These are private formats of the NPU, and they are now not supported by most aclnn APIs. If an API declares its supported data formats, follow the actual description of the API.
