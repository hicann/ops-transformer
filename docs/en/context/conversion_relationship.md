# conversion relationship

When the output data type of aclTensor of an API (such as **aclnnAdd** or **aclnnMul**) is inconsistent with the deduced result of the input data type, the API internally converts the computation result to the desired data type.

Data type conversion must meet the following rules; otherwise, parameter verification fails when APIs are called.

- Floating-point types: ACL\_FLOAT16, ACL\_FLOAT, ACL\_DOUBLE, and ACL\_BF16
- Integer types: ACL\_INT8, ACL\_UINT8, ACL\_INT16, ACL\_UINT16, ACL\_INT32, ACL\_UINT32, ACL\_INT64, and ACL\_UINT64
- Complex number types: ACL\_COMPLEX64 and ACL\_COMPLEX128
- Integer types can be converted to each other, and can also be converted to floating-point and complex number types.
- Floating-point types can be converted to each other, and can also be converted to complex number types.
- Complex number types can be converted to each other.
- BOOL can be converted to integer, floating-point, and complex number types.

Conversion is supported only for the scenarios mentioned above.
