add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    # The public TensorList API is handwritten; do not generate a duplicate aclnn API.
    target_sources(op_host_aclnnExc PRIVATE group_memcpy_def.cpp)
endif()

add_ops_compile_options(OP_NAME GroupMemcpy
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE group_memcpy ACLNNTYPE aclnn_exclude)
endif()
