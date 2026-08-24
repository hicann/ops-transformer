if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_ep_combine_epilogue_def.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE moe_ep_combine_epilogue moe_ep_combine_epilogue ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(moe_ep_combine_epilogue_depends mc2/common mc2/3rd mc2/moe_ep_combine PARENT_SCOPE)

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME MoeEpCombineEpilogue
            OPTIONS --cce-auto-sync=off
                    -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    elseif("${CONDITION_UNIT}" STREQUAL "ascend910_93")
        add_ops_compile_options(OP_NAME MoeEpCombineEpilogue OPTIONS --cce-auto-sync=off)
    elseif("${CONDITION_UNIT}" STREQUAL "ascend910b")
        add_ops_compile_options(OP_NAME MoeEpCombineEpilogue OPTIONS --cce-auto-sync=off)
    endif()
else()
    add_mc2_modules_sources(OPTYPE moe_ep_combine_epilogue moe_ep_combine_epilogue ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
