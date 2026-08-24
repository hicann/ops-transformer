# ============================================================
# CMakeLists.txt — apace QuantMatmulReduceScatter ubmem ST
# ============================================================
# Direct kernel launch (<<<>>>) with HCCL context obtained on
# the host side via HcclAllocComResourceByTiling (driven by
# Mc2CcTilingConfig) and passed as the first kernel parameter.
# main.cpp is compiled in kernel mode (-xasc).

set(APACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(OP_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
set(BLAZE_COMMON "${OPS_TENSOR_INCLUDE_DIR}")

set(KERNEL_INCLUDES
    ${OP_KERNEL_ROOT}
    ${APACE_ROOT}
    ${APACE_ROOT}/kernel/matmul/quant_batch_matmul
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${BLAZE_COMMON}
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/include
)

set(HOST_INCLUDES
    ${APACE_ROOT}
    ${APACE_ROOT}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
)

set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-D_GLIBCXX_USE_CXX11_ABI=0" "-w" "-O3")
set(HOST_OPTS "-std=c++17" "-w" "-O2")

set(COMMON_LIBS dl platform tiling_api register ascendcl runtime hccl stdc++ pthread)

add_executable(apace_qmm_rs_ubmem_st
    src/main.cpp
)

set_source_files_properties(src/main.cpp PROPERTIES
    COMPILE_OPTIONS "${KERNEL_OPTS}"
    INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}"
)

target_link_directories(apace_qmm_rs_ubmem_st PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_DIR}/lib64
)

target_link_libraries(apace_qmm_rs_ubmem_st PRIVATE ${COMMON_LIBS} apace_blaze_api)
target_link_libraries(apace_qmm_rs_ubmem_st PRIVATE "-Wl,--no-as-needed" hccl_fwk "-Wl,--as-needed")
