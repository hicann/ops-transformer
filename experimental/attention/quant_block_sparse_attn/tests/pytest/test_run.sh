#!/bin/bash

# 脚本路径
BSA_PT_SAVE_SCRIPT="./batch/test_quant_block_sparse_attn_pt_save.py"
TEST_BSA_PT_BATCH_SCRIPT="test_quant_block_sparse_attn_batch.py"
TEST_BSA_SINGLE_SCRIPT="test_quant_block_sparse_attn_single.py"
TEST_BSA_GRAPH_SCRIPT="test_quant_block_sparse_attn_batch_graph.py"

# ====================== 执行区======================

# 单用例算子调测
run_single() {
    echo "===== 执行单用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_BSA_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 用例批量生成调试
run_batch() {
    echo "===== 执行用例批量生成测试 ====="

    echo -e "\n===== 第一步：执行test_quant_block_sparse_attn_pt_save.py ====="
    python3 -m pytest -rA -s $BSA_PT_SAVE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "test_quant_block_sparse_attn_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest命令 ====="
    python3 -m pytest -rA -s $TEST_BSA_PT_BATCH_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 图模式（aclgraph reduce-overhead）批量测试
run_graph() {
    echo "===== 执行图模式（aclgraph）批量测试 ====="

    echo -e "\n===== 第一步：执行test_quant_block_sparse_attn_pt_save.py ====="
    python3 -m pytest -rA -s $BSA_PT_SAVE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "test_quant_block_sparse_attn_pt_save.py 执行失败，退出"
        exit 1
    fi

    echo -e "\n===== 第二步：执行pytest aclgraph命令 ====="
    python3 -m pytest -rA -s $TEST_BSA_GRAPH_SCRIPT -v -m graph -W ignore::UserWarning -W ignore::DeprecationWarning
    if [ $? -ne 0 ]; then
        echo "pytest执行失败"
        exit 1
    fi

    echo -e "\n=====执行完成！====="
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  batch     执行用例批量生成调试"
    echo "  graph     执行图模式（aclgraph）批量测试"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single  # 执行single模式"
    echo "  $0 batch   # 执行batch模式"
    echo "  $0 graph   # 执行graph模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/batch/graph/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    batch)
        run_batch
        ;;
    graph)
        run_graph
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/batch/graph/help"
        show_help
        exit 1
        ;;
esac

exit 0