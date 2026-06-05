#!/bin/bash
# NLP-RAG-11 测试运行脚本
# 在 llamafactory conda 环境下运行所有单元测试

set -e

PROJECT_DIR="/mnt/d/Desktop/NLP-RAG-11"
TEST_DIR="$PROJECT_DIR/homework/test"
PYTHON="/home/ztt/miniconda3/envs/llamafactory/bin/python"

cd "$PROJECT_DIR"

echo ""
echo "================================================================"
echo "  NLP-RAG-11 测试套件"
echo "  运行环境: llamafactory"
echo "  开始时间: $(date)"
echo "================================================================"
echo ""

TESTS=(
    "test_data_generation"
    "test_model_loading"
    "test_evaluation"
    "test_loss_functions"
    "test_training"
)

PASSED=0
FAILED=0
TOTAL=${#TESTS[@]}

for TEST in "${TESTS[@]}"; do
    SCRIPT="$TEST_DIR/${TEST}.py"
    if [ ! -f "$SCRIPT" ]; then
        echo "[SKIP] $SCRIPT 不存在"
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  运行: $TEST"
    echo "────────────────────────────────────────────────────────────────"

    if $PYTHON "$SCRIPT"; then
        echo "[PASS] $TEST"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $TEST"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "================================================================"
echo "  测试完成"
echo "  时间: $(date)"
echo "  结果: $PASSED/$TOTAL 通过, $FAILED 失败"
echo "================================================================"

# 生成测试报告
REPORT_DIR="$TEST_DIR/report"
mkdir -p "$REPORT_DIR"

cat > "$REPORT_DIR/test_report_$(date +%Y%m%d_%H%M%S).json" << JSONEOF
{
  "project": "NLP-RAG-11",
  "test_date": "$(date -Iseconds)",
  "environment": "llamafactory (torch 2.8.0+cu128)",
  "total": $TOTAL,
  "passed": $PASSED,
  "failed": $FAILED,
  "status": "$([ $FAILED -eq 0 ] && echo 'PASS' || echo 'FAIL')",
  "modules": [
$(
    FIRST=true
    for TEST in "${TESTS[@]}"; do
        $FIRST || echo ","
        FIRST=false
        echo -n "    {\"name\": \"$TEST\", \"result\": \"pending\"}"
    done
)
  ]
}
JSONEOF

echo ""
echo "报告已保存至: $REPORT_DIR"

exit $FAILED
