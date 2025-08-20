#!/bin/bash

# PrismRAG-2 发布前检查脚本
# 验证项目状态，确保可以成功发布到GitHub

set -e

echo "🔍 PrismRAG-2 发布前检查开始..."
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查结果计数器
PASS=0
FAIL=0
WARN=0

# 函数：打印检查结果
log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASS++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAIL++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARN++))
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 检查文件是否存在
check_file_exists() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        log_pass "$description: $file"
        return 0
    else
        log_fail "$description: $file (文件不存在)"
        return 1
    fi
}

# 检查目录是否存在
check_dir_exists() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        log_pass "$description: $dir"
        return 0
    else
        log_fail "$description: $dir (目录不存在)"
        return 1
    fi
}

# 检查文件内容包含特定文本
check_file_contains() {
    local file=$1
    local pattern=$2
    local description=$3
    
    if [ -f "$file" ] && grep -q "$pattern" "$file"; then
        log_pass "$description: 找到 '$pattern'"
        return 0
    else
        log_fail "$description: 未找到 '$pattern'"
        return 1
    fi
}

# 检查Python包配置
check_python_config() {
    log_info "检查Python包配置..."
    
    # 检查setup.py或pyproject.toml
    if [ -f "pyproject.toml" ]; then
        check_file_contains "pyproject.toml" "name.*=" "Python包名称配置"
        check_file_contains "pyproject.toml" "version.*=" "Python包版本配置"
    elif [ -f "setup.py" ]; then
        check_file_contains "setup.py" "name.*=" "Python包名称配置"
        check_file_contains "setup.py" "version.*=" "Python包版本配置"
    else
        log_warn "未找到Python包配置文件 (pyproject.toml 或 setup.py)"
    fi
}

# 检查依赖文件
check_dependencies() {
    log_info "检查依赖文件..."
    
    check_file_exists "requirements.txt" "Python依赖文件"
    
    # 检查requirements.txt内容
    if [ -f "requirements.txt" ]; then
        if grep -q "torch" requirements.txt; then
            log_pass "requirements.txt 包含PyTorch"
        else
            log_warn "requirements.txt 未包含PyTorch"
        fi
        
        if grep -q "transformers" requirements.txt; then
            log_pass "requirements.txt 包含transformers"
        else
            log_warn "requirements.txt 未包含transformers"
        fi
    fi
}

# 检查文档文件
check_documentation() {
    log_info "检查文档文件..."
    
    check_file_exists "README.md" "项目说明文档"
    check_file_exists "LICENSE" "许可证文件"
    check_file_exists "CONTRIBUTING.md" "贡献指南"
    check_file_exists "CHANGELOG.md" "更新日志"
    
    # 检查docs目录
    check_dir_exists "docs/" "文档目录"
    if [ -d "docs/" ]; then
        check_file_exists "docs/CONFIGURATION_GUIDE.md" "配置指南"
        check_file_exists "docs/API_DOCUMENTATION.md" "API文档"
    fi
}

# 检查源代码结构
check_source_code() {
    log_info "检查源代码结构..."
    
    check_dir_exists "src/" "源代码目录"
    check_dir_exists "src/data_acquisition/" "数据获取模块"
    check_dir_exists "src/data_generation/" "数据生成模块"
    check_dir_exists "src/training/" "训练模块"
    check_dir_exists "src/evaluation/" "评估模块"
    
    # 检查核心文件
    check_file_exists "src/__init__.py" "源代码包初始化"
    check_file_exists "src/data_acquisition/__init__.py" "数据获取模块初始化"
    check_file_exists "src/data_generation/__init__.py" "数据生成模块初始化"
}

# 检查配置文件
check_config_files() {
    log_info "检查配置文件..."
    
    check_dir_exists "config/" "配置目录"
    check_file_exists "config/default.yaml" "默认配置文件"
    
    # 检查生产配置
    check_dir_exists "deployment/" "部署目录"
    check_file_exists "deployment/Dockerfile.production" "生产Dockerfile"
    check_file_exists "deployment/docker-compose.production.yml" "生产Docker Compose配置"
    check_file_exists "deployment/production_config.yaml" "生产环境配置"
}

# 检查测试文件
check_test_files() {
    log_info "检查测试文件..."
    
    check_dir_exists "tests/" "测试目录"
    check_file_exists "tests/test_data_generation.py" "数据生成测试"
    check_file_exists "tests/test_evaluation.py" "评估测试"
}

# 检查CI/CD配置
check_ci_cd() {
    log_info "检查CI/CD配置..."
    
    check_dir_exists ".github/workflows/" "GitHub Actions目录"
    check_file_exists ".github/workflows/ci.yml" "CI工作流配置"
    
    if [ -f ".github/workflows/ci.yml" ]; then
        if grep -q "pytest" ".github/workflows/ci.yml"; then
            log_pass "CI配置包含测试运行"
        else
            log_warn "CI配置未包含测试运行"
        fi
    fi
}

# 检查Git配置
check_git_config() {
    log_info "检查Git配置..."
    
    # 检查.gitignore
    check_file_exists ".gitignore" "Git忽略文件"
    
    # 检查pre-commit配置
    if [ -f ".pre-commit-config.yaml" ]; then
        log_pass "pre-commit配置存在"
    else
        log_warn "pre-commit配置不存在"
    fi
}

# 主检查函数
main() {
    echo "=========================================="
    echo "        PrismRAG-2 发布前完整性检查"
    echo "=========================================="
    
    # 执行所有检查
    check_python_config
    check_dependencies
    check_documentation
    check_source_code
    check_config_files
    check_test_files
    check_ci_cd
    check_git_config
    
    # 显示检查结果
    echo ""
    echo "=========================================="
    echo "             检查结果汇总"
    echo "=========================================="
    echo -e "${GREEN}通过: $PASS${NC}"
    echo -e "${YELLOW}警告: $WARN${NC}"
    echo -e "${RED}失败: $FAIL${NC}"
    echo ""
    
    if [ $FAIL -eq 0 ]; then
        if [ $WARN -eq 0 ]; then
            echo -e "${GREEN}✅ 所有检查通过！项目可以发布。${NC}"
        else
            echo -e "${YELLOW}⚠️  检查完成，但有警告。建议修复警告后再发布。${NC}"
        fi
    else
        echo -e "${RED}❌ 检查失败！请修复上述问题后再发布。${NC}"
        exit 1
    fi
    
    echo ""
    echo "下一步操作:"
    echo "1. 运行发布脚本: ./publish_prismrag2.sh"
    echo "2. 或查看详细指南: cat GITHUB_PUBLISH_GUIDE.md"
    echo ""
}

# 执行主函数
main "$@"