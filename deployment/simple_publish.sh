#!/bin/bash

# PrismRAG-2 GitHub发布简化脚本

set -e

echo "🚀 开始发布 PrismRAG-2 到 GitHub..."

# 检查必要命令
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "错误: $1 未安装"
        exit 1
    fi
}

check_command git
check_command curl

# 备份原有git信息
if [ -d ".git" ]; then
    echo "📦 备份原有git配置..."
    mv .git .git_backup
fi

# 初始化新仓库
echo "🔄 初始化新的git仓库..."
git init
git checkout -b main

# 更新项目名称
echo "✏️ 更新项目名称..."
sed -i '' 's/PrismRAG/PrismRAG-2/g' README.md
sed -i '' 's/prismrag/prismrag-2/g' pyproject.toml 2>/dev/null || true
sed -i '' 's/prismrag/prismrag-2/g' setup.py 2>/dev/null || true

# 创建新的README
echo "📝 创建新的README..."
cat > README.md << 'EOF'
# PrismRAG-2

基于PrismRAG论文的完整实现，提供高质量的RAG数据生成和训练管道。

## 特性

- 📚 真实数据获取（Wikipedia + Web搜索）
- 🎯 智能QA对生成
- 🎭 干扰项抵抗生成
- 🧠 策略化推理链生成
- ✅ 多层次质量评估
- 🚀 生产就绪部署

## 快速开始

```bash
# 安装
pip install poetry
poetry install

# 运行
python -m src.main --config config/default.yaml
```

## 文档

- [配置指南](docs/CONFIGURATION_GUIDE.md)
- [API文档](docs/API_DOCUMENTATION.md)
- [部署指南](deployment/README.md)

## 许可证

MIT License
EOF

# 添加所有文件
echo "📁 添加文件到git..."
git add .

# 提交初始版本
echo "💾 提交初始版本..."
git commit -m "feat: Initial release of PrismRAG-2

Complete implementation including:
- Real data acquisition pipelines
- Advanced data generation with distractor resilience
- Strategic Chain-of-Thought reasoning
- Comprehensive quality assessment system
- Production-ready deployment configuration
- Multi-benchmark evaluation framework"

# 创建标签
git tag -a v1.0.0 -m "Initial release"

echo "✅ 本地准备完成！"
echo ""
echo "下一步操作:"
echo "1. 在 GitHub 上创建新仓库: prismRAG-2"
echo "2. 添加远程仓库: git remote add origin https://github.com/your-username/prismRAG-2.git"
echo "3. 推送代码: git push -u origin main"
echo "4. 推送标签: git push --tags"
echo ""
echo "或者运行自动创建脚本:"
echo "bash deployment/publish_to_github.sh"