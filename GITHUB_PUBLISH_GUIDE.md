# PrismRAG-2 GitHub发布指南

## 概述

本文档指导如何将PrismRAG-2项目发布到GitHub。我们提供了一个自动化脚本 `publish_prismrag2.sh` 来处理整个发布流程。

## 前置要求

1. **GitHub账户**: 确保您有一个GitHub账户
2. **Git安装**: 系统已安装Git
3. **仓库权限**: 有权限创建新的GitHub仓库

## 快速开始

### 方法一：使用自动化脚本（推荐）

```bash
# 1. 确保脚本有执行权限
chmod +x publish_prismrag2.sh

# 2. 运行发布脚本
./publish_prismrag2.sh

# 或者设置GitHub用户名环境变量
export GITHUB_USERNAME=your_github_username
./publish_prismrag2.sh
```

### 方法二：手动步骤

如果您想手动操作，以下是详细步骤：

```bash
# 1. 备份原有git配置（如果有）
if [ -d ".git" ]; then
    mv .git .git_backup
fi

# 2. 初始化新仓库
git init
git checkout -b main

# 3. 更新项目名称
sed -i '' 's/PrismRAG/PrismRAG-2/g' README.md
sed -i '' 's/prismrag/prismrag-2/g' pyproject.toml

# 4. 添加所有文件
git add .

# 5. 提交初始版本
git commit -m "feat: Initial release of PrismRAG-2"

# 6. 创建标签
git tag -a v1.0.0 -m "Initial release"

# 7. 添加远程仓库
git remote add origin https://github.com/your_username/prismRAG-2.git

# 8. 推送到GitHub
git push -u origin main --force
git push --tags --force
```

## 脚本功能详解

### 自动化脚本执行的操作

1. **环境检查**
   - 验证Git和Python是否安装
   - 获取GitHub用户名

2. **Git配置管理**
   - 备份原有git配置（如果存在）
   - 初始化新的git仓库

3. **项目配置更新**
   - 更新README.md中的项目名称
   - 更新pyproject.toml/setup.py中的包名
   - 创建项目概述文档

4. **GitHub Actions配置**
   - 更新CI/CD工作流配置

5. **版本控制**
   - 添加所有文件到git
   - 提交初始版本
   - 创建v1.0.0标签

6. **发布到GitHub**
   - 设置远程仓库
   - 强制推送到GitHub

### 脚本输出示例

```
🚀 PrismRAG-2 GitHub发布流程开始...
==========================================
[INFO] 备份原有git配置...
[SUCCESS] Git配置已备份到 .git_backup/
[INFO] 初始化新的git仓库...
[SUCCESS] Git仓库初始化完成
[INFO] 更新项目配置为PrismRAG-2...
[SUCCESS] 项目配置更新完成
...
🎉 PrismRAG-2 发布完成!
```

## 发布后操作

### 1. 验证发布成功

访问您的GitHub仓库页面：
```
https://github.com/your_username/prismRAG-2
```

检查以下内容：
- ✅ 所有文件都已上传
- ✅ README.md显示正确
- ✅ 版本标签v1.0.0存在
- ✅ 初始提交信息正确

### 2. 配置GitHub Secrets

在仓库设置中配置必要的环境变量：

| Secret名称 | 描述 | 示例值 |
|-----------|------|--------|
| `WIKIPEDIA_API_KEY` | Wikipedia API密钥 | your_wikipedia_key |
| `OPENAI_API_KEY` | OpenAI API密钥 | sk-... |
| `ANTHROPIC_API_KEY` | Anthropic API密钥 | sk-ant-... |
| `SERPER_API_KEY` | Serper API密钥 | your_serper_key |

### 3. 验证CI/CD工作流

1. 在GitHub仓库中转到"Actions"标签页
2. 确认CI工作流自动运行
3. 检查测试是否通过

### 4. 设置协作和权限

- 邀请协作者
- 设置分支保护规则
- 配置代码审查要求

## 故障排除

### 常见问题

1. **权限错误**
   ```bash
   # 确保有创建仓库的权限
   # 检查GitHub个人访问令牌权限
   ```

2. **网络连接问题**
   ```bash
   # 检查网络连接
   ping github.com
   ```

3. **Git配置问题**
   ```bash
   # 检查Git配置
   git config --list
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

4. **脚本执行权限**
   ```bash
   # 如果脚本无法执行
   chmod +x publish_prismrag2.sh
   ```

### 恢复备份

如果发布过程中出现问题，可以恢复原有配置：

```bash
# 恢复git配置
if [ -d ".git_backup" ]; then
    rm -rf .git
    mv .git_backup .git
fi

# 恢复项目名称（如果需要）
git checkout HEAD -- README.md pyproject.toml setup.py
```

## 高级配置

### 环境变量配置

您可以通过环境变量自定义发布过程：

```bash
# 设置GitHub用户名
export GITHUB_USERNAME="your_username"

# 设置自定义版本号
export RELEASE_VERSION="2.0.0"

# 设置自定义分支名
export DEFAULT_BRANCH="main"
```

### 自定义发布选项

编辑 `publish_prismrag2.sh` 脚本中的以下变量：

```bash
# 项目名称
PROJECT_NAME="prismRAG-2"

# 版本号
VERSION="1.0.0"

# 默认分支
DEFAULT_BRANCH="main"

# 远程仓库URL模板
REMOTE_URL_TEMPLATE="https://github.com/%s/prismRAG-2.git"
```

## 支持的功能

✅ 自动化GitHub仓库创建和配置  
✅ 项目名称和配置更新  
✅ 版本标签管理  
✅ CI/CD工作流配置  
✅ 详细的错误处理和日志  
✅ 备份和恢复机制  
✅ 环境变量支持  
✅ 颜色化的输出显示  

## 联系方式

如果在发布过程中遇到问题，请：

1. 检查本指南的故障排除部分
2. 查看脚本输出的错误信息
3. 在GitHub Issues中创建问题报告

---

**注意**: 发布脚本会强制推送代码到GitHub，请确保您了解此操作的影响。建议在发布前备份重要数据。