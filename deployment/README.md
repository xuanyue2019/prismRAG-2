# Agent Lightning 部署工具集

本目录包含 Agent Lightning 项目的完整部署工具和配置。

## 📁 文件结构

```
deployment/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖列表
├── server_config.yaml           # 服务器配置模板
├── agent_config.yaml            # 智能体配置模板
├── docker-compose.yml           # Docker Compose 配置
├── Dockerfile.server            # 服务器 Dockerfile
├── Dockerfile.agent             # 智能体 Dockerfile
├── entrypoint.sh                # 容器入口点脚本
├── .env.example                 # 环境变量模板
├── start_server.py              # 服务器启动脚本
├── health_check.py              # 健康检查工具
└── backup_script.py             # 备份管理工具
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv agent-env
source agent-env/bin/activate

# 安装核心依赖
pip install -r deployment/requirements.txt
```

### 2. 配置环境

```bash
# 复制环境变量模板
cp deployment/.env.example .env

# 编辑 .env 文件，填写实际的 API 密钥
# OPENAI_API_KEY=your_actual_key
# AGENTOPS_API_KEY=your_actual_key
```

### 3. 启动服务器

```bash
# 使用启动脚本（推荐）
python deployment/start_server.py --config deployment/server_config.yaml

# 或直接使用命令行
agentlightning server start --config deployment/server_config.yaml
```

### 4. 健康检查

```bash
# 检查服务器状态
python deployment/health_check.py --server-url http://localhost:8000

# 周期性检查
python deployment/health_check.py --interval 60 --format json
```

## 🐳 容器化部署

### 构建镜像

```bash
# 构建服务器镜像
docker build -f deployment/Dockerfile.server -t agent-lightning-server:latest .

# 构建智能体镜像  
docker build -f deployment/Dockerfile.agent -t agent-lightning-agent:latest .
```

### 使用 Docker Compose

```bash
# 启动所有服务
docker-compose -f deployment/docker-compose.yml up -d

# 查看日志
docker-compose -f deployment/docker-compose.yml logs -f

# 停止服务
docker-compose -f deployment/docker-compose.yml down
```

## ⚙️ 配置说明

### 服务器配置 (server_config.yaml)

主要配置项：
- **server**: 网络和连接设置
- **model**: 模型选择和提供商配置
- **training**: 训练参数和优化器
- **monitoring**: 监控和日志配置
- **security**: 安全相关设置

### 智能体配置 (agent_config.yaml)

主要配置项：
- **agent**: 基本信息和框架选择
- **tools**: 工具配置和权限
- **training**: 训练相关参数
- **error_handling**: 错误处理策略

## 🛠️ 工具使用

### 启动脚本 (start_server.py)

```bash
# 基本用法
python deployment/start_server.py --config deployment/server_config.yaml

# 自定义参数
python deployment/start_server.py --host 0.0.0.0 --port 8080 --log-level debug

# 只检查环境
python deployment/start_server.py --check-env

# 只创建目录
python deployment/start_server.py --setup-dirs
```

### 健康检查 (health_check.py)

```bash
# 检查服务器健康
python deployment/health_check.py --server-url http://localhost:8000

# 检查智能体健康
python deployment/health_check.py --agent-url http://agent-host:3000

# JSON 格式输出
python deployment/health_check.py --format json

# 周期性检查
python deployment/health_check.py --interval 30
```

### 备份管理 (backup_script.py)

```bash
# 创建完整快照
python deployment/backup_script.py --snapshot

# 只备份配置
python deployment/backup_script.py --configs

# 只备份检查点
python deployment/backup_script.py --checkpoints

# 列出所有备份
python deployment/backup_script.py --list

# 清理旧备份
python deployment/backup_script.py --cleanup 10      # 保留10个最新备份
python deployment/backup_script.py --cleanup-days 7  # 删除7天前的备份
```

## 🔧 故障排除

### 常见问题

1. **依赖冲突**
   ```bash
   # 按照推荐顺序安装
   pip install torch first
   pip install flash-attn --no-build-isolation
   pip install vllm
   pip install verl
   ```

2. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep :8000
   
   # 或使用其他端口
   python deployment/start_server.py --port 8080
   ```

3. **GPU 内存不足**
   - 减少批量大小
   - 使用梯度累积
   - 启用混合精度训练

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 使用调试模式启动
python -m pdb deployment/start_server.py --config deployment/server_config.yaml
```

## 📊 监控指标

健康检查工具监控的指标：
- ✅ 服务器健康状态 (HTTP 200)
- ⏱️ 响应时间 (< 1s)
- 💻 CPU 使用率 (< 80%)
- � 内存使用率 (< 85%)
- � GPU 内存使用率 (< 90%)
- 📊 磁盘使用率 (< 90%)

## 🔒 安全建议

1. **网络安全**
   - 使用内网或 VPN
   - 配置防火墙规则
   - 启用 TLS 加密

2. **数据安全**
   - 加密敏感配置
   - 定期轮换 API 密钥
   - 实施访问控制

3. **备份策略**
   - 每日自动备份配置
   - 每周完整快照
   - 异地备份重要数据

## 📝 部署检查清单

- [ ] 环境变量配置完成 (.env)
- [ ] 依赖安装成功
- [ ] 服务器启动正常
- [ ] 健康检查通过
- [ ] 监控配置生效
- [ ] 备份机制测试

## 🆘 支持资源

- [官方文档](https://github.com/microsoft/agent-lightning)
- [问题追踪](https://github.com/microsoft/agent-lightning/issues)
- [Discord 社区](https://discord.gg/RYk7CdvDR7)

## 🔄 更新日志

### v1.0.0 (2025-08-19)
- 初始版本发布
- 完整的部署工具集
- 容器化支持
- 健康检查和备份工具

---

**维护团队**: 部署运维组  
**最后更新**: 2025-08-19