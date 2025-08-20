# Microsoft Agent Lightning 部署指南

## 📖 概述

本文档提供 Microsoft Agent Lightning 项目的完整部署指南，包括环境准备、配置、部署步骤和故障排除。

## 🎯 部署目标

- [x] 单机开发环境部署
- [x] 多机生产环境部署  
- [x] 容器化部署
- [x] 监控和运维配置

## 📋 前置要求

### 硬件要求
- **CPU**: 8+ 核心
- **GPU**: NVIDIA GPU (推荐，用于 vLLM 推理)
- **内存**: 16GB+ RAM
- **存储**: 50GB+ 可用空间

### 软件要求
- **操作系统**: Linux/Windows/macOS
- **Python**: 3.10+
- **Docker**: 20.10+ (可选)
- **CUDA**: 11.8+ (如使用 GPU)

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv agent-lightning-env
source agent-lightning-env/bin/activate  # Linux/macOS
# 或
agent-lightning-env\Scripts\activate     # Windows

# 安装核心依赖
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install vllm==0.9.2
pip install verl==0.5.0

# 安装 Agent Lightning
pip install agentlightning
```

### 2. 配置环境变量

复制环境变量模板并配置：
```bash
cp deployment/.env.example .env
# 编辑 .env 文件，填写实际的 API 密钥和配置
```

### 3. 启动训练服务器

```bash
# 使用默认配置启动服务器
agentlightning server start --config deployment/server_config.yaml

# 或使用自定义配置
agentlightning server start --host 0.0.0.0 --port 8000 --log-level info
```

### 4. 启动智能体客户端

在新的终端中启动智能体：
```bash
# 激活虚拟环境
source agent-lightning-env/bin/activate

# 启动 AutoGen 智能体
python examples/autogen/agent.py --server-url http://localhost:8000

# 启动 LangChain 智能体  
python examples/langchain/agent.py --server-url http://localhost:8000
```

## 🐳 容器化部署

### 1. 构建 Docker 镜像

```bash
# 构建服务器镜像
docker build -f deployment/Dockerfile.server -t agent-lightning-server:latest .

# 构建智能体镜像
docker build -f deployment/Dockerfile.agent -t agent-lightning-agent:latest .
```

### 2. 使用 Docker Compose 部署

```bash
# 复制环境变量配置
cp deployment/.env.example .env
# 编辑 .env 文件填写实际值

# 启动所有服务
docker-compose -f deployment/docker-compose.yml up -d

# 查看日志
docker-compose -f deployment/docker-compose.yml logs -f

# 停止服务
docker-compose -f deployment/docker-compose.yml down
```

### 3. 验证部署

```bash
# 检查服务器健康状态
curl http://localhost:8000/health

# 检查智能体状态
docker ps | grep agent-lightning

# 查看监控仪表板 (如果启用)
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

## ☸️ Kubernetes 部署

### 1. 创建命名空间
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-lightning
```

### 2. 部署服务器
```yaml
# k8s/server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-lightning-server
  namespace: agent-lightning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agent-lightning-server
  template:
    metadata:
      labels:
        app: agent-lightning-server
    spec:
      containers:
      - name: server
        image: agent-lightning-server:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        envFrom:
        - secretRef:
            name: agent-lightning-secrets
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: "4"
          requests:
            memory: 4Gi
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: agent-lightning-server
  namespace: agent-lightning
spec:
  selector:
    app: agent-lightning-server
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
```

### 3. 部署智能体
```yaml
# k8s/agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-lightning-agent
  namespace: agent-lightning
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-lightning-agent
  template:
    metadata:
      labels:
        app: agent-lightning-agent
    spec:
      containers:
      - name: agent
        image: agent-lightning-agent:latest
        envFrom:
        - secretRef:
            name: agent-lightning-secrets
        env:
        - name: SERVER_URL
          value: "http://agent-lightning-server:8000"
        resources:
          limits:
            memory: 2Gi
            cpu: "1"
          requests:
            memory: 1Gi
            cpu: "0.5"
```

### 4. 创建配置 Secret
```bash
# 创建 Kubernetes Secret
kubectl create secret generic agent-lightning-secrets \
  --namespace=agent-lightning \
  --from-env-file=.env
```

### 5. 应用配置
```bash
# 应用所有配置
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/server-deployment.yaml
kubectl apply -f k8s/agent-deployment.yaml

# 检查部署状态
kubectl get pods -n agent-lightning
kubectl get svc -n agent-lightning
```

## 🔧 配置详解

### 服务器配置 (server_config.yaml)

主要配置项：
- **server**: 服务器网络和连接设置
- **model**: 模型选择和提供商配置
- **training**: 训练参数和优化器设置
- **monitoring**: 监控和日志配置
- **security**: 安全相关配置

### 智能体配置 (agent_config.yaml)

主要配置项：
- **agent**: 智能体基本信息和框架选择
- **tools**: 工具配置和权限设置
- **training**: 训练相关参数
- **monitoring**: 监控配置
- **error_handling**: 错误处理和重试策略

## 📊 监控和运维

### 1. 指标监控

启用 Prometheus 指标：
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent-lightning'
    static_configs:
      - targets: ['agent-lightning-server:9090']
```

### 2. 日志收集

配置日志轮转和收集：
```yaml
# 使用 ELK 或 Loki 进行日志收集
logging:
  level: INFO
  format: json
  file: /app/logs/server.log
```

### 3. 告警配置

设置关键指标告警：
- CPU 使用率 > 80%
- 内存使用率 > 85%
- 请求错误率 > 5%
- 响应时间 > 1000ms

## 🐛 故障排除

### 常见问题

1. **依赖冲突**
   ```bash
   # 按照推荐顺序安装依赖
   pip install torch first
   pip install flash-attn --no-build-isolation
   pip install vllm
   pip install verl
   ```

2. **GPU 内存不足**
   - 减少批量大小
   - 使用梯度累积
   - 启用混合精度训练

3. **连接问题**
   ```bash
   # 检查服务器状态
   curl http://localhost:8000/health
   
   # 检查网络连接
   ping server-host
   
   # 检查防火墙设置
   sudo ufw status
   ```

4. **训练失败**
   - 检查数据格式
   - 验证模型兼容性
   - 查看详细错误日志

### 调试技巧

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 使用调试模式启动
python -m pdb your_script.py

# 检查系统资源
top
nvidia-smi
df -h
```

## 🔄 维护和升级

### 日常维护
- 监控系统状态和资源使用
- 定期备份检查点和配置
- 清理日志和临时文件

### 版本升级
```bash
# 升级 Agent Lightning
pip install --upgrade agentlightning

# 升级依赖包
pip install --upgrade -r deployment/requirements.txt

# 验证升级后功能
python -m pytest tests/ -v
```

### 数据备份
```bash
# 备份检查点
tar -czf checkpoints-backup-$(date +%Y%m%d).tar.gz ./checkpoints

# 备份配置
tar -czf config-backup-$(date +%Y%m%d).tar.gz ./config

# 备份日志 (可选)
tar -czf logs-backup-$(date +%Y%m%d).tar.gz ./logs
```

## 🎯 性能优化

### 1. 硬件优化
- 使用 NVMe SSD 存储
- 增加 GPU 内存
- 优化网络带宽

### 2. 软件优化
- 启用模型量化
- 使用批处理推理
- 优化数据加载

### 3. 配置优化
```yaml
# 优化训练参数
training:
  batch_size: 64
  gradient_accumulation_steps: 2
  learning_rate: 2e-5

# 优化推理参数
model:
  vllm:
    gpu_memory_utilization: 0.95
    max_model_len: 8192
```

## 📝 部署检查清单

### 前置检查
- [ ] 硬件资源满足要求
- [ ] 软件环境准备完成
- [ ] API 密钥和权限配置
- [ ] 网络连接正常

### 部署检查
- [ ] 服务器启动成功
- [ ] 智能体连接正常
- [ ] 训练任务可执行
- [ ] 监控系统工作正常

### 验证检查
- [ ] 健康检查通过
- [ ] 性能指标正常
- [ ] 错误处理有效
- [ ] 备份机制可靠

## 📞 支持资源

- [官方文档](https://github.com/microsoft/agent-lightning)
- [Discord 社区](https://discord.gg/RYk7CdvDR7)
- [问题追踪](https://github.com/microsoft/agent-lightning/issues)
- [示例代码](https://github.com/microsoft/agent-lightning/tree/main/examples)

## 🔒 安全建议

1. **网络安全**
   - 使用 VPN 或私有网络
   - 配置防火墙规则
   - 启用 TLS 加密

2. **数据安全**
   - 加密敏感数据
   - 定期轮换密钥
   - 实施访问控制

3. **操作安全**
   - 使用最小权限原则
   - 启用审计日志
   - 定期安全扫描

---

**最后更新**: 2025-08-19  
**版本**: 1.0.0  
**维护团队**: 部署运维组