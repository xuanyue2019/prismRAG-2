#!/usr/bin/env python3
"""
Agent Lightning 服务器启动脚本
支持多种配置方式和环境检测
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在")
        return None
    except yaml.YAMLError as e:
        print(f"配置文件格式错误: {e}")
        return None

def check_environment():
    """检查环境依赖"""
    print("检查环境依赖...")
    
    # 检查 Python 版本
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"警告: Python 版本 {python_version.major}.{python_version.minor} 低于推荐版本 3.10")
    
    # 检查必要依赖
    required_packages = ['torch', 'vllm', 'verl', 'agentlightning']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少必要依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    print("环境检查通过")
    return True

def setup_directories(config):
    """创建必要的目录"""
    base_dir = Path.cwd()
    directories = [
        base_dir / "checkpoints",
        base_dir / "logs", 
        base_dir / "data",
        base_dir / "profiles"
    ]
    
    # 从配置中读取额外的目录
    if config and 'training' in config and 'checkpoint_dir' in config['training']:
        directories.append(base_dir / config['training']['checkpoint_dir'])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {directory}")

def start_server(config_path, host=None, port=None, log_level=None):
    """启动训练服务器"""
    
    # 加载配置
    config = load_config(config_path) if config_path else {}
    
    # 设置命令行参数
    cmd = [sys.executable, "-m", "agentlightning.server"]
    
    if config_path:
        cmd.extend(["--config", str(config_path)])
    
    if host:
        cmd.extend(["--host", host])
    elif config and 'server' in config and 'host' in config['server']:
        cmd.extend(["--host", config['server']['host']])
    else:
        cmd.extend(["--host", "0.0.0.0"])
    
    if port:
        cmd.extend(["--port", str(port)])
    elif config and 'server' in config and 'port' in config['server']:
        cmd.extend(["--port", str(config['server']['port'])])
    else:
        cmd.extend(["--port", "8000"])
    
    if log_level:
        cmd.extend(["--log-level", log_level])
    elif config and 'server' in config and 'log_level' in config['server']:
        cmd.extend(["--log-level", config['server']['log_level']])
    else:
        cmd.extend(["--log-level", "info"])
    
    print(f"启动命令: {' '.join(cmd)}")
    
    try:
        # 启动服务器
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n收到中断信号，停止服务器...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"启动服务器失败: {e}")
        return 1

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动 Agent Lightning 训练服务器')
    parser.add_argument('--config', '-c', default='deployment/server_config.yaml',
                       help='配置文件路径 (默认: deployment/server_config.yaml)')
    parser.add_argument('--host', help='服务器主机地址')
    parser.add_argument('--port', '-p', type=int, help='服务器端口')
    parser.add_argument('--log-level', '-l', choices=['debug', 'info', 'warning', 'error'],
                       help='日志级别')
    parser.add_argument('--check-env', action='store_true',
                       help='只检查环境，不启动服务器')
    parser.add_argument('--setup-dirs', action='store_true',
                       help='只创建目录，不启动服务器')
    
    args = parser.parse_args()
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    if args.check_env:
        print("环境检查完成")
        sys.exit(0)
    
    # 加载配置创建目录
    config = load_config(args.config)
    setup_directories(config)
    
    if args.setup_dirs:
        print("目录创建完成")
        sys.exit(0)
    
    # 启动服务器
    print("启动 Agent Lightning 服务器...")
    return_code = start_server(
        args.config, 
        args.host, 
        args.port, 
        args.log_level
    )
    
    sys.exit(return_code)

if __name__ == "__main__":
    main()