#!/usr/bin/env python3
"""
Agent Lightning 健康检查脚本
用于检查服务器和智能体的健康状态
"""

import requests
import time
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def check_server_health(self) -> Dict:
        """检查服务器健康状态"""
        endpoints = {
            'health': '/health',
            'ready': '/ready', 
            'metrics': '/metrics',
            'info': '/info'
        }
        
        results = {}
        for name, endpoint in endpoints.items():
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.get(url, timeout=self.timeout)
                
                results[name] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
                if name == 'health' and response.status_code == 200:
                    try:
                        results[name]['data'] = response.json()
                    except:
                        results[name]['data'] = response.text
                        
            except requests.exceptions.RequestException as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def check_agent_health(self, agent_url: Optional[str] = None) -> Dict:
        """检查智能体健康状态"""
        if not agent_url:
            return {'status': 'skipped', 'reason': 'No agent URL provided'}
            
        try:
            response = self.session.get(agent_url, timeout=self.timeout)
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_system_resources(self) -> Dict:
        """检查系统资源使用情况"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'cores': psutil.cpu_count(),
                    'cores_logical': psutil.cpu_count(logical=True)
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except ImportError:
            return {'status': 'skipped', 'reason': 'psutil not installed'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_gpu_resources(self) -> Dict:
        """检查 GPU 资源使用情况"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {'status': 'skipped', 'reason': 'CUDA not available'}
                
            gpu_count = torch.cuda.device_count()
            gpu_info = {}
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory
                
                gpu_info[f'gpu_{i}'] = {
                    'name': props.name,
                    'memory_total': memory_total,
                    'memory_allocated': memory_allocated,
                    'memory_reserved': memory_reserved,
                    'memory_allocated_percent': (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                    'memory_reserved_percent': (memory_reserved / memory_total) * 100 if memory_total > 0 else 0
                }
                
            return gpu_info
            
        except ImportError:
            return {'status': 'skipped', 'reason': 'PyTorch not installed'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def comprehensive_check(self, agent_url: Optional[str] = None) -> Dict:
        """执行全面的健康检查"""
        print("开始全面健康检查...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'server_url': self.base_url,
            'agent_url': agent_url,
            'server_health': self.check_server_health(),
            'system_resources': self.check_system_resources(),
            'gpu_resources': self.check_gpu_resources()
        }
        
        if agent_url:
            results['agent_health'] = self.check_agent_health(agent_url)
        
        return results
    
    def print_results(self, results: Dict, format: str = 'text'):
        """打印检查结果"""
        if format == 'json':
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return
            
        print(f"\n{'='*60}")
        print("Agent Lightning 健康检查报告")
        print(f"时间: {results['timestamp']}")
        print(f"服务器: {results['server_url']}")
        if results.get('agent_url'):
            print(f"智能体: {results['agent_url']}")
        print(f"{'='*60}")
        
        # 服务器健康状态
        print("\n📊 服务器健康状态:")
        for endpoint, status in results['server_health'].items():
            emoji = "✅" if status['status'] == 'healthy' else "❌" if status['status'] == 'unhealthy' else "⚠️"
            print(f"  {emoji} {endpoint}: {status['status']} "
                  f"(代码: {status.get('status_code', 'N/A')}, "
                  f"响应: {status.get('response_time', 'N/A'):.3f}s)")
        
        # 系统资源
        if 'system_resources' in results and results['system_resources']['status'] != 'skipped':
            sys = results['system_resources']
            print(f"\n💻 系统资源:")
            print(f"  CPU: {sys['cpu']['percent']}% (核心: {sys['cpu']['cores']})")
            print(f"  内存: {sys['memory']['percent']}% "
                  f"(可用: {sys['memory']['available'] // (1024**3)}GB/"
                  f"{sys['memory']['total'] // (1024**3)}GB)")
            print(f"  磁盘: {sys['disk']['percent']}% "
                  f"(可用: {sys['disk']['free'] // (1024**3)}GB/"
                  f"{sys['disk']['total'] // (1024**3)}GB)")
        
        # GPU 资源
        if 'gpu_resources' in results and isinstance(results['gpu_resources'], dict):
            gpus = results['gpu_resources']
            print(f"\n🎮 GPU 资源:")
            for gpu_id, gpu in gpus.items():
                if 'status' in gpu:  # 跳过错误信息
                    continue
                print(f"  {gpu_id}: {gpu['name']}")
                print(f"    内存使用: {gpu['memory_allocated_percent']:.1f}% "
                      f"(分配: {gpu['memory_allocated'] // (1024**3)}GB/"
                      f"{gpu['memory_total'] // (1024**3)}GB)")
        
        # 智能体健康
        if 'agent_health' in results:
            agent = results['agent_health']
            emoji = "✅" if agent['status'] == 'healthy' else "❌" if agent['status'] == 'unhealthy' else "⚠️"
            print(f"\n🤖 智能体健康: {emoji} {agent['status']}")
        
        print(f"\n{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Agent Lightning 健康检查工具')
    parser.add_argument('--server-url', '-s', default='http://localhost:8000',
                       help='服务器地址 (默认: http://localhost:8000)')
    parser.add_argument('--agent-url', '-a', help='智能体地址')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                       help='请求超时时间 (秒)')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                       help='输出格式')
    parser.add_argument('--interval', '-i', type=int, default=0,
                       help='检查间隔 (秒)，0表示只检查一次')
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.server_url, args.timeout)
    
    if args.interval > 0:
        print(f"开始周期性健康检查，间隔: {args.interval}秒")
        print("按 Ctrl+C 停止...")
        
        try:
            while True:
                results = checker.comprehensive_check(args.agent_url)
                checker.print_results(results, args.format)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n停止健康检查")
    else:
        results = checker.comprehensive_check(args.agent_url)
        checker.print_results(results, args.format)

if __name__ == "__main__":
    main()