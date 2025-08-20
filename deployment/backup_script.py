#!/usr/bin/env python3
"""
Agent Lightning 备份脚本
用于备份配置、检查点和日志文件
"""

import os
import sys
import argparse
import tarfile
import zipfile
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import shutil

class BackupManager:
    """备份管理器"""
    
    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup_name(self, prefix: str = "backup") -> str:
        """创建备份文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    def backup_directory(self, source_dir: str, backup_name: str, 
                        exclude_patterns: Optional[List[str]] = None) -> str:
        """备份整个目录"""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"源目录不存在: {source_dir}")
            
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        with tarfile.open(backup_path, "w:gz") as tar:
            for item in source_path.rglob('*'):
                # 跳过排除模式
                if exclude_patterns and any(item.match(pattern) for pattern in exclude_patterns):
                    continue
                    
                # 跳过备份目录本身
                if item.is_relative_to(self.backup_dir):
                    continue
                    
                arcname = item.relative_to(source_path.parent)
                tar.add(item, arcname=arcname)
        
        return str(backup_path)
    
    def backup_files(self, file_paths: List[str], backup_name: str) -> str:
        """备份特定文件"""
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    zipf.write(file_path_obj, file_path_obj.name)
        
        return str(backup_path)
    
    def backup_configurations(self, backup_name: str) -> Dict:
        """备份所有配置文件"""
        config_files = []
        
        # 查找所有配置文件
        config_patterns = ['*.yaml', '*.yml', '*.json', '*.env', '*.ini', '*.cfg']
        
        for pattern in config_patterns:
            config_files.extend(Path('.').rglob(pattern))
        
        # 排除备份目录中的文件
        config_files = [f for f in config_files if not f.is_relative_to(self.backup_dir)]
        
        if not config_files:
            return {'status': 'skipped', 'reason': 'No config files found'}
        
        backup_path = self.backup_dir / f"{backup_name}_configs.zip"
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for config_file in config_files:
                zipf.write(config_file, f"configs/{config_file.relative_to('.')}")
        
        return {
            'status': 'success',
            'backup_path': str(backup_path),
            'file_count': len(config_files),
            'files': [str(f) for f in config_files]
        }
    
    def backup_checkpoints(self, checkpoint_dir: str = "./checkpoints", 
                          backup_name: Optional[str] = None) -> Dict:
        """备份模型检查点"""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return {'status': 'skipped', 'reason': f'Checkpoint directory not found: {checkpoint_dir}'}
        
        if backup_name is None:
            backup_name = self.create_backup_name("checkpoints")
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        # 获取检查点文件
        checkpoint_files = []
        for pattern in ['*.pt', '*.pth', '*.bin', '*.safetensors', '*.json', '*.yaml']:
            checkpoint_files.extend(checkpoint_path.rglob(pattern))
        
        if not checkpoint_files:
            return {'status': 'skipped', 'reason': 'No checkpoint files found'}
        
        with tarfile.open(backup_path, "w:gz") as tar:
            for file_path in checkpoint_files:
                arcname = file_path.relative_to(checkpoint_path.parent)
                tar.add(file_path, arcname=arcname)
        
        return {
            'status': 'success',
            'backup_path': str(backup_path),
            'file_count': len(checkpoint_files),
            'total_size': sum(f.stat().st_size for f in checkpoint_files)
        }
    
    def backup_logs(self, log_dir: str = "./logs", 
                   backup_name: Optional[str] = None) -> Dict:
        """备份日志文件"""
        log_path = Path(log_dir)
        if not log_path.exists():
            return {'status': 'skipped', 'reason': f'Log directory not found: {log_dir}'}
        
        if backup_name is None:
            backup_name = self.create_backup_name("logs")
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        # 获取日志文件（排除当前正在写入的日志）
        log_files = []
        for pattern in ['*.log', '*.txt', '*.jsonl']:
            for log_file in log_path.rglob(pattern):
                # 跳过最近1小时内修改过的文件（可能正在写入）
                if log_file.stat().st_mtime > (time.time() - 3600):
                    continue
                log_files.append(log_file)
        
        if not log_files:
            return {'status': 'skipped', 'reason': 'No log files found (excluding recently modified)'}
        
        with tarfile.open(backup_path, "w:gz") as tar:
            for file_path in log_files:
                arcname = file_path.relative_to(log_path.parent)
                tar.add(file_path, arcname=arcname)
        
        return {
            'status': 'success', 
            'backup_path': str(backup_path),
            'file_count': len(log_files),
            'total_size': sum(f.stat().st_size for f in log_files)
        }
    
    def create_snapshot(self, snapshot_name: Optional[str] = None) -> Dict:
        """创建系统快照"""
        if snapshot_name is None:
            snapshot_name = self.create_backup_name("snapshot")
        
        snapshot_info = {
            'timestamp': datetime.now().isoformat(),
            'snapshot_name': snapshot_name,
            'components': {}
        }
        
        # 备份配置
        config_result = self.backup_configurations(f"{snapshot_name}_configs")
        snapshot_info['components']['configurations'] = config_result
        
        # 备份检查点
        checkpoint_result = self.backup_checkpoints(backup_name=f"{snapshot_name}_checkpoints")
        snapshot_info['components']['checkpoints'] = checkpoint_result
        
        # 备份日志
        log_result = self.backup_logs(backup_name=f"{snapshot_name}_logs")
        snapshot_info['components']['logs'] = log_result
        
        # 保存快照信息
        snapshot_file = self.backup_dir / f"{snapshot_name}_info.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_info, f, indent=2, ensure_ascii=False)
        
        snapshot_info['snapshot_file'] = str(snapshot_file)
        return snapshot_info
    
    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        backups = []
        
        for backup_file in self.backup_dir.glob('*'):
            stats = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'size': stats.st_size,
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'type': 'tar.gz' if backup_file.suffix == '.gz' else 'zip' if backup_file.suffix == '.zip' else 'other'
            })
        
        return sorted(backups, key=lambda x: x['modified'], reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 10, 
                           keep_days: Optional[int] = None) -> Dict:
        """清理旧备份"""
        backups = self.list_backups()
        
        if keep_days:
            cutoff_time = time.time() - (keep_days * 86400)
            backups_to_delete = [b for b in backups 
                               if datetime.fromisoformat(b['modified']).timestamp() < cutoff_time]
        else:
            backups_to_delete = backups[keep_count:]
        
        deleted = []
        for backup in backups_to_delete:
            backup_path = self.backup_dir / backup['name']
            try:
                backup_path.unlink()
                deleted.append(backup['name'])
            except Exception as e:
                print(f"删除备份失败 {backup['name']}: {e}")
        
        return {
            'deleted_count': len(deleted),
            'deleted_files': deleted,
            'remaining_count': len(backups) - len(deleted)
        }

def main():
    import time
    
    parser = argparse.ArgumentParser(description='Agent Lightning 备份工具')
    parser.add_argument('--backup-dir', '-d', default='./backups',
                       help='备份目录 (默认: ./backups)')
    parser.add_argument('--snapshot', '-s', action='store_true',
                       help='创建系统快照')
    parser.add_argument('--configs', '-c', action='store_true',
                       help='只备份配置文件')
    parser.add_argument('--checkpoints', '-p', action='store_true', 
                       help='只备份检查点')
    parser.add_argument('--logs', '-l', action='store_true',
                       help='只备份日志')
    parser.add_argument('--list', action='store_true',
                       help='列出所有备份')
    parser.add_argument('--cleanup', type=int, metavar='COUNT',
                       help='清理旧备份，保留指定数量的最新备份')
    parser.add_argument('--cleanup-days', type=int, metavar='DAYS',
                       help='清理指定天数前的备份')
    parser.add_argument('--name', help='自定义备份名称')
    
    args = parser.parse_args()
    
    manager = BackupManager(args.backup_dir)
    
    if args.list:
        backups = manager.list_backups()
        print(f"\n备份列表 ({len(backups)} 个):")
        for backup in backups:
            size_mb = backup['size'] / (1024 * 1024)
            print(f"  {backup['name']} ({size_mb:.1f}MB) - {backup['modified']}")
        return
    
    if args.cleanup:
        result = manager.cleanup_old_backups(keep_count=args.cleanup)
        print(f"清理完成: 删除了 {result['deleted_count']} 个备份，剩余 {result['remaining_count']} 个")
        if result['deleted_files']:
            print("删除的文件:", result['deleted_files'])
        return
    
    if args.cleanup_days:
        result = manager.cleanup_old_backups(keep_days=args.cleanup_days)
        print(f"清理完成: 删除了 {result['deleted_count']} 个备份，剩余 {result['remaining_count']} 个")
        return
    
    backup_name = args.name or manager.create_backup_name()
    
    if args.snapshot or (not args.configs and not args.checkpoints and not args.logs):
        # 默认创建完整快照
        print("创建系统快照...")
        result = manager.create_snapshot(backup_name)
        print(f"快照创建完成: {result['snapshot_file']}")
        
    else:
        # 选择性备份
        if args.configs:
            result = manager.backup_configurations(backup_name)
            print(f"配置备份: {result['backup_path']} ({result['file_count']} 个文件)")
        
        if args.checkpoints:
            result = manager.backup_checkpoints(backup_name=backup_name)
            if result['status'] == 'success':
                size_gb = result['total_size'] / (1024**3)
                print(f"检查点备份: {result['backup_path']} ({result['file_count']} 个文件, {size_gb:.2f}GB)")
            else:
                print(f"检查点备份: {result['reason']}")
        
        if args.logs:
            result = manager.backup_logs(backup_name=backup_name)
            if result['status'] == 'success':
                size_mb = result['total_size'] / (1024**2)
                print(f"日志备份: {result['backup_path']} ({result['file_count']} 个文件, {size_mb:.1f}MB)")
            else:
                print(f"日志备份: {result['reason']}")

if __name__ == "__main__":
    main()