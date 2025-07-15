#!/usr/bin/env python3
"""
扩展parquet文件数据的脚本
用法: python expand_parquet_data.py --directory /path/to/data --factor 3
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def expand_parquet_data(directory: str, factor: int, backup: bool = True):
    """
    扩展指定目录下train.parquet文件的数据
    
    Args:
        directory: 包含train.parquet文件的目录路径
        factor: 扩展倍数（必须大于0）
        backup: 是否创建备份文件
    """
    if factor <= 0:
        raise ValueError("扩展倍数必须大于0")
    
    # 构建文件路径
    directory_path = Path(directory)
    parquet_file = directory_path / "train.parquet"
    
    # 检查文件是否存在
    if not parquet_file.exists():
        raise FileNotFoundError(f"文件不存在: {parquet_file}")
    
    print(f"正在读取文件: {parquet_file}")
    
    try:
        # 读取原始parquet文件
        df = pd.read_parquet(parquet_file)
        original_size = len(df)
        print(f"原始数据行数: {original_size}")
        
        # 创建备份（如果需要）
        if backup:
            backup_file = directory_path / "train.parquet.backup"
            print(f"创建备份文件: {backup_file}")
            shutil.copy2(parquet_file, backup_file)
        
        # 扩展数据
        print(f"正在扩展数据 {factor} 倍...")
        expanded_df_list = [df] * factor
        expanded_df = pd.concat(expanded_df_list, ignore_index=True)
        
        new_size = len(expanded_df)
        print(f"扩展后数据行数: {new_size}")
        
        # 保存扩展后的数据到临时文件
        temp_file = directory_path / "train.parquet.tmp"
        print(f"正在保存扩展后的数据...")
        expanded_df.to_parquet(temp_file, index=False)
        
        # 替换原文件
        print(f"正在替换原文件...")
        shutil.move(temp_file, parquet_file)
        
        print(f"✅ 成功完成！数据已从 {original_size} 行扩展到 {new_size} 行")
        
        if backup:
            print(f"📁 备份文件保存在: {backup_file}")
            
    except Exception as e:
        # 清理临时文件
        temp_file = directory_path / "train.parquet.tmp"
        if temp_file.exists():
            temp_file.unlink()
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="扩展parquet文件数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python expand_parquet_data.py --directory ./data/robovqa --factor 4
  python expand_parquet_data.py -d /home/user/data -f 5 --no-backup
        """
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        required=True,
        help="包含train.parquet文件的目录路径"
    )
    
    parser.add_argument(
        "--factor", "-f",
        type=int,
        required=True,
        help="数据扩展倍数（必须大于0）"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建备份文件"
    )
    
    args = parser.parse_args()
    
    try:
        expand_parquet_data(
            directory=args.directory,
            factor=args.factor,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 