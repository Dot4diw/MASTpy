#!/usr/bin/env python3
import os
import subprocess
import sys

def check_git_repository():
    """检查当前目录是否是git仓库"""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        print("错误: 当前目录不是git仓库")
        return False

def get_changed_files():
    """获取所有已修改和新增的文件"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        changed_files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                status, file_path = line.split(' ', 1)
                # 忽略以.开头的隐藏文件和__pycache__目录
                if not file_path.startswith('.') and '__pycache__' not in file_path:
                    changed_files.append((status, file_path))
        return changed_files
    except Exception as e:
        print(f"获取变更文件时出错: {e}")
        return []

def generate_commit_message(file_path, status):
    """为文件生成commit message"""
    # 根据文件类型和状态生成不同的commit message
    status_map = {
        'A': 'Add',
        'M': 'Update',
        'D': 'Delete'
    }
    action = status_map.get(status[0], 'Modify')
    
    # 根据文件路径确定文件类型
    if file_path.endswith('.py'):
        file_type = 'Python script'
    elif file_path.endswith('.md'):
        file_type = 'Markdown file'
    elif file_path.endswith('.txt'):
        file_type = 'Text file'
    elif file_path.endswith('.csv'):
        file_type = 'CSV file'
    elif file_path.endswith('.R'):
        file_type = 'R script'
    elif file_path.endswith('.rds'):
        file_type = 'R data file'
    elif file_path.endswith('.toml'):
        file_type = 'TOML file'
    else:
        file_type = 'File'
    
    return f"{action}: {file_type} {file_path}"

def commit_files(changed_files):
    """提交所有变更的文件"""
    if not changed_files:
        print("没有需要提交的文件")
        return False
    
    for status, file_path in changed_files:
        try:
            # 添加文件到暂存区
            subprocess.run(['git', 'add', file_path], check=True)
            # 生成commit message
            commit_msg = generate_commit_message(file_path, status)
            # 提交文件
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            print(f"已提交: {file_path} - {commit_msg}")
        except subprocess.CalledProcessError as e:
            print(f"提交文件 {file_path} 时出错: {e}")
            return False
    
    return True

def push_to_github():
    """推送到GitHub"""
    try:
        # 提示用户输入远程仓库名称，默认为origin
        remote = input("请输入远程仓库名称 (默认为origin): ") or "origin"
        # 提示用户输入分支名称，默认为main
        branch = input("请输入分支名称 (默认为main): ") or "main"
        
        print(f"正在推送到 {remote}/{branch}...")
        subprocess.run(['git', 'push', remote, branch], check=True)
        print("推送成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"推送到GitHub时出错: {e}")
        return False

def main():
    """主函数"""
    # 检查当前目录是否是git仓库
    if not check_git_repository():
        sys.exit(1)
    
    # 获取变更的文件
    changed_files = get_changed_files()
    
    # 提交文件
    if commit_files(changed_files):
        # 询问是否推送到GitHub
        push = input("是否推送到GitHub? (y/n): ")
        if push.lower() == 'y':
            push_to_github()

if __name__ == "__main__":
    main()
