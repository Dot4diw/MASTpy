#!/usr/bin/env python3
import os
import subprocess
import sys

def check_git_repository():
    """Check if current directory is a git repository"""
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        print("Error: Current directory is not a git repository")
        return False

def get_changed_files():
    """Get all modified and added files"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        changed_files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                # Correctly parse git status output, ensuring only file path part is taken
                status = line[:2].strip()
                file_path = line[2:].strip()
                # Ignore hidden files except .gitignore and __pycache__ directories
                if (not file_path.startswith('.') or file_path == '.gitignore') and '__pycache__' not in file_path:
                    changed_files.append((status, file_path))
        return changed_files
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return []

def generate_commit_message(file_path, status):
    """Generate commit message for file"""
    # Generate different commit messages based on file type and status
    status_map = {
        'A': 'Add',
        'M': 'Update',
        'D': 'Delete'
    }
    action = status_map.get(status[0], 'Modify')
    
    # Determine file type based on file path
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
    """Commit all changed files"""
    if not changed_files:
        print("No files need to be committed")
        return False
    
    for status, file_path in changed_files:
        try:
            # Add file to staging area
            subprocess.run(['git', 'add', file_path], check=True)
            # Generate commit message
            commit_msg = generate_commit_message(file_path, status)
            # Commit file
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            print(f"Committed: {file_path} - {commit_msg}")
        except subprocess.CalledProcessError as e:
            print(f"Error committing file {file_path}: {e}")
            return False
    
    return True

def push_to_github():
    """Push to GitHub"""
    try:
        # Prompt user for remote repository name, default to origin
        remote = input("Enter remote repository name (default: origin): ") or "origin"
        # Prompt user for branch name, default to main
        branch = input("Enter branch name (default: main): ") or "main"
        
        print(f"Pushing to {remote}/{branch}...")
        subprocess.run(['git', 'push', remote, branch], check=True)
        print("Push successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to GitHub: {e}")
        return False

def main():
    """Main function"""
    # Check if current directory is a git repository
    if not check_git_repository():
        sys.exit(1)
    
    # Get changed files
    changed_files = get_changed_files()
    
    # Commit files
    if commit_files(changed_files):
        # Ask if user wants to push to GitHub
        push = input("Push to GitHub? (y/n): ")
        if push.lower() == 'y':
            push_to_github()

if __name__ == "__main__":
    main()
