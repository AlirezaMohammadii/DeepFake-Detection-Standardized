#!/usr/bin/env python3
"""
Project Cleanup Tool for Deepfake Detection Framework
Removes all unnecessary directories and files for clean architecture
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Remove all unnecessary directories and files"""
    
    # Get current directory (should be deepfake-detection)
    project_root = Path.cwd()
    
    print("🧹 COMPREHENSIVE PROJECT CLEANUP")
    print("=" * 60)
    print(f"Working directory: {project_root}")
    
    # Directories to completely remove
    dirs_to_remove = [
        'quarantine',
        'deployment',
        'evaluation',
        'training',
        'tests/unit',
        'tests/integration', 
        'tests/system',
        'api/grpc',
        'inference/postprocessors',
        'models/detection',
        'data/augmentation',
        'checkpoints',
        'logs',
        'plots',
        'output',
        'cache',
    ]
    
    # Files to remove
    files_to_remove = [
        'comprehensive_analysis_report.html',
        'FOLDER_INDEX.md',
        'FOLDER_CLEANUP_SUMMARY.md',
    ]
    
    removed_dirs = 0
    removed_files = 0
    
    # Remove directories
    print("\n1️⃣ Removing unnecessary directories...")
    for dir_path in dirs_to_remove:
        full_path = project_root / dir_path
        if full_path.exists():
            try:
                shutil.rmtree(full_path)
                removed_dirs += 1
                print(f"    ✓ Removed: {dir_path}")
            except Exception as e:
                print(f"    ❌ Failed to remove {dir_path}: {e}")
    
    # Remove files
    print("\n2️⃣ Removing unnecessary files...")
    for file_path in files_to_remove:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                removed_files += 1
                print(f"    ✓ Removed: {file_path}")
            except Exception as e:
                print(f"    ❌ Failed to remove {file_path}: {e}")
    
    # Clean __pycache__ directories
    print("\n3️⃣ Cleaning Python cache...")
    pycache_dirs = list(project_root.rglob('__pycache__'))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            removed_dirs += 1
            print(f"    ✓ Removed: {pycache_dir.relative_to(project_root)}")
        except Exception as e:
            print(f"    ❌ Failed to remove {pycache_dir}: {e}")
    
    # Clean results directory but keep structure
    print("\n4️⃣ Cleaning results directory...")
    results_dir = project_root / 'results'
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and item.name.startswith('session_'):
                try:
                    shutil.rmtree(item)
                    removed_dirs += 1
                    print(f"    ✓ Removed old session: {item.name}")
                except Exception as e:
                    print(f"    ❌ Failed to remove {item}: {e}")
            elif item.is_file() and item.suffix == '.csv':
                try:
                    item.unlink()
                    removed_files += 1
                    print(f"    ✓ Removed: {item.name}")
                except Exception as e:
                    print(f"    ❌ Failed to remove {item}: {e}")
    
    print(f"\n✅ CLEANUP COMPLETED!")
    print(f"  📁 Directories removed: {removed_dirs}")
    print(f"  📄 Files removed: {removed_files}")
    print(f"  🚀 Project ready for fresh main.py run!")

if __name__ == "__main__":
    cleanup_project() 