#!/usr/bin/env python3
"""
organize_screenshots.py

Utility script to move screenshots from root directory to results/screenshots/
and clean up the project structure.
"""

import os
import shutil
import glob
from pathlib import Path

def organize_screenshots():
    """Move screenshots from root to proper directory"""
    print("🧹 Organizing screenshot files...")
    
    # Create screenshots directory if it doesn't exist
    screenshots_dir = Path("results/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find screenshot files in root directory
    root_screenshots = glob.glob("screenshot_*.png")
    
    if not root_screenshots:
        print("✅ No screenshots found in root directory")
        return
    
    moved_count = 0
    for screenshot in root_screenshots:
        try:
            # Move to proper directory
            destination = screenshots_dir / screenshot
            shutil.move(screenshot, destination)
            print(f"📁 Moved: {screenshot} → {destination}")
            moved_count += 1
        except Exception as e:
            print(f"❌ Failed to move {screenshot}: {e}")
    
    print(f"✅ Moved {moved_count} screenshot(s) to results/screenshots/")

def cleanup_temp_files():
    """Clean up other temporary files"""
    print("\n🧹 Cleaning up temporary files...")
    
    # Patterns of files to clean up
    cleanup_patterns = [
        "*.tmp",
        "*.temp", 
        ".DS_Store",
        "Thumbs.db",
        "*.log.1",
        "*.log.2"
    ]
    
    cleaned_count = 0
    for pattern in cleanup_patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"🗑️  Removed: {file}")
                cleaned_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} temporary file(s)")
    else:
        print("✅ No temporary files to clean up")

def create_directory_structure():
    """Ensure all required directories exist"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "results",
        "results/screenshots",
        "results/gradio_sessions", 
        "results/batch_runs",
        "results/exports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

def show_directory_status():
    """Show current directory organization"""
    print("\n📊 Directory Status:")
    
    # Check important directories
    dirs_to_check = [
        ("results/screenshots", "Screenshot storage"),
        ("results/gradio_sessions", "Session exports"),
        ("results/batch_runs", "Batch processing results"),
        ("logs", "Log files")
    ]
    
    for dir_path, description in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            file_count = len(list(path.glob("*")))
            print(f"✅ {dir_path}: {file_count} files ({description})")
        else:
            print(f"❌ {dir_path}: Missing ({description})")
    
    # Check for screenshots in wrong places
    root_screenshots = len(glob.glob("screenshot_*.png"))
    if root_screenshots > 0:
        print(f"⚠️  {root_screenshots} screenshot(s) in root directory (should be moved)")

def main():
    print("=" * 60)
    print("🗂️  Project File Organization Utility")
    print("=" * 60)
    
    # Show current status
    show_directory_status()
    
    # Create proper directory structure
    create_directory_structure()
    
    # Move screenshots to proper location
    organize_screenshots()
    
    # Clean up temporary files
    cleanup_temp_files()
    
    # Show final status
    print("\n" + "=" * 60)
    print("📊 Final Status:")
    show_directory_status()
    
    print("\n🎉 File organization complete!")
    print("   Screenshots will now be saved to results/screenshots/")

if __name__ == "__main__":
    main()
