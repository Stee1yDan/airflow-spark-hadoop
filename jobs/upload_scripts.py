import subprocess
from pathlib import Path

LOCAL_DIR = Path("./jobs")
HDFS_DIR = "/ml/scripts"


def hdfs(cmd):
    subprocess.run(["hdfs", "dfs"] + cmd, check=True)


def upload_recursive_with_filter(local_path, hdfs_path, patterns=("*.py", "*.sh", "*.yaml")):
    """Recursively upload files matching specific patterns"""
    # Create target directory
    hdfs(["-mkdir", "-p", hdfs_path])

    # Collect all matching files
    all_files = []
    for pattern in patterns:
        all_files.extend(local_path.rglob(pattern))

    # Upload files
    for file_path in all_files:
        # Calculate relative path
        rel_path = file_path.relative_to(local_path)
        hdfs_file_path = f"{hdfs_path}/{rel_path}"

        # Ensure parent directory exists in HDFS
        hdfs_parent = str(Path(hdfs_file_path).parent)
        hdfs(["-mkdir", "-p", hdfs_parent])

        print(f"Uploading: {rel_path}")
        hdfs(["-put", "-f", str(file_path), hdfs_file_path])

    return len(all_files)


# Main execution
print(f"Starting upload from {LOCAL_DIR} to {HDFS_DIR}")
count = upload_recursive_with_filter(
    LOCAL_DIR,
    HDFS_DIR,
    patterns=("*.py", "*.ipynb", "*.txt", "*.yaml", "*.yml", "*.json")
)

print(f"\nUploaded {count} files")
print("Final directory structure:")
hdfs(["-ls", "-R", HDFS_DIR])

upload