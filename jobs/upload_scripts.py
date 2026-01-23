import subprocess
from pathlib import Path

LOCAL_DIR = Path("./jobs")
HDFS_DIR = "/ml/scripts"

def hdfs(cmd):
    subprocess.run(["hdfs", "dfs"] + cmd, check=True)

hdfs(["-mkdir", "-p", HDFS_DIR])

for script in LOCAL_DIR.glob("*.py"):
    print(f"Uploading {script.name}")
    hdfs(["-put", "-f", str(script), f"{HDFS_DIR}/{script.name}"])

print("Upload finished")
hdfs(["-ls", HDFS_DIR])