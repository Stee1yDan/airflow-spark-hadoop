from hdfs import InsecureClient
import sys
import subprocess

script_name = sys.argv[1]

client = InsecureClient("http://namenode:9870", user="hdfs")
client.download(f"/scripts/{script_name}", "/tmp/job.py")

subprocess.run(["python", "/tmp/job.py"], check=True)