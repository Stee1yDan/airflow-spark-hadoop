from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from hdfs import InsecureClient
import os
import sys

TMP_DIR = "/tmp"
if TMP_DIR not in sys.path:
    sys.path.insert(0, TMP_DIR)

HDFS_URL = "http://namenode:9870"
LOCAL_JOBS_DIR = "/jobs"
HDFS_SCRIPTS_DIR = "/ml/scripts"

def upload_scripts():
    client = InsecureClient(HDFS_URL, user="hdfs")

    # Ensure base directory exists
    client.makedirs(HDFS_SCRIPTS_DIR)

    for root, dirs, files in os.walk(LOCAL_JOBS_DIR):
        for file in files:
            if not file.endswith(".py"):
                continue

            local_path = os.path.join(root, file)

            # preserve relative path
            rel_path = os.path.relpath(local_path, LOCAL_JOBS_DIR)
            hdfs_path = os.path.join(HDFS_SCRIPTS_DIR, rel_path)

            # ensure HDFS subdir exists
            hdfs_dir = os.path.dirname(hdfs_path)
            client.makedirs(hdfs_dir)

            client.upload(
                hdfs_path,
                local_path,
                overwrite=True
            )

            print(f"Uploaded: {rel_path}")

dag = DAG(
    dag_id="upload_scripts",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
)

upload_scripts_to_hdfs = PythonOperator(
    task_id="upload_scripts_to_hdfs",
    python_callable=upload_scripts,
    dag=dag
)

start = PythonOperator(
    task_id="start",
    python_callable = lambda: print("Jobs started"),
    dag=dag
)


end = PythonOperator(
    task_id="end",
    python_callable = lambda: print("Jobs completed successfully"),
    dag=dag
)

start >> upload_scripts_to_hdfs >> end