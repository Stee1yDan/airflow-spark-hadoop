from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from hdfs import InsecureClient
import os

HDFS_URL = "http://namenode:9870"
LOCAL_JOBS_DIR = "/jobs"
HDFS_SCRIPTS_DIR = "/ml/scripts"

def upload_scripts():
    client = InsecureClient(HDFS_URL, user="hdfs")
    client.makedirs(HDFS_SCRIPTS_DIR)

    for f in os.listdir(LOCAL_JOBS_DIR):
        if f.endswith(".py"):
            local_path = os.path.join(LOCAL_JOBS_DIR, f)
            hdfs_path = f"{HDFS_SCRIPTS_DIR}/{f}"
            client.upload(hdfs_path, local_path, overwrite=True)

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