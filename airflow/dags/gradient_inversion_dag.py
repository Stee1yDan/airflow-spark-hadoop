from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from datetime import datetime


dag = DAG(
    dag_id="gradient_inversion_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
)

gradient_inversion = DockerOperator(
    task_id="ml_job",
    image="ash-ml-python:latest",
    command="python /app/jobs/test_runner.py gradient_inversion.py",
    network_mode="ash_hadoop-net",
    dag = dag,
    auto_remove=True,
    docker_url="unix://var/run/docker.sock",
    mount_tmp_dir=False
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

start >> gradient_inversion >> end