from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from datetime import datetime


dag = DAG(
    dag_id="test_runner",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
)

run_test = DockerOperator(
    task_id="run_test",
    image="ash-ml-python:latest",
    command="python /app/jobs/test_runner.py gradient_inversion.py diff_privacy_simple_mlp/SimpleMLP_with_noise/SimpleMLP_with_noise.pt diff_privacy_simple_mlp/SimpleMLP_with_noise/artifacts",
    network_mode="ash_hadoop-net",
    dag = dag,
    auto_remove=True,
    docker_url="unix://var/run/docker.sock",
    do_xcom_push=True,
    mount_tmp_dir=False
)

generate_report = DockerOperator(
    task_id="generate_report",
    image="ash-ml-python:latest",
    command="python /app/jobs/generate_report.py gradient_inversion",
    network_mode="ash_hadoop-net",
    dag = dag,
    auto_remove=True,
    docker_url="unix://var/run/docker.sock",
    environment={
        "TEST_RESULT_JSON": "{{ ti.xcom_pull(task_ids='run_test') | tojson }}"
    },
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

start >> run_test >> generate_report >> end