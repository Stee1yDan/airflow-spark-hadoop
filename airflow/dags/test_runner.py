from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id="test_runner",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={"retries": 1},
) as dag:

    # Задача запуска теста с параметрами из Jenkins
    run_test = DockerOperator(
        task_id="run_test",
        image="ash-ml-python:latest",
        # Команда остаётся почти как была, но теперь можно сделать динамической
        command=[
            "python",
            "/app/jobs/test_runner.py",
            "{{ dag_run.conf.get('test_module') }}.py",
            # Пример: если нужно подставить MODEL_NAME в путь к модели
            "{{ dag_run.conf.get('model_name') }}.pt",
            "diff_privacy_simple_mlp/SimpleMLP_with_noise/artifacts"
        ],
        network_mode="ash_hadoop-net",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        do_xcom_push=True,
        mount_tmp_dir=False,
        # ← Самый удобный способ передать ВСЕ параметры внутрь контейнера
        environment={
            "TEST_MODULE": "{{ dag_run.conf.get('test_module') }}",
            "MODEL_NAME": "{{ dag_run.conf.get('model_name') }}",
            "RUN_NOTE": "{{ dag_run.conf.get('run_note') }}",
            "SEED": "{{ dag_run.conf.get('seed', 42) }}",
            "BUILD_NUMBER": "{{ dag_run.conf.get('build_number', 'unknown') }}",
        },
    )

    # Генерация отчёта (оставляем как было)
    generate_report = DockerOperator(
        task_id="generate_report",
        image="ash-ml-python:latest",
        command="python /app/jobs/generate_report.py gradient_inversion",
        network_mode="ash_hadoop-net",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        environment={
            "TEST_RESULT_JSON": "{{ ti.xcom_pull(task_ids='run_test') | tojson }}"
        },
        mount_tmp_dir=False,
    )

    start = PythonOperator(
        task_id="start",
        python_callable=lambda: print("🚀 Jobs started"),
    )

    end = PythonOperator(
        task_id="end",
        python_callable=lambda: print("✅ Jobs completed successfully"),
    )

    start >> run_test >> generate_report >> end