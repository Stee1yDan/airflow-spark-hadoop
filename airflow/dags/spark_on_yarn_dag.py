from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG(
    dag_id="spark_on_yarn",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
)

spark_job = SparkSubmitOperator(
    task_id="spark_example",
    application="/jobs/example_spark_job.py",
    conn_id="spark_default",
    dag = dag
)

write_to_hdfs_job = SparkSubmitOperator(
    task_id="hadoop_write",
    application="/jobs/write_to_hdfs.py",
    conn_id="spark_default",
    dag = dag
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

start >> spark_job >> write_to_hdfs_job >> end