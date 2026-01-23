FROM apache/airflow:2.7.1-python3.11

USER root

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

USER root

RUN apt-get update && \
    apt-get install -y \
        libkrb5-dev \
        krb5-config \
        build-essential \
        openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

# Use constraint URL to match Airflow version
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.1/constraints-3.11.txt" \
    -r /requirements.txt