FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    matplotlib \
    hdfs

RUN pip install --no-cache-dir \
    matplotlib \
    hdfs

RUN pip install --no-cache-dir \
    scikit-learn

RUN pip install --no-cache-dir \
    scikit-image

RUN pip install --no-cache-dir \
    pathlib

RUN pip install --no-cache-dir \
    reportlab

RUN pip install --no-cache-dir \
    numpy


WORKDIR /app

COPY test_runner.py /app/jobs/
COPY generate_report.py /app/jobs/
COPY fonts /fonts
