FROM eclipse-temurin:11-jre-jammy

ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

RUN apt-get update && apt-get install -y curl procps && \
    curl -fL https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz \
    -o /tmp/hadoop.tgz && \
    tar -xzf /tmp/hadoop.tgz -C /opt && \
    mv /opt/hadoop-${HADOOP_VERSION} $HADOOP_HOME && \
    rm /tmp/hadoop.tgz

COPY conf/*.xml $HADOOP_HOME/etc/hadoop/

# 🔑 THIS IS CRITICAL
RUN mkdir -p /data/hdfs/namenode /data/hdfs/datanode && \
    chmod -R 777 /data/hdfs
