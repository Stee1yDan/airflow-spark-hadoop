FROM eclipse-temurin:11-jre-jammy

ENV SPARK_VERSION=3.5.0
ENV SPARK_HOME=/opt/spark
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=$PATH:$SPARK_HOME/bin:$PATH

# Install required dependencies
RUN apt-get update && \
    apt-get install -y \
      curl \
      ca-certificates \
      tar \
      python3 \
      python3-pip \
      openjdk-11-jdk \
      procps \
      net-tools \
      vim \
      wget \
      rsync \
      ssh && \
    rm -rf /var/lib/apt/lists/*

# Download and unpack Spark
RUN curl -fL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz -o /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop3 ${SPARK_HOME} && \
    rm /tmp/spark.tgz

# Make sure Spark scripts are executable
RUN chmod -R +x $SPARK_HOME/bin

# Copy Hadoop configs if needed
# COPY ./hadoop/etc/hadoop $HADOOP_CONF_DIR

# Default command to keep container alive
CMD ["tail", "-f", "/dev/null"]
