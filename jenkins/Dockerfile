FROM jenkins/jenkins:lts-jdk17

USER root

# Install Docker CLI + sudo
RUN apt-get update && \
    apt-get install -y docker.io sudo && \
    rm -rf /var/lib/apt/lists/*

# Add jenkins user to docker group (recommended)
RUN usermod -aG docker jenkins

# Allow sudo only for docker command (more secure than ALL)
RUN echo "jenkins ALL=(ALL) NOPASSWD: /usr/bin/docker" >> /etc/sudoers

# Create userContent directory and copy your images
RUN mkdir -p /var/jenkins_home/userContent/images
COPY images/ /var/jenkins_home/userContent/images/

# Fix permissions so jenkins user can read the images
RUN chown -R jenkins:jenkins /var/jenkins_home/userContent

USER jenkins