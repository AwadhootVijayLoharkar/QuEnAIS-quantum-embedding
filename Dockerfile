FROM mambaorg/micromamba:latest

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml

WORKDIR /app
COPY . /app

CMD ["bash"]
