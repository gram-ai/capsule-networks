# Build the docker image with 
#
#  $ docker build -t capsnet .
#
# Run with
#
#  $ nvidia-docker run --rm -it --ipc=host capsnet:latest
#
# At the container prompt, start the visdom server, and the capsnet processing
#
#  # python -m visdom.server & python capsule_network.py
#
# In a separate terminal on the docker host...
#
# Obtain the CONTAINER_ID
#
#  $ docker ps  
#
# Get the container IP address
#
#  $ docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <CONTAINER_ID>
#
# On the host, browse to <returned_IP>:8097

FROM pytorch 

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /workspace 
WORKDIR /workspace

