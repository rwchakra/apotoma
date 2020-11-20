FROM tensorflow/tensorflow:2.0.0-gpu

RUN apt-get update

# ===========
# Update pip and install all pip dependencies
# ===========

# Required dependency to upgrade pip below
RUN python -m pip install six

RUN python -m pip install --upgrade pip
COPY ./requirements.txt /opt/project/requirements.txt
RUN pip install -r /opt/project/requirements.txt
