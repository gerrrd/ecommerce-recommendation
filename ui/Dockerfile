FROM public.ecr.aws/docker/library/python:3.10-slim-bullseye
# or on different devices, you may need another version of python10:
#FROM public.ecr.aws/docker/library/python:3.10-slim
#FROM python:3.10-slim

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential

WORKDIR /ui
COPY ui-requirements.txt .
COPY setup.py .

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip-tools
RUN pip install -r ui-requirements.txt

ADD ecommercerecommendation/ ./ecommercerecommendation
ADD setup.py .
RUN pip install .

COPY ui/ .

EXPOSE 8080

ENTRYPOINT ["./start.sh"]
