FROM python:3.8.10-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv

COPY api/ api
COPY models/ models

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
