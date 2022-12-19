FROM python:3.9.12-slim

RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["server_flask.py", "model.bin", "thr.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "serv:app"]