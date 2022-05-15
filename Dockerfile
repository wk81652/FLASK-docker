FROM python:alpine3.8

LABEL maintainer="Sebastian Zajac <sebastian.zajac@sgh.waw.pl>"
LABEL description="Simpy Flask App"

COPY . /app/

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]
