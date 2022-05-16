FROM python:alpine3.8

LABEL description="Flask App"

COPY . /app/

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]
