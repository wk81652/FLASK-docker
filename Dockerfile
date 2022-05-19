FROM python:3.8-slim

WORKDIR /app

COPY app.py .
COPY Perceptron.py . 
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN python3 Perceptron.py

ENTRYPOINT ["python3"]
CMD ["app.py"]
