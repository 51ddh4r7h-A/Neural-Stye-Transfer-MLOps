FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI="https://dagshub.com/shatter-star/musical-octo-dollop.mlflow"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]