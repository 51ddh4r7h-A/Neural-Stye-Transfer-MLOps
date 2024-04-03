FROM public.ecr.aws/lambda/python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN pip install gunicorn mangum --target "${LAMBDA_TASK_ROOT}"

COPY . .

ENV MLFLOW_TRACKING_URI="https://dagshub.com/shatter-star/musical-octo-dollop.mlflow"
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV MLFLOW_TRACKING_USERNAME="shatter-star"
ENV MLFLOW_TRACKING_PASSWORD="411996890a0df0c0ccf65dbd848d454f40ad3cbb"
ENV MODEL_URI="mlflow-artifacts:/366666ce4dc8413383fd5d9a1ce802f9/8c9c0df67b1d4151886eec4a77c36417/artifacts/model"


COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]