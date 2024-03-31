FROM public.ecr.aws/lambda/python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY . .
ENV MLFLOW_TRACKING_URI="https://dagshub.com/shatter-star/musical-octo-dollop.mlflow"
COPY app.py ${LAMBDA_TASK_ROOT}

# Define the entry point for AWS Lambda
CMD ["app.lambda_handler"]