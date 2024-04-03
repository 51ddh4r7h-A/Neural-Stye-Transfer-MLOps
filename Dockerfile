FROM public.ecr.aws/lambda/python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN pip install gunicorn mangum --target "${LAMBDA_TASK_ROOT}"

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/app"


COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]