FROM public.ecr.aws/lambda/python:3.12

COPY src/lambda_retrain2.py ${LAMBDA_TASK_ROOT}/

RUN pip install --no-cache-dir pymongo scikit-learn pandas numpy

CMD ["lambda_retrain2.lambda_handler"]
