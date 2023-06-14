FROM python:alpine
RUN apk add --no-cache python3-dev build-base

ADD src/* /app/
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "./main.py" ]
