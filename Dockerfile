FROM python:alpine
ADD src/* /app/
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "./main.py" ]
