FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV PORT=80
ENV HOST=0.0.0.0

EXPOSE 80

CMD ["python", "index.py"]