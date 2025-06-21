FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e .
RUN python pipeline/training_pipeline.py

EXPOSE 5000

ENV FLASK_APP=application.py

CMD ["python", "application.py"]