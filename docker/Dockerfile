FROM python:3.11.7

RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip install -r app/requirements.txt \
    && ls -alh /app

EXPOSE 5001
# EXPOSE 5000
ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/app/scaling.py"]