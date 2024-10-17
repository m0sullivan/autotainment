FROM python:3.10

ADD autotainment.py .

RUN pip install pydub moviepy

CMD ["python3", "./autotainment.py"]