FROM python:3.7.9

WORKDIR /image-enhance

COPY requirements.txt requirements.txt
RUN pip3 install torch
RUN pip3 install -r requirements.txt

COPY . .

ENV FLASK_APP=enhance.py

CMD ["flask", "run", "--host=0.0.0.0"]


