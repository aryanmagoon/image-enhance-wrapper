FROM python:3.7.9

WORKDIR /image-enhance

RUN apt-get update
RUN apt-get -y install ffmpeg libsm6 libxext6 libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip3 install torch
RUN pip3 install -r requirements.txt

COPY . .

ENV FLASK_APP=enhance.py

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]


