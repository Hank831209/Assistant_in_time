FROM ubuntu:18.04
WORKDIR /LilyBubble
COPY . ./
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y python3-pip tzdata
RUN dpkg-reconfigure -f noninteractive tzdata
RUN python3 -m pip install --upgrade pip
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN python3 -m pip install -r requirements.txt 
CMD uwsgi -w app:app --http-socket :$PORT