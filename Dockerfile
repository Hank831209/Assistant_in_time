FROM python:3.9.9

# Copy local code to the container image.
WORKDIR /app
COPY . /app

# 如果裝太久會停掉
RUN pip install --default-timeout=100 -r requirements.txt

# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
# python不穩定得用gunicorn來啟
CMD gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
# CMD uwsgi -w app:app -s :3000 -d app.log