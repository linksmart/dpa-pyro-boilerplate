FROM python:3.7

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
COPY pyroAdapter.py /usr/src/app
COPY agent /usr/src/app/agent
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-u", "pyroAdapter.py"]

VOLUME /model
EXPOSE 9100
