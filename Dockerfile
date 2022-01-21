FROM gmontamat/python-darknet:cpu

WORKDIR app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3" ]

CMD [ "index.py" ]