FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords snowball_data

COPY . /app/

EXPOSE 5000

CMD ["python", "app.py"]