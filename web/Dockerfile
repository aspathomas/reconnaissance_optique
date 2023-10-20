FROM python:3.10.12

# Install Tesseract-OCR and other dependencies
RUN apt-get update && apt-get install -y tesseract-ocr
RUN apt-get clean

RUN mkdir /usr/src/app/
COPY . /usr/src/app/
WORKDIR /usr/src/app/
EXPOSE 5000
RUN apt install tesseract-ocr
RUN pip install -r requirements.txt
CMD ["python", "app.py"]