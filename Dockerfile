FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel

WORKDIR /

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY pdfChat.py .

CMD ["python", "-u", "pdfChat.py"]