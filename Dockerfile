FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y \
    portaudio19-dev \
    libportaudio2 \
    build-essential \
    espeak-ng \
    espeak-ng-data
    
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0