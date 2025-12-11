FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Mevcut satırı silin ve şunu yapıştırın:
# --timeout 120 (120 saniye bekleme süresi verir, Earth Engine için gereklidir)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 main:app
