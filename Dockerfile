FROM python:3.11-slim

# install system deps needed for image libs and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . /app

# create upload dir
RUN mkdir -p /app/uploads

ENV FLASK_ENV=production
EXPOSE 5000

# using gunicorn for production
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]