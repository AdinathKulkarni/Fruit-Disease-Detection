FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=-1
ENV PORT=4000
RUN apt-get update && apt-get install -y\
    libgl1-mesa-glx \
    libglib2.0.0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt
COPY . .
EXPOSE ${PORT}
# CMD ["python", "app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:4000", "app:app", "--reload"]