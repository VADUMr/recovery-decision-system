FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY reference_policy.json .
COPY app.py .
EXPOSE 8100
CMD ["python", "app.py"]