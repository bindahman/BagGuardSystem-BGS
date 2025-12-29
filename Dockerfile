FROM python:3.10-slim

# 2. optimize Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Then set working directory in the container
WORKDIR /app

# 4. copy dependency file
COPY requirements.txt .

# 5. install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. copy project files
COPY . .

# 7. start the application
CMD ["python", "helloworld.py"]