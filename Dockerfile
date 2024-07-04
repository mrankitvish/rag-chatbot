# Use the official Python image as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Copy the .env file
COPY .env_example .env

# Copy sample data files
COPY sample.csv sample.csv
COPY sample.txt sample.txt

# Expose the port the app runs on
EXPOSE 5577

# Command to run the application
CMD ["fastapi","run", "app.py", "--host", "0.0.0.0", "--port", "5577"]
