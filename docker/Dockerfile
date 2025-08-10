# Use Python 3.9 slim image
FROM python:3.9-slim
 
# Set working directory inside container
WORKDIR /app
 
# Copy requirements.txt from root into container
COPY requirements.txt .
 
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the rest of the project (from build context, i.e., root folder)
COPY ./api /app/api
COPY ./models /app/models
 
# If you have other folders needed (data, src, etc.), add them:
# COPY ./data /app/data
# COPY ./src /app/src
 
# Expose port for FastAPI
EXPOSE 8000
 
# Run FastAPI app using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
