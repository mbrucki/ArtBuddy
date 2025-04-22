# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Upgrade pip, setuptools, and wheel first to ensure secure build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # setuptools version verified during debug, now proceeding normally
    # pip show setuptools && # Removed debug line
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose the default Cloud Run port
# Although Cloud Run determines the port via the PORT env var,
# exposing it is good practice and useful for local testing.
EXPOSE 8080

# Define environment variable to tell uvicorn where to listen
# ENV PORT=8000 # Optional: Cloud Run sets this automatically

# Run main.py when the container launches using a shell to expand $PORT
# Use 0.0.0.0 to make it accessible from outside the container
# Use $PORT environment variable provided by Cloud Run
# Assuming your FastAPI app instance is named 'app' in 'main.py'
CMD exec /bin/sh -c "uvicorn main:app --host 0.0.0.0 --port $PORT" 