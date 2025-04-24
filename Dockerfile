# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Upgrade pip, setuptools, and wheel first to ensure secure build tools
# Adding comment to invalidate cache
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # setuptools version verified during debug, now proceeding normally
    # pip show setuptools && # Removed debug line
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Revert to copying everything, will adjust PYTHONPATH in CMD
COPY . /app
# COPY ./app /app # Copy contents of local app dir to container /app
# COPY ./static /static # Copy static files to /static
# COPY ./templates /templates # Copy templates to /templates

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
# Adjust CMD to work with /app/app structure by setting PYTHONPATH
# CMD exec /bin/sh -c "cd /app && uvicorn app.main:app --host 0.0.0.0 --port $PORT"
# CMD exec /bin/sh -c "python -m uvicorn app.main:app --host 0.0.0.0 --port 80"
# Simplest Python test CMD
# CMD ["python", "-c", "import sys; print('Python runs!'); sys.stdout.flush()"]
# Restore correct CMD
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"] 