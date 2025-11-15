# Step 1: Start from an official Python base image
# This is our clean operating system with Python 3.10
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
# This is like creating a folder 'app' and running all commands from there
WORKDIR /app

# Step 3: Copy the files from your laptop into the container's 'app' folder
# We copy only the files needed for the app to run
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY savedmodel.pth savedmodel.pth
COPY templates/ templates/

# Step 4: Install the Python dependencies
# This runs 'pip install -r requirements.txt' inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port the app runs on
# Tell Docker that our Flask app will run on port 5000
EXPOSE 5000

# Step 6: Define the command to run the application
# This is the command that starts the Flask server when the container launches
CMD ["python", "app.py"]