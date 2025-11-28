# --- Dockerfile (Use this content) ---

# 1. Use a Python 3.10 slim base image for minimal size
FROM python:3.10-slim

# 2. Set environment variables for Render/AWS compatibility
# This ensures Streamlit binds correctly to the host machine.
ENV STREAMLIT_SERVER_PORT 8080
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements and install dependencies
# We copy requirements.txt from the root to install everything first.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire application source code
# We copy the entire 'streamlit_app' folder into the container's root working directory (/app)
COPY streamlit_app /app/streamlit_app

# 6. Expose Streamlit's port
EXPOSE 8080

# 7. Run the Streamlit application
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]