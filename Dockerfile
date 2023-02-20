FROM anibali/pytorch:1.8.1-cuda11.1
WORKDIR /app
USER root

# Install Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy Files and Secrets
COPY ./ .

# Create a dockeruser to own the app
RUN useradd -r dockeruser
RUN chown dockeruser: /app
USER dockeruser

ENV PYTHONPATH "${PYTHONPATH}:/app"
