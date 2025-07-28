# Use official lightweight Python runtime
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt with all necessary dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data and sentence-transformers model (optional but recommended)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of your project files (make sure src/, input/, etc. are present)
COPY . .

# Create default input/output folders inside container
RUN mkdir -p input output

# Set environment variables to disable internet access and hint transformers offline usage
ENV NO_PROXY="*"
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# Run your main.py script on container startup
CMD ["python", "-m", "src.main", "--input", "/app/input", "--output", "/app/output"]