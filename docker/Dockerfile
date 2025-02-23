# python version used when developing
FROM python:3.11.11-slim-bookworm

# set working directoy
WORKDIR /app

# Expose the secrets during build time and use them as environment variables
RUN --mount=type=secret,id=GROQ_API_KEY,mode=0444,required=true
RUN --mount=type=secret,id=GOOGLE_API_KEY,mode=0444,required=true

# copy requirements file and install libraries and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the files
COPY ./rag_app.py /app/rag_app.py
COPY ./vector_db_alta_v2 /app/vector_db_alta_v2
COPY ./images /app/images

# Set permissions for vector_db_alta_v2 directory (all files inside it)
RUN chmod -R 777 /app/vector_db_alta_v2

EXPOSE 8501

# from streamlit's documentation:
# "The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working."
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
