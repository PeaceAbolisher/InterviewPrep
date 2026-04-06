pip install -r requirements.txt
python train.py
mlflow ui -------> Open http://127.0.0.1:5000 in your browser.
uvicorn main:app --reload
Open http://127.0.0.1:8000/docs in your browser. ---> FastAPI generates this automatically — click "Try it out" on `/predict`.

Docker:
docker build -t glintt-model . ----> Build a Docker image from the Dockerfile in this folder and call it glintt-model
docker run -p 8000:8000 glintt-model ----> Maps port 8000 on your machine to port 8000 inside the container. Allows the browser to reach the API
