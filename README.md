pip install -r requirements.txt
python train.py
mlflow ui -------> Open http://127.0.0.1:5000 in your browser.
uvicorn main:app --reload
Open http://127.0.0.1:8000/docs in your browser. ---> FastAPI generates this automatically — click "Try it out" on `/predict`.
