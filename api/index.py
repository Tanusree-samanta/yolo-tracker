from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "YOLO Tracker API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
