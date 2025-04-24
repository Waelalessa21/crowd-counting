from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import shutil
from core.crowd_counting_model import process_video

app = FastAPI()

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        shutil.copyfileobj(file.file, temp)
        video_path = temp.name

    state = {
        "tracked_people": {},
        "next_id": 0,
        "total_unique_people": 0
    }

    for _ in process_video(video_path, 0, state):
        pass

    return JSONResponse(content={
        "people_count": state["total_unique_people"]
    })
