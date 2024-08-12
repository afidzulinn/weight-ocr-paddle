from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2
import numpy as np
import io
import uvicorn

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.get("/")
async def root():
    return {"message": "Buat Weight OCR API with PaddleOCR API"}

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    # Baca Images
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = ocr.ocr(img, cls=True)

    # Extract text dan confidence
    extracted_text = []
    for line in result:
        for word_info in line:
            text = word_info[1][0]
            confidence = word_info[1][1]
            # , "confidence": float(confidence)
            extracted_text.append({"text": text, 
                                   "confidence": confidence})

    return JSONResponse(content={"results": extracted_text})


if __name__ == "__main__":
    uvicorn.run(app, port=8000, reload=True)