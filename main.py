from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
import imghdr
import logging
import cv2

from utils.ocr import ocr

app = FastAPI()

# CORS Middleware
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


@app.get("/")
async def home():
    return {"message": "PaddleOCR aman ngab"}


@app.post("/perform-ocr")
async def extract_image(image: UploadFile = File(...)):
    """
    Optical Character Recognition (OCR) for uploaded images.

    Args:
        file (UploadFile): The image file to be processed.

    Returns:
        JSONResponse: A JSON response containing the extracted text from the image.
    """
    contents = await image.read()

    image_format = imghdr.what(None, contents)
    logger.info(f"detected image format: {image_format}")

    if image_format not in ["jpg", "png", "jpeg", "webp"]:
        return JSONResponse(content={"error": "file gambar tidak support"}, status_code=400)

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    result = ocr(img)  # Panggil fungsi OCR ngab

    logger.debug(f"ocr result: {result}")

    if not result:
        return JSONResponse(content={"result": {"text": result}}, status_code=200)

    extracted_text = ""
    for line in result:
        if line is None:
            continue
        for word_info in line:
            if word_info and len(word_info) > 1:
                text = word_info[1][0]
                extracted_text += text + " "

    extracted_text = extracted_text.strip()

    return JSONResponse(content={"result": {"text": extracted_text}}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5523)
