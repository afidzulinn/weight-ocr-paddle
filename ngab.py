from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn
import imghdr
import logging
import io
import cv2
import os

from utils.ocr import ocr

app = FastAPI()

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
async def root():
    return {"message": "Weight OCR API menggunakan PaddleOCR"}


@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Optical Character Recognition (OCR) for uploaded images.

    Args:
        file (UploadFile): The image file to be processed.

    Returns:
        JSONResponse: A JSON response containing the extracted text from the image.
    """
    try:
        contents = await file.read()

        image_format = imghdr.what(None, contents)
        logger.info(f"detected image format: {image_format}")

        if image_format not in ["jpg", "png", "jpeg", "webp", "gif"]:
            logger.error(f"unsupported image format: {image_format}")
            return JSONResponse(content={"error": "unsupported image format"}, status_code=415)

        try:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

        result = ocr(img)  # Panggil fungsi OCR ngab

        extracted_text = ""
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                extracted_text += text + " "

        extracted_text = extracted_text.strip()

        return JSONResponse(content={"results": {"text": extracted_text}}, status_code=200)
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9123))
    uvicorn.run(app, port=port)
