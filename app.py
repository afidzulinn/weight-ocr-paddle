from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import numpy as np
import uvicorn
import imghdr
import logging
import io
import cv2

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='id')

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
    return {"message": "Weight OCR API with PaddleOCR"}


@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Optical Character Recognition (OCR) uploaded images.

    Args:
        file (UploadFile): The image file to be processed.

    Returns:
        JSONResponse: A JSON response containing the extracted text from the image.
    """

    contents = await file.read()

    image_format = imghdr.what(None, contents)
    logger.info(f"Detected image format ngab: {image_format}")


    if image_format == "png":
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    elif image_format in ["jpeg", "jpg"]:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_format == "webp":
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f"format gambar tidak suppport ngab: {image_format}")
        return JSONResponse(content={"error": "format gambar tidak support ngab"}, status_code=415)

    result = ocr.ocr(img, cls=True)

    # Extract text hasil dari gambar
    extracted_text = ""
    for line in result:
        for word_info in line:
            text = word_info[1][0]
            extracted_text += text + " "

    extracted_text = extracted_text.strip()

    return JSONResponse(content={"results": {"text": extracted_text}})



if __name__ == "__main__":
    uvicorn.run(app, port=8000, reload=True)