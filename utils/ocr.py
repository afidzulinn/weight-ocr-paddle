from paddleocr import PaddleOCR

conf_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='id', # Ganti ke id atau en kalo ngga bisa yahhh
    use_gpu=True,
    det_algorithm='DB',
    rec_algorithm='CRNN',
    # rec_algorithm='SVTR_LCNet', # Ini algoritma nya yahhh
    det_model_dir='models/det_model',
    rec_model_dir='models/rec_model',
    cls_model_dir='models/cls_model',
    drop_score=0.7
)

def ocr(image):
    """
    Perform OCR on the given image using PaddleOCR.

    Args:
        image (numpy.ndarray): The image on which to perform OCR.

    Returns:
        list: A list of detected text with bounding box information.
    """
    result = conf_ocr.ocr(image, cls=True)
    return result
