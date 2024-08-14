from paddleocr import PaddleOCR

conf_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='id', # Ganti ke id atau en kalo ngga bisa yahhh
    use_gpu=True,
    det_algorithm='DB',
    # Ini algoritma nya yahhh
    rec_algorithm='CRNN',
    # rec_algorithm='SVTR_LCNet',
    det_model_dir='models/det_model',
    rec_model_dir='models/rec_model',
    cls_model_dir='models/cls_model',
    drop_score=0.7
)

def ocr(image):
    result = conf_ocr.ocr(image, cls=True)
    return result
