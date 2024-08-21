from paddleocr import PaddleOCR

class PaddleOCRService:
    def __init__(self, use_angle_cls=True, lang='id', use_gpu=True, det_algorithm='DB',
                 rec_algorithm='SVTR_LCNet', det_model_dir='models/det_model',
                 rec_model_dir='models/rec_model', cls_model_dir='models/cls_model',
                 drop_score=0.65):
        """
        Initialize the PaddleOCR service with the given configuration.

        Args:
            use_angle_cls (bool): Whether to use angle classification.
            lang (str): Language for OCR. Use 'id' for Indonesian, 'en' for English.
            use_gpu (bool): Whether to use GPU for inference.
            det_algorithm (str): The detection algorithm to use.
            rec_algorithm (str): The recognition algorithm to use.
            det_model_dir (str): Path to the detection model directory.
            rec_model_dir (str): Path to the recognition model directory.
            cls_model_dir (str): Path to the classification model directory.
            drop_score (float): Score threshold for dropping low-confidence results.
        """
        self.ocr_engine = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            det_algorithm=det_algorithm,
            rec_algorithm=rec_algorithm,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            drop_score=drop_score
        )

    def perform_ocr(self, image):
        """
        Perform OCR on the given image.

        Args:
            image (np.ndarray): The image on which to perform OCR.

        Returns:
            list: The OCR results, including detected text and bounding boxes.
        """
        result = self.ocr_engine.ocr(image, cls=True)
        return result

# ocr_service = PaddleOCRService()
# result = ocr_service.perform_ocr(image)
