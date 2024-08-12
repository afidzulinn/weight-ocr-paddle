from paddleocr import PaddleOCR, draw_ocr
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Specify language English

img_path = 'images/15.JPG'
img = cv2.imread(img_path)

result = ocr.ocr(img_path)

for line in result:
    print(line)

boxes = [elements[0] for elements in result[0]]
txts = [elements[1][0] for elements in result[0]]
scores = [elements[1][1] for elements in result[0]]

img_with_boxes = draw_ocr(img, boxes, txts, scores)
cv2.imshow('OCR Result', img_with_boxes)
cv2.waitKey(0)
