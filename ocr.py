import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

image = cv2.imread("car.jpg")

result, _ = ocr(image)

if result:
    for line in result:
        print(line)