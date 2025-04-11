import numpy as np
import cv2

def get_boxes(model, image):
    prediction = __get_prediction(model, image)
    return __find_bounding_rectangles(prediction)

def get_text(image, reader, position, padding=(3, 10)):
    crp_image = image[position[0]-padding[0]:position[1]+padding[0], position[2]-padding[1]:position[3]+padding[1]]
    results_text = []
    results = reader.readtext(crp_image)
    if len(results) == 0:
        return [""]
    for result in results:
        results_text.append(result[1])
    return results_text

def __get_prediction(model, image):
    threshold = 127

    scaled_image = image / 255.0
    scaled_image = np.expand_dims(scaled_image, axis=0)
    prediction = model.predict(scaled_image)[0]
    prediction *= 255
    prediction = prediction.astype(np.uint8)

    return cv2.threshold(prediction, threshold, 255, cv2.THRESH_BINARY)[1]

def __find_bounding_rectangles(image, size=300):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_rectangles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= size:
            result_rectangles.append((y, y + h, x, x + w))

    return result_rectangles