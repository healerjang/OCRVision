import numpy as np
import cv2

def get_boxes(model, image, reader):
    prediction = __get_prediction(model, image)
    results = reader.readtext(image)
    return results

def get_text(text_info):
    coord = text_info[0]
    text = text_info[1]
    y_start, y_end, x_start, x_end = coord[0][1], coord[2][1], coord[0][0], coord[1][0]
    return int(y_start), int(y_end), int(x_start), int(x_end), text

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