from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS  # ✅ CORS 추가

import cv2
import numpy as np
import json
import tensorflow as tf
from models.get_text import get_boxes, get_text
import easyocr
from functools import wraps

app = Flask(__name__)
CORS(app)  # ✅ 모든 도메인 허용 기본 설정

model_path = "models/word_with_icon_MASKING_model_CNN_ver_4.h5"
model = tf.keras.models.load_model(model_path, compile=False)
reader = easyocr.Reader(['ko', 'en'], gpu=True)

app.config['ADMIN_KEY'] = "your_secret_admin_key"

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_key = request.headers.get('Admin_key')
        if admin_key != app.config.get('ADMIN_KEY'):
            return jsonify({"error": "Invalid key"}), 403
        return f(*args, **kwargs)
    return decorated_function

# @admin_required  # 관리용 인증 데코레이터 (일단 주석 처리)
@app.route('/image/upload', methods=['POST'])
def upload_and_stream():
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일을 첨부해줘."}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    text_boxes = get_boxes(model, img, reader)
    if len(text_boxes) == 0:
        return jsonify({
            "statusCode": 1,
            "status": "Can't find any text"
        }), 200

    def generate():
        progress_count = 0
        result = []
        response = {
            "statusCode": 0,
            "status": "Success",
            "total": len(text_boxes),
            "progressCount": progress_count,
            "result": []
        }
        yield json.dumps(response) + "\n"

        for text_box in text_boxes:
            progress_count += 1
            response["progressCount"] = progress_count
            try:
                y_start, y_end, x_start, x_end, text = get_text(text_box)
                result.append({
                    "start_y": y_start,
                    "end_y": y_end,
                    "start_x": x_start,
                    "end_x": x_end,
                    "text": text
                })

                yield json.dumps(response) + "\n"
            except Exception as e:
                print(e)

        response["result"] = result
        yield json.dumps(response) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
