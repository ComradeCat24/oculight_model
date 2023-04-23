import os
import subprocess
from flask import Flask, request

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_caption():
    file = request.files['image']
    filename = file.filename
    folder = os.environ.get('UPLOAD_FOLDER_PATH')
    file_path = os.path.join(folder, filename)
    file.save(file_path)
    output = subprocess.run(
        ['python', 'gen.py', file_path], capture_output=True)
    caption = output.stdout.decode('utf-8').strip()
    return caption


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
