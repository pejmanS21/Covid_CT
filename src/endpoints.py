from service import app
from service_warmup import init
import jsonpickle
from PIL import Image
from io import BytesIO
from inc.config.utils import transformer
from flask import request, Response, render_template, jsonify
from inc.IO.grad_output import visualizer


settings = init()


# home page for Flask
@app.route('/')
def hello():
    return "<h1>Check http://0.0.0.0:8000/process</h1>"


# Process unit
@app.route('/process', methods=['POST', 'GET'])
def get_segmented():
    """
        process input image for Detection
    """
    model_type = request.files['image'].filename

    img = Image.open(BytesIO(request.files['image'].read())).convert('RGB')
    img = transformer(img).unsqueeze(0)

    pred, pred_name = visualizer(img, settings[model_type], model_type=model_type, result_path=settings['RESULTS'])

    response = {'Prediction': pred.item(), 'Class': pred_name}
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)