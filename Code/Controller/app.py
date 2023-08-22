import logging
import warnings
import base64

from flask import Flask, jsonify, request, Response

from Code.Application.Services.feedback_services import feedback_service
from Code.Application.Services.model_services import init_model_service, predict_model_service
from Code.Utils.env_variables import Env

global pickle_model


warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(module)s-%(processName)s-%(threadName)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/services/is-alive', methods=['GET'])
def service_is_alive():
    return jsonify({'status': 'alive!'})


@app.route('/services/model-predict', methods=['POST'])
def get_model_predic():
    cnc_file = request.files['']
    file_content = cnc_file.read().decode('ISO-8859-1').replace('\r\n', '\n')
    return jsonify(predict_model_service(file_content))


@app.route('/services/dash-model-predict', methods=['POST'])
def get_dash_machine_configuration():
    file_data = request.values['']
    file_content = base64.b64decode(file_data)

    file_string = str(file_content, 'ISO-8859-1')
    return jsonify(predict_model_service(file_string.replace('\r\n', '\n')))


@app.route('/services/feedback', methods=['POST'])
def send_feedback():
    feedback_service(request.json)
    return Response(status=200)


@app.before_first_request
def init():
    init_model_service()


if __name__ == '__main__':
    e = Env()
    app.run(host=e.be_host, port=e.be_port)
