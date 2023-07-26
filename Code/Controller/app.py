import base64

from flask import Flask, jsonify, request, Response

from Code.Application.Services.feedback_services import feedback_service
from Code.Application.Services.model_services import fit_model_service, predict_model_service
from Code.Utils.env_variables import Env


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/services/machine-configuration', methods=['GET'])
def machine_configuration():
    return jsonify({'status': 'alive!'})


@app.route('/services/machine-configuration', methods=['POST'])
def get_machine_configuration():
    cnc_file = request.files['']
    file_content = cnc_file.read().decode('ISO-8859-1')
    return jsonify(predict_model_service(file_content))


@app.route('/services/dash-machine-configuration', methods=['POST'])
def get_dash_machine_configuration():
    file_data = request.values['']
    file_content = base64.b64decode(file_data)
    file_string = str(file_content, 'ISO-8859-1')
    return jsonify(predict_model_service(file_string))


@app.route('/services/feedback', methods=['POST'])
def send_feedback():
    feedback_service(request.json)
    return Response(status=200)


fit_model_service()

if __name__ == '__main__':
    e = Env()
    app.run(host=e.get_host, port=e.get_port)
