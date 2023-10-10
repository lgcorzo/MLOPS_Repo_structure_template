import logging
import warnings
import base64
import uvicorn
import json
from fastapi import FastAPI, File, Form, Response
from fastapi.responses import JSONResponse

from Code.Application.Services.feedback_services import feedback_service
from Code.Application.Services.model_services import init_model_service, predict_model_service
from Code.Utils.env_variables import Env

from uuid import UUID


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


global pickle_model


warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(module)s-%(processName)s-%(threadName)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)


app = FastAPI()


@app.get('/services/is-alive')
def service_is_alive():
    return JSONResponse({'status': 'alive!'})


@app.post('/services/model-predict')
def get_model_predic(cnc_file: bytes = File(...)):
    file_content = cnc_file.decode('ISO-8859-1').replace('\r\n', '\n')
    reponse_remote = predict_model_service(file_content)
    reponse_json = json.dumps(reponse_remote, cls=UUIDEncoder)
    reponse_json = json.loads(reponse_json)
    return JSONResponse(reponse_json)


@app.post('/services/dash-model-predict')
def get_dash_machine_configuration(file_data: str = Form(...)):
    file_content = base64.b64decode(file_data)
    file_string = str(file_content, 'ISO-8859-1')
    reponse_remote = predict_model_service(file_string.replace('\r\n', '\n'))
    reponse_json = json.loads(json.dumps(reponse_remote, cls=UUIDEncoder))
    return JSONResponse(reponse_json)


@app.post('/services/feedback')
def send_feedback(feedback: dict):
    feedback_service(feedback)
    return Response(status_code=200)


@app.on_event('startup')
def init():
    init_model_service()


if __name__ == '__main__':
    e = Env()
    uvicorn.run(app, host=e.be_host, port=int(e.be_port))
