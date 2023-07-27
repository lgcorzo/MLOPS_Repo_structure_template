import base64
import dash
import dash_extensions as de
import json
import pandas as pd
import requests
import flask

from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from Code.Utils.env_variables import Env


server = flask.Flask(__name__)  # define flask app.server

machine_config_service_url = ''
feedback_service_url = ''
smart_id = ''
user_comment_str = ''

ASSETS_FOLDER = 'assets/'
animation_img = ASSETS_FOLDER + 'loading.json'
star_img = ASSETS_FOLDER + 'star.svg'
check_img = ASSETS_FOLDER + 'check.svg'
merlin_icon_img = ASSETS_FOLDER + 'merlin.svg'

animation_options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))

table_style = {'visibility': 'hidden'}

star_style = {
    'width': '25px',
    'height': '25px',
    'margin-top': '43px',
    'display': 'none'
}

check_style = {
    'filter': 'grayscale(100%)',
    'opacity': '0.4',
}

initialization_df = pd.DataFrame({
    "col1": [1],
    "col2": ["A"],
    "col3": [True]
})

final_table = dash_table.DataTable(id='data_table_info',
                                   columns=[{"name": i, "id": i} for i in initialization_df.columns],
                                   data=initialization_df.to_dict(),
                                   active_cell={'row': 0, 'column': 0, 'column_id': "col1"})


check_icons_table = html.Table(id='check-icons-table-id', children=[
    # pack de iconos
    html.Tr(children=[
        html.Td(children=[
            html.Img(id='0check', src=check_img, style=check_style)
        ]),
    ]),
    html.Tr(children=[
        html.Td(children=[
            html.Img(id='1check', src=check_img, style=check_style)
        ]),
    ]),
    html.Tr(children=[
        html.Td(children=[
            html.Img(id='2check', src=check_img, style=check_style)
        ]),
    ]),
    html.Tr(children=[
        html.Td(children=[
            html.Img(id='3check', src=check_img, style=check_style)
        ]),
    ]),
    html.Tr(children=[
        html.Td(children=[
            html.Img(id='4check', src=check_img, style=check_style)
        ]),
    ])
], style=table_style)

dash_html_layout = html.Div(
    id='main', children=[
        html.Div(id='header', children=[

            html.Div(id='logo', children=[
                html.Img(src=merlin_icon_img),
                html.H1(children='fronted POC for testing ProjectName'),
                html.P('BETA')
            ]),
            html.Button("Send", id="send-button-id", disabled=True)
        ]),
        html.Div(id='content', children=[
            html.Div(id='content-1', children=[
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.P('Drag and drop CNC or browse')]
                    ),
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id='uploaded-file', children=[
                    html.Div(id='output-data-upload')
                ]),
                html.Button('Find', id='find-button-id', n_clicks=0, disabled=True),
            ]),

            html.Div(id='result-div', children=[
                html.P('Select the best option'),
                html.Div(id='result-content-div', children=[
                    html.Div(id='animation_area', children=[
                        html.Div(id='animation', children=[]),
                    ]),
                    html.Img(id='star', src=star_img, style=star_style),
                    html.Div(id='data-table'),
                    check_icons_table,
                ]),
                html.Div(id='comment-text'),
            ])
        ])
    ])


def dash_app_layout() -> html.Div:
    return dash_html_layout


def configure_dash_app(services_url: str) -> dash.Dash:
    global machine_config_service_url, feedback_service_url

    machine_config_service_url = services_url + 'dash-model-predict'
    feedback_service_url = services_url + 'feedback'

    dash_app = dash.Dash(__name__,
                         server=server,
                         suppress_callback_exceptions=True,
                         url_base_pathname='/frontend-service-dev/',
                         assets_folder=ASSETS_FOLDER)

    dash_app.layout = dash_app_layout()
    register_dash_callback(dash_app, update_output)
    register_dash_animation_callback(dash_app, animation_output)
    register_dash_confirm_callback(dash_app, confirm_output)
    register_dash_comment_update_callback(dash_app, update_comment)
    return dash_app


def update_comment(value) -> str:
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'comment-textarea' in changed_id:
        global user_comment_str
        user_comment_str = value

    return value


def get_style_data_conditional(selected_rows: list = []) -> list:
    return [
        {
            'if': {'row_index': selected_rows},
            'backgroundColor': 'rgba(0,145,143,0.05)',
            'fontWeight': 'bold',
            'color': 'black',
        }
    ]


def confirm_output(t0, t1, t2, t3, t4, cell, nc):
    cond_style = get_style_data_conditional(cell['row'])
    check_styles = [t0, t1, t2, t3, t4]
    selected_style = check_style.copy()
    selected_style['filter'] = 'grayscale(0%)'
    selected_style['opacity'] = '1'
    table_style['visibility'] = 'visible'
    for i in range(0, len(check_styles)):
        if i == cell['row']:
            check_styles[i] = selected_style
        else:
            check_styles[i] = check_style
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'send-button-id' in changed_id:
        if cell['row'] == 0:
            validation = True
        else:
            validation = False

        global user_comment_str
        feedback_json = {"id": smart_id, "validation": validation, "comments": user_comment_str}

        check_styles[cell['row']]['filter'] = 'grayscale(100%)'
        check_styles[cell['row']]['opacity'] = '0.4'

        print(feedback_json)
        requests.post(feedback_service_url, feedback_json)
        table_style['visibility'] = 'hidden'
        user_comment_str = ''

    return check_icons_table.children, check_styles[0], check_styles[1], check_styles[2], check_styles[3], \
        check_styles[4], cond_style, star_style


def animation_output(list_of_contents: list, list_of_names: list, n_clicks: any) -> (list, str):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    anima_children = []
    if 'find-button-id' in changed_id:
        anima_children = html.Div(de.Lottie(options=animation_options, width="50%", height="50%", url=animation_img))

    return anima_children


def update_output(list_of_contents: list, list_of_names: list, n_cli_button: any, n_cli_update: any) -> (list, str):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    children = []
    anim_children = html.Div(id='animation', children=[])
    message = ''
    comment = []
    disabled = True
    disabled_update = True
    button_style = {}
    style = button_style.copy()
    style_update = button_style.copy()

    if list_of_contents is not None:
        children.append(list_of_names[0])
        [children.append(parse_contents(c)) for c, n in zip(list_of_contents, list_of_names)]
        disabled = False
        style['background-color'] = '#00918f'
        table_style['visibility'] = 'hidden'

    if 'find-button-id' in changed_id and children:
        content_type, content_string = list_of_contents[0].split(',')
        response = requests.post(machine_config_service_url, data={'': content_string})
        jason = json.loads(response.text)
        global smart_id
        smart_id = jason.pop('id')
        message = load_result_table(jason)
        anim_children = html.Div(id='animation', children=[])
        comment.append(html.P('Write your comment here'))
        comment.append(dcc.Textarea(
            id='comment-textarea',
            placeholder='Add comment...'
        ))
        table_style['visibility'] = 'visible'
        style_update['background-color'] = '#00918f'
        disabled_update = False

    if 'send-button-id' in changed_id:
        table_style['visibility'] = 'hidden'
        style_update['background-color'] = '#dddddd'
        style['background-color'] = '#dddddd'
        comment = []
        children = []
    return children, message, disabled, style, style_update, anim_children, comment, disabled_update, table_style


def register_dash_callback(dash_app: dash.Dash, function: object):
    dash_app.callback(Output('output-data-upload', 'children'),
                      Output('data-table', 'children'),
                      Output('find-button-id', 'disabled'),
                      Output('find-button-id', 'style'),
                      Output('send-button-id', 'style'),
                      Output('animation_area', 'children'),
                      Output('comment-text', 'children'),
                      Output('send-button-id', 'disabled'),
                      Output('check-icons-table-id', 'style'),
                      Input('upload-data', 'contents'),
                      State('upload-data', 'filename'),
                      Input('find-button-id', 'n_clicks'),
                      Input('send-button-id', 'n_clicks'),
                      )(function)


def register_dash_animation_callback(dash_app: dash.Dash, function: object):
    dash_app.callback(Output('animation', 'children'),
                      Input('upload-data', 'contents'),
                      State('upload-data', 'filename'),
                      Input('find-button-id', 'n_clicks'))(function)


def register_dash_confirm_callback(dash_app: dash.Dash, function: object):
    dash_app.callback(Output('check-icons-table-id', 'children'),
                      Output('0check', 'style'),
                      Output('1check', 'style'),
                      Output('2check', 'style'),
                      Output('3check', 'style'),
                      Output('4check', 'style'),
                      Output('data_table_info', 'style_data_conditional'),
                      Output('star', 'style'),
                      Input('0check', 'style'),
                      Input('1check', 'style'),
                      Input('2check', 'style'),
                      Input('3check', 'style'),
                      Input('4check', 'style'),
                      Input('data_table_info', 'active_cell'),
                      Input('send-button-id', 'n_clicks')
                      )(function)


def register_dash_comment_update_callback(dash_app: dash.Dash, function: object):
    dash_app.callback(
        Output('comment-textarea', 'value'),
        Input('comment-textarea', 'value'))(function)


def parse_contents(contents: str) -> html.Div:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        data = str(decoded, 'utf-8')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div(id='read-data', children=[
        dcc.Textarea(
            id='textarea',
            value=data
        )
    ])


def load_result_table(jason: dict) -> html.Div:
    jason = {key: eval(value) for key, value in jason.items()}
    lista = [{key: jason[key][i] for key in jason.keys()} for i in jason['machine'].keys()]
    global final_table
    final_table = dash_table.DataTable(
        id='data_table_info',
        data=lista,
        columns=[{'name': i, 'id': i} for i in jason.keys()],
        style_cell={'textAlign': 'center'},
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        active_cell={'row': 0, 'column': 0, 'column_id': 'metric'},
        fill_width=False
    )
    result_table = html.Div(id='machine-config-result', style={}, children=[final_table])
    return result_table


def crete_service_url() -> dash.Dash:
    e = Env()
    services_url = f'http://127.0.0.1:{e.port}/services/'
    print(services_url)
    return configure_dash_app(services_url)


def main():
    app.run_server(debug=False)


app = crete_service_url()

if __name__ == '__main__':  # pragma: no cover
    main()
