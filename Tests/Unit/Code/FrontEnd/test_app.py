import base64
import os

from dash import dash, html
from unittest import mock
from Code.FrontEnd.app import parse_contents, load_result_table, dash_app_layout, main, \
    register_dash_callback, update_output, configure_dash_app, update_comment, get_style_data_conditional, \
    confirm_output, animation_output, register_dash_animation_callback, register_dash_confirm_callback, \
    register_dash_comment_update_callback

TEST_CNC_FILE = 'Fixtures/0000.GCD'

cell = {'row': 0, 'column': 1, 'column_id': 'machine'}
style = {'filter': 'grayscale(100%)', 'opacity': '0.4'}
c_styles = [style, style, style, style, style]

cwd = os.path.dirname(os.path.abspath(__file__))
path_comp = os.path.join(cwd, TEST_CNC_FILE)


class ResponseFixture:
    def __init__(self, text):
        self.text = text


@mock.patch('Code.FrontEnd.app.dash_html_layout', html.Div())
def test_dash_app_layout():
    return_layout = dash_app_layout()
    assert str(return_layout) == str(html.Div())


@mock.patch('Code.FrontEnd.app.register_dash_comment_update_callback')
@mock.patch('Code.FrontEnd.app.register_dash_confirm_callback')
@mock.patch('Code.FrontEnd.app.register_dash_animation_callback')
@mock.patch('Code.FrontEnd.app.register_dash_callback')
@mock.patch('Code.FrontEnd.app.dash_app_layout')
@mock.patch('Code.FrontEnd.app.dash.Dash')
def test_configure_dash_app(mock_app_dash: mock,
                            mock_dash_app_layout: mock,
                            mock_register_dash_callback: mock,
                            mock_register_dash_animation_callback: mock,
                            mock_register_dash_confirm_callback: mock,
                            mock_register_dash_comment_update_callback: mock,):
    services_url = 'http://127.0.0.1:8080/services/'
    configure_dash_app(services_url)
    mock_app_dash.assert_called()
    mock_dash_app_layout.assert_called()
    mock_register_dash_callback.assert_called()
    mock_register_dash_animation_callback.assert_called()
    mock_register_dash_confirm_callback.assert_called()
    mock_register_dash_comment_update_callback.assert_called()


@mock.patch('Code.FrontEnd.app.dash.Dash')
def test_register_dash_callback(mock_dash_app: mock):
    register_dash_callback(mock_dash_app, update_output)
    mock_dash_app.callback.assert_called()


@mock.patch('Code.FrontEnd.app.dash.Dash')
def test_register_dash_animation_callback(mock_dash_app: mock):
    register_dash_animation_callback(mock_dash_app, animation_output)
    mock_dash_app.callback.assert_called()


@mock.patch('Code.FrontEnd.app.dash.Dash')
def test_register_dash_confirm_callback(mock_dash_app: mock):
    register_dash_confirm_callback(mock_dash_app, confirm_output)
    mock_dash_app.callback.assert_called()


@mock.patch('Code.FrontEnd.app.dash.Dash')
def test_register_dash_comment_update_callback(mock_dash_app: mock):
    register_dash_comment_update_callback(mock_dash_app, update_comment)
    mock_dash_app.callback.assert_called()


@mock.patch('Code.FrontEnd.app.get_style_data_conditional')
@mock.patch('Code.FrontEnd.app.dash.callback_context')
def test_confirm_output(mock_dash_app_context: mock,
                        mock_get_style_data_conditional: mock):
    mock_get_style_data_conditional.return_value = [{'if': {'row_index': 0}, 'backgroundColor': '#dddddd',
                                                     'fontWeight': 'bold', 'color': 'black'}]
    mock_dash_app_context.triggered = [{'prop_id': []}]

    (table_child, c_styles[0], c_styles[1], c_styles[2], c_styles[3], c_styles[4], cond_style, star_style) = \
        confirm_output(c_styles[0], c_styles[1], c_styles[2], c_styles[3], c_styles[4], cell, 0)
    assert c_styles[0] == {'filter': 'grayscale(0%)', 'opacity': '1'}
    for i in range(1, len(c_styles)):
        assert c_styles[i] == style


@mock.patch('Code.FrontEnd.app.requests.post')
@mock.patch('Code.FrontEnd.app.get_style_data_conditional')
@mock.patch('Code.FrontEnd.app.dash.callback_context')
def test_confirm_output_send_feedback(mock_dash_app_context: mock,
                                      mock_get_style_data_conditional: mock,
                                      mock_requests_post: mock):
    response_fixture = '{"id": 111}'
    mock_requests_post.return_value = ResponseFixture(response_fixture)
    mock_dash_app_context.triggered = [{'prop_id': ['send-button-id']}]
    mock_get_style_data_conditional.return_value = [
        {'if': {'row_index': 0}, 'backgroundColor': '#dddddd', 'fontWeight': 'bold', 'color': 'black'}]

    (table_child, c_styles[0], c_styles[1], c_styles[2], c_styles[3], c_styles[4], cond_style, star_style) = \
        confirm_output(c_styles[0], c_styles[1], c_styles[2], c_styles[3], c_styles[4], cell, 0)
    for i in range(0, len(c_styles)):
        assert c_styles[i] == style
    mock_requests_post.assert_called()
    assert star_style == {'display': 'none', 'height': '25px', 'margin-top': '43px', 'width': '25px'}


@mock.patch('Code.FrontEnd.app.load_result_table')
@mock.patch('Code.FrontEnd.app.requests.post')
@mock.patch('Code.FrontEnd.app.parse_contents')
@mock.patch('Code.FrontEnd.app.dash.callback_context')
def test_register_dash_update_list_of_contents(mock_dash_app_context: mock,
                                               mock_parse_contents: mock,
                                               mock_requests_post: mock,
                                               mock_load_result_table: mock):

    normal_button = {'background-color': '#00918f'}
    response_fixture = '{"id": 111}'
    list_of_contents = ['context_1, connection string test']
    list_of_names = ['contex_name_1']
    mock_dash_app_context.triggered = [{'prop_id': ['find-button-id']}]
    mock_parse_contents.return_value = 'find-button-id'
    mock_requests_post.return_value = ResponseFixture(response_fixture)
    mock_load_result_table.return_value = 'expected message'
    (children, message, disabled, style, style_update, anim_children, comment, disabled_update, table_style) = \
        update_output(list_of_contents, list_of_names, 0, 0)
    mock_requests_post.assert_called()
    assert message == 'expected message'
    assert not disabled
    assert style == normal_button
    assert style_update == normal_button
    assert not disabled_update
    assert table_style == {'visibility': 'visible'}


@mock.patch('Code.FrontEnd.app.de.Lottie')
@mock.patch('Code.FrontEnd.app.dash.callback_context')
def test_animation_output(mock_dash_app_context: mock, mock_lottie: mock):
    mock_dash_app_context.triggered = [{'prop_id': ['find-button-id']}]
    mock_lottie.return_value = []
    child = animation_output([], [], 0)
    assert type(child) == dash.html.Div
    mock_lottie.assert_called()


@mock.patch('Code.FrontEnd.app.dash.callback_context')
def test_register_dash_update_button_pressed(mock_dash_app_context: mock):
    disabled_button = {'background-color': '#dddddd'}
    list_of_contents = ['context_1, connection string test']
    list_of_names = ['contex_name_1']
    mock_dash_app_context.triggered = [{'prop_id': ['send-button-id']}]
    (children, message, disabled, style, style_update, anim_children, comment, disabled_update, table_style) = \
        update_output(list_of_contents, list_of_names, 0, 0)
    assert not disabled
    assert style == disabled_button
    assert style_update == disabled_button
    assert disabled_update
    assert table_style == {'visibility': 'hidden'}


def test_parse_contents():
    file_read = open(path_comp, 'r', encoding='ISO-8859-1').read()
    coded = file_read.encode("ascii")
    file_content = base64.b64encode(coded)
    file_string = str(file_content)
    file_string = file_string[2:566]
    contents = 'data:application/octet-stream;base64,'
    list_contents = contents + file_string
    read = parse_contents(list_contents)
    assert type(read) == dash.html.Div


@mock.patch('Code.FrontEnd.app.base64.b64decode')
def test_parse_contents_exception(mock_b64decode: mock):
    mock_b64decode.return_value = None
    return_expected = html.Div(['There was an error processing this file.'])
    list_contents = 'a,b'
    return_real = parse_contents(list_contents)
    assert str(return_real) == str(return_expected)


def test_load_result_table():
    diccionario = {'file': '{"0":"file1"}',
                   'machine': '{"0":"machine1"}',
                   'metric': '{"0":1.0}',
                   'part': '{"0":"pieza1"}',
                   'post': '{"0":"post1"}'}
    result = load_result_table(diccionario)
    assert type(result) == dash.html.Div
    assert type(result.children[0].data[0]) == dict


def test_get_style_data_conditional():
    con_style = get_style_data_conditional([0])[0]
    assert con_style['if'] == {'row_index': [0]}
    assert con_style['backgroundColor'] == 'rgba(0,145,143,0.05)'
    assert con_style['fontWeight'] == 'bold'
    assert con_style['color'] == 'black'

