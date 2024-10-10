from io import StringIO, BytesIO
from typing import TypedDict

import pandas as pd
import requests
from flask import Flask, render_template, send_file, request, abort
from pandas import DataFrame
from requests import Response
from werkzeug.datastructures import FileStorage

TrainingJSON: type['TrainingJSON'] = TypedDict('TrainingJSON', {'spanish-texts': str, 'sdg': int})
TrainingInputJSON: type['TrainingInputJSON'] = TypedDict('TrainingInputJSON', {'data': list[TrainingJSON]})

TrainingOutputJSON: type['TrainingOutputJSON'] = TypedDict('TrainingOutputJSON', {'precision': float, 'recall': float, 'f1': float})

AnalyticsJSON: type['AnalyticsJSON'] = TypedDict('AnalyticsJSON', {'spanish-texts': str}) # ''
AnalyticsInputJSON: type['AnalyticsInputJSON'] = TypedDict('AnalyticsInputJSON', {'data': list[AnalyticsJSON]}) # ''

AnalyticsPredictionJSON: type['AnalyticsPredictionJSON'] = TypedDict('AnalyticsPredictionJSON', {'spanish-texts': str, 'sdg': int, 'probability': float})
AnalyticsOutputJSON: type['AnalyticsOutputJSON'] = TypedDict('AnalyticsOutputJSON', {'predictions': list[AnalyticsPredictionJSON]})


UNSAFE_CHAINING_DATAFRAME: DataFrame | None = None

app: Flask = Flask('frontend-predictions-ODS')


@app.errorhandler(404)
def notfound(error):
    return render_template('notfound.html', active_page='notfound'), 404


@app.route('/')
def index() -> str:
    return render_template(r'index.html', active_page='index')


@app.route('/analytics', methods=['GET', 'POST'])
def analytics() -> str:
    global UNSAFE_CHAINING_DATAFRAME

    if request.method == 'GET':
        return render_template('analytics.html', results=None, active_page='analytics', section='text-input')

    advanced: str = request.form.get('advanced-mode', 'off')
    section: str = request.form['section']  # Cache the current section

    if 'opinionText' in request.form:  # Changed to match input name
        opinion: str = request.form['opinionText']
        if not opinion:  # Empty opinions are not allowed
            return render_template('analytics.html', results=None, message='No hay texto para analizar', active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted
        json: AnalyticsInputJSON = {
            'data': [
                {
                    'spanish-texts': opinion
                }
            ]
        }
    elif 'opinions-file' in request.files:  # Ensure the correct input name is used
        source: FileStorage = request.files['opinions-file']
        if not source:
            return render_template('analytics.html', results=None, message='No se detectaron archivos de entrada', active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted

        try:
            if source.filename.endswith('.csv'):
                dataframe: DataFrame = pd.read_csv(source.stream)
            elif source.filename.endswith(('.xlsx', '.xls')):
                dataframe: DataFrame = pd.read_excel(source.stream)
            elif source.filename.endswith('.json'):
                dataframe: DataFrame = pd.read_json(source.stream)
            else:
                return render_template('analytics.html', results=None, message='Extensión de archivo no válida detectada', active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted
        except Exception:  # NOQA: Too broad exception clause.
            return render_template('analytics.html', results=None, message='Formato de archivo no válido detectado',  active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted

        if {'spanish-texts'}.difference(dataframe.columns):
            return render_template('analytics.html', results=None, message='Los datos para análisis solo pueden contener la columna "spanish-texts"', active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted

        json: AnalyticsInputJSON = {
            'data': [
                {
                    'spanish-texts': opinion
                } for opinion in dataframe['spanish-texts']
            ]
        }
    else:
        abort(501)  # Return a 501 error if neither input is detected

    response: Response = requests.post('http://127.0.0.1:8080/predict', json=json)
    if response.status_code != 200:
        return render_template('analytics.html', results=None, message='Ocurrió un error durante la predicción', active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted
    results: AnalyticsOutputJSON = response.json()
    for prediction in results['predictions']:
        prediction['probability'] = round(prediction['probability'] * 100, 2)
    UNSAFE_CHAINING_DATAFRAME = DataFrame(results['predictions'], columns=['spanish-texts', 'sdg', 'probability'] if advanced == 'on' else ['spanish-texts', 'sdg'])
    return render_template('analytics.html', results=results, message='Predicción exitosa', active_page='analytics', section=section, advanced=advanced)  # NOQA: Spanish grammar highlighted


@app.route('/analytics/training', methods=['GET', 'POST'])
def analytics_training() -> str:
    if request.method == 'GET':
        return render_template(r'analytics-training.html', results=None, active_page='analytics-training', advanced='off')

    advanced: str = request.form.get('advanced-mode', 'off')
    source: FileStorage = request.files['training-file']

    if not source:
        return render_template(r'analytics-training.html', results=None, message='No se detectaron archivos de entrada', active_page='analytics-training', advanced=advanced)  # NOQA: Spanish grammar highlighted
    try:
        if source.filename.endswith('.csv'):
            dataframe: DataFrame = pd.read_csv(source.stream)
        elif source.filename.endswith(('.xlsx', '.xls')):
            dataframe: DataFrame = pd.read_excel(source.stream)
        elif source.filename.endswith('.json'):
            dataframe: DataFrame = pd.read_json(source.stream)
        else:
            return render_template(r'analytics-training.html', results=None, message='Extensión de archivo no válida detectada', active_page='analytics-training', advanced=advanced)  # NOQA: Spanish grammar highlighted
    except Exception:  # NOQA: Too broad exception clause.
        return render_template(r'analytics-training.html', results=None, message='Formato de archivo no válido detectado', active_page='analytics-training', advanced=advanced)  # NOQA: Spanish grammar highlighted
    if {'spanish-texts', 'sdg'}.difference(dataframe.columns):
        return render_template(r'analytics-training.html', results=None, message='Los datos de entrenamiento solo pueden contener las columnas "spanish-texts" y "sdg"', active_page='analytics-training', advanced=advanced)  # NOQA: Spanish grammar highlighted

    json: TrainingInputJSON = {
        'data': [
            {
                'spanish-texts': opinion,
                'sdg': sdg
            } for opinion, sdg in zip(dataframe['spanish-texts'], dataframe['sdg'])
        ]
    }
    response: Response = requests.post('http://127.0.0.1:8080/retrain', json=json)
    if response.status_code != 200:
        return render_template(r'analytics-training.html', results=None, message='Ocurrió un error durante el entrenamiento', active_page='analytics-training', advanced=advanced)  # NOQA: Spanish grammar highlighted
    results: TrainingOutputJSON = response.json()
    results['precision'] = round(results['precision'] * 100, 2)
    results['recall'] = round(results['recall'] * 100, 2)
    results['f1'] = round(results['f1'] * 100, 2)
    return render_template(r'analytics-training.html', results=results, message='Entrenamiento del modelo completado con éxito', active_page='analytics-training', advanced=advanced)  # NOQA: Spanish grammar highlighted


@app.route('/analytics/download', methods=['GET'])
def download():
    global UNSAFE_CHAINING_DATAFRAME
    if UNSAFE_CHAINING_DATAFRAME is None:
        abort(501)

    extension = request.args.get('extension', 'csv')

    stream = StringIO()
    filename = f'predictions.{extension}'

    match extension:
        case 'csv':
            UNSAFE_CHAINING_DATAFRAME.to_csv(stream, index=False)
            stream.seek(0)
            return send_file(BytesIO(stream.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=filename)
        case 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                UNSAFE_CHAINING_DATAFRAME.to_excel(writer, index=False)
            output.seek(0)
            return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name=filename)
        case 'json':
            json_data = UNSAFE_CHAINING_DATAFRAME.to_json(orient='records')
            return send_file(BytesIO(json_data.encode()), mimetype='application/json', as_attachment=True, download_name=filename)
        case _:
            abort(403)


@app.route('/about')
def about() -> str:
    return render_template(r'about.html', active_page='about')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
