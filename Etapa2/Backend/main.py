import itertools
import re

import joblib
import numpy as np
from flask import Flask, Response, request, jsonify, abort
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Iterable, Tuple, List, Set

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

app: Flask = Flask('backend-predictions-ODS')


class ODSInsightPreprocessor(BaseEstimator, TransformerMixin):
    stopwords: Set[str]
    lemmatizer: WordNetLemmatizer

    def __init__(self) -> None:
        self.stopwords = set(stopwords.words('spanish'))
        self.lemmatizer = WordNetLemmatizer()

    # Speed improved
    def transform(self, iterable: Iterable[str]) -> List[str]:
        return [' '.join(map(self.lemmatizer.lemmatize,
                             itertools.filterfalse(self.stopwords.__contains__, re.sub(r'\d+', '', text).split())))
                for text in iterable]


pipeline: Pipeline = joblib.load('ODSInsightModel.joblib')
encoder: LabelEncoder = joblib.load('ODSInsightEncoder.joblib')

def training(dataset: DataFrame) -> Tuple[float, float, float]:
    x: Series = dataset['spanish-texts']
    y: Series = dataset['sdg']

    x_transformed: csr_matrix[str] = pipeline.named_steps['tfidf'].transform(pipeline.named_steps['preprocessor'].transform(x))
    y_transformed: NDArray[int] = encoder.transform(y)

    try:
        pipeline.named_steps['model'].partial_fit(x_transformed, y_transformed, classes=np.unique(y_transformed))
    except ValueError:
        pipeline.named_steps['model'].fit(x_transformed, y_transformed)

    joblib.dump(pipeline, 'ODSInsightModel.joblib')

    y_predicted: NDArray[int] = pipeline.named_steps['model'].predict(x_transformed)
    precision: float = precision_score(y_transformed, y_predicted, average='weighted')
    recall: float = recall_score(y_transformed, y_predicted, average='weighted')
    f1: float = f1_score(y_transformed, y_predicted, average='weighted')
    return precision, recall, f1


@app.route('/predict', methods=['POST'])
def predict():
    # Handle the POST request
    data = request.json['data']
    df = DataFrame(data)

    # Preprocess the input data
    preprocessed_data = pipeline['preprocessor'].transform(df['spanish-texts'])

    # Get predictions
    predictions = pipeline.predict(preprocessed_data)

    # Get probabilities of the predictions
    predictions_proba = pipeline.predict_proba(preprocessed_data)

    # Get the highest probability for each prediction and round it to two decimal places
    predictions_probabilities = predictions_proba.max(axis=1).round(2)

    # Decode the predictions (convert from 0, 1, 2 to 3, 4, 5)
    decoded_predictions = encoder.inverse_transform(predictions)

    return jsonify({
        'predictions': [
            {
                'spanish-texts': text,
                'sdg': pred,
                'probability': prob
            } for text, pred, prob in zip(df['spanish-texts'],  decoded_predictions.tolist() , predictions_probabilities.tolist())]
    })



@app.route('/retrain', methods=['POST'])
def retrain() -> Response:
    try:
        precision, recall, f1 = training(DataFrame(request.json['data']))
        return jsonify({
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    except Exception as a:  # NOQA: Too broad exception clause.
        print(a)
        abort(500)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
