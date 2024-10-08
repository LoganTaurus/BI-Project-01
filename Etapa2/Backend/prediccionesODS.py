from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

app = Flask(__name__)

# Definir la clase CustomPreprocessor
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words('spanish'))
        self.lemmatizer = WordNetLemmatizer()
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        cleaned_data = []
        for text in x:
            text = re.sub(r'\d+', '', text)  # Remove digits
            text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stopwords])
            cleaned_data.append(text)
        return cleaned_data
     

# Definir la función para ajustar el modelo existente
def ajustar_modelo(df):
    X = df['Textos_espanol']  # Características de entrada
    y = df['sdg']  # Etiquetas

    # Codificar las nuevas etiquetas con el mismo encoder
    y_encoded = encoder.transform(y)

    # Preprocesar los textos utilizando el preprocesador del pipeline
    preprocessed_texts = pipeline.named_steps['preprocessor'].transform(X)

    # Convertir los textos preprocesados en vectores numéricos usando TfidfVectorizer
    vectorized_texts = pipeline.named_steps['tfidf'].transform(preprocessed_texts)

    # Ajustar el modelo existente parcialmente con los nuevos datos
    pipeline.named_steps['model'].partial_fit(vectorized_texts, y_encoded, classes=np.unique(y_encoded))

    # Guardar el modelo actualizado
    joblib.dump(pipeline, 'ModeloODS.joblib')

    # Calcular las métricas de desempeño en el conjunto de datos actualizado
    y_pred = pipeline.named_steps['model'].predict(vectorized_texts)
    precision = precision_score(y_encoded, y_pred, average='weighted')
    recall = recall_score(y_encoded, y_pred, average='weighted')
    f1 = f1_score(y_encoded, y_pred, average='weighted')

    return precision, recall, f1




# Cargar el modelo y el encoder
pipeline = joblib.load('ModeloODS.joblib')
encoder = joblib.load('Encoder.joblib')

# Endpoint para predicciones
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({"message": "Use POST method to predict."})

    # Manejar la solicitud POST
    data = request.json['data']
    df = pd.DataFrame(data)
    
    # Preprocesar los datos de entrada
    preprocessed_data = pipeline['preprocessor'].transform(df['Textos_espanol'])
    
    # Obtener las predicciones
    predictions = pipeline.predict(preprocessed_data)
    
    # Obtener las probabilidades de las predicciones
    predictions_proba = pipeline.predict_proba(preprocessed_data)
    
    # Obtener la probabilidad más alta para cada predicción y redondearla a dos decimales
    predictions_probabilities = predictions_proba.max(axis=1).round(2)
    
    # Decodificar las predicciones (convertir de 0, 1, 2 a 3, 4, 5)
    decoded_predictions = encoder.inverse_transform(predictions)
    
    # Convertir a tipos de datos serializables por JSON
    decoded_predictions = decoded_predictions.tolist()  # Convertir a lista
    predictions_probabilities = predictions_probabilities.tolist()  # Convertir a lista
    
    # Preparar el resultado con las predicciones, probabilidades y el texto original
    result = [{'Textos_espanol': text, 'sdg': pred, 'probabilidad': prob} for text, pred, prob in zip(df['Textos_espanol'], decoded_predictions, predictions_probabilities)]
    
    # Devolver el resultado como JSON
    return jsonify({'predictions': result})


@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.json['data']  # Asegúrate de que el formato de datos sea correcto
    df = pd.DataFrame(data)
    
    # Llamar a la función de reentrenamiento
    precision, recall, f1 = ajustar_modelo(df)

    # Devolver las métricas de desempeño
    return jsonify({
        'message': 'Modelo reentrenado con éxito',
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

if __name__ == '__main__':
    app.run(debug=True)
