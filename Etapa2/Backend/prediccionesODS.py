from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import io
from num2words import num2words
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

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
    
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Definir la función para entrenar el modelo
def entrenar_modelo(df):
    X = df['Textos_espanol']  # Características de entrada
    y = df['sdg']  # Etiquetas
    
    # Codificación de etiquetas
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Crear el pipeline de preprocesamiento y clasificación
    pipeline = Pipeline([
    ('preprocessor', CustomPreprocessor()),  # Preprocesamiento personalizado
    ('tfidf', TfidfVectorizer(max_features=3000)),  # Vectorización TF-IDF
    ('model', MultinomialNB())  # Modelo Naive Bayes
], memory="cachedir")
    
    # Dividir los datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    # Entrenar el modelo
    pipeline.fit(x_train, y_train)

    # Guardar el modelo y el codificador
    joblib.dump(pipeline, 'ModeloODS.joblib')
    joblib.dump(encoder, 'Encoder.joblib')

    # Calcular las métricas de desempeño
    y_pred = pipeline.predict(x_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Devolver las métricas de desempeño
    return precision, recall, f1

# Cargar el modelo y el encoder
pipeline = joblib.load('ModeloODS.joblib')
encoder = joblib.load('Encoder.joblib')

# Endpoint para predicciones
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({"message": "Use POST method to predict."})

    # Resto del código para manejar POST
    data = request.json['data']
    df = pd.DataFrame(data)
    preprocessed_data = pipeline['preprocessor'].transform(df['Textos_espanol'])
    predictions = pipeline.predict(preprocessed_data)
    decoded_predictions = encoder.inverse_transform(predictions)
    
    return jsonify({'predictions': decoded_predictions.tolist()})


@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.json['data']  # Asegúrate de que el formato de datos sea correcto
    df = pd.DataFrame(data)
    
    # Llamar a la función de reentrenamiento
    precision, recall, f1 = entrenar_modelo(df)

    # Devolver las métricas de desempeño
    return jsonify({
        'message': 'Modelo reentrenado con éxito',
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

if __name__ == '__main__':
    app.run(debug=True)
