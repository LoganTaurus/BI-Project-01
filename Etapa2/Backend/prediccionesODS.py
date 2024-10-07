from flask import Flask, request, jsonify
import joblib
import pandas as pd
import io
from num2words import num2words
import unicodedata
import re
import stanza
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load("ModeloODS.joblib")

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return self.customPreprocessing(x)

    def replace_numbers(self, words):
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = num2words(word, lang='es')
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_nonlatin(self, words):
        new_words = []
        for word in words:
            new_word = ''.join([ch for ch in word if unicodedata.name(ch).startswith(('LATIN', 'DIGIT', 'SPACE'))])
            new_words.append(new_word)
        return new_words

    def remove_stopwords(self, words):
        stop_words = set(stopwords.words('spanish'))
        return [word for word in words if word not in stop_words]

    def remove_punctuation(self, words):
        return re.sub(r'[^\w\s]', ' ', words)

    def tokenLemma(self, text):
        # Verificación para evitar advertencias de GPU si no está disponible
        try:
            nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', use_gpu=False)  # Establecer use_gpu=False
        except Exception as e:
            print(f"Error initializing stanza pipeline: {e}")
            nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma', use_gpu=False)
        return nlp(text)

    def customPreprocessing(self, data):
        cleaned_data = []
        for text in data['Textos_espanol']:
            text = self.remove_punctuation(text)
            doc = self.tokenLemma(text)
            for sentence in doc.sentences:
                words = [word.lemma.lower() for word in sentence.words if word.pos not in ('PUNCT', 'SYM')]
                words = self.remove_nonlatin(words)
                words = self.replace_numbers(words)
                words = self.remove_stopwords(words)
                cleaned_data.append(' '.join(words))
        return cleaned_data

preprocessor = CustomPreprocessor()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Archivo CSV no encontrado"}), 400
    
    file = request.files['file']
    if file.filename.endswith(".csv"):
        # Leer el CSV
        csv_data = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
        
        # Preprocesar
        preprocessed_text = preprocessor.customPreprocessing(csv_data)

        # Hacer predicciones
        predictions = model.predict(preprocessed_text)
        
        return jsonify({"predictions": predictions.tolist()})
    else:
        return jsonify({"error": "Formato de archivo no soportado. Solo CSV."}), 400

if __name__ == '__main__':
    app.run(port=5000)
