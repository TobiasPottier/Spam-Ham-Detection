from flask import Flask, render_template, request
from joblib import load
import os
import logging
from logging.handlers import RotatingFileHandler
from preprocessing import text_preprocessing
import re

base_path = os.path.dirname(os.path.abspath(__file__))

NBclassifier = load(os.path.join(base_path, './models/NBclassifier.joblib'))
lregclassifier = load(os.path.join(base_path, './models/lregclassifier.joblib'))
count_vect = load(os.path.join(base_path, './models/count_vect.joblib'))
tf_transformer = load(os.path.join(base_path, './models/tf_transformer.joblib'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        sentence = request.form['sentence']
        pattern = r'\bbias\b$'
        result = re.search(pattern, sentence)
        sentence = re.sub(pattern, '', sentence)
        app.logger.debug("Received sentence: %s", sentence)
        processed_text = text_preprocessing(sentence, 'english', 2)
        
        test_bow = count_vect.transform([processed_text])
        test_tfidf = tf_transformer.transform(test_bow)
        
        nb_result = NBclassifier.predict(test_tfidf)
        lr_result = lregclassifier.predict(test_tfidf)

        NB_confidence = NBclassifier.predict_proba(test_tfidf)[0][0]
        LR_confidence = lregclassifier.predict_proba(test_tfidf)[0][0]
        if (bool(result)):
            final_result = nb_result[0] if abs(NB_confidence - 0.5) > abs(LR_confidence - 0.5) else lr_result[0]
            final_confidence = NB_confidence if abs(NB_confidence - 0.5) > abs(LR_confidence - 0.5) else LR_confidence
            model = 'Naive Bayes' if abs(NB_confidence - 0.5) > abs(LR_confidence - 0.5) else 'Logistic Regression'
            return render_template('index.html', message=f'{final_result} | confidence: {round(final_confidence, 2)} | model: {model} ')

        else:
            final_result = nb_result[0] if abs(NB_confidence - 0.5) > abs(LR_confidence - 0.5) else lr_result[0]
            return render_template('index.html', message=f'{final_result}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
