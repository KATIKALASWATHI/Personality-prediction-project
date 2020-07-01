import flask
import pickle
import re
import nltk
nltk.download('all')
nltk.download()


from flask import Flask,render_template,url_for,request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
from nltk.corpus import stopwords
stop=stopwords.words('english')

import os
os.getcwd()
os.chdir( 'C:\\Users\\Swathi\\Desktop\\NAOP project')
vectorizer=pickle.load(open('Transform.pkl','rb'))
LR=pickle.load(open('Model.pkl', 'rb'))


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    
    if request.method=='POST':
        message=request.form['message']
        data=message.lower()
        data=re.sub('[^A-Za-z]', ' ', data)
        lemmas=[' '.join(wnl.lemmatize(w) for w in data.split() if w not in stop)]
        vect=vectorizer.transform(lemmas)
        my_prediction=LR.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True, use_reloader=False)
