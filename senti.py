from flask import Flask,render_template,url_for,request
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
model= pickle.load(open('model.pkl','rb'))
tfidf= pickle.load(open('tfidf.pkl','rb'))



app=Flask(__name__)
def stemmed_content(comment):
    sc=re.sub('[^a-zA-z]',' ',comment)
    sc=sc.lower()
    sc=sc.split()
    port=PorterStemmer()
    sc=[port.stem(words) for words in sc if not words in stopwords.words('english')]
    sc=' '.join(sc)
    
    return sc

@app.route("/")
def home():
    return render_template("frontend.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        comment = request.form['comment']
        cleaned_comment=stemmed_content(comment)
        comment_vector=tfidf.transform([cleaned_comment])
        prediction=model.predict(comment_vector)[0]

        return render_template("frontend.html",prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
