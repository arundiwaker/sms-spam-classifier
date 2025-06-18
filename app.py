import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps=PorterStemmer()

def transform_text(text):
    # to convert entire text in lower case
    text = text.lower()

    # to split the text into words
    text = nltk.word_tokenize(text)

    # to remove symbols/special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # to remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # to remove suffix such as 'ing','tion', 'ed' &c...
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    # finally returning the string of remaing text
    return ' '.join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')
input_sms=st.text_area('Enter the message...')

if st.button('Predict'):

    #1.preprocess
    transform_sms=transform_text(input_sms)
    #2.vectorize
    vector_input=tfidf.transform([transform_sms])
    #3.predict
    result=model.predict(vector_input)[0]
    #4.display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')

