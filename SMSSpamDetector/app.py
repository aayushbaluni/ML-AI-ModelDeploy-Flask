from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask((__name__))
model = pickle.load(open('modelll.pkl', 'rb'))
cv=pickle.load(open('cv1.pkl','rb'))
dic={1:'SPAM',0:'NOT SPAM'}
@app.route('/')
def home():
	return render_template('index.html')



@app.route('/check',methods=['POST'])
def check():
	msg=request.form['message']
	#msg.replace('[^a-zA-Z]',' ',regex=True,inplace=True)
	msg=[msg]
	msg=cv.transform(msg).toarray()
	pre=model.predict(msg)
	return render_template('index.html',prediction="According to Me: {}".format(dic[pre[0]]))

if __name__ == '__main__':
	app.run(debug=True)



