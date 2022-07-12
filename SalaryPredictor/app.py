from flask import Flask, jsonify,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	features=[int(x) for x in request.form.values()]
	final=[np.array(features)]
	prediction=model.predict(final)
	output=round(prediction[0],2)

	return render_template('index.html',prediction_text='Employee Salary Should be ${}'.format(output))

if __name__=='__main__':
	app.run(debug=True)