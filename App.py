import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)

model_close = pickle.load(open("model_close.pkl","rb"))

model_high = pickle.load(open("model_high.pkl","rb"))

model_low = pickle.load((open("model_low.pkl","rb")))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_close",methods=["POST"])
def predict_close():
    
    int_features = [int(i) for i in request.form.values()]
    final_feature = [np.array(int_features)]
    prediction = model_close.predict(final_feature)
    
    output = round(prediction[0],2)
    
    return render_template("index.html",prediction_text = f"Predictive close will be {output}")
    



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model_close.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
    



















    

    