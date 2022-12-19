import pickle
from flask import Flask
from flask import request
from flask import jsonify

with open('model.bin', 'rb') as f_in:  
    model = pickle.load(f_in)

with open('thr.bin', 'rb') as f_in:  
    thr = pickle.load(f_in)

app = Flask(__name__) 
@app.route('/', methods=['POST'])


def predict():    
    data = request.get_json()
    y_pred = model.predict_proba(list(data.values()))[1]
    if y_pred >= thr:
        return jsonify('loss')
    else:
        return jsonify('not loss')

if __name__=="__main__":
   app.run(debug=True, host='0.0.0.0', port=9696) 