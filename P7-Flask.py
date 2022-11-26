from flask import Flask
from flask import jsonify
import pandas as pd
import pickle

X_test=pd.read_csv('X_test.csv')
y_test=pd.read_csv('y_test.csv')
df=X_test.join(y_test)
df['ID']=df.index

model = pickle.load(open('test_RFC_model_all.pkl','rb'))
thresh=pickle.load(open('best_th.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def hello():
    return 'This is an API'

@app.route('/clients')
def getFirstClient():
    ''' Gets the list of client ids'''
    return jsonify({
                    'ids': list(df['ID'])
                    })

@app.route('/clients/<int:id>')
def getClient(id):
    ''' Gets client prediction information for a specific client
        input = id
        output= dictionaly with 
                predicted value, probabilites of classes, the threshhold used '''
    
    client_prob=model.predict_proba(X_test[X_test.index==id])
    y_pred=(client_prob[:,1]>thresh).astype('int')
    return jsonify({
                    'prediction': int(y_pred),
                    'proba': list(client_prob[0]),
                    'thresh': float(thresh)
                    })

app.run(debug=True)