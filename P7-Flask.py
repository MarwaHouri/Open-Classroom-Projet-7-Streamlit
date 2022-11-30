from flask import Flask
from flask import jsonify
import pandas as pd
import pickle
import shap

imp=pd.read_csv('importances.csv')
#top10_feat=imp.head(10).Features.values
top10_feat=pickle.load(open('top10_feat.pkl', 'rb'))
X_test=pd.read_csv('X_test.csv')
y_test=pd.read_csv('y_test.csv')
df=X_test.join(y_test)
df['ID']=X_test.index

#model = pickle.load(open('test_RFC_model_all.pkl','rb'))
model, thresh=pickle.load(open('model_best_th.pkl', 'rb'))
#explainer, shap_values =pickle.load(open('explainer_shap_values.pkl','rb'))

clf=model.steps[2][1]
explainer= shap.TreeExplainer(clf)    

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
    ''' Gets the prediction using personalised threshhold, the threshhold and the prediction probabilities for a given client'''
    client_prob=model.predict_proba(X_test[X_test.index==id])
    y_pred=(client_prob[:,1]>thresh).astype('int')
    return jsonify({
                    'prediction': int(y_pred),
                    'proba': list(client_prob[0]),
                    'thresh': float(thresh)
                    })

@app.route('/clients-info/<int:id>')
def getClientInfo(id):
    ''' Gets the information of a given client from the complete dataset'''
    client=df[df['ID']==id]
    client=client.drop(columns=['ID', 'TARGET'])
    return (client.to_json(orient='index'))

@app.route('/shap/<int:id>')
def getShapValues(id) :
    ''' Gets shap values of a given client, returns a zip list with the features, values and shap values of the client of the positive class (default credit)'''
    shap_values = explainer.shap_values(X_test.loc[id])
    shaps=list(zip(X_test.columns.values,shap_values[1],X_test.loc[id]))
      
    return jsonify({
        'expected': float(explainer.expected_value[1]),
        'shap' : shaps      
             })
    

app.run(debug=True)