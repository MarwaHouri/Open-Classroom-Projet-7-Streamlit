from flask import Flask
from flask import jsonify
import pandas as pd
import pickle
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from flask import Response

imp=pd.read_csv('importances.csv')
minmax=pd.read_csv('minmax.csv')
#top10_feat=imp.head(10).Features.values
top10_feat=pickle.load(open('top10_feat.pkl', 'rb'))
X_test=pd.read_csv('X_test.csv')
y_test=pd.read_csv('y_test.csv')
id_test=pd.read_csv("id_test.csv")
df=X_test.join(id_test)
df=df.join(y_test)
df0=pd.read_csv('df0.csv')
df1=pd.read_csv('df1.csv')
#model = pickle.load(open('test_RFC_model_all.pkl','rb'))
model, thresh=pickle.load(open('lgbm_model_custom_noscale_best_th.pkl','rb'))
#explainer, shap_values =pickle.load(open('explainer_shap_values.pkl','rb'))


explainer= shap.TreeExplainer(model)    

app = Flask(__name__)

@app.route('/')
def hello():
    return 'This is an API'

@app.route('/clients')
def getFirstClient():
    ''' Gets the list of client ids'''
    return jsonify({
                    'ids': list(df['SK_ID_CURR'])
                    })

@app.route('/clients/<int:id>')
    
def getClient(id):
    ''' Gets the prediction using personalised threshhold, the threshhold and the prediction probabilities for a given client'''
    idx=df[df["SK_ID_CURR"]==id].index.values[0]
    client_prob=model.predict_proba(X_test[X_test.index==idx])
    y_pred=(client_prob[:,1]>thresh).astype('int')
    return jsonify({
                    'prediction': int(y_pred),
                    'proba': list(client_prob[0]),
                    'thresh': float(thresh)
                    })

@app.route('/clients-info/<int:id>')
def getClientInfo(id):
    ''' Gets the information of a given client from the complete dataset'''
    client=df[df["SK_ID_CURR"]==id]
    client=client.drop(columns=['SK_ID_CURR', 'TARGET'])
    return (client.to_json(orient='index'))

@app.route('/feature-info/<feature>')
def getFeatureInfo(feature):
    ''' Gets the distribution information for a given feature'''
    ismin=minmax[feature][0]
    ismax=minmax[feature][4]
   # list_max=df[df[feature]==1]['SK_ID_CURR']
    #lis_min=
    return jsonify({'feat': list(minmax[feature]), 
                    'min_clients' : list(df[df[feature]==ismin]['SK_ID_CURR']),
                    'max_clients' : list(df[df[feature]==ismax]['SK_ID_CURR'])                })

@app.route('/kde/<int:id>/<feature>')    
def kde(id, feature):
    client=df[df["SK_ID_CURR"]==id]
    client_feat=float(client[feature].values)
    bw_method=0.5
    xmin =minmax[feature][0]
    xmax = minmax[feature][4]
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(4, 4))
    g=df0[feature].plot(kind='kde',
                   c='g',
                   label='Non-defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    df1[feature].plot(kind='kde',
                   c='r',
                   label='Defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    ax=g.axes
    ax.axvline(client_feat, ls='--', color='r')
    #fig.suptitle(
     #   f'Distribution de {feature} par rapport a la vrai classe des clients',
      #  y=0.95, fontsize=9)
    plt.legend()
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.xlim(xmin, xmax)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
   

@app.route('/shap/<int:id>')
def getShapValues(id) :
    ''' Gets shap values of a given client, returns a zip list with the features, values and shap values of the client of the positive class (default credit)'''
    idx=df[df["SK_ID_CURR"]==id].index.values[0]
    shap_values = explainer.shap_values(X_test[X_test.index==idx])
    shaps=list(zip(X_test.columns.values,shap_values[0][0],
                   X_test.loc[idx]))
      
    return jsonify({
        'expected': float(explainer.expected_value[0]),
        'shap' : shaps      
             })



top9_feat=np.delete(top10_feat,-1)
@app.route('/gender/<int:id>')
def gender_dist_plot(id):
    client=df[df["SK_ID_CURR"]==id]
    fig, axes = plt.subplots(3,3, figsize=(15,20)) # create figure and axes
    
    for i,el in enumerate(list(top9_feat)):
        client_feat=float(client[el].values)
        ax=axes.flatten()[i]
        a = sns.boxplot(y=el, x='CODE_GENDER', data=X_test,showfliers = False, ax=ax)
        ax.axhline(y=client_feat, c='red', ls='--')
        labels = ['Gender 0', 'Gender 1']
        ax.set_xticklabels(labels)
        ax.set_xlabel('')
    #plt.savefig('./static/gender_'+str(id)+'.png')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


app.run(debug=True)
