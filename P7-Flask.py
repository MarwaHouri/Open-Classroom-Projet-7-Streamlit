from flask import Flask, url_for
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

imp=pd.read_csv('Ressources/datasets/importances.csv')
minmax=pd.read_csv('Ressources/datasets/minmax.csv')
#top10_feat=imp.head(10).Features.values
top10_feat=pickle.load(open('Ressources/datasets/top10_feat.pkl', 'rb'))
X_test=pd.read_csv('Ressources/datasets/X_test.csv')
y_test=pd.read_csv('Ressources/datasets/y_test.csv')
id_test=pd.read_csv("Ressources/datasets/id_test.csv")
df=X_test.join(id_test)
df=df.join(y_test)
df0=pd.read_csv('Ressources/datasets/df0.csv')
df1=pd.read_csv('Ressources/datasets/df1.csv')
#model = pickle.load(open('test_RFC_model_all.pkl','rb'))
model, thresh=pickle.load(open('Ressources/model/lgbm_model_custom_noscale_best_th.pkl','rb'))
#explainer, shap_values =pickle.load(open('explainer_shap_values.pkl','rb'))


explainer= shap.TreeExplainer(model)    

app = Flask(__name__)


@app.route('/') 
def routes():  # pragma: no cover
    """Main page - Prints endpoints and their documentation"""
    func_list = []
    for rule in app.url_map.iter_rules():
        endpoint = rule.rule
        #methods = ", ".join(list(rule.methods))
        doc = app.view_functions[rule.endpoint].__doc__

        route = {
            "endpoint": endpoint
                    }
        if doc:
            route["doc"] = doc
        func_list.append(route)

    func_list = sorted(func_list, key=lambda k: k['endpoint'])
    return jsonify(func_list) 

@app.route('/clients')
def getClientList():
    ''' Returns the list of client ids'''
    return jsonify({
                    'ids': list(df['SK_ID_CURR'])
                    })


@app.route('/clients/<int:id>')  
def getClientPred(id):
    ''' Returns the prediction using personalised threshhold, the threshhold and the prediction probabilities for a given client'''
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
    ''' Returns the information of a given client from the complete dataset'''
    client=df[df["SK_ID_CURR"]==id]
    client=client.drop(columns=['SK_ID_CURR', 'TARGET'])
    return (client.to_json(orient='index'))

@app.route('/feature-info/<feature>')
def getFeatureInfo(feature):
    ''' Returns for the given feature the distribution of the feature (min, 25%, median, 75%, max), list of client ids having the minimum value and list of client ids having the maximum value for the feature, feature examples: PAYMENT_RATE, EXT_SOURCE_2,EXT_SOURCE_3, DAYS_BIRTH... '''
    
    ismin=minmax[feature][0]
    ismax=minmax[feature][4]
   # list_max=df[df[feature]==1]['SK_ID_CURR']
    #lis_min=
    return jsonify({'feat': list(minmax[feature]), 
                    'min_clients' : list(df[df[feature]==ismin]['SK_ID_CURR']),
                    'max_clients' : list(df[df[feature]==ismax]['SK_ID_CURR'])                })

@app.route('/kde/<int:id>/<feature>')    
def kde(id, feature):
    ''' Display the kde graph for the actual classes of clients for a given feature '''
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
    fig.suptitle(
        f'Distribution de {feature} par rapport a la vrai classe des clients',
        y=0.95, fontsize=9)
    plt.legend()
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.xlim(xmin, xmax)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
   

@app.route('/shap/<int:id>')
def getShapValues(id) :
    ''' Returns shap informations of a given client '''
    idx=df[df["SK_ID_CURR"]==id].index.values[0]
    shap_values = explainer.shap_values(X_test[X_test.index==idx])
    shaps=list(zip(X_test.columns.values,shap_values[1][0],
                   X_test.loc[idx]))
      
    return jsonify({
        'expected': float(explainer.expected_value[1]),
        'shap' : shaps      
             })

top9_feat=np.delete(top10_feat,-1)
@app.route('/gender/<int:id>')
def gender_dist_plot(id):
    ''' Displays boxplots by gender for the best nine indicators along with the position of the client with respect to the dataset'''
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
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


app.run(debug=True)