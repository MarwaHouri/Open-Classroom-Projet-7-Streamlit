import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pickle
import shap

model = pickle.load(open('test_RFC_model_all.pkl','rb'))
thresh=pickle.load(open('best_th.pkl', 'rb'))
X_test=pd.read_csv('X_test.csv')
y_test=pd.read_csv('y_test.csv')
X_train=pd.read_csv('X_train.csv')
imp=pd.read_csv('importances.csv')
#df=X_test.join(y_test)
top10_feat=imp.head(10).Features.values

st.title('Prêt à dépenser : Calculateur de droit au crédit')

def graph_imp():
    test=imp.head(25)
    feat=test.Features.values
    fig, ax = plt.subplots()
    
    ax=test.plot.bar(x='Features', y='Importances', figsize=(10,5), legend=False)
    fig=ax.figure
    fig.suptitle("Best 25 general feature importances",
                 size=20,
                 y=1.1)
    return(fig)
def tachymetre(client_probability, best_th):
    fig, ax = plt.subplots(figsize=(5, 1))
    fig.suptitle("probability of credit default (%)",
                 size=10,
                 y=1.1)
    ax.add_patch(Rectangle((0, 0), width=best_th * 100, height=1, color='limegreen'))
    ax.add_patch(Rectangle((best_th * 100, 0), width=100 - best_th * 100, height=1, color='orangered'))
    ax.add_patch(FancyArrowPatch((client_probability * 100, 1), (client_probability * 100, 0), mutation_scale=10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 105, 10))
    ax.set_yticks([])
    return fig
def boxplot(client_id):
    df=X_test.join(y_test)
    fig, axes = plt.subplots(4,3, figsize=(8,20)) # create figure and axes
    client=df.loc[client_id]
    for i,el in enumerate(list(top10_feat)):
        ax=axes.flatten()[i]
        a = df.boxplot(el, ax=ax, showfliers=False)
        ax.axhline(y=client[el], c='red', ls='--') 
    fig.suptitle("Situating the client with respect to other on best 10 feature importances ",
                 size=15,
                 y=0.9)
    fig.delaxes(axes[3][2])
    fig.delaxes(axes[3][1])
    return(fig)

def kde(client_id, feature):
    df=X_test.join(y_test)
    d=df.loc[client_id][feature]
    bw_method=0.5
    df0 = df.loc[df['TARGET'] == 0, feature]
    df1 = df.loc[df['TARGET'] == 1, feature]
    xmin = df[feature].min()
    xmax = df[feature].max()
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4, dpi=100)
    g=df0.plot(kind='kde',
                   c='g',
                   label='Non-defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    df1.plot(kind='kde',
                   c='r',
                   label='Defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    ax=g.axes
    ax.axvline(d, ls='--', color='r')
    fig.suptitle(
        f'Observed distribution of {feature} based on clients true class',
        y=0.95)
    plt.legend()
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.xlim(xmin, xmax)
    return(fig)
with st.sidebar:
    topic = st.radio(
    'Select a topic',
    ('General', 'Decision', 'Client information'))

if topic=='General':
    st.write('This page is dedicated for the general information to help you understand our decision regarding the acceptance or denial of you credit demand')
    fig=graph_imp()
    st.pyplot(fig)

if topic=='Decision':    
    col1, col2=st.columns(2)
    with col1:
        st.subheader('Client ID')
        option=st.selectbox('Select client id: ', X_test.index)
    with col2:

        st.subheader('Informations client')
        df=X_test[X_test.index==option][top10_feat]
        st.dataframe(df.transpose())
    
    st.subheader('Prediction droit au credit')
    client_prob=model.predict_proba(X_test[X_test.index==option])
    y_pred=(client_prob[:,1]>thresh).astype('int')
#y_pred=model.predict(X_test[X_test.index==option])
    if y_pred[0]==1:
        decision='Refuse'
    else:
        decision='Accepter'
    st.write('La decision est: ',' ',  decision)
    fig=tachymetre(client_prob.flat[1], thresh)
    st.pyplot(fig)
    

if topic=='Client information':
    col1, col2=st.columns(2)
    with col1:
        st.subheader('Positionnement par rapport au autres clients')
        fig=boxplot(3)
        st.pyplot(fig)
    
#features= df.columns.values
    with col2:
        st.subheader('Probability density by feature and decision type')
        sel_features = st.multiselect("Choose features to visualize", top10_feat, top10_feat[:2])
        for feat in sel_features:
            fig=kde(3, feat)
            st.pyplot(fig)
    



