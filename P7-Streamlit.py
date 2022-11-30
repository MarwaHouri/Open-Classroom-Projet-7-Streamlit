import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pickle
import shap
import matplotlib as mpl
from PIL import Image
from lime import lime_tabular
import requests
import json
import streamlit.components.v1 as components   

#model = pickle.load(open('test_RFC_model_all.pkl','rb'))
#explainer, shap_values =pickle.load(open('explainer_shap_values.pkl','rb'))
minmax=pd.read_csv('minmax.csv')
shap_general=Image.open('shap.png')
logo=Image.open('logo.png')
summary=Image.open('summary_plot.png')
summary_mean=Image.open('summary_plot_mean.png')
imp=pd.read_csv('importances.csv')
top10_feat=imp.head(10).Features.values
df0=pd.read_csv('df0.csv')
df1=pd.read_csv('df1.csv')
X_test=pd.read_csv('X_test.csv')
baseURL = "http://127.0.0.1:5000"



##########################Les fonctions utilisees#######################################
#######################################################################################
def graph_imp():
    ''' Representation des features importances 
        globales definies par le model de classification '''
    test=imp.head(25)
    feat=test.Features.values
    fig, ax = plt.subplots()
    
    ax=test.plot.bar(x='Features', y='Importances', figsize=(10,5), legend=False, color=sns.color_palette())
    fig=ax.figure
    fig.suptitle("Best 25 general feature importances",
                 size=20,
                 y=1.1)
    return(fig)

def tachymetre(client_probability, best_th):
    ''' Representation de la probailite d'acceptation 
    du credit d'un client donne par rapport au seuil '''
    fig, ax = plt.subplots(figsize=(4, 0.3))
    fig.suptitle(f"Le score de defaut est : {client_probability*100:.2f} ",
                 size=8,
                 y=1.5)
    ax.add_patch(Rectangle((0, 0), width=best_th * 100, height=1, color='green'))
    ax.add_patch(Rectangle((best_th * 100, 0), width=100 - best_th * 100, height=1, color='red'))
    ax.add_patch(FancyArrowPatch((client_probability * 100, 1), (client_probability * 100, 0), mutation_scale=10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 105, 10))
    ax.set_yticks([])
   
    return fig


def minmax_plt(df, feature):
    ''' Representation d'un curseur qui permet l'affichage de l'emplacement des info d'un client donne par rapport a l'intervall des valeurs'''
    client_feat=float(df.loc[feature].values)
    fig, ax = plt.subplots(figsize=(5, 1.5))
    fig.subplots_adjust(bottom=0.6)
    fig.suptitle(f'Valeur client pour feature {feature} est {client_feat:.3f}' ,
                 size=10,
                 y=1)
    cmap = (mpl.colors.ListedColormap(['firebrick','darkred','firebrick' ]))
    bounds = [minmax[feature].values[0],minmax[feature].values[1],minmax[feature].values[3] ,minmax[feature].values[4]]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=ax,
        #extend='both',
        extendfrac='auto',
        ticks=[minmax[feature].values[0],minmax[feature].values[1], minmax[feature].values[2],minmax[feature].values[3],minmax[feature].values[4]],
        spacing='uniform',
        orientation='horizontal',
    )
      
    ax.axvline(minmax[feature].values[2], ls='-', color='silver')
    ax.add_patch(FancyArrowPatch((client_feat, 1), (client_feat, 0), mutation_scale=20))
    label=['min','25%', '50%','75%', 'max']
    for i, x in enumerate(ax.get_xticks()):
        plt.text(x, -0.8, label[i], size=10, ha='center')
    return fig


#def boxplot(client_id):
#    fig, axes = plt.subplots(4,3, figsize=(8,20)) # create figure and axes
#    client=df.loc[client_id]
#    for i,el in enumerate(list(top10_feat)):
#        ax=axes.flatten()[i]
#        a = df.boxplot(el, ax=ax, showfliers=False)
#        ax.axhline(y=client[el], c='red', ls='--') 
#    fig.suptitle("Situating the client with respect to other on best 10 feature importances ",
#                 size=15,
#                 y=0.9)
#    fig.delaxes(axes[3][2])
#    fig.delaxes(axes[3][1])
#    return(fig)


def kde(df, feature):
    client_feat=float(df.loc[feature].values)
    bw_method=0.5
    xmin = minmax[feature].values[0]
    xmax = minmax[feature].values[4]
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(5, 5))
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
        y=0.95)
    plt.legend()
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.xlim(xmin, xmax)
    return(fig)

#def st_shap(plot, height=None):
 #   shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
  #  components.html(shap_html, height=height)

############################## Menu global a gauche################################
##################################################################################
with st.sidebar:
    st.image(logo)
    
    st.subheader('Menu')
    topic = st.radio(
    'Select a topic',
    ( 'Decision','General', 'Interpretabilite', 'Analyse comparative'))
    
    st.subheader('Client ID')
    urlToCall = "http://127.0.0.1:5000/clients"
    response = requests.get(urlToCall)
    data_dic = response.json()  
    option=st.selectbox('Select client id: ', data_dic['ids'])
    
    urlToCall = baseURL + '/clients-info/' + str(option)
    client=requests.get(urlToCall)
    client=pd.read_json(client.text)


st.title('Prêt à dépenser : Calculateur de droit au crédit')
#st.subheader('Prediction droit au credit')
    #client_prob=model.predict_proba(X_test[X_test.index==option])
    #y_pred=(client_prob[:,1]>thresh).astype('int')
    #baseURL = "http://127.0.0.1:5000"
######################################################################################
                    # Decision#
######################################################################################

if topic=='Decision':
    st.subheader('Decision sur l\'eligibilite du client a un credit') 
    urlToCall = baseURL + '/clients/' + str(option)
    dic=requests.get(urlToCall)
    data_dic=dic.json()
    pred=data_dic['prediction']
    if pred==0:
        st.write('Le credit est accepte')
    else:
        st.write('Le credit est refuse')
        
######################Representation graphique tachymetre#################################
    
    proba_client=data_dic['proba'][1]
    thresh=data_dic['thresh']
    
    fig=tachymetre(proba_client,thresh)
    st.pyplot(fig)
    
###################################################################################
                    #General#
###################################################################################    
if topic=='General':
    st.write('This page is dedicated for the general information to help you understand our decision regarding the acceptance or denial of you credit demand')
########################Graphe des feautures importances du model#######################
    fig=graph_imp()
    st.pyplot(fig)
###################### Graphe des feature importances par Shap ########################    
    st.subheader('Shap summary plots')
    #col1, col2=st.columns(2)
    #with col1:
    st.image(summary)
    #with col2:
     #   st.image(summary_mean)

    
###########################################################################################
                #INTERPRETABILITE SHAP#
###########################################################################################

if topic=='Interpretabilite':
################# Force plot GENERAL################################################    
    st.subheader('General shap values')
    st.image(shap_general,caption='Shap force plot for the test set') 
    
#################Force plot d'un client selectionne###############################    
    st.subheader('Shap values for a given client')
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    baseURL = "http://127.0.0.1:5000/shap"
    urlToCall = baseURL + '/' + str(option)
    
    dic=requests.get(urlToCall)
    data_dic=dic.json()
    expected=data_dic['expected']
    zipped=data_dic['shap']
    features, shap_values, values=zip(*zipped)
   
    #test=pd.DataFrame(data_dic['shap'])
    #test.columns=['Features', 'Shap_values', 'Values'] 
    #data=pd.DataFrame(data_dic['data'].items())
   
    #shap_values=pd.DataFrame(shap_values)
   
    #st.write(test)
    #st.write(pd.DataFrame(data.items()))

    shap.initjs()    
    plot=shap.force_plot(expected, np.array(shap_values), list(features))
    st_shap(plot, 200)
   
    
################### Waterfall plot d'un client selectionne############################    
    
    
    st.subheader('Shap waterfall')
    explanation=(shap.Explanation(values=np.array(shap_values), base_values=expected, 
                              data=np.array(values), feature_names=list(features)))
   
    st.components.v1.html(shap.waterfall_plot(explanation, max_display = 15), width=10, height=0, scrolling=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.pyplot(shap.waterfall_plot(explanation, max_display = 15))
    
###########################################################################################
                    #Analyse comparative avec les autres clients#
###########################################################################################

if topic=='Analyse comparative':    
    client.columns=['Information client '+ str(option)] 
    st.subheader('Informations client')
    st.dataframe(client.loc[top10_feat])
    
    st.subheader('Positionnement du client par rapport aux autres clients')
    sel_features = st.multiselect("Choisir les features a visualiser", top10_feat, top10_feat[:2])             
    for feat in sel_features:
        
        st.write('Position du client', str(option),' par rapport a', feat)
        col1, col2 = st.columns(2)
        with col1 :
            
            fig=minmax_plt(client, feat)
            st.pyplot(fig)         
            
        with col2 :
            #st.subheader('Situation du client par rapport a la distribution par feature et classe reelle ')
            fig=kde(client, feat)
            st.pyplot(fig)
    



