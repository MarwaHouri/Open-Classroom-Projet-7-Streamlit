import streamlit as st
import streamlit.components.v1 as components   
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#import matplotlib
from matplotlib.patches import Rectangle, FancyArrowPatch
import pickle
import shap
from PIL import Image
import requests
import json

from PIL import Image
from io import BytesIO


logo=Image.open('logo.png')
genre=Image.open('genre.png')
summary_mean=Image.open('summary_plot_mean.png')
importance=Image.open('importance.png')
summary=Image.open('summary_plot.png')
top10_feat=pickle.load(open('top10_feat.pkl', 'rb'))
#df0=pd.read_csv('df0.csv')
#df1=pd.read_csv('df1.csv')
baseURL = "http://127.0.0.1:5000"
top9_feat=np.delete(top10_feat,-1)

##########################Les fonctions utilisees#######################################
#######################################################################################
#def graph_imp():
 #   ''' Representation des features importances 
  #      globales definies par le model de classification '''
  #  test=imp.head(25)
   # feat=test.Features.values
   # fig, ax = plt.subplots()
 #   
 #   ax=test.plot.bar(x='Features', y='Importances', figsize=(10,5), legend=False, color=sns.color_palette())
#    fig=ax.figure
    #fig.suptitle("Best 25 general feature importances",
                # size=20,
               #  y=1.1)
 #   return(fig)

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


def minmax_plt(df, feature, feat_disc):
    ''' Representation d'un curseur qui permet l'affichage de l'emplacement des info d'un client donne par rapport a l'intervall des valeurs'''
    client_feat=float(df.loc[feature].values)
    fig, ax = plt.subplots(figsize=(5, 1.5))
    fig.subplots_adjust(bottom=0.6)
    fig.suptitle(f'Valeur client pour feature {feature} est {client_feat:.3f}' ,
                 size=10,
                 y=1)
    cmap = (mpl.colors.ListedColormap(['firebrick','darkred','firebrick' ]))
    bounds = [feat_disc[0],feat_disc[1],feat_disc[3] ,feat_disc[4]]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=ax,
        #extend='both',
        extendfrac='auto',
        ticks=[feat_disc[0],feat_disc[1], feat_disc[2],feat_disc[3],feat_disc[4]],
        spacing='uniform',
        orientation='horizontal',
    )
      
    ax.axvline(feat_disc[2], ls='-', color='silver')
    ax.add_patch(FancyArrowPatch((client_feat, 1), (client_feat, 0), mutation_scale=20))
    label=['min','25%', '50%','75%', 'max']
    for i, x in enumerate(ax.get_xticks()):
        plt.text(x, -0.8, label[i], size=10, ha='center')
    return fig


def kde(df, feature, feat_disc):
    client_feat=float(df.loc[feature].values)
    bw_method=0.5
    xmin = feat_disc[0]
    xmax = feat_disc[4]
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


###############################################################################
############################## Menu global a gauche############################
#################################################################################
with st.sidebar:
    st.image(logo)
    
    st.subheader('Menu')
    urlToCall = baseURL+'/clients'
    response = requests.get(urlToCall)
    data_dic = response.json()  
    option=st.selectbox('Choisir un client ID: ', data_dic['ids'])
    
    #st.subheader('Menu')
    topic = st.radio(
    'Choisir un theme',
    ( 'Décision','Informations générales', 'Interpretabilité', 'Analyse comparative', 'Analyse par genre'))
    
        
    urlToCall = baseURL + '/clients-info/' + str(option)
    client=requests.get(urlToCall)
    client=pd.read_json(client.text)




######################################################################################
                    # Decision#
######################################################################################

if topic=='Décision':
    st.title('Prêt à dépenser : Calculateur de droit au crédit')
    st.subheader('Décision sur l\'eligibilité du client à un crédit') 
    urlToCall = baseURL + '/clients/' + str(option)
    dic=requests.get(urlToCall)
    data_dic=dic.json()
    pred=data_dic['prediction']
    if pred==0:
        st.write('La demande de crédit est acceptée')
    else:
        st.write('La demande de crédit est refusée')
        
######################Representation graphique tachymetre#################################
    
    proba_client=data_dic['proba'][1]
    thresh=data_dic['thresh']
    
    fig=tachymetre(proba_client,thresh)
    st.pyplot(fig)
    
###################################################################################
                    #General#
###################################################################################    
if topic=='Informations générales':
    st.title('Informations générales sur le modele')
########################Graphe des feautures importances du model#######################
    st.subheader('Top 15 features importances générées par le modele')
    #fig=graph_imp()
    st.image(importance, width=700)
    
################# Force plot GENERAL################################################    
   # st.subheader('Shap force plot general')
   # st.image(shap_general,caption='Shap force plot for the test set') 
    
###################### Graphe des feature importances par Shap ########################    
    
    st.subheader('Shap summary plot : impact des indicateurs sur la prédiction de rejet par instance:')
    st.write ()
    st.image(summary)
    st.subheader('Impact moyen des indicateurs sur la décision')
    st.image(summary_mean)
    
    
    
    
    
    
###########################################################################################
                #INTERPRETABILITE SHAP#
###########################################################################################

if topic=='Interpretabilité':
    st.title('Interprétation locale de la décision')
#################Force plot d'un client selectionne###############################    
    st.subheader('Shap force plot du client')
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    urlToCall = baseURL + '/shap/' + str(option)
    
    dic=requests.get(urlToCall)
    data_dic=dic.json()
    expected=data_dic['expected']
    zipped=data_dic['shap']
    features, shap_values, values=zip(*zipped)
   
    #shap.initjs()    
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
    st.title('Analyse comparative du client par rapport aux autres clients')
       
    st.subheader('Positionnement du client par rapport aux autres clients')
    sel_features = st.multiselect("Choisir les indicateurs :", top9_feat, top9_feat[:2])             
    for i, feat in enumerate(sel_features):
        urlToCall = baseURL + '/feature-info/' + feat
        response = requests.get(urlToCall)
        data_dic = response.json()
        feat_disc=data_dic['feat']
        min_clients=data_dic['min_clients']
        max_clients=data_dic['max_clients']
        list_min=pd.DataFrame(min_clients, columns=['Client ID'])
        list_max=pd.DataFrame(max_clients, columns=['Client ID'])
        
        st.write('Distribution de ' + feat + 'par rapport a la vrai classe des clients')
        col1, col2, col3=st.columns([1, 3, 1])
        urlToCall1 = baseURL + '/kde/' +str(option)  +'/'+ feat    
        response=requests.get(urlToCall1)
        img = Image.open(BytesIO(response.content))
        with col2:
          st.image(img, width=500)
           
        #fig=kde(client, feat, feat_disc)
        #st.pyplot(fig)
        st.write('Position du client', str(option),' par rapport a', feat)
        fig=minmax_plt(client, feat, feat_disc)
        st.pyplot(fig)
            
        col1, col2 = st.columns(2)
        with col1 :   
        
            show_min = st.checkbox('Montrer la liste des clients ayant une valeur minimale', key=feat)
            if show_min:
                st.dataframe(list_min)
            
        with col2:
            show_max = st.checkbox('Montrer la liste des clients ayant une valeur maximale', key=i)
            if show_max:
                st.dataframe(list_max)
        
       
        
        
###############################################################################
####            Analyse par genre
###############################################################################
if topic == 'Analyse par genre':
    st.title('Exploration des indicateurs en fonction du genre du client')
    col1, col2=st.columns(2)
    with col1:
        client.columns=['Informations client '+ str(option)] 
        st.subheader('Informations client')
        st.dataframe(client.loc[top10_feat])
    
        
    #fig=gender_dist_plot(client)
    #st.pyplot(fig)
    
    with col2:
        st.subheader('Distribution des clients par genre')
        st.image(genre, width=500)

    st.subheader('Situation du client par rapport aux indicateurs principals en fonction du genre')
    urlToCall = baseURL + '/gender/' + str(option)    
    response=requests.get(urlToCall)
    img = Image.open(BytesIO(response.content))
    st.image(img, width=800)



