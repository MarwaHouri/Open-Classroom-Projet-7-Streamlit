import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

model = pickle.load(open('model_RFC_all.pkl','rb'))
X_test=pd.read_csv('Xs_median.csv')
X_train=pd.read_csv('X_train.csv')

st.title('Prêt à dépenser : Calculateur de droit au crédit')

option=st.selectbox('Numero client: ', X_test.index)

st.subheader('Informations client')
df=X_test[X_test.index==option].transpose()
st.dataframe(df[:10])

st.subheader('Prediction droit au credit')

y_pred=model.predict(X_test[X_test.index==option])
if y_pred[0]==1:
    decision='Refuse'
else:
    decision='Accepter'
st.write('La decision est: ',' ',  decision)

st.subheader('model predict_proba')
m=model.predict_proba(X_test[X_test.index==option])
st.write(m)

st.subheader('Explication des feature importances par shap')
explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X_test[X_test.index==option])
#st.write(shap_values)

#shap.force_plot(explainer.expected_value[1], shap_values[1], X_test[X_test.index==option])