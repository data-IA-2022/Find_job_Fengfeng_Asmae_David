import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

@st.cache
def load_data(data):
    df = data.copy()
    df = df.drop('Unnamed: 0', axis=1)
    cat_cols = df.select_dtypes([object]).columns
    num_cols = df.select_dtypes(exclude="object").columns
    return df, cat_cols, num_cols

def cv_comp(df):
    cv = CountVectorizer()
    df_tf = cv.fit_transform(df['competences']).toarray()
    df_tf = pd.DataFrame(data=df_tf, columns=cv.get_feature_names_out())
    return df_tf

def add_cols(df1, df2):
    for col in df1.columns:
        df2[col] = df1[col]
    return df2

data = pd.read_csv('../clean_data.csv', sep=',')
df, cat_cols, num_cols = load_data(data)

df_tf = cv_comp(df)
cols_comp = df_tf.columns

occurrence_comp = df_tf.sum(axis=0)
df_occ_comp = pd.DataFrame(occurrence_comp)
df_occ_comp["competences"] = df_occ_comp.index
df_occ_comp.rename(columns={0 : "occurrence"}, inplace = True)

df_merge = df.drop('competences', axis=1)
df_final = add_cols(df_merge, df_tf)


# Pages

st.sidebar.title('Navigation')
page = st.sidebar.radio("Choose your page", ["Home", "Information on the job market", "Prediction"])

if page == "Home":
    line_pre = """L'objectif de cette application est de vous fournir une prediction de votre salaire maximum et minimum en fonction de
                divers paramètres que vous renseignerez"""
    line_exp = """
                    - Votre :red[**poste**]
                    - Votre :red[**entreprise**]
                    - Le type votre :red[**contrat**]
                    - Le liste de vos :red[**compétences**]
                """
    st.title('Trouve ton job')
    st.subheader('Présentation')
    st.markdown(f"{line_pre}")
    image = Image.open('images/recruit-crm-talent-process-ma.jpg')
    st.image(image, caption='find a job is hard')
    st.subheader('How does it work')
    st.write('Notre application fonctionne grâce à un algorithme de regression qui determine votre salaire. Ces paramètres sont les suivants :')
    st.markdown(f"{line_exp}")

if page == "Information on the job market":
    st.title('Information on the job market')
    st.sidebar.title('Settings')
    check_box = st.sidebar.checkbox(label='Display dataset')
    # Display details of page 1
    if check_box:
        st.dataframe(df)
    
    select_chart = st.sidebar.radio(label="Select a type of data", options=['Salaires', 'Competences'])

    if select_chart == 'Salaires':
        try:
            x_values = st.sidebar.selectbox("X axis", options=['Intitulé du poste', 'lieu', 'Nom de la société'])
            y_values = st.sidebar.selectbox("Y axis", options=['salaire_maximum', 'salaire_minimum'])
            plot = px.bar(df, x_values, y_values)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if select_chart == 'Competences':
        try:
            check_box = st.sidebar.checkbox(label="Display Most in demand skills")
            if check_box:
                plot = px.bar(df_occ_comp, x="competences", y='occurrence', title="Most in demand skills")
                st.plotly_chart(plot) 
            x_values = st.sidebar.selectbox("X axis", options=['Intitulé du poste', 'lieu', 'Nom de la société'])
            df_to_plot = df_final.groupby(x_values)[cols_comp].sum()
            fig = px.scatter(df_to_plot, title=f"Répartition des compétences en fonction de la colonne '{x_values}'")
            fig.update_traces(marker={'size': 15})
            st.plotly_chart(fig)
        except Exception as e:
            print(e)
        
elif page == "Prediction":
    # Display details of page 2
    st.title('Prediction')
    try:
        st.subheader('Predict your salary range')
        #import model 
        pickle_max = open("../RFR_max.pkl", "rb")
        model_max = pickle.load(pickle_max)
        
        pickle_min = open("../RFR_min.pkl", "rb")
        model_min = pickle.load(pickle_min)
        # Prediction Salaire max
        poste = st.text_input('Saisie ton poste : ')
        companie= st.text_input('Saisie ton entreprise : ')
        contract = st.text_input('Saisie ton type de contrat : ')
        competences = st.text_input('Saisie la liste de tes compétences : ')
        df_submit = pd.DataFrame(
            data=[[poste,companie,contract,competences]], 
            columns=['Intitulé du poste', 'Nom de la société', 'Type de contrat', 'competences']
        )
        
        submit_max = st.button('Predict')
        if submit_max:
            prediction_max = model_max.predict(df_submit)
            prediction_min = model_min.predict(df_submit)
            st.success(f"your salary range will be {np.rint(prediction_min)[0]} - {np.rint(prediction_max)[0]} $/year")
            st.balloons()
            
    except Exception as e:
        st.error(e)