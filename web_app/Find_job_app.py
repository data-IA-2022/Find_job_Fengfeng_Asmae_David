import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pickle
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

@st.cache
def load_data(data):
    df = data.copy()
    df = df.drop('Unnamed: 0', axis=1)
    df['Type de contrat'] = df['Type de contrat'].fillna('cdi')
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

def get_inputs(df):
    postes = df['Intitulé du poste'].unique()
    companies = df['Nom de la société'].unique()
    contrats = df['Type de contrat'].unique()
    competences = df[df.columns[0:-7]].columns
    return postes, companies, contrats, competences

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

postes, companies, contrats, competences = get_inputs(df_final)

# Pages

st.sidebar.title('Navigation')
page = st.sidebar.radio("Choisis une page", ["Home", "Information sur le marché du travail", "Prédiction", "Partager votre offre d'emploi"])

if page == "Home":
    line_pre = """L'objectif de cette application est de vous fournir une prediction sur votre salaire maximum et minimum en fonction de
                divers paramètres que vous renseignerez"""
    line_exp = """
                    - Votre :red[**poste**]
                    - Votre :red[**entreprise**]
                    - Le type votre :red[**contrat**]
                    - La  liste de vos :red[**compétences**]
                """
    st.title('Trouve ton job')
    st.subheader('Présentation')
    st.markdown(f"{line_pre}")
    image = Image.open('images/recruit-crm-talent-process-ma.jpg')
    st.image(image)
    st.subheader('Comment peut-on prédire ton salaire')
    st.write('Notre application fonctionne grâce à un algorithme de regression qui determine votre salaire. Ces paramètres sont les suivants :')
    st.markdown(f"{line_exp}")

if page == "Information sur le marché du travail":
    st.title('Information sur le marché du travail')
    st.sidebar.title('Paramètres')
    check_box = st.sidebar.checkbox(label='Voir les données')
    # Display details of page 1
    if check_box:
        st.dataframe(df)
    
    select_chart = st.sidebar.radio(label="Selectionner un paramètre", options=['Salaires', 'Compétences'])

    if select_chart == 'Salaires':
        try:
            x_values = st.sidebar.selectbox("X axis", options=['Intitulé du poste', 'lieu', 'Nom de la société'])
            y_values = st.sidebar.selectbox("Y axis", options=['salaire_maximum', 'salaire_minimum'])
            plot = px.bar(df, x_values, y_values, title=f"Masse salariale en fonction de la colonne '{x_values}'")
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if select_chart == 'Compétences':
        try:
            check_box = st.sidebar.checkbox(label="Les compétences les plus demandées")
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
        
elif page == "Prédiction":
    # Display details of page 2
    st.title('Prédiction')
    try:
        st.subheader('Prédire votre tranche de salaire')
        #import model 
        pickle_max = open("../RFR_max.pkl", "rb")
        model_max = pickle.load(pickle_max)
        
        pickle_min = open("../RFR_min.pkl", "rb")
        model_min = pickle.load(pickle_min)
        # Prediction Salaire max
        poste = st.selectbox('Choisis ton poste', postes)
        companie = st.selectbox('Choisis ton entreprise', companies)
        contrat = st.selectbox('Choisis ton contrat', contrats)
        competence = st.multiselect('Choisis tes competences', competences)
        
        
        df_submit = pd.DataFrame(
            data=[[poste,companie,contrat,', '.join(competence)]], 
            columns=['Intitulé du poste', 'Nom de la société', 'Type de contrat', 'competences']
        )
        
        submit_max = st.button('Predict')
        if submit_max:
            prediction_max = model_max.predict(df_submit)
            prediction_min = model_min.predict(df_submit)
            st.success(f"Votre salaire se situera entre {np.rint(prediction_min)[0]} - {np.rint(prediction_max)[0]} €/an")
            st.balloons()
            
    except Exception as e:
        st.error(e)
elif page == "Partager votre offre d'emploi":
    try:
        st.title("Améliorer notre service en fornissant une offre d'emploi")
        poste = st.text_input('Saisir un intitulé de poste').lower()
        date = st.text_input("Saisir la date de publication de l'offre (AAAA-MM-JJ)") # attention au format de la date
        lieu = st.text_input('Saisir la localisation').lower()
        competences = st.text_input('Saisir la liste des competences').lower()
        salaire_min = st.text_input('Saisir le salaire minimum')
        salaire_max = st.text_input('Saisir le salaire maximum')
        companie = st.text_input("Saisir le nom de l'entreprise").lower()
        contrat = st.text_input('Saisir le type de contrat').lower()
        
        submit = st.button('Submit')
        if submit:
            df = data.copy()
            df = df.drop('Unnamed: 0', axis=1)
            date = date.split('-')
            date = dt.date(int(date[0]), int(date[1]), int(date[2]))
            df_add = pd.DataFrame(data=[[poste, date, lieu, competences, salaire_min, salaire_max, companie, contrat]], columns=df.columns)
            df_final = df.append(df_add, ignore_index=True)
            df_final.to_csv('new_data.csv') # dataset comprenant des nouvelles données a nettoyés
            st.success("les nouvelles données ont été sauvegardées")
    except Exception as e:
        st.error(e)