import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import model_selection

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

# Title

st.title('Trouve ton job')

# Sidebar

page = st.sidebar.selectbox("Choose your page", ["Information on data", "Prediction", "Page 3"]) 

if page == "Information on data":
    st.sidebar.title('Settings')
    check_box = st.sidebar.checkbox(label='Display dataset')
    # Display details of page 1
    if check_box:
        st.dataframe(df)
    
    select_chart = st.sidebar.selectbox(label="Select a type of data", options=['Salaires', 'Competences'])

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
        df = data.copy()
        df = df.drop(['Unnamed: 0', 'Date de publication', 'lieu'], axis=1)
        
        y_max = df['salaire_maximum']
        y_min = df['salaire_minimum']
        X_cat = df.select_dtypes(include=[object])
        
        pipe_cat = Pipeline(
            steps=[
                ('pipe_imp', SimpleImputer(strategy='most_frequent')),
                ('pipe_enc', OneHotEncoder(sparse=False))
            ]
        )
        tf_cat = ColumnTransformer(
            transformers=[
                ('tf_cat', pipe_cat, ['Intitulé du poste', 'Nom de la société', 'Type de contrat']),
                ('tf_comp', CountVectorizer(), 'competences')
            ]
        )

        RFR_pipe_max = Pipeline(
            steps=[
                ('transformation', tf_cat),
                ('model', RandomForestRegressor(n_estimators=26, random_state=66, criterion='absolute_error'))
            ]
        )

        RFR_pipe_min = Pipeline(
            steps=[
                ('transformation', tf_cat),
                ('model', RandomForestRegressor(n_estimators=9, random_state=40, criterion='friedman_mse'))
            ]
        )
        
        X = X_cat
        X_train, X_test, y_train, y_test = train_test_split(X, y_max, test_size=0.25, random_state=10)
        
        # Prediction Salaire max
        RFR_pipe_max.fit(X_train, y_train)
        y_max_pred = RFR_pipe_max.predict(X_test)
        poste = st.text_input('Saisie ton poste : ')
        conpanie = st.text_input('Saisie ton entreprise : ')
        contract = st.text_input('Saisie ton type de contract : ')
        competences = st.text_input('Saisie la liste de tes compétences : ')
        st.write(RFR_pipe_max.predict([[poste, conpanie, contract, competences]]))
        
    except Exception as e:
        print(e)
