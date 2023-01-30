import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

from plotly.subplots import make_subplots
import plotly.graph_objects as go


data = pd.read_csv('../clean_data.csv')
df = data.copy()
df = df.drop('Unnamed: 0', axis=1)
st.title('Trouve ton job dans la data science')
st.dataframe(df)
fig = make_subplots(rows=2, cols=1)

fig.append_trace(go.Bar(
    x=(df["Nom de la société"]).astype(str),
    y=(df["salaire_maximum"]).astype(int), name="salaire_maximum"
), row=1, col=1)

fig.append_trace(go.Bar(
    x=(df["Nom de la société"]).astype(str),
    y=(df["salaire_minimum"]).astype(int), name="salaire_minimum"
), row=2, col=1)
fig.update_layout(height=800, width=1000, title_text="Distribution des salaires selon l'entreprise", showlegend=True)
st.plotly_chart(fig)

cv = CountVectorizer()
comp_tf = cv.fit_transform(df['competences']).toarray()
comp_tf = pd.DataFrame(data = comp_tf, columns = cv.get_feature_names_out()) # permet d'obtenir un df avec les bons noms de colonne
names_comp = comp_tf.columns
comp_tf['Intitulé du poste'] = df['Intitulé du poste']
postes = df['Intitulé du poste'].unique() # donne la liste des valeurs possibles
df_gb = comp_tf.groupby('Intitulé du poste')[names_comp].sum()

# caculate the occurrence of competences
occurrence_comp = comp_tf.sum(axis=0)

df_occ_comp = pd.DataFrame(occurrence_comp)
df_occ_comp["competences"] = df_occ_comp.index
df_occ_comp.rename(columns={0 : "occurrence"}, inplace = True)

#df_occ_comp.drop(['Intitulé du poste'], axis=0, inplace=True)

fig = px.bar(df_occ_comp, x="competences", y='occurrence', title="Occurence des compétences")
st.plotly_chart(fig)


# neighborhood = st.radio("competences", names_comp)
fig = px.scatter(df_gb, title="Répartition des compétences pour les différents Intitulés du poste")
fig.update_traces(marker={'size': 15})
st.plotly_chart(fig)

