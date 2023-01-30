import pandas as pd
import streamlit as st
import plotly.express as px
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer

from plotly.subplots import make_subplots
import plotly.graph_objects as go

@st.cache
def load_data(data):
    df = data.copy()
    df = df.drop('Unnamed: 0', axis=1)
    cat_cols = df.select_dtypes([object]).columns
    num_cols = df.select_dtypes(exclude="object").columns
    return df, cat_cols, num_cols

data = pd.read_csv('../clean_data.csv', sep=',')
df, cat_cols, num_cols = load_data(data)
print(cat_cols)
print(num_cols)
# Title

st.title('Trouve ton job')


# Sidebar
st.sidebar.title('Settings')
check_box = st.sidebar.checkbox(label='Display dataset')
if check_box:
    st.dataframe(df)
    
select_chart = st.sidebar.selectbox(label="Select the chart type", options=['Histogram'])

if select_chart == 'Histogram':
    st.sidebar.subheader("Histogram setttings")
    try:
        x_values = st.sidebar.selectbox("X axis", options=['Intitulé du poste', 'lieu', 'Nom de la société'])
        y_values = st.sidebar.selectbox("Y axis", options=num_cols)
        plot = px.bar(df, x_values, y_values)
        st.plotly_chart(plot)
    except Exception as e:
        print(e)