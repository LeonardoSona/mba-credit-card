# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:23:34 2024

"""

# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
from threadpoolctl import threadpool_limits

pio.renderers.default='browser'

# Streamlit App
st.title('Credit Card Customer Segmentation')

st.text('Source: https://www.kaggle.com/datasets/aryashah2k/credit-card-customer-data')

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Read the uploaded file
        dados_cartao = pd.read_csv(uploaded_file)
        
        # Check the content of the uploaded file
        if dados_cartao.empty:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            st.stop()

        st.write("File Uploaded Successfully")

        # Convert numeric columns from strings with commas to float
        for col in ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']:
            dados_cartao[col] = dados_cartao[col].replace({',': ''}, regex=True).astype(float)
        
        # Visualizing data information
        st.subheader('Data Information')
        st.write(dados_cartao.head())  # Display the first few rows of the dataframe
        st.write(dados_cartao.info())  # Display information about the dataframe

        # Descriptive statistics of the variables
        if 'Sl_No' in dados_cartao.columns and 'Customer Key' in dados_cartao.columns:
            cartao_cluster = dados_cartao.drop(columns=['Sl_No', 'Customer Key'])
        else:
            st.error("The necessary columns 'Sl_No' and 'Customer Key' are not found in the uploaded file.")
            st.stop()

        tab_descritivas = cartao_cluster.describe().T

        st.subheader('Descriptive Statistics')
        st.write(tab_descritivas)

        # Standardizing the variables using Z-Score
        cartao_pad = cartao_cluster.apply(zscore, ddof=1)
        
        st.subheader('Standardization Results')
        st.write("Mean of standardized data:")
        st.write(round(cartao_pad.mean(), 3))
        st.write("Standard Deviation of standardized data:")
        st.write(round(cartao_pad.std(), 3))

        # 3D Scatter Plot of observations
        st.subheader('3D Scatter Plot')
        fig = px.scatter_3d(cartao_pad, 
                            x='Avg_Credit_Limit', 
                            y='Total_Credit_Cards', 
                            z='Total_visits_bank')
        st.plotly_chart(fig)

        # Elbow Method for identifying the number of clusters
        st.subheader('Elbow Method')
        elbow = []
        K = range(1,11)
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                for k in K:
                    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(cartao_pad)
                    elbow.append(kmeanElbow.inertia_)
        except Exception as e:
            st.error(f"Error during Elbow Method calculation: {e}")
            st.stop()

        fig, ax = plt.subplots()
        ax.plot(K, elbow, marker='o')
        ax.set_xlabel('Nº Clusters')
        ax.set_xticks(range(1,11))
        ax.set_ylabel('WCSS')
        ax.set_title('Método de Elbow')
        st.pyplot(fig)

        # Silhouette Method for identifying the number of clusters
        st.subheader('Silhouette Method')
        silhueta = []
        I = range(2,11)
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                for i in I: 
                    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(cartao_pad)
                    silhueta.append(silhouette_score(cartao_pad, kmeansSil.labels_))
        except Exception as e:
            st.error(f"Error during Silhouette Method calculation: {e}")
            st.stop()

        fig, ax = plt.subplots()
        ax.plot(range(2, 11), silhueta, color = 'purple', marker='o')
        ax.set_xlabel('Nº Clusters')
        ax.set_ylabel('Silhueta Média')
        ax.set_title('Método da Silhueta')
        ax.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red')
        st.pyplot(fig)

        # K-means clustering
        st.subheader('K-means Clustering')
        kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(cartao_pad)
        kmeans_clusters = kmeans_final.labels_
        cartao_cluster['cluster_kmeans'] = kmeans_clusters
        cartao_pad['cluster_kmeans'] = kmeans_clusters
        cartao_cluster['cluster_kmeans'] = cartao_cluster['cluster_kmeans'].astype('category')
        cartao_pad['cluster_kmeans'] = cartao_pad['cluster_kmeans'].astype('category')

        # ANOVA analysis
        st.subheader('ANOVA Analysis')
        for col in ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']:
            st.write(f"ANOVA for {col}")
            st.write(pg.anova(dv=col, 
                              between='cluster_kmeans', 
                              data=cartao_pad,
                              detailed=True).T)

        # 3D Scatter Plots of clusters
        st.subheader('3D Scatter Plot of Clusters')
        fig1 = px.scatter_3d(cartao_cluster, 
                             x='Avg_Credit_Limit', 
                             y='Total_Credit_Cards', 
                             z='Total_visits_online',
                             color='cluster_kmeans')
        st.plotly_chart(fig1)

        fig2 = px.scatter_3d(cartao_cluster, 
                             x='Avg_Credit_Limit', 
                             y='Total_Credit_Cards', 
                             z='Total_visits_bank',
                             color='cluster_kmeans')
        st.plotly_chart(fig2)

        fig3 = px.scatter_3d(cartao_cluster, 
                             x='Avg_Credit_Limit', 
                             y='Total_Credit_Cards', 
                             z='Total_calls_made',
                             color='cluster_kmeans')
        st.plotly_chart(fig3)

        # Group characteristics
        st.subheader('Cluster Characteristics')
        cartao_grupo = cartao_cluster.groupby(by=['cluster_kmeans'])
        tab_desc_grupo = cartao_grupo.describe().T
        st.write(tab_desc_grupo)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file.")





