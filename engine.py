import tensorflow as tf
import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def carregar_vae():
    weights_path = 'models/vae_pneumonia.weights.h5'
    if os.path.exists(weights_path):
        return "Modelo VAE Carregado"
    return "Erro: Pesos não encontrados"

def calcular_confianca(mse):
    return max(0, min(100, int((1 - mse) * 100)))

def registrar_analise(mse, classificacao, confianca):
    if "history" not in st.session_state:
        st.session_state.history = []
        
    st.session_state.history.append({
        "Execução": len(st.session_state.history) + 1,
        "Erro MSE": round(mse, 6),
        "Resultado": classificacao,
        "Confiança (%)": confianca
  })