import tensorflow as tf
import numpy as np
import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def gerar_laudo_ia(resultado, mse):
    prompt = f"Paciente com resultado {resultado} e erro de reconstrução {mse}. Gere um laudo médico curto e formal."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except:
        return "Laudo indisponível no momento."