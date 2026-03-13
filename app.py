import os
import json
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==============================================
# App Streamlit para VAE PneumoniaMNIST
# ==============================================
# Funcionalidades:
# - Triagem de pneumonia baseada no erro de reconstrução
# - Geração de novas imagens de raio-X
# - Upload e reconstrução de imagens
# ==============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')


def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')


class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)


def load_model():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    latent_dim = int(config.get('latent_dim', 16))
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    # Construir o modelo chamando uma passagem dummy antes de carregar pesos
    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)
    vae.load_weights(WEIGHTS_PATH)
    return vae, None


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Converter para grayscale e 28x28
    if image.mode != 'L':
        image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28))
    arr = np.array(image).astype('float32')
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return arr


def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    # Erro MSE por imagem
    return float(np.mean((x - x_recon) ** 2))


def classify_pneumonia(reconstruction_error: float) -> tuple:
    """
    Classifica se há possível pneumonia baseado no erro de reconstrução.
    Erro alto = possível pneumonia (imagem fora do padrão normal aprendido).
    """
    # Thresholds baseados em experiência com o dataset (ajustar conforme necessário)
    if reconstruction_error < 0.01:
        return "NORMAL", "Baixo risco de pneumonia", "green"
    elif reconstruction_error < 0.02:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red"


def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    """Gera novas imagens de raio-X usando o VAE treinado."""
    latent_dim = vae.encoder.output_shape[0][-1]  # Pega a dimensão do z_mean
    
    # Amostrar do espaço latente normal padrão
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    
    # Decodificar para gerar imagens
    generated_images = vae.decode(z_samples, training=False).numpy()
    
    return generated_images


st.set_page_config(page_title='VAE PneumoniaMNIST - Triagem e Geração', layout='wide')
st.title('VAE PneumoniaMNIST - Triagem de Pneumonia e Geração de Imagens')

