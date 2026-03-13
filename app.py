# ==============================================
# DEPENDÊNCIAS — instale com pip antes de executar:
#
#   pip install streamlit          # framework de interface web interativa
#   pip install tensorflow         # framework de deep learning (inclui Keras)
#   pip install numpy              # operações numéricas e arrays N-dimensionais
#   pip install pandas             # manipulação de dados tabulares (DataFrames)
#   pip install plotly             # gráficos interativos (px.bar, px.scatter, etc.)
#   pip install altair             # gramática declarativa de visualização
#   pip install Pillow             # manipulação de imagens (PIL.Image)
#
# Comando único para instalar tudo de uma vez:
#   pip install streamlit tensorflow numpy pandas plotly altair Pillow
#
# Versões mínimas recomendadas:
#   Python        >= 3.9
#   TensorFlow    >= 2.10
#   Streamlit     >= 1.30
#
# Para executar a aplicação:
#   streamlit run app.py
# ==============================================

# --- Biblioteca padrão do Python ---
import os       # manipulação de caminhos e variáveis de ambiente
import json     # leitura/escrita de arquivos JSON (configuração do modelo)
import io       # streams de bytes em memória (para ler upload do Streamlit)
import time     # controle de tempo (simular latência nas animações)

# --- Computação numérica e dados ---
import numpy as np          # arrays n-dimensionais e operações vetorizadas
import pandas as pd         # DataFrames: histórico de análises, estatísticas

# --- Interface web ---
import streamlit as st      # framework principal da aplicação (UI reativa)

# --- Deep learning ---
import tensorflow as tf     # construção e inferência do VAE

# --- Visualização ---
import plotly.express as px  # gráficos interativos de alto nível (bar, scatter)
import altair as alt         # gráficos declarativos com gramática Vega-Lite

# --- Imagens ---
from PIL import Image        # leitura e conversão de imagens (Pillow)


# ==============================================
# App Streamlit para VAE PneumoniaMNIST
# ==============================================
# Funcionalidades:
# - Triagem de pneumonia baseada no erro de reconstrução
# - Geração de novas imagens de raio-X
# - Upload e reconstrução de imagens
# ==============================================

# -----------------------------------------------
# CAMINHOS DOS ARQUIVOS DO MODELO
# BASE_DIR  → diretório onde este script está localizado
# MODELS_DIR → subpasta 'models/' com pesos e config
# WEIGHTS_PATH → arquivo HDF5 com pesos treinados do VAE
# CONFIG_PATH  → JSON com hiperparâmetros (ex.: latent_dim)
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')


# -----------------------------------------------
# CAMADA CUSTOMIZADA: Reparametrização do VAE
# -----------------------------------------------
# Em um VAE, o espaço latente é parametrizado por (z_mean, z_log_var).
# Para amostrar z de forma diferenciável (backprop), usamos o
# "reparametrization trick": z = z_mean + exp(0.5 * z_log_var) * ε
# onde ε ~ N(0, I). Isso permite que os gradientes fluam através da amostragem.
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        # Desempacota média e log-variância do espaço latente
        z_mean, z_log_var = inputs
        # Ruído gaussiano com mesma forma que z_mean
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        # Reparametrização: desloca e escala o ruído
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# -----------------------------------------------
# ENCODER: comprime a imagem (28×28×1) para o espaço latente
# -----------------------------------------------
# Arquitetura:
#   Input(28,28,1)
#     → Conv2D(32, 3×3, stride=2) → (14,14,32)   ← extrai features de baixo nível
#     → Conv2D(64, 3×3, stride=2) → (7,7,64)     ← features de alto nível + redução espacial
#     → Flatten → (3136,)
#     → Dense(128)                                ← representação compacta
#     → z_mean   (latent_dim,)                    ← média do espaço latente
#     → z_log_var(latent_dim,)                    ← log-variância do espaço latente
#     → Sampling → z (latent_dim,)                ← amostra via reparametrização
def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))  # imagem grayscale 28×28
    # Primeira convolução: reduz resolução espacial pela metade (stride=2)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    # Segunda convolução: nova redução + mais filtros para features complexas
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    # Achata o tensor para vetor 1D (necessário para Dense)
    x = tf.keras.layers.Flatten()(x)
    # Camada densa intermediária: representação compacta antes do gargalo latente
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # Saídas do espaço latente gaussiano
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)       # média
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x) # log-variância
    # Amostragem diferenciável usando o reparametrization trick
    z = Sampling()([z_mean, z_log_var])
    # Retorna modelo Keras com 3 saídas: z_mean, z_log_var e z amostrado
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')


# -----------------------------------------------
# DECODER: reconstrói a imagem a partir do vetor latente z
# -----------------------------------------------
# Arquitetura (espelho do encoder, usando Conv2DTranspose = "deconvolução"):
#   Input(latent_dim,)
#     → Dense(7×7×64) → Reshape(7,7,64)      ← expande para mapa de features
#     → Conv2DTranspose(64, stride=2) → (14,14,64)   ← aumenta resolução
#     → Conv2DTranspose(32, stride=2) → (28,28,32)   ← restaura tamanho original
#     → Conv2DTranspose(1, sigmoid)  → (28,28,1)     ← imagem reconstruída [0,1]
def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))  # vetor latente z
    # Projeta z para mapa de features compatível com as convoluções transpostas
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    # Reformata para tensor 3D (altura × largura × canais)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    # Dobra a resolução: 7×7 → 14×14
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    # Dobra novamente: 14×14 → 28×28
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    # Saída final com 1 canal (grayscale) e sigmoid para valores em [0, 1]
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')


# -----------------------------------------------
# MODELO VAE: une encoder + decoder em um único modelo Keras
# -----------------------------------------------
# O VAE recebe imagens, codifica no espaço latente e reconstrói.
# A loss combina:
#   - Reconstruction loss: quão bem a imagem foi reconstruída (MSE ou BCE)
#   - KL divergence: quão próxima da distribuição N(0,I) é a distribuição latente
#   (o treinamento com essas losses é feito em train_vae.py, não neste arquivo)
class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder  # responsável pela compressão
        self.decoder = decoder  # responsável pela reconstrução

    def call(self, inputs, training=False):
        # Forward pass completo: imagem → latente → reconstrução
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction  # retorna apenas a reconstrução para facilitar uso

    def encode(self, inputs, training=False):
        # Expõe o encoder separadamente (útil para inspecionar o espaço latente)
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        # Expõe o decoder separadamente (útil para gerar imagens a partir de z)
        return self.decoder(z, training=training)


# -----------------------------------------------
# CARREGAMENTO DO MODELO (com cache do Streamlit)
# -----------------------------------------------
# @st.cache_resource: executa apenas UMA vez por sessão do servidor.
# Ideal para objetos pesados como modelos de ML — evita recarregar a cada
# interação do usuário. O cache persiste enquanto o servidor Streamlit estiver ativo.
@st.cache_resource
def load_model():
    # Verifica se os arquivos necessários existem antes de tentar carregar
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'
    # Lê a configuração do modelo (inclui latent_dim e outros hiperparâmetros)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    latent_dim = int(config.get('latent_dim', 16))  # padrão 16 se não especificado
    # Reconstrói a arquitetura do modelo com a mesma latent_dim usada no treino
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    # IMPORTANTE: é preciso fazer um forward pass dummy ANTES de load_weights.
    # Isso força o Keras a construir (build) todas as camadas e alocar os pesos,
    # caso contrário load_weights falharia com erro de variáveis não inicializadas.
    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)
    # Carrega os pesos salvos (formato HDF5 .h5)
    vae.load_weights(WEIGHTS_PATH)
    return vae, None  # retorna (modelo, None) quando bem-sucedido


# -----------------------------------------------
# PRÉ-PROCESSAMENTO DA IMAGEM ENVIADA PELO USUÁRIO
# -----------------------------------------------
# Normaliza qualquer imagem recebida para o formato esperado pelo VAE:
#   - Grayscale (1 canal)
#   - 28×28 pixels (resolução do PneumoniaMNIST)
#   - Valores float32 em [0.0, 1.0]
#   - Shape (1, 28, 28, 1) com batch dimension para inferência
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Converter para grayscale e 28x28
    if image.mode != 'L':
        image = image.convert('L')   # converte RGB/RGBA → grayscale
    if image.size != (28, 28):
        image = image.resize((28, 28))  # redimensiona para 28×28 (interpolação padrão)
    arr = np.array(image).astype('float32')  # converte PIL → numpy float32
    if arr.max() > 1.0:
        arr = arr / 255.0  # normaliza de [0,255] para [0,1]
    arr = np.expand_dims(arr, axis=-1)  # adiciona canal: (28,28) → (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # adiciona batch:  (28,28,1) → (1,28,28,1)
    return arr


# -----------------------------------------------
# CÁLCULO DO ERRO DE RECONSTRUÇÃO (MSE)
# -----------------------------------------------
# @st.cache_data: cacheia resultados de funções puras baseadas nos argumentos.
# Diferente de cache_resource, é usado para dados serializáveis (arrays, DataFrames).
# O MSE (Mean Squared Error) mede pixel a pixel a diferença entre original e reconstrução.
# Em VAEs para detecção de anomalias: MSE alto → imagem anômala (fora da distribuição aprendida).
@st.cache_data
def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    # Erro MSE por imagem
    return float(np.mean((x - x_recon) ** 2))  # média sobre todos os pixels


# -----------------------------------------------
# CLASSIFICAÇÃO POR THRESHOLD DE MSE
# -----------------------------------------------
# Lógica de triagem baseada em limiar duplo:
#   MSE < threshold_normal          → NORMAL (baixo risco)
#   threshold_normal ≤ MSE < threshold_borderline → BORDERLINE (risco moderado)
#   MSE ≥ threshold_borderline      → POSSÍVEL PNEUMONIA (alto risco)
#
# Os thresholds são configurados pelo usuário na barra lateral e devem ser
# calibrados com base nos dados de validação do modelo treinado.
@st.cache_data
def classify_pneumonia(reconstruction_error: float, threshold_normal: float, threshold_borderline: float) -> tuple:
    """
    Classifica se há possível pneumonia baseado no erro de reconstrução.
    Erro alto = possível pneumonia (imagem fora do padrão normal aprendido).
    """
    if reconstruction_error < threshold_normal:
        return "NORMAL", "Baixo risco de pneumonia", "green"
    elif reconstruction_error < threshold_borderline:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red"


# -----------------------------------------------
# GERAÇÃO DE NOVAS IMAGENS SINTÉTICAS
# -----------------------------------------------
# Amostra vetores z do prior N(0,I) e os passa pelo decoder.
# Isso é possível porque o VAE aprendeu um espaço latente estruturado:
# regiões próximas do centro (0,0,...) geram imagens plausíveis.
def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    """Gera novas imagens de raio-X usando o VAE treinado."""
    latent_dim = vae.encoder.output_shape[0][-1]  # Pega a dimensão do z_mean
    
    # Amostrar do espaço latente normal padrão
    # output_shape[0] → saída z_mean; [-1] → último dim = latent_dim
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    
    # Decodificar para gerar imagens
    # .numpy() converte tensor TF para numpy array
    generated_images = vae.decode(z_samples, training=False).numpy()
    
    return generated_images  # shape: (num_images, 28, 28, 1)


# -----------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# -----------------------------------------------
# Deve ser a PRIMEIRA chamada st.* do script (antes de qualquer outro widget).
# layout='wide' aproveita toda a largura do navegador.
st.set_page_config(page_title='VAE PneumoniaMNIST - Triagem e Geração', layout='wide')


# ==========================================================
# INICIALIZAÇÃO DE ESTADO
# ==========================================================
# st.session_state persiste valores entre reruns causados por interações do usuário.
# Cada chave é inicializada apenas uma vez (padrão "if not in").

# Lista de dicionários com o histórico de todas as análises realizadas
if "history" not in st.session_state:
    st.session_state.history = []

# Dicionário com os dados do último resultado computado (evita reprocessamento)
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Lista de feedbacks do usuário (human-in-the-loop): {"classification", "mse", "correct"}
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

# Flag que controla se a análise já foi executada na sessão atual
if "analysis_ran" not in st.session_state:
    st.session_state.analysis_ran = False

# Array numpy com as imagens geradas pelo decoder (None até o usuário gerar)
if "generated_images" not in st.session_state:
    st.session_state.generated_images = None

# Quantidade de imagens geradas na última geração
if "num_generated" not in st.session_state:
    st.session_state.num_generated = 4

# DataFrame tabular para a aba de histórico — inicializado com colunas vazias
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(
        columns=["Execução", "Classificação", "Erro MSE", "Confiança (%)"]
    )

# ==========================================================
# CALLBACK — RESET AO ALTERAR CONFIGURAÇÃO
# ==========================================================
# Chamado automaticamente pelos sliders de threshold via on_change=reset_analysis.
# Garante que ao alterar thresholds o resultado não fique desatualizado na tela.
def reset_analysis():
    st.session_state.analysis_ran = False   # exige nova execução explícita
    st.session_state.last_result = None     # descarta resultado cacheado
    st.toast("Configuração alterada. Execute novamente.")


# ==========================================================
# SIDEBAR — configurações e status do modelo
# ==========================================================
st.sidebar.header("Modelo VAE")

# Carrega o modelo uma única vez (cacheado); exibe status na sidebar
vae, err = load_model()
if err:
    st.sidebar.error(err)   # exibe erro e interrompe a execução se modelo não carregar
    st.stop()               # st.stop() aborta o restante do script imediatamente
else:
    st.sidebar.success("✅ Modelo carregado com sucesso!")
    # output_shape[0][-1] → dimensão da saída z_mean = latent_dim
    st.sidebar.info(f"Dimensão latente: {vae.encoder.output_shape[0][-1]}")

st.sidebar.markdown("---")
st.sidebar.header("Configurações de Triagem")

# Slider: threshold que separa NORMAL de BORDERLINE
# on_change=reset_analysis → invalida resultados ao alterar o threshold
st.sidebar.slider(
    "Threshold Normal (MSE)",
    min_value=0.000, max_value=0.050, value=0.010, step=0.001,
    format="%.3f",
    key="threshold_normal",   # persiste valor em st.session_state.threshold_normal
    on_change=reset_analysis,
    help="MSE abaixo deste valor → NORMAL",
)

# Slider: threshold que separa BORDERLINE de POSSÍVEL PNEUMONIA
st.sidebar.slider(
    "Threshold Borderline (MSE)",
    min_value=0.000, max_value=0.100, value=0.020, step=0.001,
    format="%.3f",
    key="threshold_borderline",  # persiste em st.session_state.threshold_borderline
    on_change=reset_analysis,
    help="MSE entre Normal e este valor → BORDERLINE",
)

# Checkbox: ativa/desativa animações de progresso para simular latência de processamento
st.sidebar.checkbox(
    "Simular latência",
    value=True,
    key="simulate_latency",   # persiste em st.session_state.simulate_latency
)

st.sidebar.markdown("---")

# Botão para limpar o cache de dados (@st.cache_data) — útil ao trocar imagem
if st.sidebar.button("Limpar Cache"):
    st.cache_data.clear()   # invalida apenas cache_data; cache_resource é mantido
    st.sidebar.success("Cache limpo com sucesso.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre")
st.sidebar.info(
    "Triagem de pneumonia via VAE — erro de reconstrução como sinal de anomalia. "
    "Sempre consulte um médico para diagnóstico definitivo."
)


# ==========================================================
# TÍTULO & EMPTY STATE
# ==========================================================
st.title("VAE PneumoniaMNIST — Triagem de Pneumonia e Geração de Imagens")

# Widget de upload: aceita PNG e JPG; retorna objeto UploadedFile ou None
uploaded = st.file_uploader(
    "Envie uma imagem de raio-X para análise (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
)

# Guarda do fluxo: se nenhuma imagem foi enviada, exibe mensagem e para o script
if not uploaded:
    st.info("Envie uma imagem de raio-X para iniciar a análise.")
    st.stop()  # interrompe o restante do script — nada abaixo será renderizado


# ==========================================================
# BOTÃO COMO GATILHO (AÇÃO, NÃO ESTADO)
# ==========================================================
# Padrão recomendado no Streamlit: o botão define o estado, não executa lógica.
# A lógica fica no bloco "if st.session_state.analysis_ran" abaixo,
# garantindo que o resultado permaneça visível mesmo após reruns por outros widgets.
if st.button("🔍 Executar Triagem"):
    st.session_state.analysis_ran = True
    # Chave única por arquivo: nome + tamanho detecta troca de imagem sem hash completo
    st.session_state.run_file_key = uploaded.name + str(uploaded.size)


# ==========================================================
# EXECUÇÃO CONTROLADA PELO ESTADO
# ==========================================================
# Só executa e renderiza resultados se o botão foi clicado nesta sessão.
if st.session_state.analysis_ran:

    # Identifica unicamente o arquivo em análise (nome + tamanho)
    file_key = st.session_state.get("run_file_key", "")

    # --------------------------------------------------------
    # LOADING STATE / LATÊNCIA — só simulação da primeira execução do arquivo
    # --------------------------------------------------------
    # Compara o file_key atual com o da última execução concluída.
    # Animações só aparecem quando é um arquivo novo (evita re-spinner a cada rerun).
    if st.session_state.get("last_file_key") != file_key:
        if st.session_state.simulate_latency:
            # st.spinner: exibe mensagem de loading enquanto o bloco with executa
            with st.spinner("Pré-processando imagem..."):
                time.sleep(0.5)
            with st.spinner("Codificando no espaço latente..."):
                time.sleep(0.5)
            with st.spinner("Reconstruindo imagem..."):
                # st.progress: barra de progresso de 0 a 100 para feedback visual
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # simula processamento gradual
                    bar.progress(i + 1)
            st.toast("Análise concluída.")  # notificação não-bloqueante no canto
        # Marca este arquivo como já processado (evita re-animação)
        st.session_state.last_file_key = file_key

    # --------------------------------------------------------
    # PROCESSAMENTO PRINCIPAL
    # --------------------------------------------------------
    # Lê o arquivo enviado como bytes e abre com Pillow
    image = Image.open(io.BytesIO(uploaded.read()))
    # Normaliza para o formato esperado pelo VAE: (1, 28, 28, 1) float32 [0,1]
    x = preprocess_image(image)
    # Inferência: passa a imagem pelo VAE e obtém a reconstrução
    recon = vae(x, training=False).numpy()
    # Calcula o erro de reconstrução (MSE) entre original e reconstrução
    mse = compute_reconstruction_error(x, recon)

    # Classifica com base nos thresholds configurados na sidebar
    classification, description, color = classify_pneumonia(
        mse,
        st.session_state.threshold_normal,
        st.session_state.threshold_borderline,
    )
    # Confiança estimada: complemento do MSE em %, saturado em [0, 100]
    # Heurística simples: MSE=0 → 100%, MSE=1 → 0%
    confidence_percent = max(0, int((1 - mse) * 100)) if mse < 1 else 0

    # Atualiza resultado e histórico apenas se for nova execução
    # (evita duplicar linhas no histórico ao rerenderizar por outros widgets)
    if st.session_state.last_result is None or st.session_state.last_result.get("file_key") != file_key:
        # Armazena o resultado atual no estado da sessão
        st.session_state.last_result = {
            "x": x, "recon": recon, "mse": mse,
            "classification": classification,
            "confidence": confidence_percent,
            "file_key": file_key,
        }
        # Cria nova linha para o DataFrame histórico
        new_row = pd.DataFrame([{
            "Execução":       len(st.session_state.history) + 1,
            "Classificação":  classification,
            "Erro MSE":       round(mse, 6),
            "Confiança (%)":  confidence_percent,
        }])
        # pd.concat: concatena linha nova ao DataFrame existente (substitui .append deprecated)
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, new_row], ignore_index=True
        )
        # Adiciona ao histórico de dicts (usado em gráficos e monitoramento)
        st.session_state.history.append({
            "classification": classification,
            "mse": mse,
            "confidence": confidence_percent,
        })

    # ==========================================================
    # TABS DE RESULTADOS
    # ==========================================================
    # st.tabs: organiza o conteúdo em abas clicáveis sem recarregar a página
    tab_triagem, tab_dados, tab_monitor, tab_sobre = st.tabs([
        "🔍 Triagem & Validação Humana",
        "📊 Dados & Histórico",
        "📈 Monitoramento",
        "ℹ️ Sobre o Modelo",
    ])

    # ==========================================================
    # TAB 1 — TRIAGEM & VALIDAÇÃO HUMANA
    # ==========================================================
    with tab_triagem:
        # Layout de duas colunas: imagem original vs. reconstruída
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            # squeeze() remove dimensões de tamanho 1: (1,28,28,1) → (28,28)
            st.image(x[0].squeeze(), clamp=True,width=100)
        with col2:
            st.subheader("Reconstrução VAE")
            # clamp=True: garante que valores fora de [0,1] sejam recortados
            st.image(recon[0].squeeze(), clamp=True,width=100)

        st.markdown("---")
        st.subheader("📊 Resultado da Triagem")

        # Delta: diferença em relação à execução anterior (None na primeira)
        prev_mse = st.session_state.history[-2]["mse"] if len(st.session_state.history) >= 2 else None
        delta_mse = f"{(mse - prev_mse):+.6f}" if prev_mse is not None else None

        # KPIs em 3 colunas com st.metric (exibe valor + variação delta)
        m1, m2, m3 = st.columns(3)
        # delta_color="inverse": vermelho se delta positivo (MSE alto = pior)
        m1.metric("Erro de Reconstrução (MSE)", f"{mse:.6f}", delta=delta_mse, delta_color="inverse")
        m2.metric("Classificação", classification)
        m3.metric("Confiança estimada", f"{confidence_percent}%")

        # Barra de progresso como indicador visual de confiança (0-100)
        st.progress(confidence_percent)

        # Alerta contextual com cor semântica (success=verde, warning=amarelo, error=vermelho)
        if color == "green":
            st.success(f"✅ {classification} — {description}")
        elif color == "orange":
            st.warning(f"⚠️ {classification} — {description}")
        else:
            st.error(f"🚨 {classification} — {description}")

        # Banner HTML customizado com a cor da classificação
        # unsafe_allow_html=True: necessário para renderizar HTML cru no Streamlit
        st.markdown(f"""
        <div style="padding:1rem; border-radius:0.5rem;
                    background-color:{color}20; border-left:4px solid {color}; margin-top:0.5rem;">
            <h4 style="color:{color}; margin:0;">{classification}</h4>
            <p style="margin:0.5rem 0 0 0;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

        # Aviso legal obrigatório: sistema é auxiliar, não diagnóstico definitivo
        st.caption("⚠️ **Importante:** Este é apenas um auxiliar de triagem. Sempre consulte um médico para diagnóstico definitivo.")

        st.markdown("---")

        # --------------------------------------------------------
        # HUMAN-IN-THE-LOOP: feedback do especialista médico
        # --------------------------------------------------------
        # Permite que o usuário valide ou corrija a classificação.
        # Os feedbacks ficam em st.session_state.feedback_log e são
        # exibidos na aba de Monitoramento para calcular acurácia percebida.
        st.subheader("Validação Humana")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("✅ Classificação correta"):
                st.session_state.feedback_log.append(
                    {"classification": classification, "mse": mse, "correct": True}
                )
                st.success("Feedback registrado.")
        with fc2:
            if st.button("❌ Classificação incorreta"):
                st.session_state.feedback_log.append(
                    {"classification": classification, "mse": mse, "correct": False}
                )
                st.error("Feedback registrado.")

    # ==========================================================
    # TAB 3 — DADOS & HISTÓRICO
    # ==========================================================
    with tab_dados:
        st.subheader("Histórico de Análises")
        st.caption(
            "Tabela interativa: ordene, redimensione e inspecione. "
            "A coluna *Confiança* é exibida como barra visual."
        )

        if not st.session_state.history_df.empty:
            # st.dataframe com column_config para personalizar tipos e visuais
            st.dataframe(
                st.session_state.history_df,
                use_container_width=True,  # ocupa 100% da largura disponível
                hide_index=True,           # oculta o índice numérico do DataFrame
                column_config={
                    # ProgressColumn: renderiza valores numéricos como barra de progresso
                    "Confiança (%)": st.column_config.ProgressColumn(
                        "Confiança",
                        help="Confiança estimada pelo modelo",
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                    ),
                    # NumberColumn: formata com casas decimais específicas
                    "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
                },
            )

            st.markdown("#### Estatísticas descritivas")
            st.caption("Antes de confiar no gráfico, inspecione o dado bruto.")
            # .describe() gera count, mean, std, min, quartis, max
            st.dataframe(
                st.session_state.history_df[["Erro MSE", "Confiança (%)"]].describe().round(6),
                use_container_width=True,
            )

    # ==========================================================
    # TAB 4 — MONITORAMENTO
    # ==========================================================
    with tab_monitor:
        st.subheader("Monitoramento do Sistema")

        total_fb = len(st.session_state.feedback_log)
        if total_fb > 0:
            # Conta feedbacks marcados como corretos pelo usuário
            correct = sum(1 for f in st.session_state.feedback_log if f["correct"])
            accuracy = correct / total_fb  # proporção de acertos

            mon1, mon2, mon3 = st.columns(3)
            mon1.metric("Feedbacks recebidos", total_fb)
            mon2.metric("Acertos validados", correct)
            mon3.metric("Acurácia percebida", f"{int(accuracy * 100)}%")

            # Alerta de possível degradação se acurácia < 70%
            if accuracy < 0.7:
                st.warning("⚠️ Possível degradação do modelo detectada.")
        else:
            st.info("Ainda não há feedback suficiente para monitoramento.")

        st.markdown("---")

        # Gráfico nativo: evolução do MSE ao longo das execuções
        # st.line_chart: gráfico de linha nativo do Streamlit, rápido para prototipação
        if len(st.session_state.history) > 1:
            st.markdown("#### Evolução do Erro de Reconstrução (MSE)")
            st.caption("Gráfico nativo do Streamlit (`st.line_chart`): ideal para prototipação rápida de séries temporais.")
            mse_series = pd.DataFrame(
                {"Erro MSE": [h["mse"] for h in st.session_state.history]},
                index=range(1, len(st.session_state.history) + 1),
            )
            mse_series.index.name = "Execução"
            st.line_chart(mse_series)

 
        # Filtro interativo com aplicação explícita (evita reruns a cada ajuste do slider)
        if not st.session_state.history_df.empty:
            st.markdown("---")
            st.markdown("#### Filtrar histórico por Erro MSE máximo")
            st.caption("O filtro só é aplicado ao clicar em **Aplicar**, evitando reprocessamento a cada ajuste do slider.")

            # Slider do filtro — não usa on_change para evitar filtragem automática
            max_mse_filter = st.slider(
                "MSE máximo",
                min_value=0.000, max_value=0.100, value=0.050, step=0.001,
                format="%.3f",
                key="mse_filter",
            )

            # Filtro aplicado apenas ao clicar (ação explícita, não reativa)
            if st.button("Aplicar filtro"):
                # Persiste o DataFrame filtrado no estado para sobreviver a reruns
                st.session_state["filtered_history"] = st.session_state.history_df[
                    st.session_state.history_df["Erro MSE"] <= max_mse_filter
                ]

            # Exibe o resultado filtrado se o botão já foi clicado nesta sessão
            if "filtered_history" in st.session_state:
                st.dataframe(
                    st.session_state["filtered_history"],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confiança (%)": st.column_config.ProgressColumn(
                            "Confiança (%)", min_value=0, max_value=100, format="%d%%"
                        ),
                        "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
                    },
                )

    # ==========================================================
    # TAB 5 — SOBRE O MODELO
    # ==========================================================
    with tab_sobre:
        st.header("ℹ️ Sobre o Modelo VAE")
        # Documentação inline da arquitetura e lógica do sistema
        st.markdown("""
        ### Arquitetura do Modelo

        **Encoder:**
        Conv2D(32) → Conv2D(64) → Flatten → Dense(128) → Espaço Latente (z_mean, z_log_var, z)

        **Decoder:**
        Dense(7×7×64) → Reshape → Conv2DTranspose(64) → Conv2DTranspose(32) → Output(sigmoid)

        ### Como Funciona a Triagem

        1. **Imagens Normais:** O VAE foi treinado em imagens normais — erro de reconstrução baixo.
        2. **Imagens com Pneumonia:** Padrões diferentes do aprendido → maior erro de reconstrução.
        3. **Thresholds configuráveis** na barra lateral:
           - MSE < Threshold Normal → **NORMAL**
           - Threshold Normal ≤ MSE < Threshold Borderline → **BORDERLINE**
           - MSE ≥ Threshold Borderline → **POSSÍVEL PNEUMONIA**

        ### Limitações

        - Treinado apenas em PneumoniaMNIST (imagens 28×28 grayscale)
        - Não substitui diagnóstico médico profissional
        - Sensibilidade depende da qualidade e resolução da imagem enviada
        """)

        st.markdown("---")
        st.subheader("Estatísticas do Modelo")
        col1, col2 = st.columns(2)
        with col1:
            # count_params(): conta o total de parâmetros treináveis do sub-modelo
            st.metric("Parâmetros Encoder", f"{vae.encoder.count_params():,}")
            st.metric("Parâmetros Decoder", f"{vae.decoder.count_params():,}")
        with col2:
            st.metric("Total de Parâmetros", f"{vae.count_params():,}")
            st.metric("Dimensão Latente", vae.encoder.output_shape[0][-1])

else:
    # Estado inicial: nenhuma análise executada ainda nesta sessão
    st.info("Configure os parâmetros na barra lateral e clique em **🔍 Executar Triagem**.")


# ==========================================================
# FOOTER
# ==========================================================
# st.markdown("---") gera uma linha horizontal <hr> no HTML
st.markdown("---")
st.caption(
    "🔬 **Modelo VAE para Triagem de Pneumonia** | "
    "Desenvolvido com TensorFlow e Streamlit | "
    "Sempre consulte um médico para diagnóstico definitivo."
)