# CP1 — Detecção de Pneumonia com VAE

Aplicação Streamlit que utiliza um Variational Autoencoder (VAE) treinado para detectar pneumonia em radiografias de tórax.

## Equipe

| Nome | RM |
|---|---|
| Carlos Bucker | 555812 |
| Filipe Melo da Silva | 564571 |
| Luca Schmidt | 560255 |

## Pré-requisitos

- Python 3.9+
- Git

## Rodando localmente

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd CP1-Front-End-Mobile-Dev

# 2. Crie e ative o virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
streamlit run app.py
```

Acesse em `http://localhost:8501`.

## Deploy — Streamlit Community Cloud

1. Faça push do repositório para o GitHub (branch `main`)
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte sua conta GitHub
3. Clique em **New app** → selecione o repositório, branch `main` e arquivo `app.py`
4. Clique em **Deploy** — o Streamlit Cloud instala as dependências automaticamente via `requirements.txt`

> **Nota:** o arquivo de pesos `models/vae_pneumonia.weights.h5` está versionado no repositório (6 MB) e será disponibilizado automaticamente no deploy.

## Estrutura

```
app.py                  # Interface Streamlit
engine.py               # Funções auxiliares do modelo
requirements.txt        # Dependências Python
models/
  config.json           # Configuração da arquitetura do VAE
  vae_pneumonia.weights.h5  # Pesos treinados
.streamlit/
  config.toml           # Configuração do servidor Streamlit
```