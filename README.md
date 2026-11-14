# 🎬 Tradutor & Dublador Automático de Vídeos com IA (`tradutor-video`)

Sistema completo, moderno e **100% gratuito** para **Tradução e Dublagem Automática de Vídeos com Inteligência Artificial**, projetado e otimizado para rodar localmente em computadores com **16 GB de RAM e GPU integrada ou de entrada** (utilizando quantização `int8`, filtro de voz `VAD` e processamento assíncrono leve).

---

## ✨ Principais Funcionalidades

- 🚀 **Executor Zero-Dependência (`Iniciar_Tradutor.bat`):** Basta dar **dois cliques**! Mesmo que o computador esteja recém-formatado e **não tenha Python nem FFmpeg instalados**, o sistema baixa e configura automaticamente o Python portátil e o FFmpeg dentro da pasta local `.runtime/`, cria o ambiente virtual `.venv/` e abre a interface no navegador.
- 🧠 **Transcrição Inteligente (`faster-whisper`):** Extração de falas com timestamps precisos usando `compute_type="int8"` e `vad_filter=True` para máxima velocidade e baixo consumo de memória RAM.
- 🌍 **Tradução Multilíngue (`deep-translator`):** Detecção automática do idioma original (`auto`, Inglês, Espanhol, Francês, Alemão, Japonês, Chinês, Italiano, etc.) e tradução natural para Português (Brasil) ou diversos outros idiomas.
- 🎙️ **Vozes Neurais Ultra-Naturais de 2ª Geração (`edge-tts`):**
  - `en-US-AndrewMultilingualNeural` *(Masculina Ultra-Natural — fala PT-BR e múltiplos idiomas com entonação humana)*
  - `en-US-BrianMultilingualNeural` *(Masculina Narrador Natural — Multilíngue)*
  - `pt-BR-ThalitaMultilingualNeural` *(Feminina Ultra-Natural PT-BR)*
  - `pt-BR-AntonioNeural` *(Masculina Clássica PT-BR)*
  - `pt-BR-FranciscaNeural` *(Feminina Clássica PT-BR)*
- ⏱️ **Sincronia Temporal Dinâmica (Time-Stretching sem Distorcer o Tom):**
  - Mede a duração de cada fala traduzida em relação à janela disponível até a próxima fala do vídeo.
  - Se couber no tempo original, mantém a velocidade **100% natural (`1.0x`)**.
  - Se a frase traduzida ficar mais longa, aplica automaticamente o filtro `atempo` do FFmpeg (limitado entre `1.0x` e `1.35x`) para encaixar perfeitamente sem deixar a voz robótica.
- 🎚️ **Mixagem com Áudio Ambiente (Ducking):** Reduz suavemente o volume da faixa original (configurável de `-5 dB` a `-40 dB`, padrão `-20 dB`) e sobrepõe cada fala traduzida no milissegundo exato (`seg.start * 1000`).
- ⚡ **Renderização Instantânea (`-c:v copy -c:a aac`):** Substitui a faixa de áudio sem re-renderizar o vídeo original, mantendo 100% da qualidade de imagem em segundos.
- 🖥️ **Player HTML5 Integrado com Tela Cheia (Fullscreen):**
  - Assista ao vídeo dublado direto na interface assim que ficar pronto.
  - Alterne em tempo real entre **🔊 Vídeo Dublado** e **🎬 Vídeo Original** mantendo o mesmo segundo de reprodução.
  - Suporte completo a **Tela Cheia (Fullscreen)**, **Picture-in-Picture**, controle de velocidade e clique direto nas falas traduzidas para pular para o trecho correspondente.
  - Botão dedicado **"Exportar / Baixar Vídeo Traduzido (.MP4)"** (com seletor de pasta para salvar onde desejar) + exportação automática de legendas **`.SRT`**.

---

## 🚀 Como Usar (Passo a Passo Simples)

### Opção 1: Baixar o ZIP (Recomendado para qualquer usuário)
1. Nesta página do GitHub, clique no botão verde **`<> Code`** e depois em **`Download ZIP`**.
2. Extraia o arquivo `.zip` em qualquer pasta do seu computador (ex.: `Documentos\tradutor-video`).
3. Dê **dois cliques** no arquivo **`Iniciar_Tradutor.bat`**.
4. Aguarde a configuração automática na primeira execução — o navegador abrirá sozinho em **`http://127.0.0.1:8000`**!

### Opção 2: Via Git no Terminal
```bash
git clone https://github.com/euefrai/tradutor-video.git
cd tradutor-video
Iniciar_Tradutor.bat
```

---

## 📂 Estrutura do Projeto

| Arquivo | Descrição |
| :--- | :--- |
| [`Iniciar_Tradutor.bat`](Iniciar_Tradutor.bat) | Lançador universal para Windows (Zero-Dependência: basta dar dois cliques) |
| [`setup_runtime.ps1`](setup_runtime.ps1) | Script PowerShell auxiliar que instala/baixa Python 3.11 e FFmpeg portáteis em `.runtime/` caso necessário |
| [`app.py`](app.py) | Servidor FastAPI e Interface Gráfica Dark Mode completa com Player Fullscreen, comparador e exportação |
| [`dubber_engine.py`](dubber_engine.py) | Motor de transcrição (`faster-whisper`), tradução, síntese (`edge-tts`), sincronia `atempo` e mixagem (`pydub` + `ffmpeg`) |
| [`requirements.txt`](requirements.txt) | Lista completa de dependências Python do projeto |
| [`.gitignore`](.gitignore) | Ignora `.venv/`, `.runtime/`, `temp_tts/`, `outputs/` e `__pycache__/` |

---

## 🛠️ Requisitos de Sistema

- **Sistema Operacional:** Windows 10 ou Windows 11 (64-bit)
- **Memória RAM:** 8 GB mínimo (Otimizado para **16 GB de RAM**)
- **Placa de Vídeo:** Funciona 100% em CPU ou GPU integrada/entrada (quantização `int8` automática)
- **Conexão com a Internet:** Necessária para buscar a tradução e sintetizar as vozes neurais gratuitas da Microsoft (`edge-tts`)
