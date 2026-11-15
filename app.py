import asyncio
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

from dubber_engine import (
    BASE_DIR,
    OUTPUTS_DIR,
    TEMP_TTS_DIR,
    configure_ffmpeg_environment,
    generate_voice_preview,
    process_video_dubbing,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tradutor_app")

UPLOADS_DIR = BASE_DIR / "outputs" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Tradutor & Dublador Automático de Vídeos IA",
    description="Sistema 100% gratuito de tradução e dublagem neural sincronizada com faster-whisper + edge-tts + FFmpeg",
    version="1.0.0",
)

# Armazenamento em memória do estado dos jobs
JOBS: Dict[str, Dict] = {}

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@app.get("/api/health")
async def health_check():
    ffmpeg_ok = False
    ffmpeg_path = ""
    try:
        ffmpeg_path, _ = configure_ffmpeg_environment()
        ffmpeg_ok = True
    except Exception as exc:
        ffmpeg_path = str(exc)

    return {
        "status": "ok",
        "ffmpeg_ready": ffmpeg_ok,
        "ffmpeg_path": ffmpeg_path,
    }


@app.post("/api/preview-voice")
async def preview_voice(
    voice: str = Form("en-US-AndrewMultilingualNeural"),
    text: str = Form(
        "Olá! Esta é uma demonstração da voz neural ultra-natural configurada no seu tradutor de vídeos."
    ),
):
    try:
        audio_path = await generate_voice_preview(voice=voice, sample_text=text)
        return FileResponse(
            path=str(audio_path),
            media_type="audio/mpeg",
            filename=f"preview_{voice}.mp3",
        )
    except Exception as exc:
        logger.exception("Erro ao gerar preview de voz")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/dub")
async def start_dubbing_job(
    video: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("pt"),
    voice: str = Form("en-US-AndrewMultilingualNeural"),
    ducking_db: float = Form(-20.0),
    whisper_model: str = Form("small"),
):
    orig_filename = video.filename or "video.mp4"
    ext = Path(orig_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato '{ext}' não suportado. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job_id = uuid.uuid4().hex[:10]
    safe_stem = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in Path(orig_filename).stem
    )[:50]
    input_path = UPLOADS_DIR / f"{job_id}_{safe_stem}{ext}"
    output_filename = f"{safe_stem}_dublado_{target_lang}_{job_id}.mp4"
    output_path = OUTPUTS_DIR / output_filename

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "percent": 2,
        "stage": "Iniciando",
        "message": f"Vídeo '{orig_filename}' recebido. Preparando pipeline...",
        "original_filename": orig_filename,
        "output_filename": output_filename,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "srt_path": str(output_path.with_suffix(".srt")),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "voice": voice,
        "ducking_db": ducking_db,
        "whisper_model": whisper_model,
        "segments": [],
        "logs": [f"[0%] Vídeo '{orig_filename}' carregado com sucesso."],
        "error": None,
        "elapsed_seconds": None,
    }

    def progress_cb(percent: int, stage: str, message: str, seg_dict: Optional[Dict] = None):
        job = JOBS.get(job_id)
        if not job:
            return
        job["percent"] = percent
        job["stage"] = stage
        job["message"] = message
        job["logs"].append(f"[{percent}%] {stage}: {message}")
        if seg_dict:
            job["segments"].append(seg_dict)

    async def _run_job():
        try:
            result = await process_video_dubbing(
                video_path=input_path,
                output_video_path=output_path,
                source_lang=source_lang,
                target_lang=target_lang,
                voice=voice,
                ducking_db=ducking_db,
                whisper_model_size=whisper_model,
                progress_callback=progress_cb,
            )
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["percent"] = 100
            JOBS[job_id]["stage"] = "Concluído"
            JOBS[job_id]["detected_language"] = result.get("detected_language")
            JOBS[job_id]["elapsed_seconds"] = result.get("elapsed_seconds")
        except Exception as exc:
            logger.exception(f"Erro no processamento do job {job_id}")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["stage"] = "Erro"
            JOBS[job_id]["message"] = str(exc)
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["logs"].append(f"[ERRO] {exc}")

    asyncio.create_task(_run_job())
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return job


@app.get("/api/video/original/{job_id}")
async def stream_original_video(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    path = Path(job["input_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo original não encontrado.")
    return FileResponse(path=str(path), filename=job["original_filename"])


@app.get("/api/video/dubbed/{job_id}")
async def stream_dubbed_video(job_id: str, download: bool = False):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Vídeo dublado ainda não disponível.")
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{job["output_filename"]}"'
    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename=job["output_filename"],
        headers=headers,
    )


@app.get("/api/subtitle/{job_id}")
async def download_srt(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    srt_path = Path(job["srt_path"])
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo SRT não encontrado.")
    return FileResponse(
        path=str(srt_path),
        media_type="text/plain; charset=utf-8",
        filename=srt_path.name,
    )


@app.post("/api/open-folder/{job_id}")
async def open_output_folder(job_id: str):
    job = JOBS.get(job_id)
    target_path = Path(job["output_path"]) if job and Path(job["output_path"]).exists() else OUTPUTS_DIR
    try:
        if os.name == "nt":
            if target_path.is_file():
                subprocess.Popen(["explorer", f"/select,{str(target_path)}"])
            else:
                subprocess.Popen(["explorer", str(OUTPUTS_DIR)])
        return {"status": "opened", "path": str(target_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


HTML_INTERFACE = r"""<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tradutor & Dublador IA — Estúdio Neural Local</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#eef2ff',
              400: '#818cf8',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
            }
          }
        }
      }
    }
  </script>
  <style>
    body {
      background: radial-gradient(circle at top right, #1e1b4b 0%, #090d16 45%, #05070b 100%);
      color: #f1f5f9;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .glass-card {
      background: rgba(15, 23, 42, 0.78);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .drop-active {
      border-color: #6366f1 !important;
      background: rgba(99, 102, 241, 0.12) !important;
    }
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
      background: #334155;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #475569;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col selection:bg-indigo-500 selection:text-white">

  <!-- Cabeçalho -->
  <header class="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md sticky top-0 z-30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
          </svg>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-lg font-bold tracking-tight text-white">Tradutor & Dublador Automático de Vídeos</h1>
            <span class="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">100% Local & Gratuito</span>
          </div>
          <p class="text-xs text-slate-400">Otimizado para 16 GB RAM (int8 + VAD) • Vozes Neurais 2ª Geração • Sincronia Dinâmica FFmpeg</p>
        </div>
      </div>

      <div id="healthBadge" class="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span>Motor FFmpeg & Whisper Prontos</span>
      </div>
    </div>
  </header>

  <!-- Conteúdo Principal -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Coluna Esquerda: Importação e Configurações (5 colunas) -->
    <section class="lg:col-span-5 flex flex-col gap-5">

      <!-- 1. Importação de Vídeo -->
      <div class="glass-card rounded-2xl p-5 shadow-xl">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-sm font-semibold uppercase tracking-wider text-indigo-400 flex items-center gap-2">
            <span>1. Importar Vídeo</span>
          </h2>
          <span class="text-xs text-slate-400">.MP4, .MKV, .AVI, .MOV, .WEBM</span>
        </div>

        <label id="dropZone" for="videoInput"
          class="group cursor-pointer border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-6 flex flex-col items-center justify-center text-center transition-all duration-200 bg-slate-900/50 hover:bg-slate-900/90">
          <input id="videoInput" type="file" accept=".mp4,.mkv,.avi,.mov,.webm,video/*" class="hidden" />
          <div class="w-12 h-12 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
            <svg class="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
          </div>
          <p id="fileNameDisplay" class="text-sm font-medium text-slate-200">
            Arraste e solte o vídeo aqui ou <span class="text-indigo-400 underline">clique para selecionar</span>
          </p>
          <p id="fileSizeDisplay" class="text-xs text-slate-400 mt-1">
            Processamento direto no seu computador sem limite de tempo
          </p>
        </label>
      </div>

      <!-- 2. Configurações de Tradução e Vozes Ultra-Naturais -->
      <div class="glass-card rounded-2xl p-5 shadow-xl flex flex-col gap-4">
        <h2 class="text-sm font-semibold uppercase tracking-wider text-indigo-400">
          2. Idiomas, Voz Neural & Sincronia
        </h2>

        <!-- Idiomas -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          <div>
            <label class="block text-xs font-medium text-slate-300 mb-1.5">Idioma de Origem</label>
            <select id="sourceLang" class="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
              <option value="auto" selected>✨ Detecção Automática (Auto)</option>
              <option value="en">🇺🇸 Inglês (English)</option>
              <option value="es">🇪🇸 Espanhol (Español)</option>
              <option value="fr">🇫🇷 Francês (Français)</option>
              <option value="de">🇩🇪 Alemão (Deutsch)</option>
              <option value="it">🇮🇹 Italiano (Italiano)</option>
              <option value="ja">🇯🇵 Japonês (日本語)</option>
              <option value="zh-CN">🇨🇳 Chinês (中文)</option>
              <option value="ko">🇰🇷 Coreano (한국어)</option>
              <option value="ru">🇷🇺 Russo (Русский)</option>
              <option value="pt">🇧🇷 Português</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-300 mb-1.5">Idioma de Destino</label>
            <select id="targetLang" class="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
              <option value="pt" selected>🇧🇷 Português (Brasil)</option>
              <option value="en">🇺🇸 Inglês (English)</option>
              <option value="es">🇪🇸 Espanhol (Español)</option>
              <option value="fr">🇫🇷 Francês (Français)</option>
              <option value="de">🇩🇪 Alemão (Deutsch)</option>
              <option value="it">🇮🇹 Italiano (Italiano)</option>
              <option value="ja">🇯🇵 Japonês (日本語)</option>
              <option value="zh-CN">🇨🇳 Chinês Simplificado</option>
            </select>
          </div>
        </div>

        <!-- Seletor de Vozes Neurais Realistas -->
        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="block text-xs font-medium text-slate-300">Voz Neural Realista (Edge-TTS 2ª Geração)</label>
            <button id="previewVoiceBtn" type="button"
              class="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium transition-colors">
              <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"/>
              </svg>
              <span>Ouvir Prévia da Voz</span>
            </button>
          </div>
          <select id="voiceSelect" class="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
            <option value="en-US-AndrewMultilingualNeural" selected>🎙️ Andrew Multilingual (Masculina Ultra-Natural - PT-BR / Multilíngue)</option>
            <option value="en-US-BrianMultilingualNeural">🎙️ Brian Multilingual (Masculina Narrador Natural - Multilíngue)</option>
            <option value="pt-BR-ThalitaMultilingualNeural">🎙️ Thalita Multilingual (Feminina Ultra-Natural PT-BR)</option>
            <option value="pt-BR-AntonioNeural">🎙️ Antonio Neural (Masculina Clássica PT-BR)</option>
            <option value="pt-BR-FranciscaNeural">🎙️ Francisca Neural (Feminina Clássica PT-BR)</option>
          </select>
          <audio id="voicePreviewAudio" class="hidden"></audio>
        </div>

        <!-- Controle de Ducking -->
        <div class="bg-slate-900/70 border border-slate-800 rounded-xl p-3.5">
          <div class="flex items-center justify-between mb-1.5">
            <label for="duckingSlider" class="text-xs font-medium text-slate-300">
              Volume do Áudio Original ao Fundo (Ducking)
            </label>
            <span id="duckingValueBadge" class="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              -20 dB
            </span>
          </div>
          <input id="duckingSlider" type="range" min="-40" max="-5" step="1" value="-20"
            class="w-full accent-indigo-500 cursor-pointer" />
          <div class="flex justify-between text-[11px] text-slate-500 mt-1">
            <span>-40 dB (Quase Mudo)</span>
            <span>-20 dB (Padrão Ideal)</span>
            <span>-5 dB (Ambiente Alto)</span>
          </div>
        </div>

        <!-- Modelo Faster-Whisper -->
        <div>
          <label class="block text-xs font-medium text-slate-300 mb-1.5">
            Modelo de Transcrição IA (<code class="text-indigo-300">faster-whisper int8 + VAD</code>)
          </label>
          <select id="whisperModel" class="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
            <option value="tiny">⚡ Tiny (Ultra-Rápido • ~1 GB RAM)</option>
            <option value="base">🚀 Base (Equilibrado Leve • ~1.5 GB RAM)</option>
            <option value="small" selected>🎯 Small (Recomendado • Alta Precisão int8 • Ideal 16 GB RAM)</option>
          </select>
        </div>

        <!-- Botão Iniciar Dublagem -->
        <button id="startDubBtn" type="button"
          class="mt-1 w-full py-3.5 px-5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 active:scale-[0.99] text-white font-semibold text-sm shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <span id="startDubBtnText">Traduzir e Dublar Vídeo Agora</span>
        </button>
      </div>
    </section>

    <!-- Coluna Direita: Player Integrado, Exportação e Progresso (7 colunas) -->
    <section class="lg:col-span-7 flex flex-col gap-5">

      <!-- 3. Player de Vídeo Integrado com Comparador Original vs Dublado e Fullscreen -->
      <div class="glass-card rounded-2xl p-5 shadow-xl flex flex-col gap-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div class="flex items-center gap-2">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-indigo-400">
              3. Player de Vídeo Integrado
            </h2>
            <span id="playerModeBadge" class="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hidden">
              Dublado Ativo
            </span>
          </div>

          <!-- Alternador de Abas: Vídeo Dublado vs Vídeo Original -->
          <div class="inline-flex rounded-xl bg-slate-900 p-1 border border-slate-800">
            <button id="tabDubbedBtn" type="button"
              class="px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow">
              🔊 Vídeo Dublado
            </button>
            <button id="tabOriginalBtn" type="button"
              class="px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white">
              🎬 Vídeo Original
            </button>
          </div>
        </div>

        <!-- Container do Vídeo -->
        <div id="videoContainer" class="relative w-full aspect-video rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center group">
          <video id="mainVideoPlayer" class="w-full h-full object-contain hidden" controls playsinline preload="metadata"></video>

          <!-- Placeholder inicial -->
          <div id="videoPlaceholder" class="flex flex-col items-center justify-center text-center p-6 text-slate-500">
            <svg class="w-14 h-14 mb-3 text-slate-700" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
            </svg>
            <p class="text-sm font-medium text-slate-400">Selecione um vídeo ao lado para pré-visualizar e iniciar a dublagem</p>
            <p class="text-xs text-slate-600 mt-1">Suporte a Tela Cheia (Fullscreen), Picture-in-Picture, Velocidade e Comparação Instantânea</p>
          </div>
        </div>

        <!-- Barra de Controles Extras do Player (Velocidade, PiP, Fullscreen dedicados) -->
        <div class="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-slate-800/80 text-xs text-slate-300">
          <div class="flex items-center gap-3">
            <label for="playbackSpeed" class="text-slate-400">Velocidade do Player:</label>
            <select id="playbackSpeed" class="rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1 text-xs text-white focus:outline-none focus:border-indigo-500">
              <option value="0.75">0.75x</option>
              <option value="1" selected>1.0x (Normal)</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2.0x</option>
            </select>
          </div>

          <div class="flex items-center gap-2">
            <button id="pipBtn" type="button"
              class="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 flex items-center gap-1.5 transition-colors">
              <span>🗔 Picture-in-Picture</span>
            </button>
            <button id="fullscreenBtn" type="button"
              class="px-3.5 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 font-medium flex items-center gap-1.5 transition-colors">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
              </svg>
              <span>Tela Cheia (Fullscreen)</span>
            </button>
          </div>
        </div>

        <!-- 4. Área de Exportação e Download -->
        <div id="exportPanel" class="mt-1 pt-4 border-t border-slate-800 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <div class="text-xs text-slate-400">
            <p class="font-medium text-slate-200">Exportação Rápida sem Perda de Qualidade (<code class="text-indigo-300">-c:v copy</code>)</p>
            <p id="exportSubtext">O botão de download será habilitado assim que a dublagem terminar.</p>
          </div>
          <div class="flex flex-wrap items-center gap-2.5">
            <button id="openFolderBtn" type="button" disabled
              class="px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium text-slate-200 border border-slate-700 transition-all">
              📂 Abrir Pasta Local
            </button>
            <button id="downloadSrtBtn" type="button" disabled
              class="px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium text-slate-200 border border-slate-700 transition-all">
              📝 Baixar Legenda (.SRT)
            </button>
            <button id="exportVideoBtn" type="button" disabled
              class="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-semibold text-white shadow-lg shadow-emerald-600/25 flex items-center gap-2 transition-all">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
              </svg>
              <span>Exportar / Baixar Vídeo Traduzido (.MP4)</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Progresso em Tempo Real e Log de Falas Traduzidas -->
      <div class="glass-card rounded-2xl p-5 shadow-xl flex flex-col gap-3 flex-1">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-indigo-400">
              Progresso & Sincronia das Falas
            </h2>
            <span id="stageBadge" class="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              Aguardando
            </span>
          </div>
          <span id="percentText" class="text-sm font-mono font-bold text-indigo-400">0%</span>
        </div>

        <!-- Barra de Progresso -->
        <div class="w-full h-3 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
          <div id="progressBar" class="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-400 transition-all duration-300" style="width: 0%"></div>
        </div>

        <p id="statusMessage" class="text-xs text-slate-400">
          Nenhum processamento em andamento.
        </p>

        <!-- Lista de Segmentos Traduzidos e Sincronizados -->
        <div id="segmentsContainer" class="mt-1 bg-slate-950/80 border border-slate-800/90 rounded-xl p-3 h-56 overflow-y-auto flex flex-col gap-2 text-xs">
          <div class="text-slate-500 text-center my-auto">
            As falas transcritas, traduzidas e a taxa de sincronia temporal (<code class="text-indigo-400">1.00x - 1.35x</code>) aparecerão aqui em tempo real.
          </div>
        </div>
      </div>

    </section>
  </main>

  <script>
    const videoInput = document.getElementById('videoInput');
    const dropZone = document.getElementById('dropZone');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const fileSizeDisplay = document.getElementById('fileSizeDisplay');
    const duckingSlider = document.getElementById('duckingSlider');
    const duckingValueBadge = document.getElementById('duckingValueBadge');
    const previewVoiceBtn = document.getElementById('previewVoiceBtn');
    const voiceSelect = document.getElementById('voiceSelect');
    const voicePreviewAudio = document.getElementById('voicePreviewAudio');
    const startDubBtn = document.getElementById('startDubBtn');
    const startDubBtnText = document.getElementById('startDubBtnText');

    const mainVideoPlayer = document.getElementById('mainVideoPlayer');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const videoContainer = document.getElementById('videoContainer');
    const tabDubbedBtn = document.getElementById('tabDubbedBtn');
    const tabOriginalBtn = document.getElementById('tabOriginalBtn');
    const playerModeBadge = document.getElementById('playerModeBadge');
    const playbackSpeed = document.getElementById('playbackSpeed');
    const pipBtn = document.getElementById('pipBtn');
    const fullscreenBtn = document.getElementById('fullscreenBtn');

    const exportVideoBtn = document.getElementById('exportVideoBtn');
    const downloadSrtBtn = document.getElementById('downloadSrtBtn');
    const openFolderBtn = document.getElementById('openFolderBtn');
    const exportSubtext = document.getElementById('exportSubtext');

    const stageBadge = document.getElementById('stageBadge');
    const percentText = document.getElementById('percentText');
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');
    const segmentsContainer = document.getElementById('segmentsContainer');

    let selectedFile = null;
    let localPreviewUrl = null;
    let currentJobId = null;
    let pollTimer = null;
    let renderedSegmentsCount = 0;
    let activeTab = 'dubbed'; // 'dubbed' | 'original'
    let dubbedVideoUrl = null;
    let originalServerUrl = null;
    let currentOutputFilename = 'video_dublado.mp4';

    // Slider de Ducking
    duckingSlider.addEventListener('input', () => {
      duckingValueBadge.textContent = `${duckingSlider.value} dB`;
    });

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(evt => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drop-active');
      });
    });

    ['dragleave', 'drop'].forEach(evt => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drop-active');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        handleSelectedVideo(files[0]);
      }
    });

    videoInput.addEventListener('change', () => {
      if (videoInput.files && videoInput.files.length > 0) {
        handleSelectedVideo(videoInput.files[0]);
      }
    });

    function handleSelectedVideo(file) {
      selectedFile = file;
      const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
      fileNameDisplay.innerHTML = `<span class="text-emerald-400 font-semibold">✔ ${file.name}</span>`;
      fileSizeDisplay.textContent = `Tamanho: ${sizeMB} MB • Pronto para traduzir e dublar`;

      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
      }
      localPreviewUrl = URL.createObjectURL(file);
      originalServerUrl = localPreviewUrl;
      dubbedVideoUrl = null;

      switchPlayerTab('original');
    }

    function switchPlayerTab(mode) {
      const prevTime = mainVideoPlayer.currentTime || 0;
      const wasPlaying = !mainVideoPlayer.paused;
      activeTab = mode;

      if (mode === 'dubbed') {
        tabDubbedBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow';
        tabOriginalBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white';
        if (dubbedVideoUrl) {
          videoPlaceholder.classList.add('hidden');
          mainVideoPlayer.classList.remove('hidden');
          mainVideoPlayer.src = dubbedVideoUrl;
          mainVideoPlayer.currentTime = prevTime;
          playerModeBadge.textContent = '🔊 Reproduzindo: Vídeo Dublado';
          playerModeBadge.classList.remove('hidden');
          if (wasPlaying) mainVideoPlayer.play().catch(() => {});
        }
      } else {
        tabOriginalBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow';
        tabDubbedBtn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white';
        if (originalServerUrl) {
          videoPlaceholder.classList.add('hidden');
          mainVideoPlayer.classList.remove('hidden');
          mainVideoPlayer.src = originalServerUrl;
          mainVideoPlayer.currentTime = prevTime;
          playerModeBadge.textContent = '🎬 Reproduzindo: Vídeo Original';
          playerModeBadge.classList.remove('hidden');
          if (wasPlaying) mainVideoPlayer.play().catch(() => {});
        }
      }
    }

    tabDubbedBtn.addEventListener('click', () => {
      if (!dubbedVideoUrl) {
        alert('O vídeo dublado ficará disponível aqui assim que o processamento terminar.');
        return;
      }
      switchPlayerTab('dubbed');
    });

    tabOriginalBtn.addEventListener('click', () => {
      if (!originalServerUrl) return;
      switchPlayerTab('original');
    });

    // Velocidade de Reprodução
    playbackSpeed.addEventListener('change', () => {
      mainVideoPlayer.playbackRate = parseFloat(playbackSpeed.value);
    });

    // Picture-in-Picture
    pipBtn.addEventListener('click', async () => {
      try {
        if (document.pictureInPictureElement) {
          await document.exitPictureInPicture();
        } else if (mainVideoPlayer.src) {
          await mainVideoPlayer.requestPictureInPicture();
        }
      } catch (err) {
        console.warn('PiP indisponível:', err);
      }
    });

    // Tela Cheia (Fullscreen)
    fullscreenBtn.addEventListener('click', () => {
      const target = !mainVideoPlayer.classList.contains('hidden') ? mainVideoPlayer : videoContainer;
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else if (target.requestFullscreen) {
        target.requestFullscreen();
      } else if (target.webkitRequestFullscreen) {
        target.webkitRequestFullscreen();
      }
    });

    // Ouvir Prévia da Voz Neural
    previewVoiceBtn.addEventListener('click', async () => {
      const voice = voiceSelect.value;
      const originalLabel = previewVoiceBtn.innerHTML;
      previewVoiceBtn.disabled = true;
      previewVoiceBtn.innerHTML = '<span>⏳ Gerando voz...</span>';
      try {
        const form = new FormData();
        form.append('voice', voice);
        const res = await fetch('/api/preview-voice', { method: 'POST', body: form });
        if (!res.ok) throw new Error('Falha ao sintetizar prévia de voz');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        voicePreviewAudio.src = url;
        await voicePreviewAudio.play();
      } catch (err) {
        alert('Erro ao testar voz: ' + err.message);
      } finally {
        previewVoiceBtn.disabled = false;
        previewVoiceBtn.innerHTML = originalLabel;
      }
    });

    // Iniciar Dublagem
    startDubBtn.addEventListener('click', async () => {
      if (!selectedFile) {
        alert('Por favor, selecione ou arraste um arquivo de vídeo primeiro!');
        return;
      }

      startDubBtn.disabled = true;
      startDubBtnText.textContent = 'Enviando e Processando...';
      exportVideoBtn.disabled = true;
      downloadSrtBtn.disabled = true;
      openFolderBtn.disabled = true;
      renderedSegmentsCount = 0;
      segmentsContainer.innerHTML = '';

      const formData = new FormData();
      formData.append('video', selectedFile);
      formData.append('source_lang', document.getElementById('sourceLang').value);
      formData.append('target_lang', document.getElementById('targetLang').value);
      formData.append('voice', voiceSelect.value);
      formData.append('ducking_db', duckingSlider.value);
      formData.append('whisper_model', document.getElementById('whisperModel').value);

      try {
        stageBadge.textContent = 'Enviando Vídeo';
        statusMessage.textContent = 'Carregando vídeo para o motor local...';
        progressBar.style.width = '4%';
        percentText.textContent = '4%';

        const res = await fetch('/api/dub', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Falha ao iniciar dublagem');
        }

        currentJobId = data.job_id;
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollJobStatus, 1000);
      } catch (err) {
        startDubBtn.disabled = false;
        startDubBtnText.textContent = 'Traduzir e Dublar Vídeo Agora';
        stageBadge.textContent = 'Erro';
        statusMessage.textContent = err.message;
        alert('Erro: ' + err.message);
      }
    });

    function formatSeconds(sec) {
      const m = Math.floor(sec / 60).toString().padStart(2, '0');
      const s = Math.floor(sec % 60).toString().padStart(2, '0');
      return `${m}:${s}`;
    }

    async function pollJobStatus() {
      if (!currentJobId) return;
      try {
        const res = await fetch(`/api/jobs/${currentJobId}`);
        if (!res.ok) return;
        const job = await res.json();

        progressBar.style.width = `${job.percent}%`;
        percentText.textContent = `${job.percent}%`;
        stageBadge.textContent = job.stage || 'Processando';
        statusMessage.textContent = job.message || '';

        if (Array.isArray(job.segments) && job.segments.length > renderedSegmentsCount) {
          for (let i = renderedSegmentsCount; i < job.segments.length; i++) {
            const seg = job.segments[i];
            const item = document.createElement('div');
            item.className = 'p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 hover:border-indigo-500/40 transition-colors cursor-pointer';
            item.title = 'Clique para pular para este trecho no vídeo';
            item.addEventListener('click', () => {
              if (mainVideoPlayer.src) {
                mainVideoPlayer.currentTime = seg.start;
                mainVideoPlayer.play().catch(() => {});
              }
            });

            const speedBadgeClass = seg.speed_factor > 1.01
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
              : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';

            item.innerHTML = `
              <div class="flex items-center justify-between gap-2 mb-1">
                <span class="font-mono text-[11px] text-indigo-400 font-semibold">
                  #${seg.index} • ${formatSeconds(seg.start)} → ${formatSeconds(seg.end)}
                </span>
                <span class="text-[10px] font-mono px-2 py-0.5 rounded border ${speedBadgeClass}">
                  Velocidade: ${Number(seg.speed_factor).toFixed(2)}x
                </span>
              </div>
              <p class="text-slate-400 text-[11px] line-clamp-1">Original: "${seg.original_text}"</p>
              <p class="text-slate-100 font-medium text-xs mt-0.5">Dublado: "${seg.translated_text}"</p>
            `;
            segmentsContainer.appendChild(item);
          }
          renderedSegmentsCount = job.segments.length;
          segmentsContainer.scrollTop = segmentsContainer.scrollHeight;
        }

        if (job.status === 'completed') {
          clearInterval(pollTimer);
          pollTimer = null;
          startDubBtn.disabled = false;
          startDubBtnText.textContent = 'Traduzir e Dublar Vídeo Agora';

          currentOutputFilename = job.output_filename || 'video_dublado.mp4';
          dubbedVideoUrl = `/api/video/dubbed/${currentJobId}?t=${Date.now()}`;
          originalServerUrl = `/api/video/original/${currentJobId}`;

          exportVideoBtn.disabled = false;
          downloadSrtBtn.disabled = false;
          openFolderBtn.disabled = false;
          exportSubtext.textContent = `Arquivo pronto: ${currentOutputFilename} (${job.elapsed_seconds}s)`;

          switchPlayerTab('dubbed');
          mainVideoPlayer.play().catch(() => {});
        } else if (job.status === 'failed') {
          clearInterval(pollTimer);
          pollTimer = null;
          startDubBtn.disabled = false;
          startDubBtnText.textContent = 'Traduzir e Dublar Vídeo Agora';
          alert('Ocorreu um erro durante a dublagem:\n' + (job.error || job.message));
        }
      } catch (err) {
        console.error('Erro ao consultar status do job:', err);
      }
    }

    // Exportar / Baixar Vídeo Traduzido (com seletor de pasta nativo se suportado pelo navegador)
    exportVideoBtn.addEventListener('click', async () => {
      if (!currentJobId) return;
      const downloadUrl = `/api/video/dubbed/${currentJobId}?download=true`;

      if (window.showSaveFilePicker) {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: currentOutputFilename,
            types: [{
              description: 'Vídeo MP4 Dublado',
              accept: { 'video/mp4': ['.mp4'] }
            }]
          });
          const response = await fetch(downloadUrl);
          const writable = await handle.createWritable();
          await response.body.pipeTo(writable);
          return;
        } catch (err) {
          if (err.name === 'AbortError') return;
        }
      }

      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = currentOutputFilename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    // Baixar Legenda .SRT
    downloadSrtBtn.addEventListener('click', () => {
      if (!currentJobId) return;
      const a = document.createElement('a');
      a.href = `/api/subtitle/${currentJobId}`;
      a.download = currentOutputFilename.replace(/\.mp4$/i, '.srt');
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    // Abrir pasta no Windows Explorer
    openFolderBtn.addEventListener('click', async () => {
      if (!currentJobId) return;
      await fetch(`/api/open-folder/${currentJobId}`, { method: 'POST' });
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_INTERFACE)


def find_available_port(host: str = "127.0.0.1", preferred_ports: Optional[List[int]] = None) -> int:
    """
    Testa portas locais para evitar erros de porta ocupada ou bloqueada pelo Hyper-V/Windows NAT
    (WinError 10013 / WinError 10048) e retorna a primeira porta livre e acessível.
    """
    candidates = list(preferred_ports or [7860, 8080, 8765, 5000, 3000, 8000])
    candidates.extend(range(7861, 7910))

    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue

    # Fallback garantido pelo sistema operacional
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def open_browser_when_ready(host: str, port: int) -> None:
    url = f"http://{host}:{port}"

    def _target():
        # Aguarda o servidor responder de fato antes de abrir o navegador
        for _ in range(30):
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning(f"Não foi possível abrir o navegador automaticamente: {exc}")

    threading.Thread(target=_target, daemon=True).start()


if __name__ == "__main__":
    host = os.environ.get("TRADUTOR_HOST", "127.0.0.1")
    env_port = os.environ.get("TRADUTOR_PORT")
    if env_port:
        port = find_available_port(host, [int(env_port)])
    else:
        port = find_available_port(host, [7860, 8080, 8765, 5000, 8000])

    app_url = f"http://{host}:{port}"
    print("\n" + "=" * 68)
    print(f"  SERVIDOR ATIVO EM: {app_url}")
    print("  O navegador abrira automaticamente nesta pagina.")
    print("  Mantenha esta janela aberta enquanto estiver usando o tradutor.")
    print("=" * 68 + "\n")
    logger.info(f"Iniciando servidor Tradutor & Dublador de Vídeos em {app_url}")

    if os.environ.get("TRADUTOR_NO_BROWSER") != "1":
        open_browser_when_ready(host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")

