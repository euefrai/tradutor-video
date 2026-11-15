import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from deep_translator import GoogleTranslator
import edge_tts
from faster_whisper import WhisperModel
import numpy as np
from pydub import AudioSegment

# Força resolução IPv4 (AF_INET) para evitar travamentos de 20s+ em redes Windows com IPv6 instável
# ao conectar nos WebSockets do Microsoft Edge-TTS e APIs de tradução.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        res = _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        if res:
            return res
    except OSError:
        pass
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dubber_engine")

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_FFMPEG_DIR = BASE_DIR / ".runtime" / "ffmpeg"
TEMP_TTS_DIR = BASE_DIR / "temp_tts"
OUTPUTS_DIR = BASE_DIR / "outputs"

TEMP_TTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def configure_ffmpeg_environment() -> Tuple[str, str]:
    """
    Garante que o FFmpeg e o FFprobe estejam acessíveis no PATH ou em .runtime/ffmpeg,
    configurando também o pydub automaticamente.
    """
    if RUNTIME_FFMPEG_DIR.exists():
        runtime_str = str(RUNTIME_FFMPEG_DIR)
        current_path = os.environ.get("PATH", "")
        if runtime_str not in current_path:
            os.environ["PATH"] = f"{runtime_str}{os.pathsep}{current_path}"

    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")

    if not ffmpeg_bin and (RUNTIME_FFMPEG_DIR / "ffmpeg.exe").exists():
        ffmpeg_bin = str(RUNTIME_FFMPEG_DIR / "ffmpeg.exe")
    if not ffprobe_bin and (RUNTIME_FFMPEG_DIR / "ffprobe.exe").exists():
        ffprobe_bin = str(RUNTIME_FFMPEG_DIR / "ffprobe.exe")

    if not ffmpeg_bin:
        raise RuntimeError(
            "FFmpeg não encontrado no sistema nem em .runtime/ffmpeg. "
            "Execute pelo Iniciar_Tradutor.bat para baixar automaticamente."
        )

    AudioSegment.converter = ffmpeg_bin
    if ffprobe_bin:
        AudioSegment.ffprobe = ffprobe_bin

    return ffmpeg_bin, ffprobe_bin or "ffprobe"


@dataclass
class DubSegment:
    index: int
    start: float
    end: float
    original_text: str
    translated_text: str
    speed_factor: float
    tts_duration: float
    available_window: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelCache:
    """Cache em memória para evitar recarregar o modelo faster-whisper a cada vídeo."""

    _models: Dict[str, WhisperModel] = {}

    @classmethod
    def get_model(cls, model_size: str = "small") -> WhisperModel:
        if model_size not in cls._models:
            logger.info(
                f"Carregando modelo faster-whisper '{model_size}' (device='auto', compute_type='int8')..."
            )
            cls._models[model_size] = WhisperModel(
                model_size,
                device="auto",
                compute_type="int8",
                cpu_threads=max(2, (os.cpu_count() or 4) - 1),
            )
        return cls._models[model_size]


def format_srt_timestamp(seconds: float) -> str:
    """Converte segundos (float) para o formato padrão SRT (HH:MM:SS,mmm)."""
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1_000
    millis %= 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_file(segments: List[DubSegment], srt_path: Path) -> None:
    """Salva os segmentos traduzidos em um arquivo de legenda .srt."""
    lines: List[str] = []
    for seg in segments:
        start_ts = format_srt_timestamp(seg.start)
        end_ts = format_srt_timestamp(max(seg.end, seg.start + 0.5))
        lines.append(f"{seg.index}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.translated_text.strip())
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def extract_original_audio(video_path: Path, wav_output_path: Path, ffmpeg_bin: str) -> None:
    """Extrai a faixa de áudio original do vídeo em WAV 44.1kHz estéreo para mixagem."""
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(wav_output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not wav_output_path.exists():
        raise RuntimeError(f"Erro ao extrair áudio original do vídeo: {proc.stderr[-500:]}")


def extract_whisper_waveform_16k(
    audio_or_video_path: Path, wav_16k_path: Path, ffmpeg_bin: str
) -> np.ndarray:
    """
    Converte o áudio para WAV 16kHz mono PCM 16-bit via FFmpeg e carrega como array
    NumPy float32 normalizado [-1.0, 1.0]. Isso evita incompatibilidades do PyAV
    (ex: av.open metadata_errors) e acelera a leitura no faster-whisper.
    """
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(audio_or_video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_16k_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not wav_16k_path.exists():
        raise RuntimeError(f"Erro ao preparar áudio 16kHz para transcrição: {proc.stderr[-500:]}")

    with wave.open(str(wav_16k_path), "rb") as wf:
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


def load_audio_segment_via_wav(
    input_audio_path: Path,
    ffmpeg_bin: str,
    speed_factor: float = 1.0,
) -> AudioSegment:
    """
    Converte qualquer arquivo de áudio (ex: MP3 do edge-tts) para WAV 44.1kHz estéreo
    usando FFmpeg — aplicando opcionalmente o filtro atempo (1.0x a 1.35x) num único passo —
    e carrega via AudioSegment.from_wav sem depender do ffprobe.
    """
    clamped_speed = min(max(speed_factor, 1.0), 1.35)
    wav_out = input_audio_path.with_suffix(f".proc_{int(clamped_speed * 100)}.wav")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_audio_path),
    ]
    if clamped_speed > 1.01:
        cmd.extend(["-filter:a", f"atempo={clamped_speed:.4f}"])

    cmd.extend(
        [
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(wav_out),
        ]
    )

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not wav_out.exists():
        raise RuntimeError(f"Falha ao processar segmento de áudio no FFmpeg: {proc.stderr[-400:]}")

    seg = AudioSegment.from_wav(str(wav_out))
    return seg


async def synthesize_edge_tts(
    text: str,
    voice: str,
    output_path: Path,
    rate: str = "+0%",
    retries: int = 4,
) -> None:
    """Gera áudio de síntese neural com edge-tts na velocidade natural (+0%) com timeout de segurança."""
    last_err: Optional[Exception] = None
    clean_text = text.strip() or "..."
    for attempt in range(1, retries + 1):
        try:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            communicate = edge_tts.Communicate(text=clean_text, voice=voice, rate=rate)
            await asyncio.wait_for(communicate.save(str(output_path)), timeout=12.0)
            if output_path.exists() and output_path.stat().st_size > 0:
                return
        except Exception as exc:
            last_err = exc
            logger.warning(f"Tentativa {attempt}/{retries} do edge-tts falhou ({ clean_text[:25]}...): {exc}")
            await asyncio.sleep(0.5 * attempt)
    raise RuntimeError(f"Não foi possível sintetizar voz via edge-tts: {last_err}")


def _translate_via_clients5_api(text: str, src: str, tgt: str) -> Optional[str]:
    """
    Traduz texto via endpoint oficial Chrome Extension (clients5.google.com),
    que não sofre bloqueio 429 do translate.google.com/m nem do translate.googleapis.com.
    """
    q = urllib.parse.quote(text)
    url = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl={src}&tl={tgt}&q={q}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, list) and first:
                translated = str(first[0])
            elif isinstance(first, str):
                translated = first
            else:
                translated = ""
            if translated and translated.strip():
                return translated.strip()
    return None


def _translate_via_mymemory_api(text: str, src: str, tgt: str) -> Optional[str]:
    """
    Fallback secundário gratuito via API MyMemory Translated (sem necessidade de chave).
    """
    if not src or src == "auto":
        return None
    q = urllib.parse.quote(text)
    url = f"https://api.mymemory.translated.net/get?q={q}&langpair={src}|{tgt}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        translated = (data.get("responseData") or {}).get("translatedText")
        if translated and isinstance(translated, str) and translated.strip():
            return translated.strip()
    return None


def _translate_via_gtx_api(text: str, src: str, tgt: str) -> Optional[str]:
    """
    Traduz texto via endpoint JSON GTX do Google Translate.
    """
    q = urllib.parse.quote(text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={tgt}&dt=t&q={q}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if data and isinstance(data, list) and data[0]:
            translated = "".join(part[0] for part in data[0] if part and part[0])
            if translated and translated.strip():
                return translated.strip()
    return None


def translate_text_segment(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Traduz um segmento individual utilizando múltiplos provedores gratuitos em cascata."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    src = "auto" if not source_lang or source_lang.lower() == "auto" else source_lang
    tgt = target_lang or "pt"

    if src.split("-")[0].lower() == tgt.split("-")[0].lower():
        return cleaned

    for attempt in range(2):
        # 1. Chrome Extension endpoint (clients5) - ultra-rápido e imune a HTTP 429
        try:
            res = _translate_via_clients5_api(cleaned, src, tgt)
            if res:
                return res
        except Exception as exc:
            logger.debug(f"clients5 tentativa {attempt + 1} falhou: {exc}")

        # 2. MyMemory API
        try:
            res = _translate_via_mymemory_api(cleaned, src, tgt)
            if res:
                return res
        except Exception as exc:
            logger.debug(f"mymemory tentativa {attempt + 1} falhou: {exc}")

        # 3. Google GTX JSON endpoint
        try:
            res = _translate_via_gtx_api(cleaned, src, tgt)
            if res:
                return res
        except Exception as exc:
            logger.debug(f"GTX tentativa {attempt + 1} falhou: {exc}")

        # 4. deep-translator GoogleTranslator
        try:
            translator = GoogleTranslator(source=src, target=tgt)
            translated = translator.translate(cleaned)
            if translated and translated.strip():
                return translated.strip()
        except Exception as exc:
            logger.warning(f"Tentativa {attempt + 1}/2 de tradução falhou: {exc}")
            time.sleep(0.4 * (attempt + 1))

    return cleaned


def translate_segments_in_batches(
    texts: List[str],
    effective_source_lang: str,
    target_lang: str,
) -> List[str]:
    """
    Traduz todos os segmentos em lotes agrupados por quebra de linha via clients5 + GTX + MyMemory
    para máxima velocidade e zero bloqueio de rate-limit (HTTP 429).
    Caso o idioma detectado seja igual ao idioma de destino, mantém o texto original.
    """
    if not texts:
        return []

    src_base = (effective_source_lang or "auto").split("-")[0].lower()
    tgt_base = (target_lang or "pt").split("-")[0].lower()

    if src_base == tgt_base:
        logger.info(
            f"Idioma de origem ({effective_source_lang}) igual ao de destino ({target_lang}); mantendo falas originais."
        )
        return [t.strip() for t in texts]

    src_code = "auto" if not effective_source_lang else effective_source_lang
    results: List[str] = []
    batch: List[str] = []
    batch_chars = 0

    def _flush_batch(current_batch: List[str]) -> List[str]:
        if not current_batch:
            return []
        joined = "\n".join(line.replace("\n", " ").strip() for line in current_batch)
        for provider_fn in (_translate_via_clients5_api, _translate_via_gtx_api):
            try:
                translated_block = provider_fn(joined, src_code, target_lang)
                if translated_block:
                    split_lines = [ln.strip() for ln in translated_block.split("\n") if ln.strip()]
                    if len(split_lines) == len(current_batch):
                        return split_lines
            except Exception as exc:
                logger.debug(f"Provedor em lote {provider_fn.__name__} falhou: {exc}")

        # Fallback individual rápido (clients5 -> mymemory -> gtx -> deep-translator)
        fallback_out: List[str] = []
        for item in current_batch:
            fallback_out.append(translate_text_segment(item, src_code, target_lang))
        return fallback_out

    for txt in texts:
        clean = txt.replace("\n", " ").strip()
        if batch and (batch_chars + len(clean) > 1800 or len(batch) >= 12):
            results.extend(_flush_batch(batch))
            batch = []
            batch_chars = 0
        batch.append(clean)
        batch_chars += len(clean) + 1

    if batch:
        results.extend(_flush_batch(batch))

    return results


async def generate_voice_preview(
    voice: str,
    sample_text: str = "Olá! Esta é uma demonstração da voz neural ultra-natural para dublagem automática de vídeos.",
) -> Path:
    """Gera um arquivo MP3 de demonstração de voz para a interface."""
    configure_ffmpeg_environment()
    preview_dir = TEMP_TTS_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_voice = "".join(c if c.isalnum() or c in "-_" else "_" for c in voice)
    out_file = preview_dir / f"preview_{safe_voice}.mp3"
    await synthesize_edge_tts(sample_text, voice, out_file, rate="+0%")
    return out_file


async def process_video_dubbing(
    video_path: Path,
    output_video_path: Path,
    source_lang: str = "auto",
    target_lang: str = "pt",
    voice: str = "en-US-AndrewMultilingualNeural",
    ducking_db: float = -20.0,
    whisper_model_size: str = "small",
    progress_callback: Optional[Callable[[int, str, str, Optional[Dict]], None]] = None,
) -> Dict:
    """
    Pipeline completo de Tradução e Dublagem Automática:
    1. Extração da trilha de áudio original.
    2. Transcrição com faster-whisper (int8 + vad_filter=True) usando waveform 16kHz.
    3. Tradução rápida em lote com deep-translator (GoogleTranslator).
    4. Síntese de voz natural com edge-tts (rate="+0%").
    5. Ajuste dinâmico de tempo (Time-Stretching via FFmpeg atempo 1.0x a 1.35x).
    6. Mixagem com pydub (ducking configurável + overlay no milissegundo exato).
    7. Renderização rápida com FFmpeg (-c:v copy -c:a aac).
    """
    start_time = time.time()
    ffmpeg_bin, _ = configure_ffmpeg_environment()

    def notify(percent: int, stage: str, message: str, seg_dict: Optional[Dict] = None):
        logger.info(f"[{percent}%] {stage}: {message}")
        if progress_callback:
            progress_callback(percent, stage, message, seg_dict)

    job_temp_dir = TEMP_TTS_DIR / f"job_{int(time.time() * 1000)}"
    job_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Etapa 1: Extrair áudio original (44.1kHz estéreo para mixagem + 16kHz mono para Whisper)
        notify(5, "Extração de Áudio", "Extraindo faixa de áudio original do vídeo...")
        original_wav = job_temp_dir / "original_audio.wav"
        whisper_16k_wav = job_temp_dir / "whisper_16k.wav"

        await asyncio.to_thread(extract_original_audio, video_path, original_wav, ffmpeg_bin)
        waveform_16k = await asyncio.to_thread(
            extract_whisper_waveform_16k, original_wav, whisper_16k_wav, ffmpeg_bin
        )

        # Etapa 2: Carregar modelo faster-whisper e transcrever
        notify(
            12,
            "Transcrição IA",
            f"Carregando modelo faster-whisper ({whisper_model_size}, int8, VAD ativo)...",
        )
        model = await asyncio.to_thread(ModelCache.get_model, whisper_model_size)

        whisper_lang = None if source_lang == "auto" else source_lang.split("-")[0]
        notify(
            18,
            "Transcrição IA",
            "Transcrevendo falas com timestamps precisos e filtro de voz (VAD)...",
        )

        def _run_transcription():
            segments_gen, info = model.transcribe(
                waveform_16k,
                language=whisper_lang,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=450),
            )
            raw_list = []
            for seg in segments_gen:
                txt = seg.text.strip()
                if txt:
                    raw_list.append((float(seg.start), float(seg.end), txt))
            return raw_list, info.language, float(info.duration or 0.0)

        raw_segments, detected_lang, audio_duration = await asyncio.to_thread(_run_transcription)

        if not raw_segments:
            raise RuntimeError(
                "Nenhuma fala foi detectada no vídeo. Verifique se o vídeo possui trilha de voz audível."
            )

        total_segs = len(raw_segments)
        effective_src_lang = detected_lang if source_lang == "auto" else source_lang

        notify(
            26,
            "Tradução Inteligente",
            f"{total_segs} falas detectadas (Idioma: {detected_lang}). Traduzindo roteiro para '{target_lang}'...",
        )

        original_texts = [seg[2] for seg in raw_segments]
        translated_texts = await asyncio.to_thread(
            translate_segments_in_batches,
            original_texts,
            effective_src_lang,
            target_lang,
        )

        # Carregar áudio base no pydub e aplicar Ducking
        base_audio = await asyncio.to_thread(AudioSegment.from_wav, str(original_wav))
        total_duration_s = len(base_audio) / 1000.0

        # Aplica redução de volume (ducking) mantendo som ambiente de fundo
        ducked_background = base_audio + float(ducking_db)

        dub_segments: List[DubSegment] = []
        prepared_overlays: List[Tuple[int, AudioSegment]] = []

        # Pré-síntese paralela controlada (2 conexões simultâneas) via edge-tts
        # evitando throttling de WebSockets simultâneos no servidor da Microsoft
        tts_sem = asyncio.Semaphore(2)
        raw_tts_paths: List[Path] = [
            job_temp_dir / f"seg_{idx:04d}_raw.mp3" for idx in range(1, total_segs + 1)
        ]

        async def _synth_worker(idx_zero: int, text_val: str):
            async with tts_sem:
                await synthesize_edge_tts(
                    text=text_val,
                    voice=voice,
                    output_path=raw_tts_paths[idx_zero],
                    rate="+0%",
                )

        # Primeiro segmento aquece o ajuste de relógio DRM (Sec-MS-GEC) antes do paralelismo
        if translated_texts:
            await _synth_worker(0, translated_texts[0])
        if len(translated_texts) > 1:
            await asyncio.gather(
                *(_synth_worker(i, txt) for i, txt in enumerate(translated_texts[1:], start=1))
            )

        # Etapa 4 e 5: Ajuste Dinâmico de Tempo (Time-Stretching) e preparação de mixagem
        for idx, ((seg_start, seg_end, orig_text), translated_text) in enumerate(
            zip(raw_segments, translated_texts), start=1
        ):
            # Janela disponível até o início da próxima fala (ou fim do vídeo)
            if idx < total_segs:
                next_start = raw_segments[idx][0]
                available_window = max(0.4, next_start - seg_start)
            else:
                available_window = max(0.5, max(seg_end - seg_start, total_duration_s - seg_start))

            raw_tts_path = raw_tts_paths[idx - 1]

            tts_seg_audio = await asyncio.to_thread(
                load_audio_segment_via_wav, raw_tts_path, ffmpeg_bin, 1.0
            )
            tts_duration = len(tts_seg_audio) / 1000.0

            # Verificar necessidade de Time-Stretching com FFmpeg atempo (1.0x a 1.35x)
            if tts_duration <= available_window or available_window <= 0:
                speed_factor = 1.0
                final_seg_audio = tts_seg_audio
            else:
                raw_ratio = tts_duration / available_window
                speed_factor = round(min(max(raw_ratio, 1.0), 1.35), 2)
                if speed_factor > 1.01:
                    final_seg_audio = await asyncio.to_thread(
                        load_audio_segment_via_wav,
                        raw_tts_path,
                        ffmpeg_bin,
                        speed_factor,
                    )
                else:
                    speed_factor = 1.0
                    final_seg_audio = tts_seg_audio

            final_tts_duration = len(final_seg_audio) / 1000.0
            position_ms = max(0, int(round(seg_start * 1000)))
            prepared_overlays.append((position_ms, final_seg_audio))

            dub_seg = DubSegment(
                index=idx,
                start=round(seg_start, 2),
                end=round(seg_start + final_tts_duration, 2),
                original_text=orig_text,
                translated_text=translated_text,
                speed_factor=speed_factor,
                tts_duration=round(final_tts_duration, 2),
                available_window=round(available_window, 2),
            )
            dub_segments.append(dub_seg)

            # Progresso proporcional entre 30% e 84%
            pct = 30 + int((idx / total_segs) * 54)
            notify(
                pct,
                f"Dublando ({idx}/{total_segs})",
                f"[{seg_start:.1f}s] {translated_text} ({speed_factor:.2f}x)",
                dub_seg.to_dict(),
            )

        # Etapa 6: Mixagem de áudio com pydub
        notify(86, "Mixagem de Áudio", "Sobrepondo falas dubladas na faixa original com ducking...")

        def _mix_all_overlays() -> Path:
            mixed = ducked_background
            for pos_ms, seg_audio in prepared_overlays:
                # Normalização leve da voz dublada para clareza em cima do áudio original reduzido
                boosted_voice = seg_audio + 1.5
                mixed = mixed.overlay(boosted_voice, position=pos_ms)
            mixed_wav_path = job_temp_dir / "mixed_final.wav"
            mixed.export(str(mixed_wav_path), format="wav")
            return mixed_wav_path

        mixed_wav_path = await asyncio.to_thread(_mix_all_overlays)

        # Gerar arquivo de legendas .srt correspondente
        srt_output_path = output_video_path.with_suffix(".srt")
        await asyncio.to_thread(write_srt_file, dub_segments, srt_output_path)

        # Etapa 7: Substituição rápida da faixa de áudio via FFmpeg (-c:v copy -c:a aac)
        notify(
            93,
            "Renderização Final",
            "Mesclando nova trilha de áudio no vídeo via FFmpeg (-c:v copy -c:a aac)...",
        )

        def _mux_video_and_audio():
            cmd_copy = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(mixed_wav_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-strict",
                "-2",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_video_path),
            ]
            res = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0 or not output_video_path.exists():
                logger.warning(
                    "Codec de vídeo de origem incompatível com copy direto para MP4. Aplicando conversão rápida H.264..."
                )
                cmd_reencode = [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    str(video_path),
                    "-i",
                    str(mixed_wav_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "21",
                    "-c:a",
                    "aac",
                    "-strict",
                    "-2",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    str(output_video_path),
                ]
                res2 = subprocess.run(
                    cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if res2.returncode != 0:
                    raise RuntimeError(f"Erro ao exportar vídeo final: {res2.stderr[-500:]}")

        await asyncio.to_thread(_mux_video_and_audio)

        elapsed = round(time.time() - start_time, 1)
        notify(
            100,
            "Concluído",
            f"Dublagem concluída com sucesso em {elapsed}s! ({total_segs} falas sincronizadas)",
        )

        return {
            "output_video": str(output_video_path),
            "output_srt": str(srt_output_path),
            "detected_language": detected_lang,
            "target_language": target_lang,
            "voice": voice,
            "total_segments": total_segs,
            "elapsed_seconds": elapsed,
            "segments": [s.to_dict() for s in dub_segments],
        }

    finally:
        # Limpeza dos arquivos temporários do job
        shutil.rmtree(job_temp_dir, ignore_errors=True)
