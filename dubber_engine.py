import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from deep_translator import GoogleTranslator
import edge_tts
from faster_whisper import WhisperModel
from pydub import AudioSegment

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


def apply_atempo_stretch(
    input_audio_path: Path,
    output_audio_path: Path,
    speed_factor: float,
    ffmpeg_bin: str,
) -> None:
    """
    Aplica o filtro atempo do FFmpeg (limitado entre 1.0x e 1.35x) para ajustar
    a duração da fala traduzida sem distorcer o tom (pitch) da voz.
    """
    clamped_speed = min(max(speed_factor, 1.0), 1.35)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_audio_path),
        "-filter:a",
        f"atempo={clamped_speed:.4f}",
        "-vn",
        str(output_audio_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not output_audio_path.exists():
        logger.warning(
            f"Falha ao aplicar atempo={clamped_speed:.2f}, usando áudio original do TTS. Detalhe: {proc.stderr[-300:]}"
        )
        shutil.copyfile(input_audio_path, output_audio_path)


async def synthesize_edge_tts(
    text: str,
    voice: str,
    output_path: Path,
    rate: str = "+0%",
    retries: int = 3,
) -> None:
    """Gera áudio de síntese neural com edge-tts na velocidade natural (+0%)."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return
        except Exception as exc:
            last_err = exc
            logger.warning(f"Tentativa {attempt}/{retries} do edge-tts falhou: {exc}")
            await asyncio.sleep(1.0 * attempt)
    raise RuntimeError(f"Não foi possível sintetizar voz via edge-tts: {last_err}")


def translate_text_segment(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Traduz um segmento de texto utilizando GoogleTranslator (deep-translator)."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    src = "auto" if not source_lang or source_lang.lower() == "auto" else source_lang
    tgt = target_lang or "pt"

    if src == tgt:
        return cleaned

    for attempt in range(3):
        try:
            translator = GoogleTranslator(source=src, target=tgt)
            translated = translator.translate(cleaned)
            if translated and translated.strip():
                return translated.strip()
        except Exception as exc:
            logger.warning(f"Tentativa {attempt + 1}/3 de tradução falhou: {exc}")
            time.sleep(0.6 * (attempt + 1))

    return cleaned


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
    2. Transcrição com faster-whisper (int8 + vad_filter=True).
    3. Tradução de cada segmento com deep-translator (GoogleTranslator).
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
        # Etapa 1: Extrair áudio original
        notify(5, "Extração de Áudio", "Extraindo faixa de áudio original do vídeo...")
        original_wav = job_temp_dir / "original_audio.wav"
        await asyncio.to_thread(extract_original_audio, video_path, original_wav, ffmpeg_bin)

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
                str(original_wav),
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
        notify(
            28,
            "Transcrição Concluída",
            f"{total_segs} segmentos de fala detectados (Idioma identificado: {detected_lang}). Iniciando tradução e dublagem...",
        )

        # Carregar áudio base no pydub e aplicar Ducking
        base_audio = await asyncio.to_thread(AudioSegment.from_wav, str(original_wav))
        total_duration_s = len(base_audio) / 1000.0

        # Aplica redução de volume (ducking) mantendo som ambiente de fundo
        ducked_background = base_audio + float(ducking_db)

        dub_segments: List[DubSegment] = []
        prepared_overlays: List[Tuple[int, AudioSegment]] = []

        # Etapa 3, 4 e 5: Tradução, TTS e Ajuste Dinâmico de Tempo (Time-Stretching)
        for idx, (seg_start, seg_end, orig_text) in enumerate(raw_segments, start=1):
            # Janela disponível até o início da próxima fala (ou fim do vídeo)
            if idx < total_segs:
                next_start = raw_segments[idx][0]
                available_window = max(0.4, next_start - seg_start)
            else:
                available_window = max(0.5, max(seg_end - seg_start, total_duration_s - seg_start))

            # Traduzir segmento
            translated_text = await asyncio.to_thread(
                translate_text_segment,
                orig_text,
                source_lang,
                target_lang,
            )

            # Sintetizar voz com edge-tts em velocidade 100% natural (+0%)
            raw_tts_path = job_temp_dir / f"seg_{idx:04d}_raw.mp3"
            await synthesize_edge_tts(
                text=translated_text,
                voice=voice,
                output_path=raw_tts_path,
                rate="+0%",
            )

            tts_seg_audio = await asyncio.to_thread(AudioSegment.from_file, str(raw_tts_path))
            tts_duration = len(tts_seg_audio) / 1000.0

            # Verificar necessidade de Time-Stretching com FFmpeg atempo (1.0x a 1.35x)
            if tts_duration <= available_window or available_window <= 0:
                speed_factor = 1.0
                final_seg_audio = tts_seg_audio
            else:
                raw_ratio = tts_duration / available_window
                speed_factor = round(min(max(raw_ratio, 1.0), 1.35), 2)
                if speed_factor > 1.01:
                    stretched_tts_path = job_temp_dir / f"seg_{idx:04d}_stretched.mp3"
                    await asyncio.to_thread(
                        apply_atempo_stretch,
                        raw_tts_path,
                        stretched_tts_path,
                        speed_factor,
                        ffmpeg_bin,
                    )
                    final_seg_audio = await asyncio.to_thread(
                        AudioSegment.from_file, str(stretched_tts_path)
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
