import os, sys
import ctypes
# --- make the CUDA 12 DLLs inside the venv visible (must run BEFORE faster_whisper) ---
_nv = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
for _lib in ("cublas", "cudnn"):
    _bin = os.path.join(_nv, _lib, "bin")
    if os.path.isdir(_bin):
        os.add_dll_directory(_bin)
        os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]

_cublas_bin = os.path.join(_nv, "cublas", "bin")
for _name in ("cublasLt64_12.dll", "cublas64_12.dll"):
    _path = os.path.join(_cublas_bin, _name)
    if os.path.isfile(_path):
        ctypes.CDLL(_path)

from faster_whisper import WhisperModel
import requests
from pydub import AudioSegment
from dotenv import load_dotenv  # uv pip install python-dotenv

load_dotenv()


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

SARVAM_PIECE_SECONDS = 25

_model= None


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("small", device= "cuda",compute_type="float16", download_root="D:\\models")
    return _model

##sending audio to sarvam : HITTING THE API with requests OF our video chunk path  - returns a json transcript
def _send_to_sarvam(chunk_path: str)-> str:  
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL,}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files={"file": (os.path.basename(chunk_path), f, "audio/wav")},
            data={"model": SARVAM_MODEL, "mode": "translate"},
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")

def transcribe_chunk_sarvam(chunk_path: str) -> str:

    """  Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts."""

    if not SARVAM_API_KEY:
        raise RuntimeError("sarvam api key not FOUND in environment .env/")

    audio= AudioSegment.from_wav(chunk_path)
    piece_ms= SARVAM_PIECE_SECONDS*1000  ## AS WE NEED ESS THAN 30 S DATA SO WE NEED TO SHORTEN IT AND AS IT IS PROCESSEDD IN MILISSECONDS 
    total = (len(audio) + piece_ms - 1) // piece_ms
    parts=[]

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{os.path.splitext(chunk_path)[0]}_sv_{i}.wav"
        piece= audio[start:start + piece_ms]
        piece.export(piece_path, format="wav")
        try:
            print(f"  -> Sarvam piece {i + 1}/{total} ...")
            parts.append(_send_to_sarvam(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(p.strip() for p in parts if p.strip())



def transcribe_chunk_whisper(chunk_path: str) -> str:
    segments, info = _get_model().transcribe(chunk_path)  ##segments is a generator --> only works when looped over otherwise empty

    chunk_text = ""   ## creating empty string to store transcribed chunk
    for seg in segments:
        chunk_text+= seg.text + ""
    
    return chunk_text.strip()

    
def transcribe_chunks(chunk_path : str, language : str = "english")->str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """

    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english")->str:

    full_transcript = "" 

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    results = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunks(chunk, language=language)
        if text:
            results.append(text)
    return " ".join(results)
