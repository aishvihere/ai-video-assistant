import os
import sys
import yt_dlp
from pydub import AudioSegment


# Explicitly tell Pydub where winget installed your FFmpeg tools!
# # (Winget installs Gyan.FFmpeg directly into your local AppData paths)
# WINGET_FFMPEG_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_85cc3011\ffmpeg-7.1-essentials_build\bin")

# # Bind them directly to Pydub's internal converters
# AudioSegment.converter = os.path.join(WINGET_FFMPEG_PATH, "ffmpeg.exe")
# AudioSegment.ffprobe   = os.path.join(WINGET_FFMPEG_PATH, "ffprobe.exe")
download_dir = "downloads" 
os.makedirs(download_dir,exist_ok=True)
  
def download_yt_video(url: str) -> str:

    output_path = os.path.join(download_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".wav"
            
    return filename

## convert to clean .wav in 16khz format with pydub audio segment(+mono audio conversion) cause whisper processes 16khz best
def convert_to_wav(input_path : str):
    output_path=os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio= audio.set_channels(1).set_frame_rate(16000)  ## mono audio =1, and framerate=16000 - best for whisperai
    audio.export(output_path, format = "wav")
    return output_path


def chunk_audio(wav_path : str, chunk_minutes : str =10 )-> list:  ## 10 minutes is default if no value of chunk_min is given
    ## get the audio from our previous func
    audio = AudioSegment.from_wav(wav_path)
    # as the audio is stored in milliseconds so to chunk it - we need miliseconds values
    chunks_ms = chunk_minutes * 60 * 1000  ## miliseconds conversion of our 1 chunk value
    chunks =[]
    ## pass the chunks over on our audio length

    for i, start in enumerate(range(0,len(audio),chunks_ms)):
        chunk = audio[start : start+ chunks_ms]  ## starting from the first value at 0th index to last chunk value
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format = "wav")

        chunks.append(chunk_path)

    return chunks

## now combining all the functions above together so they trigger one after another

def process_input(source : str) -> list:
## loading and conversion--
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected url, downloading...")

        try:
            wav_path = download_yt_video(source)
        except Exception as e:
            print(f"There was an error downloading the url: {e}, please try and paste correct url..")
            return []

    else: 
        print("Detected local audio/video file. Converting to WAV")
        try:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Local file not found at {source}")
            wav_path = convert_to_wav(source)
        except Exception as  e:
            print(f"Error converting local file to wav: {e}")
            return []       
## chunking audio now--
    print("Chunking audio/video now...")

    try:  # throw extra safety if converted wav file not found
        if not wav_path or not os.path.exists(wav_path):
            raise FileNotFoundError(f"Converted wav file not generated successfully for chunking..")
        else:
            chunks = chunk_audio(wav_path)

    except Exception as e:
        print(f"Error during audio chunking {e}")
        return []

    return chunks