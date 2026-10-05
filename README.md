# 🎬 AI Video Assistant

Turn any YouTube video, meeting recording, or audio file into structured notes you can chat with.

Paste a link or upload a file, and the app produces:

- **Title and summary** of the whole recording
- **Action items** with owner, deadline, and a supporting quote from the transcript
- **Decisions made**, with rationale and who decided
- **Questions asked**, marked as answered or unanswered
- **Full transcript** (downloadable)
- **Chat** to ask follow-up questions, answered from the transcript using RAG

Supports English and Hinglish.

<img width="1919" height="577" alt="image" src="https://github.com/user-attachments/assets/c6011c89-e1b1-468f-b83e-bef3f37064d5" />


## How it works

```
YouTube URL / file
      │
      ▼
 Audio processing  ──▶  Transcription  ──▶  Full transcript
                                                │
                    ┌───────────────────────────┼──────────────────────┐
                    ▼                           ▼                      ▼
            Title + summary          Structured extraction     Vector index (RAG)
                                (actions, decisions, questions)        │
                                                                       ▼
```

- **Structured extraction** uses LangChain with Pydantic schemas, so results come back as typed objects instead of free text. Long transcripts are split into chunks, extracted in parallel, and merged to remove duplicates. Questions are extracted from the full transcript in one pass so a question can be matched with its answer later in the video.
- **Chat** retrieves the most relevant transcript chunks for each question and passes them to the model, which is instructed to answer only from that context.

## Project structure

```
.
├── app.py                  # Streamlit web UI
├── main.py                 # Command-line version
├── core/
│   ├── transcriber.py      # Audio chunks → transcript
│   ├── summarizer.py       # Title and summary
│   ├── extractor.py        # Action items, decisions, questions
│   └── rag_engine.py       # Vector index and question answering
├── utils/
│   └── audio_processing.py # Download / split audio
├── .streamlit/
│   └── config.toml         # Theme and upload limit
├── requirements.txt
└── .env                    # API keys (not committed)
```

## Setup

### Prerequisites

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- An OpenAI API key
- *(Recommended)* An NVIDIA GPU with a recent driver. The project uses PyTorch with NVIDIA's cuBLAS and cuDNN libraries to speed up inference. It can run on CPU, but will be noticeably slower on long recordings.

### Install


bash
```
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

#### GPU setup (PyTorch + CUDA)

For GPU acceleration, install a CUDA-enabled build of PyTorch that matches your driver. Get the exact command for your system from the [PyTorch install selector](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The NVIDIA cuBLAS and cuDNN libraries are installed as dependencies of the CUDA build of PyTorch (as `nvidia-cublas-*` and `nvidia-cudnn-*` packages). To check that the GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, you are on the CPU build or your driver is out of date.

### Configure

Create a `.env` file in the project root:


OPENAI_API_KEY=your_key_here

Never commit this file. Make sure `.env` is listed in `.gitignore`.

## Usage

### Web app

# bash
streamlit run app.py


Open the URL shown in the terminal (usually http://localhost:8501), choose a YouTube URL or upload a file in the sidebar, pick a language, and click **Process**. When it finishes, browse the tabs or open **Chat** to ask questions.

### Command line

```bash
python main.py
```

You'll be prompted for a YouTube URL or file path and a language. Results print to the terminal, followed by an interactive chat loop. Type `exit` to quit.

## Tech stack

- [LangChain](https://python.langchain.com/) for chains, structured output, and retrieval
- OpenAI `gpt-4o-mini` for summarization, extraction, and chat
- [PyTorch](https://pytorch.org/) with NVIDIA CUDA libraries (cuBLAS and cuDNN) for GPU-accelerated model inference
- [Pydantic](https://docs.pydantic.dev/) for output schemas
- [Streamlit](https://streamlit.io/) for the web interface
- ffmpeg for audio handling

## Limitations

- Processing time grows with video length; long recordings can take several minutes.
- Extraction quality depends on transcript quality. Owners and deadlines are only filled in when they are stated explicitly, and show "Not specified" otherwise.
- Speaker names are only available if the transcript contains them.
- YouTube downloads can fail on cloud servers because YouTube often blocks datacenter IPs. File upload is more reliable when deployed.
- Each run uses OpenAI API credits.

## Roadmap

- [ ] Speaker diarization (who said what)
- [ ] Timestamps linked back to the video
- [ ] Export results to PDF / Markdown
- [ ] FastAPI + React frontend
- [ ] Docker deployment

## License

MIT
