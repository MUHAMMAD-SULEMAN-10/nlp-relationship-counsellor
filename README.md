---
title: AI Relationship Counsellor
emoji: ❤️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
short_description: Upload a WhatsApp chat and get an AI relationship health analysis
---

# ❤️ AI Relationship Counsellor

Upload an exported WhatsApp chat (.txt) and get an AI read on how healthy it looks.

- Toxicity score out of 10, top 3 red flags, 1 positive, 3 suggestions
- Ask questions about the chat (FAISS search + Gemini)

*AI reading of text only, not professional advice.*

## Run locally

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY="your_key_here"
python -m streamlit run app.py --server.fileWatcherType none
```

Get a free key at https://aistudio.google.com/apikey and upload `sample_chat.txt` to test.

## Deploy

Add a secret named `GEMINI_API_KEY`:

- **Hugging Face Spaces:** new Space → SDK Docker → upload all files → Settings → Variables and secrets
- **Streamlit Cloud:** push to GitHub → share.streamlit.io → Advanced settings → Secrets

## Privacy

Your chat is sent to Google's Gemini API. It is not saved by this app. Only upload chats you have the right to share.

## Files

`app.py` (app) · `chat_utils.py` (WhatsApp parsing) · `requirements.txt` · `Dockerfile` (Hugging Face)