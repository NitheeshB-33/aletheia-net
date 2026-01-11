🛡️ Aletheia-Net

Automated Credibility Assessment System

🔗 Live Demo

Click Here to Launch Aletheia-Net
https://aletheia-net-eds9tok2e7qey7iobtniv7.streamlit.app/

📖 Project Overview

Aletheia-Net is a hybrid AI application designed to assess the credibility of news articles. Unlike simple binary "True/False" detectors, this system employs a Multi-Layered Assessment Engine to evaluate linguistic patterns, identifying whether content is factual news, clickbait, or speculative opinion.

⚙️ System Architecture

The application implements a 3-layer verification pipeline:

Semantic Intent Analysis:

Utilizes a Zero-Shot Transformer (valhalla/distilbart-mnli-12-1) to classify text intent (e.g., Factual vs. Clickbait) without relying on outdated static datasets.

Logic Guardrails (Rule Engine):

A deterministic logic layer that scans for anomalies, such as future dates (e.g., "IPL 2026"), to prevent the AI from hallucinating on speculative content.

Abstractive Summarization:

A BART Transformer (facebook/bart-large-cnn) generates a neutral, concise executive summary, stripping away emotional rhetoric.

🛠️ Technical Stack

Frontend: Streamlit

AI Models: Hugging Face Transformers

Logic: Python Regex & Datetime

📦 Installation

Clone the repository:

git clone [https://github.com/NitheeshB-33/aletheia-net.git](https://github.com/NitheeshB-33/aletheia-net.git)


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


📄 License

MIT License
