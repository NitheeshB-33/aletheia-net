Aletheia-Net: Misinformation Detection System

Aletheia-Net is a machine learning application designed to assess the credibility of news articles using Natural Language Processing (NLP). It utilizes a multi-model architecture to classify text veracity and generate abstractive summaries.

System Architecture

The application implements a stacked pipeline approach:

Credibility Classification: A DistilBERT model fine-tuned on the WELFake dataset detects semantic patterns associated with misinformation.

Abstractive Summarization: A BART (Bidirectional and Auto-Regressive Transformers) model extracts core information to provide a neutral summary.

Technical Stack

Language: Python 3.9+

Framework: Streamlit

ML Libraries: Transformers (Hugging Face), PyTorch

Installation

Clone the repository.

Install dependencies:

pip install -r requirements.txt


Launch the application:

streamlit run app.py


License

MIT License
