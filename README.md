# Internship Support FAQ Bot

A lightweight FAQ assistant powered by Retrieval-Augmented Generation (RAG), built to answer internship-related support questions. Runs in Google Colab or locally.

## Features
- Retrieves answers from a structured FAQ dataset using semantic search
- Shows source references for each answer
- Lets users leave feedback to improve coverage over time
- Interactive interface built with Streamlit

## Tech Stack
Python, Streamlit, ChromaDB, Sentence-Transformers

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
- `faqs.csv` — source FAQ data used for retrieval
- `tickets.csv` — [support ticket data — describe how it's used, e.g. for feedback/gap analysis]

## Notebook
`Chatbot_Internship.ipynb` — development and experimentation notebook

## License
See [LICENSE](LICENSE) for details.