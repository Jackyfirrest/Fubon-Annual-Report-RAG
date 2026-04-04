# Fubon Annual Report RAG

A RAG-based QA system for Fubon Financial 2024 annual report, designed to answer annual report questions with citations and hallucination detection.

## Project Goal
Build a retrieval-augmented generation system that:
- answers questions from the annual report
- cites source pages
- detects or refuses unsupported answers
- evaluates performance using the provided QA set

## Project Structure
- `data/raw/`: raw PDF and QA dataset
- `src/`: core modules
- `scripts/`: runnable scripts
- `results/`: predictions and evaluation outputs
- `notebooks/`: experiments and analysis

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt