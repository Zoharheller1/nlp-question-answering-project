# NLP Question Answering for Data Structures

An academic Natural Language Processing project that explores several approaches for generating and answering questions about data-structure material.

The project compares classical NLP retrieval, a custom six-layer BiDAF-style question-answering model, Semantic Role Labeling (SRL), and transformer fine-tuning with BERT and T5.

## Authors

- Zohar Heller
- Diana Korsunsky
- Eliran Hirsch

## Main approaches

1. **Question generation and retrieval without SRL**
   - Text preprocessing with NLTK
   - Tokenization, POS tagging, and named-entity extraction
   - TF-IDF and cosine similarity for answer matching

2. **Custom BiDAF-style question-answering model**
   - Six-layer PyTorch architecture
   - Train/test evaluation with Exact Match and F1
   - Experiments with dropout, learning-rate scheduling, and weight decay
   - Analysis of overfitting caused by the limited dataset

3. **Semantic Role Labeling**
   - SRL-assisted question generation
   - Comparison between models trained with and without semantic-role information

4. **Transformer experiments**
   - Fine-tuning BERT and T5 for question answering
   - Evaluation with ROUGE, BLEU, and F1
   - Comparison with baseline transformer outputs

## Repository structure

```text
.
├── data/
│   ├── data_nlp_project.txt
│   └── data_nlp_project_extended.txt
├── docs/
│   └── project_report_he.docx
├── notebooks/
│   ├── 01_question_generation_and_bidaf.ipynb
│   ├── 02_srl_question_generation.ipynb
│   └── 03_bert_t5_experiments.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies

Python, PyTorch, Hugging Face Transformers, BERT, T5, BiDAF, NLTK, Stanza, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, TF-IDF, cosine similarity, SRL, ROUGE, BLEU, and F1.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Jupyter from the repository root and open one of the notebooks:

```bash
jupyter notebook
```

The notebooks were originally developed in Google Colab, so some upload or download cells may be Colab-specific.

## OpenAI API configuration

API keys are intentionally not stored in this repository. If an optional OpenAI experiment is used, set the key as an environment variable.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
```

Python reads it with:

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## Results and limitations

The experiments show that the small dataset strongly affects generalization and can produce misleadingly high baseline scores or overfitting. The SRL-assisted experiments improved semantic understanding compared with the non-SRL configuration, while the transformer experiments demonstrated the importance of larger, carefully separated train and evaluation datasets.

The full methodology, experiments, results, and conclusions are available in `docs/project_report_he.docx`.

