# Prompt-Engineering-in-Healthcare
Medical-QA-Prompt-Engineering/
│
├── data/                    # All medical QA datasets
│   ├── MedicalQuestionAnswering.csv
│   ├── CancerQA.csv
│   ├── ...
│
├── src/                     # Source code
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── prompt_engineering.py# Prompt template construction
│   ├── model_runner.py      # Inference pipeline
│   ├── evaluation.py        # Metrics calculation (ROUGE, BLEU, etc.)
│   └── utils.py             # Helper functions
│
├── results/                 # Output results
│   ├── outputs.csv
│   └── performance_metrics.json
│
├── figures/                 # Visuals (e.g., model architecture, training graphs)
│   └── piccc.jpeg
│
├── paper/                   # LaTeX paper source
│   └── medical_qa_paper.tex
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── run_experiments.py       # Orchestrates the full QA pipeline
