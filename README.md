# Prompt-Engineering-in-Healthcare
Medical-QA-Prompt-Engineering/
│
├── data/
│   ├── MedicalQuestionAnswering.csv
│   ├── CancerQA.csv
│   ├── Heart_Lung_and_BloodQA.csv
│   ├── Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv
│   ├── Neurological_Disorders_and_StrokeQA.csv
│   ├── Genetic_and_Rare_DiseasesQA.csv
│   ├── SeniorHealthQA.csv
│   ├── growth_hormone_receptorQA.csv
│   └── OtherQA.csv
│
├── src/
│   ├── data_loader.py              # Loads and preprocesses datasets
│   ├── prompt_engineering.py      # Template-based prompt generation
│   ├── model_runner.py            # BART model loading and inference
│   ├── evaluation.py              # Evaluation metrics (ROUGE, BLEU, etc.)
│   └── utils.py                   # Helper functions (tokenizer, batching)
│
├── results/
│   ├── outputs.csv                # Model-generated answers
│   └── performance_metrics.json   # Evaluation scores per topic
│
├── figures/
│   └── piccc.jpeg                 # Model architecture / training visualization
│
├── paper/
│   └── medical_qa_paper.tex       # Final LaTeX document
│
├── requirements.txt
├── README.md
└── run_experiments.py             # Full pipeline runner
