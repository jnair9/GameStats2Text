# GameStats2Text

> **GameStats2Text** is an end-to-end pipeline that takes basketball game statistics and post-game interview transcripts, trains an NLP model, and generates in-character responses (from a player’s perspective) based on the game data.

---

## Project Structure

```text
GameStats2Text/
│
├── data/
│   ├── data.ipynb                 # Exploratory analysis on raw data
│   ├── data-new.ipynb             # Updated data processing
│   ├── interviews.json            # Raw interview transcripts (JSON)
│   ├── processed_interviews.json  # Cleaned interview Q&A
│   ├── data.json                  # Merged stats + interviews
│   ├── cleaned_data.json          # Final cleaned dataset
│   ├── data.csv                   # Tabular export of merged data
│   └── dataset.csv                # Final CSV for model training
│
├── src/
│   ├── preprocessing/
│   │   └── stats_encoder.py       # Statistical feature encoder
│   │
│   ├── models/
│   │   ├── encoder.py             # Defines the encoder architecture
│   │   └── generator.py           # Defines the response generator
│   │
│   ├── training/
│   │   ├── train_encoder.py       # Training loop for encoder
│   │   └── train_generator.py     # Training loop for generator
│   │
│   └── evaluation/
│       └── metrics.py             # Evaluation metrics (BLEU, ROUGE, etc.)
│
├── main.ipynb                     # End-to-end demonstration notebook
└── README.md                      # Project overview and setup instructions


Dependency Steps
Install poetry
run poetry install --no-root
run: poetry lock
run poetry install --no-root
install poetry shell : poetry self add poetry-plugin-shell
run: poetry shell
python -m ipykernel install --user --name gamestats2text --display-name "GameStats2Text (Poetry)"
run poetry shell


