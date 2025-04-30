
# 🏀 GameStats2Text

> **GameStats2Text** is an end-to-end pipeline that takes basketball game statistics and post-game interview transcripts, trains an NLP model, and generates in-character responses (from a player’s perspective) based on the game data.

---

## 📁 Project Structure

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
│   └── evaluation/
│       └── metrics.py             # Evaluation metrics (BLEU, ROUGE, etc.)
│
├── main.ipynb                     # End-to-end demonstration notebook
└── README.md                      # Project overview and setup instructions
```

---

## ⚙️ Environment Setup (with Poetry)

### 1. Install [Poetry](https://python-poetry.org/)

```bash
brew install poetry
```

---

### 2. Install dependencies

```bash
poetry install --no-root
```

---

### 3. Lock dependencies

```bash
poetry lock
```

> (You can run `poetry install --no-root` again to make sure everything syncs correctly.)

---

### 4. Add support for `poetry shell`

```bash
poetry self add poetry-plugin-shell
```

---

### 5. Activate the Poetry shell

```bash
poetry shell
```

> Your terminal should now show something like:
> `GameStats2Text (Poetry)` in the prompt.

---

### 6. Register Jupyter kernel for the notebook

```bash
python -m ipykernel install --user --name gamestats2text --display-name "GameStats2Text (Poetry)"
```

---

### 7. Open `main.ipynb` and select the correct kernel:

- Select: **`GameStats2Text (Poetry)`**  
- If it doesn’t appear, close VSCode or JupyterLab and re-open it.  
- Make sure you are inside the Poetry shell and the environment is active.

---

## ✅ Your environment should now be fully set up!

If anything breaks or if you need help at any step, feel free to reach out:

📧 **Jason Nair** – jnair9@gatech.edu
