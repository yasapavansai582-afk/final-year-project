# TCSP-PC: Traffic Control System Process for Pollution Control

## Final Year Project — Based on IEEE Access 2025 Paper

> Ali Mohd Ali et al., *"Traffic Monitoring and Control System for Smart City Pollution Regulation Using IoT and Correlated Capsule Networks"*, IEEE Access, Vol. 13, 2025.

---

## What This Project Does

- Simulates IoT sensor data for vehicle emissions and traffic density
- Trains a Capsule Network (CapsNet) to classify vehicles as **eco-friendly** or **polluting**
- Compares against baseline models: VEQM, MPTC-PRP, MWIS, CONV-BI-LSTM
- Provides a web dashboard for real-time monitoring and recommendations
- Generates pollution control recommendations using IoT + CapsNet data

---

## Project Structure

```
tcsp_pc/
├── data/
│   ├── raw/                  # DAR TE dataset or synthetic data
│   └── processed/            # Preprocessed features
├── models/
│   ├── capsnet.py            # Correlated Capsule Network
│   ├── baselines.py          # VEQM, MPTC-PRP, MWIS, CONV-BI-LSTM
│   └── saved/                # Trained model weights
├── utils/
│   ├── data_generator.py     # IoT data simulation
│   ├── preprocessing.py      # Feature engineering
│   ├── metrics.py            # Evaluation metrics
│   └── visualizer.py         # Plots and charts
├── api/
│   └── app.py                # Flask REST API
├── dashboard/
│   ├── templates/            # HTML pages
│   └── static/               # CSS + JS
├── tests/
│   └── test_all.py           # Unit tests
├── notebooks/
│   └── exploration.ipynb     # EDA notebook
├── train.py                  # Model training script
├── evaluate.py               # Evaluation + comparison
├── requirements.txt          # Python dependencies
└── README.md
```

---

## IDE Recommendation

**Use VS Code** (most recommended for this project):
- Download: https://code.visualstudio.com
- Install extensions:
  - Python (Microsoft)
  - Pylance
  - Jupyter
  - GitLens

Alternative: **PyCharm Community Edition** (free)

---

## Setup Guide (Step by Step)

### Step 1 — Install Python 3.10+
```bash
# Check version
python --version   # should be 3.10 or 3.11

# Download from https://python.org if needed
```

### Step 2 — Clone / Download Project
```bash
# If using git
git clone <your-repo-url>
cd tcsp_pc

# Or simply unzip the project folder and open it
```

### Step 3 — Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Generate/Download Data
```bash
# Generate synthetic IoT data (if no real dataset)
python utils/data_generator.py

# This creates data/raw/iot_traffic_data.csv and data/processed/features.csv
```

### Step 6 — Train Models
```bash
# Train the Capsule Network
python train.py --model capsnet --epochs 50

# Train all models for comparison
python train.py --model all --epochs 50
```

### Step 7 — Evaluate and Compare
```bash
python evaluate.py
# Outputs comparison table + charts in results/
```

### Step 8 — Run Dashboard
```bash
python api/app.py
# Open browser: http://localhost:5000
```

---

## Dataset

This project uses the **DAR TE (Database of Road Transportation Emissions)** dataset:
- Source: Oak Ridge National Laboratory
- URL: https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1735
- Coverage: North America, 1980–2017, 1 km² grid
- Metric: CO₂ grams per km

If you cannot access the real dataset, `data_generator.py` creates realistic synthetic data mimicking the same distributions.

---

## Key Results (Expected)

| Metric | VEQM | MPTC-PRP | MWIS | CONV-BI-LSTM | TCSP-PC |
|---|---|---|---|---|---|
| Impact Verification | 0.558–0.614 | 0.666–0.748 | — | — | **0.70–0.89** |
| Classification Rate (/km) | 0.336–0.397 | 0.424–0.536 | — | — | **0.53–0.67** |
| Traffic Control (V/km) | — | — | 24–57 | 59–102 | **110–168** |
| Analysis Time (s) | — | — | 3.13–4.31 | 1.35–3.08 | **0.4–1.26** |