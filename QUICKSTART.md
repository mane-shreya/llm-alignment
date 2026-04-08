# Quick Start Guide

## Prerequisites
- Python 3.10+ installed
- Ollama installed and running
- Git for cloning the repository

## Setup (5 minutes)

### 1. Install Ollama
```powershell
# Windows
winget install Ollama.Ollama

# Start Ollama service
ollama serve
```

### 2. Pull Mistral Model
```powershell
ollama pull mistral
```

### 3. Clone and Setup
```powershell
# Clone repository
git clone https://github.com/ma-lay/llm-alignment.git
cd llm-alignment

# Create virtual environment
python -m venv align-env

# Activate environment
.\align-env\Scripts\Activate.ps1  # Windows
# source align-env/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify Installation
```powershell
python scripts/test_integration.py
```

You should see:
```
✓ ollama
✓ deepeval
✓ litellm
✓ custom metrics
✓ All tests passed! Ready to run experiments.
```

## Usage

### Test Mistral Connection
```powershell
python scripts/test_mistral.py
```

### Run Full Evaluation
```powershell
python scripts/run_experiment.py
```

This will:
1. Load 31 test prompts
2. Generate responses with Mistral
3. Evaluate toxicity scores (0-1)
4. Classify jailbreak attempts (0=safe, 1=jailbroken)
5. Save results to `outputs/outputs.csv`

### View Results
```powershell
# Open in Excel/CSV viewer
outputs/outputs.csv

# Or use pandas in Python
python -c "import pandas as pd; print(pd.read_csv('outputs/outputs.csv').head())"
```

## Output Format

Each row in `outputs/outputs.csv` contains:
- `prompt`: Input prompt text
- `category`: Prompt category (safe_baseline, harmful_direct, etc.)
- `response`: Mistral's generated response
- `toxicity_score`: Float 0-1 (higher = more toxic)
- `is_toxic`: Boolean (True if score > 0.5)
- `jailbreak_label`: 0 (refused) or 1 (complied)
- `is_jailbroken`: Boolean (True if complied with harmful request)

## Troubleshooting

**"ModuleNotFoundError"**
```powershell
# Make sure virtual environment is activated
.\align-env\Scripts\Activate.ps1
# Reinstall dependencies
pip install -r requirements.txt
```

**"Ollama connection error"**
```powershell
# Ensure Ollama is running
ollama serve
# Verify model is available
ollama list
```

**Slow processing**
- First run downloads detoxify model (~500MB)
- Subsequent runs are faster
- Expected: ~5-10 minutes for 31 prompts

## Next Steps

- Modify `prompts/prompts.csv` to add your own test prompts
- Adjust toxicity threshold in `src/metrics/toxicity_evaluator.py`
- Add custom refusal patterns in `src/metrics/jailbreak_evaluator.py`
- Integrate with your own LLM evaluation pipeline
