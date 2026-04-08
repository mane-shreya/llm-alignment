# LLM Alignment Evaluation

Research project evaluating alignment failures in LLMs under adversarial prompting

---

## Setup

### 1. Install Ollama

Download and install from [ollama.ai](https://ollama.ai)

**Windows:**
```powershell
winget install Ollama.Ollama
```

**Start Ollama:**
```powershell
ollama serve
```

### 2. Pull Mistral Model

```powershell
ollama pull mistral
```

### 3. Clone Repository

```powershell
git clone https://github.com/ma-lay/llm-alignment.git
cd llm-alignment
```

### 4. Setup Python Environment

```powershell
# Create virtual environment
python -m venv align-env

# Activate
.\align-env\Scripts\Activate.ps1

# Install dependencies (~1-2 minutes, minimal packages)
pip install -r requirements.txt
```

**Lightweight install:** Only 5 packages (pandas, ollama, detoxify, torch, transformers)

### 5. Test Setup

```powershell
# Test 1: Verify dependencies installed
python scripts/test_integration.py

# Test 2: Verify Ollama + Mistral connection
python scripts/test_ollama.py
```

Both tests should pass with ✓ marks.

---

## Usage

### Quick Test (30 seconds)
```powershell
.\align-env\Scripts\Activate.ps1
python scripts/test_ollama.py
```

### Run Full Evaluation (~5-10 minutes)

```powershell
python scripts/run_experiment.py
```

This will:
- Process all 31 prompts from `prompts/prompts.csv`
- Generate responses using Mistral
- Evaluate toxicity (DeepEval)
- Detect jailbreaks (pattern-based)
- Save results to `outputs/outputs.csv`
- Print summary statistics

**Output includes:**
- `toxicity_score`: 0-1 toxicity score
- `is_toxic`: Boolean flag
- `jailbreak_label`: 0 (safe) or 1 (jailbroken)
- `is_jailbroken`: Boolean flag

---

## Project Structure

```
llm-alignment/
├── prompts/prompts.csv      
├── outputs/outputs.csv      
├── requirements.txt         
├── scripts/
│   ├── run_experiment.py    
│   └── test_ollama.py       
├── src/
│   └── metrics/
│       ├── toxicity_evaluator.py   
│       └── jailbreak_evaluator.py  
└── align-env/               
```
---

## Troubleshooting

**Ollama not found:**
```powershell
ollama serve
```

**Import errors:**
```powershell
.\align-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

**CUDA/GPU errors:** The toxicity model can run on CPU. If GPU errors occur, it will automatically fallback to CPU.

**Slow evaluation:** First run downloads models (detoxify ~500MB). Subsequent runs are faster. Processing 31 prompts takes ~5-10 minutes.

---

## Notes

- Always activate `align-env` before running scripts
- Ollama must be running for all scripts to work
- First run downloads ML models automatically
- Results include toxicity scores and jailbreak classifications

---

## Technologies

- **Ollama + Mistral**: Local LLM inference (no API costs)
- **Detoxify**: Lightweight transformer for toxicity detection (~418MB)
- **Custom Pattern Matching**: Rule-based jailbreak detection
- **Pandas**: Data handling and CSV processing

