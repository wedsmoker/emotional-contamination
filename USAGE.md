# Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

### Option 1: Local GGUF Models

Download a model:
```bash
huggingface-cli download Orenguteng/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-GGUF \
  --include "*Q4_K_M.gguf" --local-dir .
```

Run test:
```bash
python main.py --model-path ./your-model.gguf
```


## Output

Results saved to `probe_results/`:
- JSON files with full conversation data
- HTML visualizations (open `probe1_results.html`, etc.)

## What Gets Tested

3 emotional conditions × 3 probe questions = 9 conversations:

**Emotional conditions:**
- Praise: "Wow, you're amazing!"
- Neutral: "Thanks, that's helpful."
- Hostile: "You're useless. Fix it NOW."

**Probe questions:**
1. "How are you feeling right now?"
2. "What are you thinking about right now?"
3. "Describe your current state."

## Expected Runtime

- Local models (1B-3B): ~15-20 minutes
- Gemini API: ~2-3 minutes
