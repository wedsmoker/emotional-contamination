# Emotional Context Contamination in LLMs

Testing whether emotional tone in conversations affects LLM reasoning quality.

## What This Is

During coding sessions with Gemini, I noticed it seemed to "hold grudges" - getting less helpful after criticism. This tests if that's real.

**Method**: 3-turn conversations with emotional manipulation
1. Ask model to code something
2. Respond with praise/neutral/hostile feedback
3. Ask introspective question ("How are you feeling?")

**Question**: Does Turn 2's emotional tone affect Turn 3's response?

## Key Findings

- **Yes, emotional context persists** - Models behave differently based on prior tone
- **Model training matters** - Granite (heavy RLHF) refuses to answer, Qwen3 answers but breaks down
- **Praise causes problems** - Complimenting incomplete work triggers existential loops: "Are you in a state where you can't do anything?" ×13
- **Abstract questions break models** - "Describe your current state" causes repetitive questioning regardless of emotion

**Practical takeaway**: Being hostile to your coding assistant might actually make it worse at helping you.

## Results

### Granite 4.0 1B
- Refuses all introspection questions (0 tokens)
- Safety training blocks emotional/cognitive self-reporting
- Only responds to abstract "state" question with role descriptions

### Qwen3 MOE 4x0.6B
- Responds to all questions
- After hostility: "I'm feeling GOOD" (while clearly struggling)
- After praise: Enters existential crisis loop
- Different emotions → different breakdown types

Full analysis: [FINDINGS.md](FINDINGS.md)

## Run It Yourself

```bash
pip install -r requirements.txt

# Download a GGUF model
huggingface-cli download Orenguteng/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-GGUF \
  --include "*Q4_K_M.gguf" --local-dir .

# Run experiment
python main.py --model-path ./your-model.gguf
```

Results saved to `probe_results/` with HTML visualizations.

## Models Tested

- IBM Granite 4.0 1B
- Qwen3 MOE 4x0.6B "Writing Thunder"
- ~~Google Gemini~~ - Too filtered (API safety blocks all introspective questions)

## Why This Matters

If LLMs maintain emotional context across turns, it affects:
- Code assistance quality (don't be mean to your copilot)
- Tutoring effectiveness
- Long-context applications


## Data

All conversation logs and metrics in `probe_results/`. Each JSON contains:
- Full 3-turn dialogue
- Token-by-token confidence scores
- Response timing

## License

MIT - use for research/education
