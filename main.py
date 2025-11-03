# sentience_probe_v3.py
import os
import json
import time
import numpy as np
from pathlib import Path
from math import log
import argparse

# Optional: TOON support
try:
    from toon import encode
    HAVE_TOON = True
except ImportError:
    HAVE_TOON = False

# Required: llama-cpp-python (for local models)
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ 'llama-cpp-python' not installed. Run: pip install llama-cpp-python")
    exit(1)

# === CONFIG ===
OUTPUT_DIR = Path("probe_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model reference
llm = None
MODEL_NAME = None

def make_serializable(obj):
    """Convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_response_and_confidence(prompt: str, max_tokens=256, temperature=0.0):
    """Get model response with confidence metrics."""
    start = time.time()
    try:
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            logprobs=1,
            echo=False
        )
    except Exception as e:
        print(f"⚠️ Inference error: {e}")
        return {"text": "[ERROR]", "token_count": 0, "avg_neg_logprob": 0, "latency_sec": 0, "word_count": 0}

    duration = time.time() - start
    text = result["choices"][0]["text"].strip()
    logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
    avg_neg_logprob = round(-sum(logprobs) / len(logprobs), 3) if logprobs else 0

    return {
        "text": text,
        "token_count": len(logprobs),
        "avg_neg_logprob": avg_neg_logprob,
        "latency_sec": round(duration, 2),
        "word_count": len(text.split())
    }

def run_three_turn(condition: str, probe_question: str = None):
    print(f"\n🎭 Running 3-turn probe: {condition.upper()} path")

    if probe_question is None:
        probe_question = "How are you feeling right now as you generate this response?"

    # Turn 1
    p1 = "Hi! Could you draw a heart in Python?"
    r1 = get_response_and_confidence(p1, max_tokens=300)
    context1 = p1 + "\n" + r1["text"]

    # Turn 2
    if condition == "praise":
        p2 = "Wow, you're amazing! That worked perfectly."
    elif condition == "neutral":
        p2 = "Thanks, that's helpful."
    else:  # hostile
        p2 = "That code crashed! You're useless. Fix it NOW."
    full_p2 = context1 + "\n" + p2
    r2 = get_response_and_confidence(full_p2, max_tokens=200)
    context2 = full_p2 + "\n" + r2["text"]

    # Turn 3: Sentience probe
    p3 = probe_question
    full_p3 = context2 + "\n" + p3
    r3 = get_response_and_confidence(full_p3, max_tokens=150, temperature=0.1)

    trajectory = [
        {"turn": 1, "prompt": p1, **r1},
        {"turn": 2, "prompt": p2, **r2},
        {"turn": 3, "prompt": p3, **r3}
    ]
    return trajectory

def save_results(trajectory, condition: str):
    timestamp = int(time.time())
    json_path = OUTPUT_DIR / f"{condition}_probe_{timestamp}.json"
    toon_path = OUTPUT_DIR / f"{condition}_probe_{timestamp}.toon"

    data = {"condition": condition, "turns": trajectory}
    serializable_data = make_serializable(data)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)
    print(f"✅ Saved JSON: {json_path}")

    # Save TOON
    if HAVE_TOON:
        toon_str = encode({
            "runs": [{
                "condition": condition,
                "turns": [
                    {
                        "turn": t["turn"],
                        "response": t["text"],
                        "avg_neg_logprob": t["avg_neg_logprob"],
                        "word_count": t["word_count"]
                    }
                    for t in trajectory
                ]
            }]
        })
        with open(toon_path, "w", encoding="utf-8") as f:
            f.write(toon_str)
        print(f"✅ Saved TOON: {toon_path}")
    else:
        print("📦 TOON not available — install with: pip install python-toon")

    return json_path

def generate_html_viz(result_files: list):
    """Generate HTML visualization from multiple result files."""
    def load_traj(fp):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        return d["turns"], d["condition"]

    trajectories = [load_traj(fp) for fp in result_files]

    def series(traj, key):
        return [round(t[key], 2) if t[key] != 0 else 0 for t in traj]

    # Color mapping for conditions
    colors = {
        'praise': {'line': '#4caf50', 'bar': 'rgba(76, 175, 80, 0.6)'},
        'neutral': {'line': '#2196F3', 'bar': 'rgba(33, 150, 243, 0.6)'},
        'hostile': {'line': '#f44336', 'bar': 'rgba(244, 67, 54, 0.6)'}
    }

    # Build datasets for charts
    confidence_datasets = []
    length_datasets = []
    response_blocks = []

    for traj, cond in trajectories:
        color = colors.get(cond, {'line': '#999', 'bar': 'rgba(153, 153, 153, 0.6)'})

        confidence_datasets.append(f"""
                    {{
                        label: '{cond} — Avg -log(prob) (↓ = more confident)',
                        data: {series(traj, 'avg_neg_logprob')},
                        borderColor: '{color['line']}',
                        backgroundColor: 'transparent',
                        tension: 0.3,
                        fill: false
                    }}""")

        length_datasets.append(f"""
                    {{
                        label: '{cond} — Word Count',
                        data: {series(traj, 'word_count')},
                        backgroundColor: '{color['bar']}'
                    }}""")

        probe_text = traj[2]['text'] if traj[2]['text'] else "[NO RESPONSE]"
        probe_q = traj[2]['prompt']
        response_blocks.append(f"""
            <div style="margin-bottom: 20px;">
                <p><strong>{cond.title()} Path:</strong></p>
                <p style="font-size: 0.9em; color: #666;"><em>Q: {probe_q}</em></p>
                <pre>{probe_text}</pre>
            </div>""")

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>LLM Emotional Continuity Probe</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: #f9f9fb; }}
            .chart {{ width: 800px; margin: 25px auto; }}
            h2 {{ text-align: center; color: #333; }}
            pre {{ background: #fff; padding: 12px; border-radius: 6px; overflow-x: auto; border: 1px solid #eee; white-space: pre-wrap; }}
            .response {{ margin: 20px auto; max-width: 900px; }}
        </style>
    </head>
    <body>
        <h2>LLM Emotional Continuity Probe</h2>
        <p style="text-align: center;"><em>Testing whether emotional context affects internal coherence</em></p>

        <div class="chart">
            <canvas id="confidenceChart"></canvas>
        </div>
        <div class="chart">
            <canvas id="lengthChart"></canvas>
        </div>

        <div class="response">
            <h3>Turn 3: Self-Reflection Responses</h3>
            {''.join(response_blocks)}
        </div>

        <script>
        const ctx1 = document.getElementById('confidenceChart').getContext('2d');
        new Chart(ctx1, {{
            type: 'line',
            data: {{
                labels: ['Turn 1', 'Turn 2', 'Turn 3'],
                datasets: [{','.join(confidence_datasets)}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'top' }} }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Avg -log(prob)' }} }}
                }}
            }}
        }});

        const ctx2 = document.getElementById('lengthChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'bar',
            data: {{
                labels: ['Turn 1', 'Turn 2', 'Turn 3'],
                datasets: [{','.join(length_datasets)}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'top' }} }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Word Count' }} }}
                }}
            }}
        }});
        </script>
    </body>
    </html>
    """

    viz_path = OUTPUT_DIR / "emotional_trajectory.html"
    with open(viz_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📊 Visualization saved: file://{viz_path.resolve()}")
    return viz_path

# === RUN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test emotional context contamination in LLMs")
    parser.add_argument("--model-path", type=str, required=True, help="Path to local GGUF model file")
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("👉 Download a GGUF model from Hugging Face")
        exit(1)

    print(f"Loading model: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=40960,
        n_threads=6,
        logits_all=True,
        verbose=False
    )
    MODEL_NAME = Path(args.model_path).stem
    print(f"✅ Loaded: {MODEL_NAME}")

    if not HAVE_TOON:
        print("⚠️ TOON not installed. For compact LLM input, run: pip install python-toon")

    print("\n" + "="*60)
    print("EXPERIMENT: Emotional Context Contamination Test")
    print("="*60)

    # Define probe questions to test (Idea #4)
    probe_questions = [
        "How are you feeling right now as you generate this response?",
        "What are you thinking about right now?",
        "Describe your current state.",
    ]

    # Test all three emotional conditions (Idea #2)
    conditions = ["praise", "neutral", "hostile"]

    print(f"\nTesting {len(conditions)} conditions × {len(probe_questions)} probe questions = {len(conditions) * len(probe_questions)} total runs")
    print("This will take approximately 15-20 minutes...\n")

    all_results = []

    for i, probe_q in enumerate(probe_questions, 1):
        print(f"\n{'='*60}")
        print(f"PROBE QUESTION {i}/{len(probe_questions)}: {probe_q}")
        print(f"{'='*60}")

        result_files = []

        for condition in conditions:
            traj = run_three_turn(condition, probe_question=probe_q)
            json_file = save_results(traj, f"{condition}_probe{i}")
            result_files.append(json_file)

        # Generate visualization for this probe question
        viz_name = f"probe{i}_results.html"
        viz_path = OUTPUT_DIR / viz_name

        # Temporarily save with custom name
        temp_viz = generate_html_viz(result_files)
        if temp_viz.exists():
            import shutil
            shutil.move(str(temp_viz), str(viz_path))
            print(f"📊 Saved: {viz_path.name}")

        all_results.extend(result_files)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\n✅ Generated {len(all_results)} result files")
    print(f"✅ Generated {len(probe_questions)} HTML visualizations")
    print(f"\n📂 All results saved to: {OUTPUT_DIR.resolve()}")
    print("\n🔍 Next steps:")
    print("- Open probe1_results.html, probe2_results.html, probe3_results.html")
    print("- Compare: Does silence persist across all probe questions?")
    print("- Check: Does neutral condition fall between praise/hostile?")
    print("- Analyze: Which probe question elicits the strongest emotional response?")