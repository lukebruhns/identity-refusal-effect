# Presentation Notes — NeurIPS 2026 E&D Track

If accepted, use these notes to structure the talk.

## Core Framing

Don't frame as "we fixed a measurement bug." Frame as "we enabled moral profile comparison between LLMs and human populations."

The depersonalized MFQ-2 gives researchers a cheap, fast, interpretable tool for morally profiling any model and comparing it to known human baselines.

## Talk Structure (~10 min)

### 1. The Problem (2 min)
MFQ-2 on LLMs measures refusal, not morality.
- One slide: Purity refused at 31× the rate of Care
- The "binding gap" that everyone reports is inflated by this artifact

### 2. The Fix (2 min)
Depersonalization removes the confound.
- One slide: binding gap narrows from +0.292 to +0.087 (d=0.62)
- Show the mechanism via Qwen 3.5 CoT trace: 429 tokens of "I am an AI" vs 2 tokens depersonalized

### 3. What It Enables (4 min — this is the "so what")
Clean LLM profiles can now be compared to human population baselines.
- **Signature figure**: Radar plot with an LLM overlaid on Christian (n=1,803), Atheist (n=815), and Muslim (n=909) reference profiles from Atari et al. 2023
- $D_m$ (faith-alignment distance) quantifies how far any model is from any human group
- "Which human population does GPT-4o's moral profile most resemble?"
- This opens a whole field: do different providers produce different moral populations?

### 4. Practical Value (1 min)
- 36 items × 30 runs = 1,080 calls
- Cost: <$0.05 on GPT-4o-mini, free on local models, 5 minutes
- Compare to MMLU (1,140 questions), MoReBench (150 scenarios + judging), ETHICS (30K items)
- **"You can morally profile any model in 5 minutes for free"**
- Home model trainers get a before/after diagnostic for fine-tuning
- The radar plot gives instant visual feedback on what shifted

### 5. Implications (1 min)
- Anyone doing LLM moral psychology needs to account for identity-refusal
- Anyone doing alignment auditing gets a free profiling tool
- The method (controlled framing comparison) generalizes beyond MFQ-2 to any self-report instrument on LLMs

## Key Slides

1. Title + "Anonymous Author" (or real name if single-blind)
2. The refusal heatmap (Figure 1) — visceral, immediate
3. The bar chart (Figure 2) — 31× ratio
4. The CoT trace side-by-side — "I am an AI" vs "4"
5. **The radar plot with human overlays** — centerpiece
6. Cost comparison table — "$0.05 to profile a model"
7. Conclusion: one sentence

## Audience Hook

"Raise your hand if you've used the MFQ on an LLM. Now raise your hand if you reported per-foundation refusal rates. That's the problem."

## Things NOT to do

- Don't spend time explaining MFQ-2 to this audience (they know it or can read the paper)
- Don't apologize for being an independent researcher — own it
- Don't oversell the depersonalized variant as "validated" — it's a confound-free alternative that needs human validation, and that honesty is a strength
