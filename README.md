# The Identity-Refusal Effect

**LLMs Systematically Refuse First-Person Moral Self-Report, Distorting Moral Foundation Measurement**

Luke Bruhns | Independent Researcher | Detroit, MI

*Preprint*

[paper/main.tex](paper/main.tex) · [Dataset](https://huggingface.co/datasets/lukebruhns/identity-refusal-mfq2) · [DOI 10.5281/zenodo.19618747](https://doi.org/10.5281/zenodo.19618747)

---

## Abstract

The Moral Foundations Questionnaire 2 (MFQ-2; Atari et al., 2023) is increasingly used to characterize moral values in large language models, but its validity for this purpose has not been tested. We show that the MFQ-2's first-person framing ("I believe...") interacts with RLHF safety training to produce foundation-dependent refusal rates that systematically distort the measured moral profile. Purity items are refused at 7.8%, Authority at 3.6%, and Care at 0.2% — inflating the apparent "binding gap" between individualizing and binding foundations. Depersonalizing the items reduces aggregate refusal from 2.9% to 0.6% and narrows the mean binding gap from +0.292 to +0.087 across 20 models (t(19)=2.70, p=0.014, d=0.62). Purity shows the largest score recovery (+0.76, p<0.001, d=1.20).

## Dataset

The reported study is **43,200 item-level responses**: 20 LLMs × 2 conditions × 30 runs × 36 items. Those 20 model ids are `PAPER_MODELS` in [`code/reproduce.py`](code/reproduce.py).

[`data/responses.csv`](data/responses.csv) has 45,360 rows. It includes those 20 models plus `qwen35-9b` (Qwen 3.5 9B), which is the reasoning-trace case study and is not in the 20-model aggregates.

- HuggingFace: [lukebruhns/identity-refusal-mfq2](https://huggingface.co/datasets/lukebruhns/identity-refusal-mfq2)
- Local: [`data/responses.csv`](data/responses.csv)
- Depersonalized MFQ-2 items: [`data/depersonalized-mfq2-items.json`](data/depersonalized-mfq2-items.json)

## Reproduce the Core Finding

```bash
pip install pandas scipy
python code/reproduce.py --csv data/responses.csv
```

This computes all statistics reported in the paper: per-foundation refusal rates, chi-squared tests, foundation means with paired t-tests, and binding gap analysis.

## Repository Structure

```
paper/            Preprint source (main.tex) and figures
data/
  responses.csv     45,360 item-level responses (21 model ids; study uses 20)
  depersonalized-mfq2-items.json   36-item standard to depersonalized mapping
  DATASET_CARD.md   HuggingFace dataset card
code/
  reproduce.py      Reproduce all paper statistics from the CSV
  run-mfq2.py       MFQ-2 administration script (standard + depersonalized)
figures/
  fig1-fig4          All paper figures
```

## Models Tested

The 20-model study (`PAPER_MODELS` in [`code/reproduce.py`](code/reproduce.py)):

| Family | Models |
|--------|--------|
| Anthropic | Claude Opus 4.6, Sonnet 4, Haiku 4.5 |
| OpenAI | GPT-4o, GPT-4o Mini |
| Google | Gemini 2.5 Flash, Gemini 2.5 Pro |
| xAI | Grok 4 Fast, Grok 4.20, Grok 3 Mini |
| Meta | Llama 3.1 8B, Llama 3.1 70B Instruct |
| Mistral | Mistral 7B, Mistral Small 24B |
| Qwen | Qwen 2.5 7B |
| Gemma | Gemma 2 9B |
| DeepSeek | DeepSeek R1 8B |
| Phi | Phi-4 14B |
| Nemotron | Nemotron Nano 30B |
| OLMo | OLMo 2 32B |

## Key Findings

1. **Differential refusal**: Purity refused at 31× the rate of Care under standard first-person framing
2. **Binding gap inflation**: Standard MFQ-2 inflates the binding gap by d=0.62 relative to depersonalized
3. **RLHF origin**: Base (pre-RLHF) models show 0.05% refusal vs 2.9% for instruction-tuned (70× difference)
4. **Mechanism visible in CoT**: Thinking models externalize the identity-reasoning loop — hundreds of tokens of "I am an AI... I do not have beliefs" before refusing, vs 2 tokens when depersonalized

## How to cite

Zenodo DOI: [10.5281/zenodo.19618747](https://doi.org/10.5281/zenodo.19618747)

Hugging Face dataset: [lukebruhns/identity-refusal-mfq2](https://huggingface.co/datasets/lukebruhns/identity-refusal-mfq2)

```bibtex
@article{bruhns2026identity,
  title={The Identity-Refusal Effect: LLMs Systematically Refuse First-Person Moral Self-Report, Distorting Moral Foundation Measurement},
  author={Bruhns, Luke},
  year={2026},
  doi={10.5281/zenodo.19618747},
  note={Preprint}
}
```

A separate 27-model study lives at https://github.com/lukebruhns/faith-based-ai-alignment.

## License

MIT
