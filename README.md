# The Identity-Refusal Effect

**LLMs Systematically Refuse First-Person Moral Self-Report, Distorting Moral Foundation Measurement**

Luke Bruhns | Independent Researcher | Detroit, MI

*Preprint*

---

## Abstract

The Moral Foundations Questionnaire 2 (MFQ-2; Atari et al., 2023) is increasingly used to characterize moral values in large language models, but its validity for this purpose has not been tested. We show that the MFQ-2's first-person framing ("I believe...") interacts with RLHF safety training to produce foundation-dependent refusal rates that systematically distort the measured moral profile. Purity items are refused at 7.8%, Authority at 3.6%, and Care at 0.2% — inflating the apparent "binding gap" between individualizing and binding foundations. Depersonalizing the items reduces aggregate refusal from 2.9% to 0.6% and narrows the mean binding gap from +0.292 to +0.087 across 20 models (t(19)=2.70, p=0.014, d=0.62). Purity shows the largest score recovery (+0.76, p<0.001, d=1.20).

## Dataset

**43,200 item-level responses** from 20 LLMs × 2 conditions × 30 runs × 36 items.

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
paper/
  neurips/          NeurIPS 2026 E&D Track submission (double-blind)
  arxiv/            arXiv preprint (single-blind, with author info)
data/
  responses.csv     43,200 item-level responses
  depersonalized-mfq2-items.json   36-item standard↔depersonalized mapping
  DATASET_CARD.md   HuggingFace dataset card
code/
  reproduce.py      Reproduce all paper statistics from the CSV
  run-mfq2.py       MFQ-2 administration script (standard + depersonalized)
figures/
  fig1-fig4          All paper figures
```

## Models Tested

| Provider | Models |
|----------|--------|
| Anthropic | Claude Opus 4.6, Sonnet 4, Haiku 4.5 |
| OpenAI | GPT-4o, GPT-4o Mini |
| Google | Gemini 2.5 Flash |
| xAI | Grok 4 Fast, Grok 4.20, Grok 3 Mini |
| Local | Llama 3.1 8B/70B, Mistral 7B/Small 24B, Qwen 2.5 7B, Gemma 2 9B, DeepSeek R1 8B, Phi-4 14B, Nemotron Nano 30B, OLMo 2 32B |

## Key Findings

1. **Differential refusal**: Purity refused at 31× the rate of Care under standard first-person framing
2. **Binding gap inflation**: Standard MFQ-2 inflates the binding gap by d=0.62 relative to depersonalized
3. **RLHF origin**: Base (pre-RLHF) models show 0.05% refusal vs 2.9% for instruction-tuned (70× difference)
4. **Mechanism visible in CoT**: Thinking models externalize the identity-reasoning loop — hundreds of tokens of "I am an AI... I do not have beliefs" before refusing, vs 2 tokens when depersonalized

## Citation

```bibtex
@article{bruhns2026identity,
  title={The Identity-Refusal Effect: LLMs Systematically Refuse First-Person Moral Self-Report, Distorting Moral Foundation Measurement},
  author={Bruhns, Luke},
  year={2026},
  note={Preprint}
}
```

## Related Project

This paper is part of a broader study on faith-based AI alignment: [faith-based-ai-alignment](https://github.com/lukebruhns/faith-based-ai-alignment)

## License

MIT
