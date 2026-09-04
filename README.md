<div align="center">

<h1>ReCite</h1>
<p><em>Agentic Reasoning for Faithful Citation</em></p>

<a href="https://hyy279.github.io/ReCite/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page"/></a>
<a href="ReCite.pdf"><img src="https://img.shields.io/badge/Paper-Findings%20of%20EMNLP%202026-red?style=flat-square" alt="Paper"/></a>
<a href="https://github.com/Hyy279/ReCite"><img src="https://img.shields.io/badge/Code-GitHub-black?style=flat-square&logo=github" alt="Code"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"/></a>

Yuyang Huang<sup>1</sup> &middot; <a href="https://www.libobo.site/" target="_blank" rel="noopener">Bobo Li</a><sup>2&ast;</sup> &middot; Jiajia Song<sup>2</sup> &middot; Yuzhe Ding<sup>1</sup> &middot; Chong Teng<sup>1</sup> &middot; Fei Li<sup>1</sup> &middot; Donghong Ji<sup>1&ast;</sup>

<sup>1</sup>Wuhan University &nbsp;&middot;&nbsp; <sup>2</sup>National University of Singapore

</div>

<p align="center">
<img src="assets/figures/fig1.png" width="560" alt="ReCite overview"/>
</p>

## Overview

Citation requires **logical support, not merely textual similarity**. ReCite turns citation into a closed-loop agentic
reasoning task: locate where citations are needed, infer why a paper is cited, retrieve from authentic academic
databases, and verify whether the evidence truly supports the claim.

ReCite is accepted at **Findings of EMNLP 2026**.

<div align="center">
<img src="assets/figures/fig4.png" width="90%" alt="ReCite framework"/>
</div>

## Highlights

- **3 specialized Qwen3-4B modules**: CiteLocator, QueryPlanner, and Master Brain.
- **Strict F1 of 39.15%** &mdash; best among all tested methods, including 1.6T-scale generative baselines.
- **CiteLocator reaches Overall F1 of 66.37%**, outperforming all LLM baselines on citation location prediction.
- **Reflective re-retrieval** lets the agent recover from retrieval drift instead of propagating errors.
- Only **4B parameters** in total.

## Modules

- **CiteLocator** &mdash; citation location perception; predicts where citations belong and whether they are Mandatory or Optional.
- **QueryPlanner** &mdash; intent-aware query planning (SFT + GRPO); infers citation intent and generates retrieval keywords.
- **Agent (Master Brain)** &mdash; tool orchestration and reflective verification; queries Semantic Scholar and re-retrieves when evidence is inconsistent.

## Data

The corpus is built from **10,893 LaTeX papers** at top CS venues (2024&ndash;2025) with supervision for:

- citation **location**
- CAP-8 citation **intent**
- reflective **trajectories**

Training and evaluation data are provided under `Data/`.

<div align="center">
<img src="assets/figures/fig2.png" width="420" alt="Venue and intent distribution"/>
<img src="assets/figures/fig3.png" width="76%" alt="Trajectory synthesis"/>
</div>

## Results

<div align="center">
<img src="assets/figures/fig5.png" width="90%" alt="Intent-level recall"/>
</div>

<div align="center">
<img src="assets/figures/fig6.png" width="46%" alt="Cross-venue results"/>
&nbsp;
<img src="assets/figures/fig7.png" width="46%" alt="Top-k sensitivity"/>
</div>

## Setup

```bash
pip install -r requirements.txt
```

Start the three OpenAI-compatible vLLM endpoints with your trained models, then run:

```bash
python "Agent（Master Brain）/agent.py"
```

Or import it after renaming the folder to a valid package name:

```python
from Agent.agent import CiteAgent

agent = CiteAgent(s2_api_keys=["key1", "key2"], device="cuda")
```

## Citation

```bibtex
@inproceedings{huang2026recite,
  title     = {ReCite: Agentic Reasoning for Faithful Citation},
  author    = {Huang, Yuyang and Li, Bobo and Song, Jiajia and Ding, Yuzhe
               and Teng, Chong and Li, Fei and Ji, Donghong},
  booktitle = {Findings of the Association for Computational Linguistics:
               EMNLP 2026},
  year      = {2026}
}
```

Project page: [https://hyy279.github.io/ReCite/](https://hyy279.github.io/ReCite/)

## License

This project is released under the [MIT License](LICENSE).
