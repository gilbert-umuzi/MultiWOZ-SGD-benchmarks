# AI Progress in Customer Service: Benchmark Analysis and Forecast

This repository contains data, code, and visualizations used to track AI progress on task-oriented dialogue benchmarks relevant to Customer Service Representative (CSR) automation. See the [original research note](https://docs.google.com/document/d/1XzHUTZR7ynu0OD88NeKkF6vqkrEzVC298PnlHLbo1m4/edit?usp=sharing).

## Overview

Large Language Models (LLMs) continue to make significant gains on complex, real-world tasks. This project analyzes AI progress across key dialogue benchmarks from 2019 through early 2025, with a focus on capabilities required for automating customer service roles.

We track three core skill areas:

- **Adaptability**: Can AI generalize to new tasks, services, and domains? *(Schema-Guided Dialogue benchmark — JGA on “Unseen” tasks)*
- **Complex Interaction Tracking**: Can AI accurately track user goals across multi-turn interactions? *(MultiWOZ benchmark — JGA)*
- **Conversational Robustness**: Can AI handle realistic multi-turn scenarios requiring memory, instruction-following, coherence, and editing? *(MultiChallenge benchmark — average score vs. compute scaling)*

## Key Findings

1. **Adaptability (SGD - JGA Unseen):**  
   Steady improvement over time; fine-tuned models are on track to reach human-equivalent adaptability (~95% JGA) by **Q2 2027** if current trends continue.

2. **Complex Tracking (MultiWOZ - JGA):**  
   More varied progress; fine-tuned models forecast to reach 95% by **Q2 2028**, though zero/few-shot methods remain inconsistent.

3. **MultiChallenge vs. Training Compute:**  
   Because MultiChallenge is newer, we model performance against training compute (FLOP).  
   - Trend: **+14.9% accuracy per 10× increase in compute**  
   - Forecast: Reaching **80%** score (competitive with top-quartile human agents) may require ~**10^28.0 FLOP** (~25.6× increase), potentially arriving **late 2026–early 2028**  
   - Hitting **95%** (full automation threshold) may require ~**10^29.0 FLOP** (~262× increase), likely **late 2027–mid 2029**

## Repository Structure

- **`data/`** — Raw benchmark results and training compute estimates  
- **`scripts/`** — Python scripts for processing data and generating visualizations  
- **`figures/`** — Output charts tracking AI progress on key metrics  

## Visualizations

- **Figure 1:** Adaptability (SGD - JGA Unseen) and Complex Tracking (MultiWOZ - JGA) over time  
- **Figure 2:** Conversational Capability vs. Training Compute (MultiChallenge)

## Usage

1. Clone this repository  
2. Install requirements: `pip install -r requirements.txt`  
3. Run: `python scripts/multiwoz_sgd.ipynb` and `python scripts/multichallenge.ipynb`

## Data Sources

Benchmark data is compiled from publicly available academic papers, leaderboard results, and major model evaluations (e.g., GPT-4, Claude 3, DeepSeek, Mixtral, LLaMA 3). MultiChallenge results from [SEAL](https://scale.com/leaderboard/multichallenge) are matched to training compute estimates from [Epoch AI](https://epoch.ai/data/notable-ai-models), model release disclosures, and triangulation of the best available public guesses.

## Future Work

We welcome community contributions to help maintain and expand this benchmark tracking effort. Key gaps remain in standardized, transparent evaluations for newer LLMs on task-oriented dialogue, especially for real-world deployment settings.

## Citation

If you use this repository, please cite:

**Pooley, G. (2025). _How Fast is AI Learning Customer Service? A Benchmark Analysis and Forecast._**

## License

This project is licensed under the MIT License.
