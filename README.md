# AI Progress in Customer Service: Benchmark Analysis and Forecast

This repository contains data, code, and analysis for tracking AI progress on task-oriented dialogue benchmarks relevant to Customer Service Representative (CSR) automation.

## Overview

Large Language Models (LLMs) continue to improve on complex, real-world tasks. This project analyzes AI progress on key dialogue benchmarks from 2019 to early 2025, focusing on capabilities relevant to customer service automation:

- **Adaptability**: Can AI handle new products/services/procedures? (Schema-Guided Dialogue benchmark)
- **Complex Interaction Tracking**: Can AI track all details in multi-part conversations? (MultiWOZ benchmark)
- **Task Completion**: Does AI successfully resolve the customer's issue? (Success Rate metrics)

## Key Findings

1. **Adaptability (SGD JGA)**: Rapid improvement with GPT-4-Turbo achieving 88.7% JGA in late 2023.
2. **Complex Tracking (MultiWOZ JGA)**: More variable progress with high sensitivity to evaluation method.
3. **Task Completion (MultiWOZ SR)**: More stable metrics in the ~67-76% range across models.

Our speculative forecast based on benchmark trends suggests AI automation capabilities could reach high levels for customer service tasks between late 2025 and late 2028.

## Repository Structure

- **`data/`**: Contains benchmark results and raw data files
- **`scripts/`**: Python scripts for creating the dataset and generating visualizations
- **`figures/`**: Generated plots showing AI progress on different metrics

## Visualizations

Adaptability: AI Progress on SGD JGA

Complex Tracking: AI Progress on MultiWOZ JGA

Task Completion: AI Progress on MultiWOZ Success Rate

## Usage

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the data creation script: `python scripts/create_data_table.py`
4. Generate the visualizations: `python scripts/create_plots.py`

## Data Sources

The benchmark data is compiled from academic papers and publications evaluating various models on the MultiWOZ and Schema-Guided Dialogue datasets. See the full reference list in the original publication.

## Future Work

We welcome contributions to keep this benchmark collection updated with results from new models as they are released. There are significant gaps in publicly available standardized benchmark results for the latest LLMs on task-oriented dialogue.

## Citation

If you use this data or code in your research, please cite:

Pooley, G. (2025). How Fast is AI Learning Customer Service? A Benchmark Analysis and Forecast.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
