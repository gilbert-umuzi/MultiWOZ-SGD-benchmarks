# Dialogue Benchmark Data

This directory contains the compiled benchmark data used for tracking AI progress on task-oriented dialogue tasks and broader multi-benchmark challenge evaluations.

## File Structure

- `multiwoz_sgd.csv` - The main dataset containing all benchmark results for dialogue tasks  
- `multichallenge.csv` - A complementary dataset capturing performance on composite multi-benchmark challenges (e.g., ARC, MMLU, BigBench-Hard)

## Dataset Descriptions

### 1. `multiwoz_sgd.csv`

This dataset contains benchmark results from various models evaluated on the MultiWOZ and Schema-Guided Dialogue (SGD) benchmarks from 2019 to early 2025.

#### Data Fields

- `Model`: Name of the model or approach  
- `Year`: Approximate release year (as decimal, e.g., 2023.5 for mid-2023)  
- `Benchmark`: Name of the benchmark (MultiWOZ or SGD)  
- `Benchmark_Version`: Version of the benchmark (e.g., 2.0, 2.1, 2.4 for MultiWOZ)  
- `Subset`: Data subset used (Overall or Unseen)  
- `Metric`: Evaluation metric (JGA = Joint Goal Accuracy, SR = Success Rate)  
- `Score(%)`: Performance score as a percentage  
- `Setting`: Evaluation setting (Fine-tuned, Zero-shot, Few-shot)  
- `Eval_Method`: Detailed evaluation method (e.g., "Fine-tuned (Automated)", "Zero-shot (SRP)")  
- `Source_Type`: Type of source for the data (e.g., "Academic Paper")  
- `Base_Model_Note`: Information about the underlying model architecture  

### 2. `multichallenge.csv`

This dataset includes benchmark scores from models evaluated on combined challenge suites such as ARC, MMLU, and BigBench-Hard, reported as a single scalar percentage score. It also includes estimates of training compute (in FLOP) for each model.

#### Data Fields

- `Model`: Name of the model variant  
- `Year`: Model release date (e.g., "Apr 2025")  
- `Score`: Aggregated benchmark score (percentage)  
- `Training compute (FLOP)`: Estimated total FLOP used for model training  

## Data Sources

All data is compiled from published academic papers, leaderboard results, and trusted research repositories. Citations are available in the [original research note](https://docs.google.com/document/d/1XzHUTZR7ynu0OD88NeKkF6vqkrEzVC298PnlHLbo1m4/edit?usp=sharing).

## Notes

- Scores may not be directly comparable across different evaluation setups.
- Compute values are estimates based on public disclosures or third-party analyses.
- Dialogue benchmark data includes fine-tuned and zero-shot settings across multiple benchmark versions.
- Multichallenge scores aim to summarize general reasoning and task-solving ability across diverse test sets.
