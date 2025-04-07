# Dialogue Benchmark Data

This directory contains the compiled benchmark data used for tracking AI progress on task-oriented dialogue tasks.

## File Structure

- `updated_dialogue_benchmarks.csv` - The main dataset containing all benchmark results

## Dataset Description

The main dataset (`updated_dialogue_benchmarks.csv`) contains benchmark results from various models evaluated on the MultiWOZ and Schema-Guided Dialogue (SGD) benchmarks from 2019 to early 2025. 

### Data Fields

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

## Data Sources

The data is compiled from published academic papers and research reports evaluating various models on the MultiWOZ and SGD benchmarks. The full citations are available in the original publication.

## Notes

- Some data points are marked as estimates where exact values were not available.
- Different evaluation methods may not be directly comparable, particularly when comparing across different benchmark versions or between human and automated evaluations.
- "Unseen" for SGD refers to the model's performance on domains not seen during training, testing adaptability.
