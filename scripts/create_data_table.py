import pandas as pd
import csv

# Create a list of dictionaries from the updated data table
updated_data = [
    # Original entries (with updates/additions from new data)
    {'Model': 'SUMBT', 'Year': 2019.5, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.0', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 42.4, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'SimpleTOD', 'Year': 2020.4, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 55.8, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-2 based'},
    {'Model': 'SimpleTOD', 'Year': 2020.4, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 70.5, 'Setting': 'Fine-tuned (E2E)', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-2 based'},
    {'Model': 'TripPy', 'Year': 2020.5, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 55.3, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'T5 (Fine-tuned)', 'Year': 2021.5, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1+', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 75.0, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper (Estimate)', 'Base_Model_Note': 'T5'},
    {'Model': 'T5 (Fine-tuned)', 'Year': 2021.5, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Unseen', 'Metric': 'JGA', 'Score(%)': 70.0, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper (Estimate)', 'Base_Model_Note': 'T5'},
    {'Model': 'Flan-T5 (Fine-tuned)', 'Year': 2022.8, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Unseen', 'Metric': 'JGA', 'Score(%)': 75.0, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper (Estimate)', 'Base_Model_Note': 'Flan-T5'},
    {'Model': 'GPT-3.5 (`text-davinci-003`)', 'Year': 2022.9, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': 'Unknown', 'Subset': 'Overall', 'Metric': 'SR (Human)', 'Score(%)': 57.1, 'Setting': 'Few-Shot', 'Eval_Method': 'Few-Shot (Human Eval)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-3 series'},
    {'Model': 'GPT-3.5-Turbo', 'Year': 2022.9, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 58.6, 'Setting': 'Zero-shot (FuncCall)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-3 series'},
    {'Model': 'GPT-4', 'Year': 2023.3, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': 'Unknown', 'Subset': 'Overall', 'Metric': 'SR (Human)', 'Score(%)': 76.0, 'Setting': 'Few-Shot', 'Eval_Method': 'Few-Shot (Human Eval)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4'},
    {'Model': 'SPLAT (Large)', 'Year': 2023.5, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Unseen', 'Metric': 'JGA', 'Score(%)': 82.2, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Longformer-large'},
    {'Model': 'SPLAT (Large)', 'Year': 2023.5, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 57.4, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Longformer-large'},
    {'Model': 'LLaMA 1/2 (Fine-tuned)', 'Year': 2023.5, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Unseen', 'Metric': 'JGA', 'Score(%)': 75.0, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper (Estimate)', 'Base_Model_Note': 'Llama 1/2'},
    {'Model': 'GPT-4-Turbo', 'Year': 2023.8, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 88.7, 'Setting': 'Zero-shot (SRP)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},
    {'Model': 'GPT-4-Turbo', 'Year': 2023.8, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 79.6, 'Setting': 'Zero-shot (SRP)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},

    # New entries from the updated data
    {'Model': 'FnCTOD (GPT-3.5-Turbo)', 'Year': 2024.1, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 38.6, 'Setting': 'Zero-shot (FuncCall)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-3.5 series'},
    {'Model': 'FnCTOD (GPT-4)', 'Year': 2024.1, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 38.7, 'Setting': 'Zero-shot (FuncCall)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},
    {'Model': 'Llama3-8B-Instruct', 'Year': 2024.3, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 55.9, 'Setting': 'Few-Shot (5ex FuncCall)', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 3 8B'},
    {'Model': 'Llama3-70B-Instruct', 'Year': 2024.3, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 72.4, 'Setting': 'Few-Shot (5ex FuncCall)', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 3 70B'},
    {'Model': 'Llama 3 (70B Instruct)', 'Year': 2024.3, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 76.3, 'Setting': 'Zero-shot (SRP)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 3 70B'},
    {'Model': 'Llama 3 (70B Instruct)', 'Year': 2024.3, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 66.2, 'Setting': 'Zero-shot (SRP)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 3 70B'},
    {'Model': 'CoALM 70B', 'Year': 2024.4, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 43.8, 'Setting': 'Zero-shot (AutoTOD)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 2 70B based'},
    {'Model': 'CoALM 70B', 'Year': 2024.4, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 69.4, 'Setting': 'Zero-shot (AutoTOD)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 2 70B based'},
    {'Model': 'CoALM 405B', 'Year': 2024.4, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 38.8, 'Setting': 'Zero-shot (AutoTOD)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 2 based (?)'},
    {'Model': 'CoALM 405B', 'Year': 2024.4, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 66.7, 'Setting': 'Zero-shot (AutoTOD)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 2 based (?)'},
    {'Model': 'GPT-4o', 'Year': 2024.5, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 36.9, 'Setting': 'Zero-shot (AutoTOD)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},
    {'Model': 'GPT-4o', 'Year': 2024.5, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 75.5, 'Setting': 'Zero-shot (AutoTOD)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},
    {'Model': 'S3-DST (GPT-4)', 'Year': 2024.8, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 45.1, 'Setting': 'Zero-shot (PAR)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},
    {'Model': 'S3-DST (GPT-4)', 'Year': 2024.8, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 53.3, 'Setting': 'Zero-shot (PAR)', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4 series'},
    {'Model': 'CORRECTIONLM (Llama 3 8B)', 'Year': 2024.8, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 37.8, 'Setting': 'Few-Shot (5%)+Corr', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 3 8B'},
    {'Model': 'CORRECTIONLM (Llama 3 8B)', 'Year': 2024.8, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 56.2, 'Setting': 'Few-Shot (5%)+Corr', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Llama 3 8B'},
    {'Model': 'IDIC-DST (CodeLlama 7B)', 'Year': 2024.9, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 44.3, 'Setting': 'Few-Shot (5% data)', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'CodeLlama 7B'},
    {'Model': 'IDIC-DST (CodeLlama 7B)', 'Year': 2024.9, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 52.7, 'Setting': 'Few-Shot (1% data)', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'CodeLlama 7B'},
    {'Model': 'NL-DST (LLM)', 'Year': 2025.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 65.9, 'Setting': 'Fine-tuned (NL Gen)', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Unknown LLM'},

    # Additional entries from MultiWOZ.pdf
    {'Model': 'MTTOD', 'Year': 2021.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 20.25, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Seq2Seq'},
    {'Model': 'GALAXY', 'Year': 2021.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 20.5, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'HDNO', 'Year': 2020.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.0', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 18.85, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'LAVA', 'Year': 2020.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 94.8, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'DORA', 'Year': 2021.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 91.1, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-2 based'},
    {'Model': 'TOATOD', 'Year': 2023.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 63.79, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BART-based'},
    {'Model': 'DAIR', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 59.98, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'D3ST', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 58.7, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},
    {'Model': 'SPACE-3', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.2', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 57.5, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'BERT-based'},

    # Additional entries from SGD data.pdf
    {'Model': 'InstructDial', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 52.0, 'Setting': 'Few-Shot', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Instruction Tuning'},
    {'Model': 'IC-DST (Codex)', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Domain-Specific', 'Metric': 'JGA', 'Score(%)': 56.45, 'Setting': 'Few-Shot ICL', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Codex'},
    {'Model': 'SG-USM', 'Year': 2023.0, 'Benchmark': 'SGD', 'Benchmark_Version': 'Original', 'Subset': 'Unseen', 'Metric': 'User Satisfaction', 'Score(%)': 78.0, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Schema-based'},
    {'Model': 'PPTOD', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 57.45, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'T5-based'},
    {'Model': 'PPTOD', 'Year': 2022.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 79.08, 'Setting': 'Fine-tuned', 'Eval_Method': 'Fine-tuned (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'T5-based'},
    {'Model': 'UnifiedTOD', 'Year': 2024.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.4', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 57.0, 'Setting': 'Zero-shot', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Autonomous agent'},
    {'Model': 'Unsupervised-TOD', 'Year': 2024.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'SR', 'Score(%)': 70.0, 'Setting': 'Unsupervised E2E', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Noisy channel'},
    {'Model': 'Mixtral 8x7B', 'Year': 2024.1, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Overall', 'Metric': 'JGA', 'Score(%)': 68.0, 'Setting': 'Zero-shot', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'Mixtral'},
    {'Model': 'InstructTODS (GPT-4)', 'Year': 2023.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Domain-Specific', 'Metric': 'JGA', 'Score(%)': 48.16, 'Setting': 'Zero-shot', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-4'},
    {'Model': 'InstructTODS (GPT-3)', 'Year': 2023.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Domain-Specific', 'Metric': 'JGA', 'Score(%)': 42.02, 'Setting': 'Zero-shot', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-3'},
    {'Model': 'FNCTOD (LLAMA2-13B)', 'Year': 2024.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Domain-Specific', 'Metric': 'JGA', 'Score(%)': 59.54, 'Setting': 'Zero-shot', 'Eval_Method': 'Zero-shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'LLAMA2-13B'},
    {'Model': 'SERI-DST (GPT-3.5)', 'Year': 2024.0, 'Benchmark': 'MultiWOZ', 'Benchmark_Version': '2.1', 'Subset': 'Domain-Specific', 'Metric': 'JGA', 'Score(%)': 60.58, 'Setting': 'Few-shot', 'Eval_Method': 'Few-Shot (Automated)', 'Source_Type': 'Academic Paper', 'Base_Model_Note': 'GPT-3.5'}
]

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(updated_data)

# Save the DataFrame to a CSV file
df.to_csv('updated_dialogue_benchmarks.csv', index=False)

# Print a message indicating the CSV file has been successfully created
print("CSV file 'updated_dialogue_benchmarks.csv' has been created successfully.")

# Display the first few rows of the DataFrame to verify the content
print("\nPreview of the DataFrame:")
print(df.head())

# Count the number of entries for each benchmark and metric combination
print("\nCount of entries by benchmark and metric:")
benchmark_metric_counts = df.groupby(['Benchmark', 'Metric']).size()
print(benchmark_metric_counts)

# Count of entries by evaluation method
print("\nCount of entries by evaluation method:")
eval_method_counts = df.groupby(['Eval_Method']).size()
print(eval_method_counts)

# Display the unique model types
print("\nUnique models in the dataset:")
print(df['Model'].nunique())

# Display the year range
print(f"\nYear range: {df['Year'].min()} to {df['Year'].max()}")

# Basic statistics on scores
print("\nBasic statistics on scores:")
print(df.groupby(['Benchmark', 'Metric'])['Score(%)'].describe())

# Return the DataFrame
df
