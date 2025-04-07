import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Setup directories ---
os.makedirs('figures', exist_ok=True)

# --- Data Loading ---
def load_data(filepath='data/updated_dialogue_benchmarks.csv'):
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    else:
        print(f"CSV file {filepath} not found. Please run create_data_table.py first.")
        return None

# --- Plotting Setup ---
plt.style.use('ggplot')
sns.set_context("talk")

# --- Plotting Function for SGD JGA ---
def plot_sgd_jga(df, show_plot=True):
    """Generates the SGD JGA plot showing adaptability progress."""
    df_sgd = df[(df['Benchmark'] == 'SGD') & (df['Metric'] == 'JGA')].copy()
    if df_sgd.empty:
        print("No data found for SGD JGA plot.")
        return None

    # Create a figure
    plt.figure(figsize=(12, 8), num=1)
    ax = plt.gca()

    # Create mappings
    setting_colors = {
        'Fine-tuned': '#1f77b4',  # Blue
        'Fine-tuned (NL Gen)': '#1f77b4',  # Blue (same as Fine-tuned)
        'Zero-shot (SRP)': '#2ca02c',  # Green
        'Zero-shot (PAR)': '#2ca02c',  # Green (same as SRP)
        'Zero-shot (FuncCall)': '#ff7f0e',  # Orange
        'Zero-shot (AutoTOD)': '#d62728',  # Red
        'Few-Shot (5%)+Corr': '#9467bd',  # Purple
        'Few-Shot (5ex FuncCall)': '#9467bd',  # Purple (same as 5%+Corr)
        'Few-Shot (5% data)': '#9467bd',  # Purple (same as 5%+Corr)
        'Few-Shot (1% data)': '#8c564b',  # Brown
    }
    subset_markers = {
        'Unseen': 'o',  # Circle
        'Overall': 's'  # Square
    }

    # Plot points
    for setting in df_sgd['Setting'].unique():
        for subset in df_sgd['Subset'].unique():
            mask = (df_sgd['Setting'] == setting) & (df_sgd['Subset'] == subset)
            if mask.any():  # Only plot if there are points with this combination
                subset_df = df_sgd[mask]
                plt.scatter(
                    subset_df['Year'], 
                    subset_df['Score(%)'], 
                    s=150,  # Marker size
                    c=[setting_colors.get(setting, '#808080')] * len(subset_df),  # Default gray
                    marker=subset_markers.get(subset, 'x'),  # Default x
                    alpha=0.8, 
                    edgecolors='black'
                )

    # Add labels
    for i, row in df_sgd.iterrows():
        x_offset = 0.05
        y_offset = 1.5 if row['Model'] != 'CORRECTIONLM (Llama 3 8B)' else -4
        plt.text(
            row['Year'] + x_offset, 
            row['Score(%)'] + y_offset, 
            row['Model'],
            fontsize=10, 
            verticalalignment='center'
        )

    # Add annotations for notable points
    if 'GPT-4-Turbo' in df_sgd['Model'].values:
        gpt4_idx = df_sgd[df_sgd['Model'] == 'GPT-4-Turbo'].index[0]
        plt.annotate(
            f"Highest score: GPT-4-Turbo ({df_sgd.loc[gpt4_idx, 'Score(%)']}%)\nZero-shot setting",
            xy=(df_sgd.loc[gpt4_idx, 'Year'], df_sgd.loc[gpt4_idx, 'Score(%)']),
            xytext=(2023.4, 95), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1),
            fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8)
        )

    if 'CORRECTIONLM (Llama 3 8B)' in df_sgd['Model'].values:
        correction_idx = df_sgd[df_sgd['Model'] == 'CORRECTIONLM (Llama 3 8B)'].index[0]
        plt.annotate(
            "Low score may be due to\nlow-resource setting (5% data)",
            xy=(df_sgd.loc[correction_idx, 'Year'], df_sgd.loc[correction_idx, 'Score(%)']),
            xytext=(2024.3, 30), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1),
            fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8)
        )

    plt.title('AI Progress on Schema-Guided Dialogue (SGD) JGA Over Time', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('SGD Joint Goal Accuracy (%)', fontsize=14)
    plt.xlim(2021, 2025.2)
    plt.ylim(30, 100)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create custom legends
    from matplotlib.lines import Line2D
    setting_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=setting)
        for setting, color in setting_colors.items() if setting in df_sgd['Setting'].unique()
    ]
    subset_handles = [
        Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', markeredgecolor='black', markersize=10, label=subset)
        for subset, marker in subset_markers.items() if subset in df_sgd['Subset'].unique()
    ]

    leg1 = ax.legend(handles=setting_handles, title="Training Setting", loc='upper left', fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=subset_handles, title="Data Subset", loc='center left', fontsize=10)

    # Add trendline for 'Unseen' fine-tuned data if possible
    unseen_ft_df = df_sgd[(df_sgd['Subset'] == 'Unseen') & (df_sgd['Setting'].str.contains('Fine-tuned'))]
    if len(unseen_ft_df) >= 2:
        z = np.polyfit(unseen_ft_df['Year'], unseen_ft_df['Score(%)'], 1)
        p = np.poly1d(z)
        years_range = np.linspace(unseen_ft_df['Year'].min(), unseen_ft_df['Year'].max(), 100)
        plt.plot(years_range, p(years_range), linestyle='--', color='gray', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/sgd_jga_progress.png', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return plt  # Return plt for display


# --- Plotting Function for MultiWOZ JGA ---
def plot_multiwoz_jga(df, show_plot=True):
    """Generates the MultiWOZ JGA plot showing complex tracking accuracy progress."""
    df_mw_jga = df[(df['Benchmark'] == 'MultiWOZ') & (df['Metric'] == 'JGA')].copy()
    if df_mw_jga.empty:
        print("No data found for MultiWOZ JGA plot.")
        return None

    # Create a figure
    plt.figure(figsize=(14, 9), num=2)

    # Create mappings
    setting_styles = {
        'Fine-tuned': {'color': '#1f77b4', 'marker': 'o'},  # Blue circles
        'Fine-tuned (NL Gen)': {'color': '#1f77b4', 'marker': 'o'},  # Blue circles (same as Fine-tuned)
        'Zero-shot (SRP)': {'color': '#2ca02c', 'marker': 's'},  # Green squares
        'Zero-shot (PAR)': {'color': '#2ca02c', 'marker': 'D'},  # Green diamonds
        'Zero-shot (FuncCall)': {'color': '#ff7f0e', 'marker': 'p'},  # Orange pentagons
        'Zero-shot (AutoTOD)': {'color': '#d62728', 'marker': '^'},  # Red triangles
        'Few-Shot (5%)+Corr': {'color': '#9467bd', 'marker': '*'},  # Purple stars
        'Few-Shot (5ex FuncCall)': {'color': '#9467bd', 'marker': 'P'},  # Purple plus
        'Few-Shot (5% data)': {'color': '#9467bd', 'marker': 'h'},  # Purple hexagons
        'Few-Shot (1% data)': {'color': '#8c564b', 'marker': 'H'},  # Brown hexagons
    }

    # Plot points
    for setting, style in setting_styles.items():
        subset_df = df_mw_jga[df_mw_jga['Setting'] == setting]
        if not subset_df.empty:
            plt.scatter(
                subset_df['Year'], 
                subset_df['Score(%)'], 
                s=150,
                c=style['color'], 
                marker=style['marker'], 
                label=setting,
                alpha=0.8, 
                edgecolors='black'
            )

    # Add labels
    for i, row in df_mw_jga.iterrows():
        x_offset, y_offset = 0.05, 1
        
        # Adjust position for specific models or if close to others
        if row['Model'] == 'GPT-4-Turbo': y_offset = 2
        elif row['Model'] == 'CoALM 70B': x_offset, y_offset = -0.35, -3
        elif row['Model'] == 'CoALM 405B': x_offset, y_offset = 0.05, -3
        elif row['Model'] == 'GPT-4o': x_offset, y_offset = 0.05, -3
        elif row['Model'] == 'SUMBT': y_offset = -2
        elif row['Model'] == 'FnCTOD (GPT-3.5-Turbo)': y_offset = 2
        elif row['Model'] == 'FnCTOD (GPT-4)': y_offset = -3
        elif row['Model'] == 'Llama3-8B-Instruct': x_offset = -0.3; y_offset = 2
        elif row['Model'] == 'Llama3-70B-Instruct': x_offset = -0.3; y_offset = -2
        elif row['Model'] == 'S3-DST (GPT-4)' and row['Benchmark_Version'] == '2.1': y_offset = -2
        elif row['Model'] == 'IDIC-DST (CodeLlama 7B)': x_offset = -0.2; y_offset = 2

        # Use Benchmark_Version if available, otherwise use a default
        version = row.get('Benchmark_Version', '')
        label_text = f"{row['Model']} ({version})" if version else row['Model']
        
        plt.text(
            row['Year'] + x_offset, 
            row['Score(%)'] + y_offset, 
            label_text,
            fontsize=10, 
            verticalalignment='center'
        )

    # Add special annotations
    plt.annotate(
        "High variance in 2024 zero-shot results\ndepending on evaluation approach\n(SRP vs AutoTOD vs FuncCall vs PAR)",
        xy=(2024.4, 43), 
        xytext=(2021.8, 40),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8)
    )

    if 'GPT-4-Turbo' in df_mw_jga['Model'].values:
        gpt4t_idx = df_mw_jga[df_mw_jga['Model'] == 'GPT-4-Turbo'].index[0]
        plt.annotate(
            "GPT-4-Turbo achieves highest\nperformance with SRP prompting",
            xy=(df_mw_jga.loc[gpt4t_idx, 'Year'], df_mw_jga.loc[gpt4t_idx, 'Score(%)']),
            xytext=(2021.8, 75),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8)
        )

    plt.title('AI Progress on MultiWOZ JGA Over Time (Overall Subset)', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('MultiWOZ Joint Goal Accuracy (%) (Overall Subset)', fontsize=14)

    # Add benchmark version transitions
    benchmark_transitions = {'2.0': 2019.0, '2.1': 2020.0, '2.2': 2023.0, '2.4': 2023.6, '2.1+': 2021.5}
    
    # Sort the transitions by year
    sorted_transitions = sorted(benchmark_transitions.items(), key=lambda x: x[1])
    
    # Plot vertical lines and text labels
    last_year = 2019.0
    for version, start_year in sorted_transitions:
        if version != '2.1+':  # Skip the special 2.1+ version for vertical lines
            plt.axvline(x=start_year, color='grey', linestyle=':', alpha=0.5)
            plt.text(
                (last_year + start_year)/2 if last_year != 2019 else start_year - 0.25, 
                88, 
                f"MWOZ {version}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                style='italic', 
                color='grey'
            )
            last_year = start_year
    
    # Label last section
    plt.text(
        (last_year + 2025.2)/2, 
        88, 
        f"MWOZ {sorted_transitions[-1][0]}", 
        ha='center', 
        va='bottom', 
        fontsize=9, 
        style='italic', 
        color='grey'
    )

    plt.xlim(2019, 2025.2)
    plt.ylim(30, 90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Evaluation Setting", loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=12)

    # Add trendline for fine-tuned models
    fine_tuned_df = df_mw_jga[df_mw_jga['Setting'].str.contains('Fine-tuned')]
    if len(fine_tuned_df) >= 2:
        z = np.polyfit(fine_tuned_df['Year'], fine_tuned_df['Score(%)'], 1)
        p = np.poly1d(z)
        years_range = np.linspace(fine_tuned_df['Year'].min(), fine_tuned_df['Year'].max(), 100)
        plt.plot(years_range, p(years_range), linestyle='--', color='#1f77b4', alpha=0.7, label='_nolegend_')
    
    # Add an annotation for NL-DST as the most recent fine-tuned model
    if 'NL-DST (LLM)' in df_mw_jga['Model'].values:
        nl_dst_idx = df_mw_jga[df_mw_jga['Model'] == 'NL-DST (LLM)'].index[0]
        plt.annotate(
            "Highest fine-tuned score in 2025\nusing NL generation approach",
            xy=(df_mw_jga.loc[nl_dst_idx, 'Year'], df_mw_jga.loc[nl_dst_idx, 'Score(%)']),
            xytext=(2023.5, 70),
            arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1, headwidth=8),
            fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.8)
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for legend
    plt.savefig('figures/multiwoz_jga_progress.png', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
        
    return plt  # Return plt for display


# --- Plotting Function for MultiWOZ SR ---
def plot_multiwoz_sr(df, show_plot=True):
    """Generates the MultiWOZ SR plot showing task completion rate progress."""
    df_mw_sr = df[df['Benchmark'] == 'MultiWOZ'].copy()
    # Filter only rows where SR score is available (not JGA)
    df_mw_sr = df_mw_sr[df_mw_sr['Metric'].str.contains("SR", na=False)]

    if df_mw_sr.empty:
        print("No data found for MultiWOZ SR plot.")
        return None

    # Create a figure
    plt.figure(figsize=(14, 9), num=3)

    # Create mappings
    eval_styles = {
        'Fine-tuned (Automated)': {'color': '#1f77b4', 'marker': 'o'},  # Blue circles
        'Few-Shot (Human Eval)': {'color': '#ff7f0e', 'marker': '*'},  # Orange stars
        'Zero-shot (Automated)': {'color': '#2ca02c', 'marker': 's'}  # Green squares
    }

    # Plot points
    for eval_method, style in eval_styles.items():
        subset_df = df_mw_sr[df_mw_sr['Eval_Method'] == eval_method]
        if not subset_df.empty:
            plt.scatter(
                subset_df['Year'], 
                subset_df['Score(%)'],
                s=200 if eval_method == 'Few-Shot (Human Eval)' else 150,
                c=style['color'], 
                marker=style['marker'], 
                label=eval_method,
                alpha=0.8, 
                edgecolors='black',
                linewidth=1.5 if eval_method == 'Few-Shot (Human Eval)' else 1
            )

    # Add labels
    for i, row in df_mw_sr.iterrows():
        x_offset, y_offset = 0.05, 2
        if row['Model'] == 'CoALM 405B': y_offset = -3
        elif row['Model'] == 'GPT-3.5 (`text-davinci-003`)': y_offset = -3; x_offset = -0.4

        benchmark_text = f" ({row['Benchmark_Version']})" if pd.notna(row['Benchmark_Version']) and row['Benchmark_Version'] != 'Unknown' else ""
        label_text = f"{row['Model']}{benchmark_text}"
        plt.text(
            row['Year'] + x_offset, 
            row['Score(%)'] + y_offset, 
            label_text,
            fontsize=10, 
            verticalalignment='center',
            fontweight='bold' if 'Human Eval' in row['Eval_Method'] else 'normal'
        )

    # Add special annotations
    plt.annotate(
        "Note: This plot combines different evaluation methodologies:\n"
        "• Automated metrics from fine-tuned models\n"
        "• Human evaluations (orange stars)\n"
        "• Automated zero-shot evaluations",
        xy=(2020.5, 55), 
        xytext=(2020.5, 55),  # Adjusted position slightly
        fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    )

    # Annotation for human evaluation points
    human_eval_points = df_mw_sr[df_mw_sr['Eval_Method'] == 'Few-Shot (Human Eval)']
    if not human_eval_points.empty and 'GPT-4' in human_eval_points['Model'].values:
        gpt4_idx = human_eval_points[human_eval_points['Model'] == 'GPT-4'].index[0]
        plt.annotate(
            "Human evaluation results\n(different methodology)",
            xy=(human_eval_points.loc[gpt4_idx, 'Year'], human_eval_points.loc[gpt4_idx, 'Score(%)']),
            xytext=(2021, 83),
            arrowprops=dict(facecolor='#ff7f0e', shrink=0.05, width=1, headwidth=8),
            fontsize=11, 
            color='#ff7f0e', 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e", alpha=0.8)
        )

    plt.title('AI Progress on MultiWOZ Success Rate Over Time (Mixed Eval Methods)', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('MultiWOZ Success Rate (%) (Overall Subset)', fontsize=14)
    plt.xlim(2020, 2025.2)
    plt.ylim(50, 90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Evaluation Method", loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

    # Add trendlines for each category
    for eval_method, style in eval_styles.items():
        method_df = df_mw_sr[df_mw_sr['Eval_Method'] == eval_method]
        if len(method_df) >= 2:
            z = np.polyfit(method_df['Year'], method_df['Score(%)'], 1)
            p = np.poly1d(z)
            years_range = np.linspace(method_df['Year'].min(), method_df['Year'].max(), 100)
            plt.plot(years_range, p(years_range), linestyle='--', color=style['color'], alpha=0.7, label='_nolegend_')

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Adjust layout
    plt.savefig('figures/multiwoz_sr_progress.png', dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
        
    return plt  # Return plt for display


# --- Main execution ---
def main():
    # Load data from CSV
    df = load_data()
    
    if df is not None:
        # Print some basic statistics
        print(f"Loaded {len(df)} data points")
        print(f"Unique benchmarks: {df['Benchmark'].unique()}")
        print(f"Unique metrics: {df['Metric'].unique()}")
        print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")
        
        # Create all plots
        print("Generating SGD JGA plot...")
        plot_sgd_jga(df, show_plot=False)  # Set to True if you want plots to display
        
        print("Generating MultiWOZ JGA plot...")
        plot_multiwoz_jga(df, show_plot=False)
        
        print("Generating MultiWOZ SR plot...")
        plot_multiwoz_sr(df, show_plot=False)
        
        print("All plots have been generated successfully and saved to the 'figures' directory!")
    else:
        print("Failed to load data. Please run create_data_table.py first.")


if __name__ == "__main__":
    main()
