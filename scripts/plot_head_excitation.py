import argparse
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def plot_entropy_by_category(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = df['category'].unique()
    means = df.groupby('category')['avg_entropy'].mean().reindex(categories)
    stds = df.groupby('category')['avg_entropy'].std().reindex(categories)

    ax.bar(categories, means, yerr=stds, capsize=5, color='skyblue')
    ax.set_title('Average Entropy by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Average Entropy')
    ax.set_xticklabels(categories, rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(output_dir / 'avg_entropy_by_category.png')
    plt.close(fig)


def plot_entropy_histogram(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['avg_entropy'], bins=20, color='mediumpurple', edgecolor='black')
    ax.set_title('Distribution of Average Entropy')
    ax.set_xlabel('Average Entropy')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(output_dir / 'entropy_distribution.png')
    plt.close(fig)


def plot_top_head_excitation(df, output_dir):
    sorted_df = df.sort_values('top_head_excitation', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(sorted_df['prompt'].astype(str), sorted_df['top_head_excitation'], color='teal')
    ax.set_title('Top 20 Prompts by Head Excitation')
    ax.set_xlabel('Top Head Excitation Score')
    ax.set_ylabel('Prompt')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / 'top_head_excitation.png')
    plt.close(fig)


def plot_layer_scatter(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df['top_head_layer'],
        df['top_head_excitation'],
        c=pd.factorize(df['category'])[0],
        cmap='tab10',
        alpha=0.75
    )
    ax.set_title('Top Head Layer vs Excitation Score')
    ax.set_xlabel('Top Head Layer')
    ax.set_ylabel('Top Head Excitation')
    fig.tight_layout()
    fig.savefig(output_dir / 'top_head_layer_vs_excitation.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot head excitation analysis results')
    parser.add_argument('--input', default='outputs/head_excitation_results.csv', help='Path to head excitation CSV')
    parser.add_argument('--output-dir', default='outputs/plots', help='Directory to save plot images')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if plt is None:
        raise RuntimeError(
            'matplotlib is required to create plots. Install it with: pip install matplotlib'
        )

    df = pd.read_csv(input_path)

    plot_entropy_by_category(df, output_dir)
    plot_entropy_histogram(df, output_dir)
    plot_top_head_excitation(df, output_dir)
    plot_layer_scatter(df, output_dir)

    print(f'Plots saved to: {output_dir.resolve()}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
