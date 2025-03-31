import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm


def set_plot_style():
    """set consistent plot style for all visualizations"""
    # Use a more modern style that works with newer seaborn versions
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def plot_sequence_length_distribution(user_sequences, save_path=None):
    """
    plot distribution of sequence lengths

    args:
        user_sequences: dictionary mapping user_id to item sequence
        save_path: path to save the figure
    """
    set_plot_style()

    # Get sequence lengths
    seq_lengths = [len(seq) for seq in user_sequences.values()]

    # Plot histogram
    plt.figure()
    sns.histplot(seq_lengths, kde=True, bins=30)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Distribution of User Sequence Lengths")

    if save_path:
        plt.savefig(save_path)
        print(f"saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_item_popularity(user_sequences, top_n=20, save_path=None):
    """
    plot item popularity distribution

    args:
        user_sequences: dictionary mapping user_id to item sequence
        top_n: number of most popular items to show
        save_path: path to save the figure
    """
    set_plot_style()

    # Count item occurrences
    all_items = [item for seq in user_sequences.values() for item in seq]
    item_counts = Counter(all_items)

    # Get top items
    top_items = item_counts.most_common(top_n)
    items, counts = zip(*top_items)

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(items)), counts)
    plt.xticks(range(len(items)), items, rotation=45)
    plt.xlabel("Item ID")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Most Popular Items")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_genre_distribution(
    item_features, user_sequences=None, top_n=15, save_path=None
):
    """
    plot genre distribution

    args:
        item_features: dictionary with item features
        user_sequences: dictionary mapping user_id to item sequence
        top_n: number of top genres to show
        save_path: path to save the figure
    """
    set_plot_style()

    # Count genres
    if user_sequences:
        # Weight by actual usage in sequences
        genre_counts = Counter()
        for user_id, sequence in user_sequences.items():
            for item in sequence:
                if item in item_features:
                    genre_counts.update(item_features[item]["genres"])
    else:
        # Just count all genres in the dataset
        genre_counts = Counter()
        for movie_id, features in item_features.items():
            genre_counts.update(features["genres"])

    # Get top genres
    top_genres = genre_counts.most_common(top_n)
    genres, counts = zip(*top_genres)

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(genres)), counts)
    plt.xticks(range(len(genres)), genres, rotation=45)
    plt.xlabel("Genre")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Most Common Genres")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_year_distribution(item_features, user_sequences=None, save_path=None):
    """
    plot movie year distribution

    args:
        item_features: dictionary with item features
        user_sequences: dictionary mapping user_id to item sequence
        save_path: path to save the figure
    """
    set_plot_style()

    # Get years
    if user_sequences:
        # Get years weighted by occurrence in sequences
        years = []
        for user_id, sequence in user_sequences.items():
            for item in sequence:
                if item in item_features and item_features[item]["year"]:
                    years.append(item_features[item]["year"])
    else:
        # Get all years in dataset
        years = [
            features["year"]
            for features in item_features.values()
            if features["year"] is not None
        ]

    # Plot histogram
    plt.figure()
    sns.histplot(years, kde=True, bins=30)
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title("Distribution of Movie Years")

    if save_path:
        plt.savefig(save_path)
        print(f"saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_user_item_matrix(user_sequences, max_users=50, max_items=50, save_path=None):
    """
    visualize user-item interaction matrix

    args:
        user_sequences: dictionary mapping user_id to item sequence
        max_users: maximum number of users to display
        max_items: maximum number of items to display
        save_path: path to save the figure
    """
    set_plot_style()

    # Get top users by sequence length
    top_users = sorted(user_sequences.items(), key=lambda x: len(x[1]), reverse=True)[
        :max_users
    ]
    user_ids = [user_id for user_id, _ in top_users]

    # Get most popular items
    all_items = [item for seq in user_sequences.values() for item in seq]
    item_counts = Counter(all_items)
    top_items = [item for item, _ in item_counts.most_common(max_items)]

    # Create interaction matrix
    matrix = np.zeros((len(user_ids), len(top_items)))
    for i, (user_id, sequence) in enumerate(top_users):
        for j, item_id in enumerate(top_items):
            if item_id in sequence:
                matrix[i, j] = 1

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="viridis", cbar_kws={"label": "Interaction"})
    plt.xlabel("Item Index (by popularity)")
    plt.ylabel("User Index (by sequence length)")
    plt.title("User-Item Interaction Matrix")

    if save_path:
        plt.savefig(save_path)
        print(f"saved figure to {save_path}")
    else:
        plt.show()

    plt.close()
