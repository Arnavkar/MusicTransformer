import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the CSV file
file_path = './data/raw/maestro-v3.0.0/maestro-v3.0.0.csv'
maestro_data = pd.read_csv(file_path)

splits = maestro_data['split'].unique()
custom_params = {"axes.spines.right": False}
sns.set_theme(style="ticks", rc=custom_params)
for split in splits:
    split_data = maestro_data[maestro_data['split'] == split]

    # Create a new figure
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Histogram of the number of compositions per composer
    composer_counts = split_data['canonical_composer'].value_counts()
    composer_counts.sort_values(ascending=True, inplace=True)

    # Histogram of the total duration of music per composer
    duration_per_composer = split_data.groupby('canonical_composer')['duration'].sum()
    duration_per_composer.sort_values(ascending=True, inplace=True)

    # Bar positions
    y_positions = np.arange(len(composer_counts))
    bar_width = 0.4

    # Ensure the same composers are on both histograms
    assert set(composer_counts.index) == set(duration_per_composer.index)
    all_composers = composer_counts.index

    # Set y-ticks to composer names
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(all_composers, fontsize=8)

    # Plotting - first plot
    bar1 = ax1.barh(y_positions - bar_width/2, composer_counts.values, height=bar_width, color='blue', label='Number of Compositions')
    ax1.set_xlabel('Number of Compositions')
    ax1.set_title(f'Number of Compositions / Total Duration of Compositions per Composer - {split}')
    ax1.tick_params(axis='y', labelsize=8)



    # Plotting - second plot
    ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
    bar2 = ax2.barh(y_positions + bar_width/2, duration_per_composer.values,height=bar_width, color='red', label='Total Duration (seconds)')
    ax2.set_xlabel('Total Duration (seconds)')

    #adjust y-lims
    ax1.set_xlim([0, max(composer_counts.values) + 10])
    ax2.set_xlim([0, max(duration_per_composer.values) + 1000])
    ax1.set_ylim([-1,len(composer_counts)])

    # Adjust layout and show plot
    plt.tight_layout()
    ax1.legend([bar1, bar2], ['Number of Compositions', 'Total Duration (seconds)'], loc='lower right')
    plt.savefig(f'./Analysis/graphs/data_analysis_{split}.jpg', format='jpg', dpi = 600)