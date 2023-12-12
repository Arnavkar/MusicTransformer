from pickle import load
import matplotlib
from matplotlib.pylab import plt
import os
import numpy as np
import pandas as pd
import seaborn as sns

# Load the CSV file
file_path = './data/raw/maestro-v3.0.0/maestro-v3.0.0.csv'
maestro_data = pd.read_csv(file_path)

splits = maestro_data['split'].unique()
for split in splits:

    split_data = maestro_data[maestro_data['split'] == split]

    # Create a new figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Histogram of the number of compositions per composer
    composer_counts = split_data['canonical_composer'].value_counts()
    #sort values
    composer_counts.sort_values(ascending=True,inplace=True)

    # Histogram of the total duration of music per composer
    duration_per_composer = split_data.groupby('canonical_composer')['duration'].sum()
    #sort values
    duration_per_composer.sort_values(ascending=True,inplace=True)

    axes[0].barh(composer_counts.index, composer_counts.values)
    axes[1].barh(duration_per_composer.index, duration_per_composer.values)

    #train has 58 composer, test and validatio only have 16
    if split == 'train':
        axes[0].set_title('Number of Compositions per Composer - ' + split)
        axes[1].set_title('Total Duration of Music by Each Composer  - ' + split)
        #make labels smaller
        axes[0].tick_params(axis='y', labelsize=4)
        axes[1].tick_params(axis='y', labelsize=4)

        #increase the spacing between each histogram bar
        axes[0].set_ylim([0, 60])
        axes[1].set_ylim([0, 60])
    else:
        axes[0].set_title('Number of Compositions per Composer - ' + split)
        axes[1].set_title('Total Duration of Music by Each Composer - ' + split)

    axes[0].set_xlabel('Number of Compositions')
    axes[1].set_xlabel('Total Duration (seconds)')

    # Adjust layout
    plt.tight_layout()

    # Display the plot

plt.show()