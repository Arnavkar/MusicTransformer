from pickle import load
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pylab import plt
from numpy import arange
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-n',"--model_name", type=str,required= True)
args = parser.parse_args()
# Load the training and validation loss dictionaries
train_loss = load(open('./models/' + args.model_name + '/train_loss.pkl', 'rb'))
val_loss = load(open('./models/' + args.model_name + '/val_loss.pkl', 'rb'))
# Retrieve each dictionary's values
train_values = train_loss.values()
val_values = val_loss.values()

# Generate a sequence of integers to represent the epoch numbers
epochs = range(len(train_values))

# Plot and label the training and validation loss values
plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')

# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs (checkpointing every 5 epochs)')
plt.ylabel('Loss')

# Set the tick locations
plt.xticks(arange(0,len(train_values) ))

# Display the plot
plt.legend(loc='best')
plt.show()