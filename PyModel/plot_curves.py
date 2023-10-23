from pickle import load
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pylab import plt
from numpy import arange
from Transformer.params import baseline_test_params, midi_test_params_v1, Params
 
p = Params(midi_test_params_v1)
# Load the training and validation loss dictionaries
train_loss = load(open('./train_loss.pkl', 'rb'))

val_loss = load(open('./val_loss.pkl', 'rb'))
 
# Retrieve each dictionary's values
train_values = train_loss.values()
val_values = val_loss.values()

 
# Generate a sequence of integers to represent the epoch numbers
epochs = range(p.epochs)
 
# Plot and label the training and validation loss values
plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')
 
# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(0, p.epochs, 2))
 
# Display the plot
plt.legend(loc='best')
plt.show()