import os
import glob
import pickle
import matplotlib.pyplot as plt

def plot_learning_results(directory):
    # Find all the pickle files in the directory
    file_paths = glob.glob(os.path.join(directory, '*.pkl'))

    # Loop over the pickle files and plot their results
    for file_path in file_paths:
        # Open the pickle file and load the results
        with open(file_path, 'rb') as f:
            results = pickle.load(f)

        # Extract the lists of values you want to plot
        epochs = list(range(1, len(results['train_loss']) + 1))
        train_loss = results['train_loss']
        val_loss = results['val_loss']
        train_acc = results['train_acc']
        val_acc = results['val_acc']

        # Create a new figure and plot the results
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, label='Train')
        plt.plot(epochs, val_loss, label='Validation')
        plt.title(f'Training and Validation Loss ({os.path.basename(file_path)})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_acc, label='Train')
        plt.plot(epochs, val_acc, label='Validation')
        plt.title(f'Training and Validation Accuracy ({os.path.basename(file_path)})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    # Display all the plots
    plt.show()

# We can use this
# plot_learning_results('path/to/directory')
