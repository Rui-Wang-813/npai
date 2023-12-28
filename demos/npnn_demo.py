# make sure to be able to import parent directory
import sys
sys.path.append('..')
sys.path.append('../npai')

import npai.machine_learning as npml
import npai.ensemble as npen
import npai.deep_learning as npdl
import npai.neural_network as npnn
import npai.optimization as npop
import npai.reinforcement_learning as nprl

from npai.neural_network.dataset import Dataset

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="D:/DATA/mnist/mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="D:/DATA/mnist/mnist_test.npz")
    p.add_argument(
        "--optim", help="training optimizer", default="SGD")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()

def run(args, train_dataset, val_dataset, graph=True, verbose=True):
    
    if args.optim == "adam" or args.optim == "Adam":
        optim = npop.Adam(learning_rate=args.lr)
    if args.optim == "Adagrad":
        optim = npop.Adagrad(learning_rate=args.lr)
    if args.optim == "Adadelta":
        optim = npop.Adadelta(learning_rate=args.lr)
    if args.optim == "AdamW":
        optim = npop.AdamW(learning_rate=args.lr)
    if args.optim == "Adamax":
        optim = npop.Adamax(learning_rate=args.lr)
    if args.optim == "NAdam":
        optim = npop.NAdam(learning_rate=args.lr)
    if args.optim == "ASGD":
        optim = npop.ASGD(learning_rate=args.lr, T=len(train_dataset) * args.epochs)
    else:
        optim = npop.SGD(learning_rate=args.lr)

    model =  npnn.Sequential(
    modules=[
        npnn.Flatten(),
        npnn.Dense(dim_in=784, dim_out=256), 
        npnn.ReLU(),
        npnn.Dense(dim_in=256, dim_out=64), 
        npnn.ReLU(),
        npnn.Dense(dim_in=64, dim_out=10)
    ],
    loss=npnn.SoftmaxCrossEntropy(),  
    optimizer=optim
)
    stats = pd.DataFrame()
    
    lowest_val_loss = float('inf')
    lowest_training_loss = float('inf')
    hightest_val_accuracy = 0
    highest_training_accuracy = 0
    # Train the model
    for epoch in tqdm(range(args.epochs), bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}', position=0):
        train_loss, train_accuracy = model.train(train_dataset)
        val_loss, val_accuracy = model.test(val_dataset)

        lowest_val_loss = min(lowest_val_loss, val_loss)
        lowest_training_loss = min(lowest_training_loss, train_loss)
        hightest_val_accuracy = max(hightest_val_accuracy, val_accuracy)
        highest_training_accuracy = max(highest_training_accuracy, train_accuracy)

        if verbose:
            tqdm.write(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f}\
                | Validation Loss: {val_loss:.3f} | Validation Accuracy: {val_accuracy:.3f}")
        

        # Record the statistics
        stats = pd.concat([stats, pd.DataFrame({
            "epoch": [epoch+1],
            "train_loss": [train_loss],
            "train_accuracy": [train_accuracy],
            "val_loss": [val_loss],
            "val_accuracy": [val_accuracy]
        })], ignore_index=True)
    
    val_loss, val_accuracy = model.test(val_dataset)
    print(f"Final Validation Loss: {val_loss} | Validation Accuracy: {val_accuracy}")

    # Save statistics to file.
    # We recommend that you save your results to a file, then plot them
    # separately, though you can also place your plotting code here.
    if args.save_stats:
        stats.to_csv("data/{}_{}.csv".format(args.opt, args.lr))

    if graph:

        plt.figure(figsize=(12, 5))

        # Plotting training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(stats['epoch'], stats['train_loss'], label='Training Loss')
        plt.plot(stats['epoch'], stats['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(stats['epoch'], stats['train_accuracy'], label='Training Accuracy')
        plt.plot(stats['epoch'], stats['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    # Save predictions.
    if args.save_pred:
        X_test, _ = npnn.load_mnist("mnist_test.npz")
        y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)

    return lowest_val_loss, lowest_training_loss, hightest_val_accuracy, highest_training_accuracy

if __name__ == '__main__':
    args = _get_args()
    X, y = npnn.load_mnist(args.dataset)

    # TODO
    # Create dataset (see npnn/dataset.py)
    # Create model (see npnn/model.py)
    # Train for args.epochs

    # Split the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    X_train, X_val = X[:50000], X[50000:]
    y_train, y_val = y[:50000], y[50000:]

    train_dataset = npnn.Dataset(X_train, y_train, batch_size=32)
    val_dataset = npnn.Dataset(X_val, y_val, batch_size=32)

    #run(args, train_dataset, val_dataset, graph=True, verbose=True)
    print(f"Optimizer: {args.optim} | Learning Rate: {args.lr}\n")

    lowest_val_loss, lowest_training_loss, hightest_val_accuracy, highest_training_accuracy = run(args, train_dataset, val_dataset, graph=True, verbose=True)
    
    print(f"Learning Rate: {args.lr} | Lowest Validation Loss: {lowest_val_loss:.3f} | Lowest Training Loss: {lowest_training_loss:.3f}")

    '''

    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    val_losses = []
    training_losses = []
    val_accuracies = []
    training_accuracies = []

    for lr in learning_rates:
        args.lr = lr
        lowest_val_loss, lowest_training_loss, hightest_val_accuracy, highest_training_accuracy = run(args, train_dataset, val_dataset, graph=False, verbose=True)
        val_losses.append(lowest_val_loss)
        training_losses.append(lowest_training_loss)
        val_accuracies.append(hightest_val_accuracy)
        training_accuracies.append(highest_training_accuracy)
        tqdm.write(f"Learning Rate: {lr} | Lowest Validation Loss: {lowest_val_loss:.3f} | Lowest Training Loss: {lowest_training_loss:.3f}\
                | Highest Validation Accuracy: {hightest_val_accuracy:.3f} | Highest Training Accuracy: {highest_training_accuracy:.3f}\n")
        
    
    # Plotting the losses
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(np.log(learning_rates), val_losses, label='Validation Loss', marker='o')
    ax.plot(np.log(learning_rates), training_losses, label='Training Loss', marker='o')
    ax.set_xticks(np.log(learning_rates))
    ax.set_xticklabels(learning_rates)
    ax.set_xlabel('Log-Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('Losses vs Log-Learning Rate')
    ax.legend()

    # Plotting the accuracies
    fig, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(np.log(learning_rates), val_accuracies, label='Validation Accuracy', marker='o')
    ax2.plot(np.log(learning_rates), training_accuracies, label='Training Accuracy', marker='o')
    ax2.set_xticks(np.log(learning_rates))
    ax2.set_xticklabels(learning_rates)
    ax2.set_xlabel('Log-Learning Rate')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracies vs Log-Learning Rate')
    ax2.legend()

    plt.show()
    '''