
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Used for saving plots
import matplotlib.image as mpimg

def training_cnn(model,trainloader,validloader,num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # Loops through the number of epochs
    for epoch in range(num_epochs):
        # Sets model to train mode
        model.train()

        # Iterates over the training data in batches

        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for i, (images, labels) in enumerate(trainloader):

            # Sets the optimizer gradients to zero to avoid accumulation of gradients
            optimizer.zero_grad()

            # Computes the output of the model
            output = model(images)

            # Computes the loss on current batch
            loss = criterion(output,labels)

            # backpropagate the loss
            loss.backward()

            # Updates the optimizer parameters
            optimizer.step()

            # Updates the train loss and accuracy
            train_loss += loss.item()
            values, indices = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (indices == labels).sum().item()

        # Computes the average training loss and accuracy and store in respective arrays
        train_loss_list.append(train_loss / len(trainloader))
        train_acc_list.append(100 * train_correct / train_total)

        # Sets the model to evaluation mode
        model.eval()

        valid_loss = 0.0
        valid_total = 0
        valid_correct = 0
        for i, (images, labels) in enumerate(validloader):
            # Computes the output of the model
            output = model(images)

            # Computes the loss
            loss = criterion(output, labels)

            # Computes the validation loss and accuracy
            valid_loss += loss.item()
            values, indices = torch.max(output.data, 1)
            valid_total += labels.size(0)
            valid_correct += (indices == labels).sum().item()

        # Computes the average validation loss and accuracy and store in respective lists
        valid_loss_list.append(valid_loss / len(validloader))
        valid_acc_list.append(100 * valid_correct / valid_total)

        print('Epoch [{}/{}]'.format(epoch+1, num_epochs))
        print('Train Loss: {:.4f}'.format(train_loss / len(trainloader)))
        print('Train Acc: {:.2f}%'.format(100 * train_correct / train_total))
        print('Valid Loss: {:.4f}'.format(valid_loss / len(validloader)))
        print('Valid Acc: {:.2f}%'.format(100 * valid_correct / valid_total))

        # Save the model parameters once in every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), './results/model_epoch_{}.ckpt'.format(epoch))

    #  Plots the training and validation loss
    fig = plt.figure(figsize=(15, 10))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(valid_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Saves the image in case the plot doesn't pop up (for example in an unconfigured WSL)
    plt.savefig('./results/training_plots/loss_plot.png')
    plt.show()

    # Plots the training and validation accuracy
    fig = plt.figure(figsize=(15, 10))
    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(valid_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Saves the image in case the plot doesn't pop up (for example in an unconfigured WSL)
    plt.savefig('./results/training_plots/accuracy_plot.png')
    plt.show()
