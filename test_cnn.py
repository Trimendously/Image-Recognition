
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Used for saving plots
import matplotlib.image as mpimg

def testing_cnn(model,trainloader,testloader,batch_size,classes, num_epochs):

    # For storing true vs predicted labels
    results = []

    # Load parameters from one of the saved model states (I just manually chose the 15th)
    model.load_state_dict(torch.load('./results/model_epoch_15.ckpt'))

    # Set this model to evaluation mode
    model.eval()

    # Load a random batch of test set images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show the images
    grid = torchvision.utils.make_grid(images)
    grid = grid / 2 + 0.5  #Avoids this WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    grid_numpy = grid.numpy()
    plt.imshow(np.transpose(grid_numpy, (1, 2, 0)))
    plt.show()

    # Saves the image in case the plot doesn't pop up (for example in an unconfigured WSL)
    mpimg.imsave("./image_set/testing_image_set.png", np.transpose(grid_numpy, (1, 2, 0)))

    # Computes the model output
    model.eval()
    output = model(images)

    # Prints the ground true/predicted class labels for these images
    values, indices  = torch.max(output, 1)
    for i in range(batch_size):
        if classes[labels[i]] != classes[indices[i]]:
            results.append([i+1,classes[labels[i]],classes[indices[i]],'False'])
        else:
            results.append([i+1,classes[labels[i]],classes[indices[i]],'True'])
        print("Image {} : \n    True Label:{}\n     Predicted Label:{}".format(i+1,classes[labels[i]],classes[indices[i]]))

    # Computes the accuracy on each batch of test set
    check = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            values , indices = torch.max(output.data, 1)
            total += labels.size(0)
            check += (indices == labels).sum().item()

    # Prints the average accuracy
    print('Average Accuracy: {:.2f}%'.format(100 * (check / total)))

    # Computes per-class accuracy on each batch of test set
    check_class = [0] * num_epochs
    total_class = [0] * num_epochs

    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            values , indices = torch.max(output.data, 1)
            eval = (indices == labels)
            for i in range(len(labels)):
                label = labels[i]
                total_class[label] += 1
                check_class[label] += eval[i].item()

    # Prints per-class accuracy for 10 output classes
    for i in range(10):
        if total_class[i] > 0:
            acc = 100 * check_class[i] / total_class[i]
            print('Accuracy of {} : {:.2f} %'.format(classes[i], acc))
        else:
            print('No samples for class {}'.format(classes[i]))

    # Create a DataFrame
    df = pd.DataFrame(results, columns=["Image Number", "True Label", "Predicted Label", 'Comparison'])
    # Display the DataFrame
    print(df)

    # Specify the absolute file path for the Excel file
    excel_file_path = os.path.join(os.getcwd(), 'results/labels.xlsx')
    # Append the DataFrame to the Excel file or create a new one
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='True vs. Predicted Analysis', index=False)

    print("Dataframe was now added to Excel sheet.")
