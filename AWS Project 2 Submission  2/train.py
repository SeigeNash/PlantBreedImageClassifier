'''
Things to do in predict.py -
1. train.py successfully trains a new network on a dataset of images and 
2. saves the model to a checkpoint
3. The training loss, validation loss, and validation accuracy are printed out as a network trains
4. The training script allows users to choose from at least two different architectures available from torchvision.models
5. The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
6. The training script allows users to choose training the model on a GPU
'''

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import json





# creating parser. Defined for taking arguments for command line
parser = argparse.ArgumentParser()

# successfully trains a new network on a dataset of images and saves the model to a checkpoint
parser.add_argument ('data_directory', type = str)
parser.add_argument ('--save_directory', help = 'Provide saving directory. Optional argument', type = str)

args = parser.parse_args()

parser.add_argument ('--model_choice', type = str)
   
# The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
parser.add_argument ('--learning_rate', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Default value = 1000', type = int)
parser.add_argument ('--epochs', type = int)

# Option to use GPU.
parser.add_argument('--GPU', type = str)



    
# all the command line arguments -

data_dir = args.data_directory
save_directory = args.save_directory
model_choice = args.model_choice
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.GPU
    
def main():
    
    # loading transformers 
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.CenterCrop(255),
                                         transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    # dataloaders 
    training_dataset = datasets.ImageFolder( train_dir, transform = training_transforms)
    validation_dataset = datasets.ImageFolder( valid_dir, transform = validation_transforms)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
    
    
    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    
    
    
    # loading model and defining classifier and loss function    
    if(args.model_choice == "vgg16"):
        model = models.vgg16(pretrained = True)
    else:
        model = models.alexnet(pretrained = True)

        
        
    for parameters in model.parameters():
    parameters.requires_grad = False 
    
    
    
    # defining classifier with ReLU and dropout    
    if(hidden_units != 1000):
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    else:
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1000)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    
    
    
    # learning rate selection for optimizer
    if(learning_rate != 0.001):
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

    
    # code for training the model (the same as image classsifier notebook)
    choice_of_device = str(input("Enter your choice, CPU or GPU: "))
    if(choice_of_device == 'cuda' || choice_of_device == 'GPU'):
        device = 'cuda'
    else:
        device = 'cpu'

        
        
        
        
    # TRAINING LOOP ALONG WITH VALIDATION LOOP STARTS    
    running_loss = 0
    training_accuracy = 0

    

    for e in range(epochs):

       
        model.to(device)
        model.train()

        for i, (inputs, labels) in enumerate(training_loader):

            # Move input and label tensors to the device (gpu or cpu)
            inputs, labels = inputs.to(device), labels.to(device)

            start = time.time()

            optimizer.zero_grad()

            outputs = model.forward(inputs)    # forward pass!
            training_loss = criterion(outputs, labels)
            training_loss.backward()   #backpropogation
            optimizer.step()

            running_loss += training_loss.item()

            # calculating training accuracy (the topk method) 
            probs = torch.exp(outputs)
            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            training_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # the following if statement is a simple check for the flow of training. 
            if(i % 50 == 0):
                print(i)

            print("Train loss: ", running_loss/len(training_loader), 
                          "training accuracy: ", training_accuracy)


        else:
            ## validation loop on validation_loader
            print("validation loop starts")

            running_validation_loss = 0
            validation_accuracy = 0

            with torch.no_grad():

                for i, (inputs, labels) in enumerate(validation_loader):

                    inputs, labels = inputs.to(device), labels.to(device)

                    validation_outputs = model.forward(inputs)
                    validation_loss = criterion(validation_outputs, labels)

                    running_validation_loss += validation_loss.item()


                    # calculating validation accuracy (the topk method) 
                    probs = torch.exp(validation_outputs)
                    top_p, top_class = probs.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    if(i % 50 == 0):
                        print(i)


            print("Validation loss: ", running_validation_loss/len(validation_loader),
                          "Validation accuracy: ", validation_accuracy)




        
    model.class_to_idx = training_dataset.class_to_idx

    # defining the dictionery checkpoint and pasing it into torch.save
    # checkpoint cointains my model's state_dict(), optimzer state_dict(), and classifier 
    torch.save(checkpoint = {
        'model_state_dict': model.state_dict(),
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_validation_loss/len(validation_loader),
    }, './model_checkpoint')






if __name__ == "__main__":
    main()

