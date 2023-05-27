'''
Things to do in predict.py -
1. The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
2. The predict.py script allows users to print out the top K classes along with associated probabilitie
3. The predict.py script allows users to load a JSON file that maps the class values to other category names
4. The predict.py script allows users to use the GPU to calculate the predictions

'''

#imports
import numpy as np
import argparse     # for making the command line
import matplotlib.pyplot as plt
from PIL import Image
import json         # for loading the JSON file 
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.utils.data
from collections import OrderedDict



# creating parser. Defined for tking arguments for command line
parser = argparse.ArgumentParser()

# Providing path to image for prediction
parser.add_argument('image_directory', type = str, required = True)

# Loading the model from the path provided for checkpoint 
parser.add_argument('load_directory', type = str, required = True )
model = loading_model(args.load_directory)

# Top K most likely classes.
parser.add_argument('--top_k', type = int)

# JSON file name for mapping classes to categories
parser.add_argument('--category_names', type = str)

# choose your model
parser.add_argument('--model_choice', type = str)

# Option to use GPU.
parser.add_argument('--GPU', type = str)
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'



    
image_directory = args.image_directory
load_directory = args.load_directory
model_choice = args.model_choice
category_names = args.category_names
device = args.GPU
top_k_classes = args.top_k





def main():
    
    
    # choose model to train
    if(args.model_choice == "vgg16"):
        model = models.vgg16(pretrained = True)
    else:
        model = models.alexnet(pretrained = True)

    
    # model
    if args.GPU == 'GPU':
        device = 'cuda'
    else:
        device = 'cpu'

    
    # loading checkpoint 
    checkpoint = torch.load(load_directory)
    
    # loading checkpoints 
    model.classifier = checkpoint['classifier']
    model.load_state_dict (checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']

    for param in model.parameters():
        param.requires_grad = False 
        
        
        
        
    
    # this code block process a PIL image for use in a PyTorch model. 
        
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image_directory) 
    width, height = img.size 
    
    
    #using thumbnail to keep aspect ratio
    if width > height: 
        height = 256
        img.thumbnail((50000, height), Image.ANTIALIAS)
    else: 
        width = 256
        img.thumbnail((width,50000), Image.ANTIALIAS)    

    
    process_image_transforms = transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor()])
    
    img = process_image_transforms(img)
     
    # img = process_image_transforms(img)
    img = np.array(img)/255
    
    
    '''
    TASK -
    individually extract all the solour channels of the image
    after that, individually subtract mean from them and divide the standard deviation from them
    after that, make these modified arrays as the new colour channels of the image 
    return this particular image
    to undo this process in the imshow function, repeat the above steps 
    '''
    
    # creating two 2D arrays mean and std 
    # mean will be subtracted 
    # std will be divided
    meanR = np.full((224, 224), 0.485)
    stdR = np.full((224, 224), 0.229)
    meanG = np.full((224, 224), 0.456)
    stdG = np.full((224, 224), 0.224)
    meanB = np.full((224, 224), 0.406)
    stdB = np.full((224, 224), 0.225)
    
    img[0] = ( img[0] - meanR ) / stdR
    img[1] = ( img[1] - meanG ) / stdG
    img[2] = ( img[2] - meanB ) / stdB
    
    
    ### the shape before reshaping is (3, 224, 224). it indicates that there are 3 colour channels, R, G, B

    
    img = img.transpose((1,2,0))
    
    
    
    
    
    '''
    NOTE :-
    The inspiration for the following code has baan taken from the following Github Repo-
    https://github.com/Kusainov/udacity-image-classification/blob/master/Image%20Classifier%20Project.ipynb
    
    '''
    
    
    
    
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    img = process_image(image_directory) #loading image and processing it using above defined function

    #we cannot pass image to model.forward 'as is' as it is expecting tensor, not numpy array
    #converting to tensor
    if device == 'cuda':
        im = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy(img).type(torch.FloatTensor)

    im = im.unsqueeze(dim = 0) #used to make size of torch as expected. as forward method is working with batches,
    #doing that we will have batch size = 1

    #enabling GPU/CPU
    model.to(device)
    im.to(device)

    with torch.no_grad ():
        output = model.forward(im)
    output_probabilities = torch.exp(output) #converting into a probability

    probs, indeces = output_probabilities.topk(top_k_classes)
    # using CPU for the following as they dont need GPU
    probs = probs.cpu()
    indeces = indeces.cpu()
    probs = probs.numpy().tolist()[0] #converting both to numpy array and then to list 
    indeces = indeces.numpy().tolist()[0]


    mapping = {value: key for key, value in
                model.class_to_idx.items()
                }

    classes = [mapping[item] for item in indeces]
    classes = np.array(classes) #converting to Numpy array



    #setting values data loading
    args = parser.parse_args()
    file_path = image_directory

    #defining device: either cuda or cpu

    #loading JSON file if provided, else load default file name
    # The predict.py script allows users to load a JSON file that maps the class values to other category names
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            pass


    #defining number of classes to be predicted. Default = 1
    if top_k_classes:
        number_of_classes = top_k_classes
    else:
        number_of_classes = 1

    #calculating probabilities and classes
    probabilities, classes = predict(file_path, model, number_of_classes, device)

    #preparing class_names using mapping with cat_to_name
    # using cat_to_name dict 
    class_names = [cat_to_name[item] for item in classes]

    
    print("print out the top K classes along with associated probabilities")
    for l in range(number_of_classes):
         print("For class number: " , l,
             " Number of classes are : " number_of_classes,
                " Along with class name: "class_names[l],
                " and probability: ", probabilities[l]*100,
                )

 




if __name__ == "__main__":
    main()
