# PlantBreedImageClassifier

Classification of 102 flower categories using Pytorch models and deep learning techniques. 

This project was in the Udacity AI ML Nanodegree curriculum. The model used for training is VGG16. 

Arguments for train.py

'data_dir'. 'Provide data directory. Mandatory argument', type = str
'--save_dir'. 'Provide saving directory. Optional argument', type = str
'--arch'. 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str
'--lrn'. 'Learning rate, default value 0.001', type = float
'--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int
'--epochs'. 'Number of epochs', type = int
'--GPU'. "Option to use GPU", type = str

Arguments for predict.py

'image_dir'. 'Provide path to image. Mandatory argument', type = str
'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str
'--top_k'. 'Top K most likely classes. Optional', type = int
'--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
'--GPU'. "Option to use GPU. Optional", type = str
