# TF - Intel Images

This project demonstrates the use of the Tensorflow framework to approach Machine Learning (ML) model building and deployment for an image recognition problem - to classify an image into one of six categories (buildings, forest, glacier, mountain, sea or street). It is intended as a learning exercise for working on a challenging unstructured deep learning problem while gaining experience in the Google/Tensorflow toolkit. This toolkit was used for the entire end-to-end workflow, from image processing and model development to deployment.    

The image data used for model training and evaluation is the Intel Image Classification set from Kaggle: https://www.kaggle.com/puneet6060/intel-image-classification.


### Experimentation

This folder contains notebooks used during the project experimentation phase. They outline steps for exploratory data analysis, organsing and processing data, and model building, testing and evaluation. 

Given the complexity of the problem, Convolutional Neural Networks (CNNs) and other Deep Neural Networks (DNNs) were trained and tested on image dataset. For each architecture, results were compared for different hyperparameters, initially with a subset of the problem, to gain an intuition as to how appropriate the model was to the task. As experimentation progressed to more complex models that required lengthier training times, Google Colab was used to gain access to a server with a GPU for model training.  

It consists of:
 - Tensorflow/Keras for building models for various DNN and CNN architectures
 - Trialing pre-built CNNs within the Keras library on the problem
 - Use of Tensorboard to track hyperparameters and model learning during Training
 - Use of Google Colab for model training on a GPU-enabled server


### Deployment

This folder contains scripts to deploy the ML model as a RESTful prediction web service to Kubernetes. Scripts have also been developed to query an open-source API (Pexels) for more images and to process them, and to execute a GitHub Actions workflow for Continuous Integration and Continous Deployment (CI/CD) of the model and associated code.  

It is made up of the following components:
 - Use of GitHub Actions to execute a series of unit tests using PyTest 
 - Use of Tensorflow model server to serve the model as a prediction web service
 - Custom-built helm chart for deploying the model to a Kubernetes instance
 - Python scripts to query the Pexels API to retrieve new images for testing the prediction service 

*Intel Images - deployment.ipynb* outlines steps to deploy a chosen TF model to Kubernetes, test its prediction capability using images retrieved from the Pexels API and analyse the results. 

Any python packages used during this project can be found listed in the *requirements.txt* file. 
