# Damage Type Analysis Using an Integrated Transformer-ResNet Model

## Overview

In this project, we present a novel multimodal deep learning framework combining tiny-BERT’s natural language processing with R-CNN and ResNet-18’s image processing for enhanced detection of images in Twtter posts that may be identified as "damage related". By integrating textual context and , it significantly boosts its accuracy and performance.

## Milestones

-   ✔ Development of CNN with ResNet as backbone and training of an image identifying model
-   ✔ Development of tiny-BERT and training of a text identifying model
-   ✔ Integration of foregoing two models, using various fusion methods
-   ✔ Optimzation for better time and space performance
-   ✔ A cheering accuracy outcome!

## Repository and Code Structure

This repository includes:

- In folder **[Dataset](https://github.com/Tidaul/True_Damage/tree/main/Dataset)** you would see:
	- [Raw dataset we use, source cited here](https://archive.ics.uci.edu/dataset/456/multimodal+damage+identification+for+humanitarian+computing)
	- Data split processor
-   In folder **[CV](https://github.com/Tidaul/True_Damage/tree/main/CV)** you would see: 
	- Code of CNN-ResNet training and evaluation for visual data processing,
	- Model trained for visual data processing
	- Predictor and prediction for the dataset
- In folder **[NLP](https://github.com/Tidaul/True_Damage/tree/main/NLP)** you would see:
	- Code of Tiny-BERT training and evaluation for textual data processing,
	- Model trained for textual data processing
	- Predictor and prediction  for the dataset
- In folder **[Integrate](https://github.com/Tidaul/True_Damage/tree/main/Integrate)** you would see:
	- Code of ensembling learning models (Linear Regression and Random Forest) for combining predictions
	- Code of metric monitor and AUC plotting
- In folder **[Result](https://github.com/Tidaul/True_Damage/tree/main/Result)** you would see:
	- Image and .csv of the final predicting result for our project

## Usage

Me and my teammate seperately handled CV and NLP parts, and due to the rush of time we weren't able to build a pipeline for the whole project. Sorry for the inconvenience, and you are still welcome to check and test our code!

## Results and Observations

|Integration Method 		|Accuracy  |Precision |Recall 	 |F1 Score  |
|---------------------------|--------- |----------|----------|----------|
|Only using NLP				|`0.886986`|`0.893676`|`0.852279`|`0.889142`|
|Only using CV				|`0.806507`|`0.813269`|`0.777819`|`0.807774`|
|Linear Regression			|`0.863014`|`0.881020`|`0.830396`|`0.868383`|
|Random Forest				|`0.895548`|`0.895288`|`0.844837`|`0.895279`|
|Logistic Regression		|`0.886986`|`0.891178`|`0.845402`|`0.888499`|
|Support Vector Classifier	|`0.364726`|`0.701497`|`0.320328`|`0.307200`|
|Weighted Average			|`0.912671`|`0.915565`|`0.886684`|`0.913261`|


Above is the accuracy of several differnt integration methods applied for our test set.
-   Accuracy for CNN+Resnet(CV) in testset is 80.65%, in trainset is 91%
    
-   Accuracy for Finetuned TinyBert(NLP) in testset is 88.70%, in trainset is 97.23%
    
-   After Model Fusion, the best preformed accuracy for integrated model is Weighted Average, with 91.26% accuracy in testset and 98.17% in trainset.
    
-   It is a great improvement from the best accuracy rate provided in [the dataset article](https://idl.iscram.org/files/husseinmouzannar/2018/2129_HusseinMouzannar_etal2018.pdf): Word2Vec’s testset accuracy 89.88% and trainset accuracy 91.18%.

