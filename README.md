# Image-Classification-for-ML-Internship-2023

This task is for the position of an machine learning internship for the company Space Sense.  
The following is an image classifier on the EuroSAT dataset used for land cover classification.

## User guide

You can clone this repository on your machine with the following line :  

    git clone https://github.com/camdore/Image-Classification-for-ML-Internship-2023.git

In the 4th cell of the training notebook and 3rd of the inference notebook you have to change the path of the directory where is stored the dataset with the adapted line :  

    path/to/your/directory/Image-Classification-for-ML-Internship-2023/EuroSAT/2750
## Structure of the repository 

saved_model/my_model : directories where the model created in training.ipynb is saved  
EuroSAT/2750 : directory of the 27,000 images dataset with all of the 10 classes/subdirectories  
README.md  
training.ipynb : creation of the classification model from the loading of the dataset to the model saving  
inference.ipynb : is results of predictions of this model on 20 random image sample  

## Observations
### 1. Constraints of the solution

- There is not enough data in the original EuroSAT, only 27 000 images. Even with the data augmentation the number of images is not high enough for training. 
- The definition of each image (64x64) is very low therefore the model is limited. This may limit the ability of the model to capture fine-grained details that could be useful for classification.
- The dataset is imbalanced, with some classes having many more images than others. This can lead to bias towards certain classes which may affect model performance and predictions.
- The model uses a fixed learning rate, which may not be optimal for convergence on this particular dataset.
- Use the GPU power for faster training.

### 2. Potential improvements to the solution

- Use a larger and more balanced dataset.
- Use a pretrained model and use its architecture as a starting point with **transfer learning**. For example, we can use models such as ResNet, VGG, AlexNet or GoogLeNet.
- Add more **preprocessing layers** such as **normalization** on the RGB channels in order to normalize the pixel values. There are some differences of contrast between some day and night pictures. We can use the following line :   
   
    `tf.keras.layers.Normalization(mean=([meanR, meanG, meanB]), variance=([stdR,stdG, stdB]))`

- Use **batch normalization** after each convolutional layer to improve the stability of the model during training and prevent overfitting.
- We can also reduce overfitting with weight regularization with **L2 regularization**.
- **Ensemble learning** : Combining multiple models trained on different subsets of the dataset can improve performance and reduce overfitting. For example, using a combination of CNNs and decision trees or SVMs could provide a more robust classification system.
- **Hyperparameter tuning** : Experiment with different values for hyperparameters such as learning rate, dropout rate, and batch size, it can improve model performance. Grid search or Bayesian optimization could be used to efficiently explore the hyperparameter space. We can use the Keras Tuner which influence model selection such as the number and width of hidden layers.

## Autors 
Camille Doré