# American_Sign_Language_Classification_Project_Phase_2

**Creating a RESTful API to classify American Sign Language gestures**
(Part 2 of the mobile computing project)

**The first phase of this project can be found here: https://github.com/prempatel97/American_Sign_Language_Classification_Using_ML_Project_Phase_1**


**Details:**

• Developed an online Application Service that accepts Human Pose Skeletal key points of a sign video and returns the label of the sign as a JSON Response. The Key points are generated using TensorFlow’s Pose Net.

• Using these key points, developed 4 machine learning models that can classify the American Sign Language signs { gift, car, sell, book, total, movie }.

• For generating the key points, followed instructions in the link -> https://github.com/prashanthnetizen/posenet_nodejs_setup

• For the dataset, we used the videos captured as part of your first assignment (https://github.com/prempatel97/American_Sign_Language_Classification_Using_ML_Project_Phase_1). 

• The JSON input given to the service follows the format of the output given by the JavaScript file mentioned in the GitHub repo.

• The output of the service is given as a JSON response for each input JSON video data in the following format.

{
“1”: “predicted_label”,
“2”: “predicted_label”,
“3”: “predicted_label”,
“4”: “predicted_label”
}

where 1 to 4 denotes the index of your ML models,
“predicted_label” is the predicted sign.

• We hosted our service using AWS EC2 cloud platform


The keypoints from the posenet are in Json format. We cannot just feed the data into the
machine learning algorithms and expect the algorithms to learn the actions. The data is needed
to be cleaned and normalized. We call it data pre-processing. Data preprocessing is done as
under:
1. The received Json data is converted to the pandas dataframe.
2. Drop the features that don’t provide any spatial data (does not move)
3. For our project, we did not consider any keypoints below waist and above shoulders.
4. We then centered all the frames by the nose’s coordinates.
5. Furthermore, we normalized all the dimensions by dividing nose-waist length.
6. We considered the angles of the arms as engineered input, but as there was no drastic
improvement in the predictions, we decided to drop it.
7. We then feed the initial 40% frames of each video to the models. (because the actor acts
for only 2-3 seconds from the total 5 seconds).

The various models used by us are as under:

**K Nearest Neighbors -**
k-Nearest Neighbor is a supervised machine learning algorithm that can be used for
classification or regression problems. The ML models that we are developing give a discrete
output (ASL sign class) rather than a continuous output and thus, we will be using the kNN for
classification. The kNN model operates under the assumption of proximity i.e. it assumes that
things that belong to the same class will lie together. The similarity or closeness is usually
measured by the euclidean distance between the points on a multi-dimensional graph. The k in
kNN stands for the number of neighbors to be considered for classifying the data point. In
classification, we use the mode of the labels of the k nearest points. The accuracy of the kNN
model will change depending on the value of k. If we set the value of k to be too high or too low,
the accuracy wil decrease. To find the right value of k, we need to run the algorithm multiple
times and then choose the value of k that minimizes the error. The advantages of using this
model is that it is fairly easy to implement and eliminates the need to build a complex model and
tune multiple parameters. This algorithm considers the closest K neighbors of the query vector
and classifies the query vector based on majority voting. The distance metric we used is
Euclidean (L2 Norm). The number of neighbors we used were 5. This is to eliminate the tie
conditions that could occur if K=even.

**Random Forest -**
We choose this model due to its greater accuracy for prediction.Concept constituted to create
the random forest model is "Wisdom of Crowds". In other terms, the core point helping the
random forest to give way better accuracy than other models is "Large number of relative
uncorrelated models(trees) operating as a group will be able to give much more accuracy than
any individual constituent models". So by this reason we chose Random Forest as one of the
models for prediction.

**MLP Classifier(Multi-Layer Perceptron) -**
Perceptron is defined as a linear classifier that classifies input by separating into two categories
with a straight line. For example: cat or not cat. When more than one perceptron is used than it
becomes a multilayer perceptron(MLP) which creates a deep artificial neural network. It is
composed of an input layer to receive the signal and an output layer that makes the prediction
or decision about the input but in between(those input & output layers) there are an arbitrary
number of hidden layers that do the true computation for the classifier. The classifier gets
trained on set of input-output pairs and learns the correlation(or dependencies) between those
inputs and outputs. Also in training it adjusts the parameters, weights and biases dynamically in
order to minimize the error. It also uses the back propogation to make those weight and bias
adjustment relative to the error and the error itself can be measured in a variety of ways,
including by root mean squared error (RMSE). This leads us to select this model for better
prediction.
We used MLP with 4 hidden layers with 100,100,60,40 neurons in respective layers.

**SVM with RBF kernel:**
Support Vector Machine achieves a good separation by the hyperplane that has the largest
distance to the nearest training data. Support Vector machines have a limiting factor that does
not allow it to be usable above 2d space. To overcome this problem, we use the RBF kernel to
project the data into high dimensional space and then use SVM to find the classifier
hyper-plane. This hyper plane is then reverted back to the 2d plane, which gives out the
classifier boundary. Furthermore, we use the slack variables to ease the outlier SVM trying to
classify the outlier. C = 0.8


P.S: The AWS EC2 instance running the models is not active.

Created by Sumit Rawat, Prem Patel, Dhruv Patel and Kum Hee Choy
