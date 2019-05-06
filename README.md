# Artificial Intelligence Handwritten Digit and Face Classification

sh -e commands.txt to run commands from commands.txt

Currently in commands.txt

python dataClassifier.py -c naiveBayes --autotune [runs classification using Naive Bayes on DEFAULT digits with autotuned k value]

python dataClassifier.py -c naiveBayes -d digits -f basic -t 1000 -k 2.5 [runs classification using Naive Bayes on SPECIFIED digits using basic feature extractor and a training set of 1000 data points and a k value of 2.5]

python dataClassifier.py -c perceptron -d faces [runs classification using Perceptron on SPECIFIED faces with default values]

python dataClassifier.py -c perceptron -w  [runs classification using Perceptron on DEFAULT digits and outputs weight values]

python dataClassifier.py -c neuralNetwork -d digits -f sobel -i 60 -p 3.0 [runs classification using Neural Network on SPECIFIED digits using "sobel" edge detection features with 60 iterations and a learning value of 3.0]

python dataClassifier.py -d digits -c naiveBayes -f enhanced -a -t 1000 [runs classification using Naive Bayes on SPECIFIED digits using "enhanced" feature detection with autotuned k value and a training set of 1000 data points]

Add more commands as you see fit

FLAGS: 

-h, --help show this help message and exit

-c CLASSIFIER, --classifier=CLASSIFIER

                    The type of classifier [Default: mostFrequent][Choices: naiveBayes, perceptron, neuralNetwork, nb, nn]
-d DATA, --data=DATA

                    Dataset to use [Default: digits][Choices: digits, faces]
                    
-t TRAINING, --training=TRAINING

                    The size of the training set [Default: 100][Ideal Choice: 1000]
                    
-f FEATURES, --features=FEATURES

                    The type of features [Default: basic][Choices: basic, sobel, enhanced]
-w, --weights

                    Whether to print weights [Default: False]
                    
-k SMOOTHING, --smoothing=SMOOTHING

                    Smoothing parameter (ignored when using --autotune)
                    [Default: 2.0]
                    
-a, --autotune=AUTOTUNE

                    Whether to automatically tune hyperparameter [Default: False]
                    
-i ITERATIONS, --iterations=ITERATIONS

                    Maximum iterations to run training [Default: 3][Ideal: 60]
                    
-s TEST, --test=TEST

                    Amount of test data to use [Default: 100]
                    
-p ALPHA, --alpha=ALPHA

                    Learning rate for neural network [Default: 1.0][Ideal: 3.0]

TODO:

Write up and analysis of the three classification algorithms [Naive Bayes, Perceptron, Neural Network]
Compare running times and accuracy
Tune models
