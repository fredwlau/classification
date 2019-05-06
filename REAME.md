sh -e commands.txt to run multiple commands in the commands file


FLAGS:
  -h, --help            show this help message and exit
  
  -c CLASSIFIER, --classifier=CLASSIFIER
  
                        The type of classifier [Default: mostFrequent][Choices: naiveBayes, perceptron, neuralNetwork, nb, nn]
                        
  -d DATA, --data=DATA  Dataset to use [Default: digits][Choices: digits, faces]
  
  -t TRAINING, --training=TRAINING
  
                        The size of the training set [Default: 100][Ideal Choice: 1000]
                        
  -f FEATURES, --features=FEATURES
  
                        The type of features [Default: basic][Choices: basic, sobel, enhanced]
                        
  -w, --weights         Whether to print weights [Default: False]
  
  -k SMOOTHING, --smoothing=SMOOTHING
  
                        Smoothing parameter (ignored when using --autotune)
                        [Default: 2.0]
                        
  -a, --autotune        Whether to automatically tune hyperparameters
                        [Default: False]
                        
  -i ITERATIONS, --iterations=ITERATIONS
                        Maximum iterations to run training [Default: 3][Ideal: 60]
                        
  -s TEST, --test=TEST  Amount of test data to use [Default: 100]
  
  -p ALPHA, --alpha=ALPHA
                        Learning rate for neural network [Default: 1.0][Ideal: 3.0]
