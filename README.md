# computer-vision-fruit-detector
 Computer Vision AI: Overripe Fruit Detector

## Background on classification algorithms
### Common Classification Algorithms
1. Logistic Regression
2. Native bayes
3. K-Nearest Neighbors (KNN)
4. Markov Decision Process (MDP)
5. Decision Trees/Random Forest
6. K-Means Clustering
7. Support Vector Machines (SVM)

### What is a classification algorithm?
A classification algorithm takes raw data and predicts its category.
e.g. fruit classification algorithm

Data(Features)
| Fruit | Color | Shape | Size | Weight |
|-------|-------|-------|------|--------|
| Apple | Red   | Round | 5cm  | 100g   |
| Blueberry | Blue   | Round | 2cm  | 50g   |

e.g. email spam classification algorithm
Data(Features)
- Sender, send time, subject, body, etc.

### How do we construct an algorithm specific to our needs?
Type of Learning:
1. Supervised Learning:
- Models 'learn' from training examples, provided by humans.
- Someone needs to tell the model what the correct answer is.
- Model can be calibrated to be more or less accurate.
- Mostly used for 'discrimination' (recognition) tasks.
- Train/test data- typically 80%/20% split.
e.g. Random Forest

Unsupervised Learning
- When model 'learns' from unlabeled data
- Not giving human-generated data to learn from
- Mostly used for 'generative' (imagination) tasks
- No split between train/test data
e.g. K-Means Clustering algorithm
Generative Adversarial Networks (GANs) can take it some parameters e.g. age, eyes, perspective, mood and generate a face.

Reinforcement Learning
- Model 'learns' the best strategy, using a scenario given by humans
- Model is given a reward for correct actions
- Scenario: actions & environment
- Mostly used for decision making tasks e.g. robotics
e.g. Markov Decision Process (MDP)
(Check out Computerphile's video on MDPs)

Select Learning Mode based on Application!

### Other factors
Number of features
- How many variables do you have? e.g. age, height
- Can you reduce the number of features to simplify your model? Always trying to reduce features to simplify model (Feature Selection)
- More features: consider a support vector machine
- Less features: consider a decision tree

Linearity
- Is your data linear? Is there a linear relationship- directly proportional between 2 things in your data? This is good as you can be confident using a linear model.
- Linear: consider a support vector machine

Training Time
- How much data do you have and how fast is your computer?
- Faster training: Logistic regression
- Slower: consider a random forest
- Generally speaking, more training = more accuracy (but not always)

Number of parameteres
- How much flexibility do you want in your training?
- Less parameters: consider a logistic regression
- More parameters: consider K-Means clustering
- Split data into train/validation/test to examine the parameter space

Using Gradient Descent Algorthm - our image has more dimensions e.g. colour, shape of fruit, foreground and background, etc.

### What is Transfer learning?
Using a previously-trained neural net to extract certain features.
The dataset could be trained on 1 million+ images using a CNN algorithm (for accuracy)
Then, adding a final classification layer (e.g. Gradient Descent).
The final layer is faster and requires less training data
Pre-trained image model (does feature selection) -> Final classification layer (Gradient Descent) -> Result

### Assessing results
Confusion Matrix - way to represent the accuracy of our classification approach
On Y-axis (rows), we have the actual condition of the data (e.g. overripe, ripe, underripe)
On X-axis (columns), we have the predicted condition of the data (e.g. overripe, ripe, underripe)
If model predicts image is ripe but the actual condition is unripe, it would be a False-Positive (FP)
This matrix tells us where the model is making mistakes.

TensorFlow Transfer Learning:
https://www.tensorflow.org/hub/tutorials/image_feature_vector

CNN are great for classifying images but they are hard to run. Pre-trained models are great for this. 

https://link.springer.com/article/10.1007/s12652-021-03267-w

