# Iris Flower Classification with KNN

A beginner machine learning project using the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers into three species.

## Project Overview

The goal of this project is to understand the basic machine learning workflow through a simple classification problem.

The model uses four measurements of Iris flowers to predict their species:

* `sepal_length`
* `sepal_width`
* `petal_length`
* `petal_width`

The target variable is `species`, with three classes:

* `setosa`
* `versicolor`
* `virginica`

## Dataset

The dataset contains 150 Iris flower samples.

Each sample contains four numerical features and one target label.

The dataset used in this project is stored in `Iris.csv`.

## Machine Learning Workflow

The project follows these steps:

1. Load and explore the dataset
2. Separate features (`X`) and target (`y`)
3. Split the data into training and test sets
4. Train a K-Nearest Neighbors classifier
5. Make predictions on the test set
6. Evaluate the model using accuracy and a confusion matrix
7. Evaluate model performance using five-fold cross-validation
8. Visualize the Iris species using petal measurements

## Model

The classification algorithm used in this project is **K-Nearest Neighbors (KNN)**.

The initial model uses:

* `n_neighbors = 5`
* 80% training data
* 20% test data
* `random_state = 42`

## Results

The model achieved:

* **100% accuracy** on the selected test set
* **97.33% mean accuracy** using five-fold cross-validation

The confusion matrix showed that all 30 samples in the selected test set were classified correctly.

The cross-validation results were:

| Fold | Accuracy |
| ---- | -------: |
| 1    |   96.67% |
| 2    |  100.00% |
| 3    |   93.33% |
| 4    |   96.67% |
| 5    |  100.00% |

## Experiments

In addition to the baseline model, I experimented with different aspects of the KNN model.

### Feature Experiment

I removed one feature at a time and evaluated the model using five-fold cross-validation.

The largest decrease in mean accuracy occurred when `petal_width` was removed.

This suggests that `petal_width` was particularly useful for classification in this experiment.

### Hyperparameter Experiment

I tested several values of `k`:

|  k | Mean CV Accuracy |
| -: | ---------------: |
|  1 |           96.00% |
|  3 |           96.67% |
|  5 |           97.33% |
|  7 |           98.00% |
|  9 |           97.33% |

Among the tested values, `k = 7` achieved the highest mean cross-validation accuracy.

### Feature Scaling

I also compared KNN with and without feature scaling.

In this dataset, scaling did not improve the model's performance. The four features are measured in centimeters and have relatively similar numerical ranges.

## Visualization

The project includes a scatter plot comparing petal length and petal width.

The visualization shows that the Iris species can be separated reasonably well using petal measurements, although some overlap exists between `versicolor` and `virginica`.

## What I Learned

This project helped me understand the basic machine learning workflow, including:

* Data exploration
* Features and targets
* Train/test splitting
* KNN classification
* Model evaluation
* Confusion matrices
* Cross-validation
* Feature experiments
* Hyperparameter testing
* Feature scaling
* Data visualization

## Tools and Libraries

* Python
* pandas
* scikit-learn
* matplotlib
* Kaggle

## Kaggle Notebook

The complete analysis and experiments are available in my Kaggle Notebook.

[Kaggle Notebook](https://www.kaggle.com/code/maxxigolden/iris-flower-classification-with-knn)
