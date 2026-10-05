import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
df = pd.read_csv("Iris.csv")

print("First five rows:")
print(df.head())


# Separate features and target
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = df["species"]


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create and train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:")
print(accuracy)


# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion matrix:")
print(cm)


# Five-fold cross-validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross-validation scores:")
print(cv_scores)

print("\nMean cross-validation accuracy:")
print(cv_scores.mean())


# Visualization
setosa = df[df["species"] == "setosa"]
versicolor = df[df["species"] == "versicolor"]
virginica = df[df["species"] == "virginica"]

plt.scatter(
    setosa["petal_length"],
    setosa["petal_width"],
    label="setosa"
)

plt.scatter(
    versicolor["petal_length"],
    versicolor["petal_width"],
    label="versicolor"
)

plt.scatter(
    virginica["petal_length"],
    virginica["petal_width"],
    label="virginica"
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris: Petal Length vs Petal Width")
plt.legend()
plt.show()
