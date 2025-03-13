import bentoml
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier

# Load training data set
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train your model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("my_rf_model", model)
print(f"Model saved to {saved_model}")
