# Import necessary libraries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

print("Loading the Olivetti faces dataset...")
# Load the dataset
faces = fetch_olivetti_faces()
data = faces.data
target = faces.target

print("Splitting the data into training (70%) and testing (30%) sets...")
# [cite_start]Split the data into 70% training and 30% testing [cite: 34]
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

print(f"Data split complete. Training set has {len(X_train)} samples.")

print("Training the Decision Tree Classifier model...")
# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

print("Model training complete.")

print("Saving the trained model to 'savedmodel.pth'...")
# [cite_start]Save the model using joblib as savedmodel.pth [cite: 35]
joblib.dump(model, 'savedmodel.pth')

print("Model saved successfully as 'savedmodel.pth'. train.py execution finished.")