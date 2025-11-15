import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading the Olivetti faces dataset for testing...")
# We need to load the data again to get the test set
faces = fetch_olivetti_faces()
data = faces.data
target = faces.target

print("Splitting data to get the *same* test set...")
# Split the data with the same parameters to get the same test set
# We only care about the 'test' variables here
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

print(f"Test set has {len(X_test)} samples.")

print("Loading the saved model from 'savedmodel.pth'...")
# [cite_start]Load the model you saved in train.py [cite: 36]
try:
    model = joblib.load('savedmodel.pth')
    print("Model loaded successfully.")
    
    print("Making predictions on the test set...")
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    print("Calculating accuracy...")
    # [cite_start]Calculate the accuracy [cite: 36]
    acc = accuracy_score(y_test, y_pred)
    
    print("---" * 10)
    print(f"TEST ACCURACY: {acc * 100:.2f}%") # Display the test accuracy
    print("---" * 10)
    print("test.py execution finished.")

except FileNotFoundError:
    print("Error: 'savedmodel.pth' not found.")
    print("Please run train.py first to create the model file.")
except Exception as e:
    print(f"An error occurred: {e}")