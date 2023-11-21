import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic data for demonstration
np.random.seed(42)
num_responses = 100
trustworthiness_scores = np.random.randint(0, 11, num_responses)
features = np.random.rand(num_responses, 3)  # Replace with your actual features

# Define a trustworthiness threshold
trustworthiness_threshold = 6

# Convert trustworthiness scores into binary labels (trustworthy or not)
labels = np.where(trustworthiness_scores >= trustworthiness_threshold, 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
