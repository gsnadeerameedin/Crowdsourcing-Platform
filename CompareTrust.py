import numpy as np
from sklearn.metrics import accuracy_score

# Simulated data: True trustworthiness values and predicted scores
true_trustworthiness = np.random.randint(0, 2, 100)  # True values (0 or 1)
predicted_consensus_scores = np.random.rand(100)  # Consensus-based predicted scores (0 to 1)
predicted_reputation_scores = np.random.rand(100)  # Reputation-based predicted scores (0 to 1)
predicted_gold_scores = np.random.rand(100)  # Gold standard predicted scores (0 to 1)

# Convert predicted scores to binary labels (trustworthy or not)
predicted_consensus_labels = np.round(predicted_consensus_scores)
predicted_reputation_labels = np.round(predicted_reputation_scores)
predicted_gold_labels = np.round(predicted_gold_scores)

# Calculate accuracy scores
accuracy_consensus = accuracy_score(true_trustworthiness, predicted_consensus_labels)
accuracy_reputation = accuracy_score(true_trustworthiness, predicted_reputation_labels)
accuracy_gold = accuracy_score(true_trustworthiness, predicted_gold_labels)

# Print accuracy scores
print("Accuracy of Consensus-based approach:", accuracy_consensus)
print("Accuracy of Reputation-based approach:", accuracy_reputation)
print("Accuracy of Gold Standard approach:", accuracy_gold)
