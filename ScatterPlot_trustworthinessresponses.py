import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate synthetic data for demonstration
np.random.seed(42)
num_entries = 30
date_today = datetime.today()

# Generate random timestamps within the last 30 days
timestamps = [date_today - timedelta(days=np.random.randint(1, 31)) for _ in range(num_entries)]
timestamps.sort()

# Generate random trustworthiness scores
trustworthiness_scores = np.random.uniform(0, 100, num_entries)

# Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(timestamps, trustworthiness_scores, color='blue')
plt.xlabel('Time')
plt.ylabel('Trustworthiness Score')
plt.title('Trustworthiness Score Over Time')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
