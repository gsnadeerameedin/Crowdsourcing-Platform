import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate synthetic data for demonstration
np.random.seed(42)
num_responses = 50
date_today = datetime.today()

# Generate random timestamps within the last 30 days
timestamps = [date_today - timedelta(days=np.random.randint(1, 31)) for _ in range(num_responses)]
timestamps.sort()

# Create a histogram of responses over time
plt.figure(figsize=(10, 6))
plt.hist(timestamps, bins=15, edgecolor='black')
plt.xlabel('Time')
plt.ylabel('Number of Responses')
plt.title('Pattern of Responses Over Time')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
