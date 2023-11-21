import matplotlib.pyplot as plt

# Sample data: contributor names and their response counts
contributors = ['Contributor A', 'Contributor B', 'Contributor C', 'Contributor D']
response_counts = [45, 62, 30, 53]

# Create a bar chart
plt.bar(contributors, response_counts, color='blue')

# Add labels and title
plt.xlabel('Contributors')
plt.ylabel('Response Counts')
plt.title('Contributor Responses')

# Display the chart
plt.show()
