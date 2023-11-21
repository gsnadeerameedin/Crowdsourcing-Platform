import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# Load your data (replace with your data)
data = pd.read_csv('worker_behavior_data.csv')

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Initialize the XGBoost model
model = xgb.XGBRegressor()

# Fit the model
model.fit(X, y)

# Get feature importance scores
importance_scores = model.feature_importances_

# Create a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance - XGBoost')
plt.show()
