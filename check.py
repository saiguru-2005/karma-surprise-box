import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON dataset
data = pd.read_json("training_data.json")

# Extract the features into a DataFrame
features = pd.DataFrame(data["features"].tolist())

# Plot histograms for multiple metrics
metrics_to_plot = ["posts_created", "upvotes_received", "comments_written"]
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))
    sns.histplot(features[metric], bins=20, kde=True)
    plt.title(f"Distribution of {metric.replace('_', ' ').title()}")
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Number of Users")
    plt.show()

import matplotlib.pyplot as plt
importances = [0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05]
plt.bar(["upvotes", "login_streak", "posts", "comments", "quizzes", "buddies", "karma_spent", "karma_earned"], importances)
plt.show()