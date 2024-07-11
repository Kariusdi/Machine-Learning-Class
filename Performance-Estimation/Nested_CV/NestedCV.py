<<<<<<< HEAD
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor  # Example model
import pandas as pd

# Load data from CSV file
df = pd.read_csv("data.csv")

# Assuming your features are in columns 'feature1', 'feature2', etc.
X = df[['feature1', 'feature2', ...]]  # Features (input variables)
y = df['target']  # Target variable (output/response variable)


# Define the outer K-Fold cross-validation
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize an empty list to store outer loop scores
outer_scores = []

for train_index, test_index in outer_kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize an inner K-Fold cross-validation
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize an empty list to store inner loop scores
    inner_scores = []

    for inner_train_index, inner_val_index in inner_kf.split(X_train):
        X_inner_train, X_val = X_train[inner_train_index], X_train[inner_val_index]
        y_inner_train, y_val = y_train[inner_train_index], y_train[inner_val_index]

        # Train your model (e.g., RandomForestRegressor) on X_inner_train, y_inner_train
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_inner_train, y_inner_train)

        # Evaluate the model on the validation set
        score = model.score(X_val, y_val)
        inner_scores.append(score)

    # Compute the mean inner loop score
    mean_inner_score = np.mean(inner_scores)
    outer_scores.append(mean_inner_score)

# Compute the mean outer loop score
final_score = np.mean(outer_scores)
print(f"Nested Cross-Validation Score: {final_score:.4f}")
=======
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import pandas as pd

df = pd.read_csv("HeightWeight.csv")
>>>>>>> dff1f8f (add csv)
