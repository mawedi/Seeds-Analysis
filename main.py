from DataCleaning import DataPreparation as dp
from DataTraining import DataTraining as dt

# Importing the dataset
data_path = 'Dataset/seeds_dataset.csv'
df = dp.import_dataset(data_path)

# Verifying the dataset
dp.verify_dataset(df)

# Repairing Classes
dp.repair_class(df)

# Separating the dataset to features and targets
dp.split_dataset(df)

# Splitting the dataset to features and targets
features, targets = dp.split_dataset(df)

# Scaling the features
scaled_features = dp.scale_features(features)

# Splitting the dataset to training and testing
X_train, X_test, Y_train, Y_test = dp.split_dataset_to_train_and_test(scaled_features, targets)

# Training the model using the KNN algorithm by providing the number of neighbors
model = dt.train_model(X_train, Y_train, 5)