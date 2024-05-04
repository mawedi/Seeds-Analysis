from preparation import prepare_data as pd
from models import train_data as td
from models import evaluate_model as ca

# Importing the dataset
data_path = '../data/seeds_dataset.csv'
df = pd.import_dataset(data_path)

# Verifying the dataset
pd.verify_dataset(df)

# Repairing Classes
pd.repair_class(df)

# Splitting the dataset to features and targets
features, targets = pd.split_dataset(df)

# Scaling the features
scaled_features = pd.scale_features(features)

# Splitting the dataset to training and testing
X_train, X_test, Y_train, Y_test = pd.split_dataset_to_train_and_test(scaled_features, targets)

# Training the data using the KNN algorithm by providing the number of neighbors
knn_model = td.train_data_with_KNN_model(X_train, Y_train, 5)

# Calculating the accuracy of the knn model and the confusion matrix
accuracy_of_knn_model = ca.calculate_model_accuracy(knn_model, X_test, Y_test)
matrix = ca.calculate_confusion_matrix(knn_model, X_test, Y_test)
print(f"Accuracy of the KNN model: {accuracy_of_knn_model}")
print(f"Confusion Matrix of the KNN model: {matrix}")

# Training the data using the multi layer perceptron algorithm
mlp_model = td.train_data_with_MLP_model(X_train, Y_train)

# Calculating the accuracy of the mlp model 
accuracy_of_mlp_model = ca.calculate_model_accuracy(knn_model, X_test, Y_test)
matrix = ca.calculate_confusion_matrix(knn_model, X_test, Y_test)
print(f"Accuracy of the MLP model: {accuracy_of_mlp_model}")
print(f"Confusion Matrix of the MLP model: {matrix}")

# Chossing the best params with grid search
params = td.choose_best_parameters_with_grid_search(X_train, Y_train)
C = params.get('C')
gamma = params.get('gamma')

# Training the data using the SVM algorithm on prviding the best params and using the rbf kernel
svm_rbf_model = td.train_data_with_SVM_model_using_rbf_kernel(X_train, Y_train, C, gamma)

# Calculating the accuracy of the svm model using rbf kernel
accuracy_of_svm_rbf_model = ca.calculate_model_accuracy(knn_model, X_test, Y_test)
matrix = ca.calculate_confusion_matrix(knn_model, X_test, Y_test)
print(f"Accuracy of the SVM model using rbf kernel: {accuracy_of_svm_rbf_model}")
print(f"Confusion Matrix of the SVM model using rbf kernel: {matrix}")

# Training the data using the SVM algorithm on prviding the best params and using the poly kernel
svm_rbf_model = td.train_data_with_SVM_model_using_poly_kernel(X_train, Y_train, C, gamma)

# Calculating the accuracy of the svm model using poly kernel\
accuracy_of_svm_poly_model = ca.calculate_model_accuracy(knn_model, X_test, Y_test)
matrix = ca.calculate_confusion_matrix(knn_model, X_test, Y_test)
print(f"Accuracy of the SVM model using poly kernel: {accuracy_of_svm_poly_model}")
print(f"Confusion Matrix of the SVM model using poly kernel: {matrix}")

# Training the data using the SVM algorithm on prviding the best params and using the sigmoid kernel
svm_rbf_model = td.train_data_with_SVM_model_using_sigmoid_kernel(X_train, Y_train, C, gamma)

# Calculating the accuracy of the svm model using sigmoid kernel
accuracy_of_svm_sigmoid_model = ca.calculate_model_accuracy(knn_model, X_test, Y_test)
matrix = ca.calculate_confusion_matrix(knn_model, X_test, Y_test)
print(f"Accuracy of the SVM model using sigmoid kernel: {accuracy_of_svm_sigmoid_model}")
print(f"Confusion Matrix of the SVM model using sigmoid kernel: {matrix}")