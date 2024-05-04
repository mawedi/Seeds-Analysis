from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Training the model on providing the variable k
def train_data_with_KNN_model(X_train, Y_train, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)

    return model

# Traning the model using the multi layer perceptron algorithm
def train_data_with_MLP_model(X_train, Y_train):
    # We set the max iteration to 550 to avoid the warning of Maximum iterations reached and the optimization hasn't converged yet.
    model = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=550, activation ='relu', solver='adam', random_state=42)
    model.fit(X_train, Y_train)
    
    return model


# Initializing the parameters for the grid search
def initialize_parameters_for_grid_search():
    return {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    }


# Choosing the best parameters with Greed Search
def choose_best_parameters_with_grid_search(X_train, Y_train):
    parameters = initialize_parameters_for_grid_search()

    grid = GridSearchCV(SVC(), parameters, refit=True, verbose=3)
    grid.fit(X_train, Y_train)
    
    return grid.best_params_


# Training the model using the SVM algorithm using rbf kernel
def train_data_with_SVM_model_using_rbf_kernel(X_train, Y_train, C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(X_train, Y_train)

    return model


# Training the model using the SVM algorithm using poly kernel
def train_data_with_SVM_model_using_poly_kernel(X_train, Y_train, C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='poly')
    model.fit(X_train, Y_train)

    return model


# Training the model using the SVM algorithm using sigmoid kernel
def train_data_with_SVM_model_using_sigmoid_kernel(X_train, Y_train, C, gamma):
    model = SVC(C=C, gamma=gamma, kernel='sigmoid')
    model.fit(X_train, Y_train)

    return model