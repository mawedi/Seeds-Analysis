from sklearn.neighbors import KNeighborsClassifier

# Training the model on providing the variable k
def train_model(X_train, Y_train, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)

    return model