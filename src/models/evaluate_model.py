from sklearn.metrics import f1_score, confusion_matrix


# Calculating the accuracy of the model KNN
def calculate_model_accuracy(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    accuracy = f1_score(Y_test, Y_pred, average='weighted')

    return accuracy


# Calculating the confusion matrix
def calculate_confusion_matrix(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test, Y_pred)

    return matrix