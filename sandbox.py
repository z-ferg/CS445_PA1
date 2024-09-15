import numpy as np

X_test = np.array([[1.6, 3.0, 4.0],
                   [1.5, 2.5, 5.0],
                   [2.0, 4.0, 2.1]])
y_test = np.array([1, 0, 1])

if __name__ == "__main__":
    print(X_test)
    print(X_test[:, 1]) # Vertical slicing at column *array arg 2*
    print(y_test[1:]) # Horizontal slicing from 0 to arg (inclusive)
    print(y_test[:1]) # Horizontal slicing from right of arg (2 in this case)