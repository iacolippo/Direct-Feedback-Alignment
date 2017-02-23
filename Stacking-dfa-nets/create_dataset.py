import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
np.random.seed(1234)

def create_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    X_train = X_train.reshape(60000, 28*28)
    X_test = X_test.reshape(10000, 28*28)

    nb_classes = 10
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    
    return (X_train, y_train), (X_test, y_test)

def main():
    (X_train, y_train), (X_test, y_test) = create_dataset()
    np.savez('mnist.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return
    

if __name__ == '__main__':
    main()