import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.special import expit
from sklearn.metrics import log_loss
np.random.seed(1234)


def load_train_data():
    trainfile = sys.argv[2]
    data = np.load(trainfile)
    linouts = data['train_linear_outputs']
    y_train = np.load('mnist.npz')['y_train']
    return linouts, y_train

    
def load_test_data():
    testfile = sys.argv[3]
    data = np.load(testfile)
    linouts = data['test_linear_outputs']
    y_test = np.load('mnist.npz')['y_test']
    return linouts, y_test


def forward_pass(W1, b1, x):
    a1 = np.matmul(W1, x)+np.tile(b1, x.shape[1])
    y_hat = expit(a1)
    return a1, y_hat

    
def backward_pass(e, x):
    dW1 = -np.matmul(e, x.T)
    db1 = -np.sum(e, axis=1)
    return dW1, db1[:, np.newaxis]

    
def train(x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-3):
    n_weak_learners = int(sys.argv[1])
    
    W1 = np.random.randn(10, 10*n_weak_learners)
    b1 = np.random.randn(10, 1)
    
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    prev_training_error = 0.
    
    for i in xrange(n_epochs):
        perm = np.random.permutation(x.shape[1])
        x = x[:, perm]
        y = y[:, perm]
        loss = 0.
        train_error = 0.
        for j in xrange(n_batches):
            samples = x[:, j*batch_size:(j+1)*batch_size]
            targets = y[:, j*batch_size:(j+1)*batch_size]
            a1, y_hat = forward_pass(W1, b1, samples)
            error = y_hat - targets
            
            preds = np.argmax(y_hat, axis=0) 
            truth = np.argmax(targets, axis=0)
            train_error += np.sum(preds!=truth)
            loss_on_batch = log_loss(targets, y_hat)
            
            dW1, db1= backward_pass(error, samples)
            W1 += lr*dW1
            b1 += lr*db1
            loss += loss_on_batch
        
        training_error = 1.*train_error/x.shape[1]
        
        print 'Loss at epoch', i+1, ':', loss/x.shape[1]
        print 'Training error:', training_error
        
        #if np.abs(training_error-prev_training_error) <= tol:
        #    break
            
        prev_training_error = training_error
    return W1, b1, training_error


def test(x, y, W1, b1):
    outs = forward_pass(W1, b1, x)[-1]
    preds = np.argmax(outs, axis=0) 
    truth = np.argmax(y, axis=0)
    test_error = 1.*np.sum(preds!=truth)/preds.shape[0]
    return test_error

    
def main():
    train_linear_outputs, y_train = load_train_data()
    W1, b1, training_error = train(train_linear_outputs, y_train, n_epochs=100, lr=1e-4, batch_size=200, tol=1e-6)
    test_linear_outputs, y_test = load_test_data()
    test_error = test(test_linear_outputs, y_test, W1, b1)
    
    print 'Test error:', test_error*100, '%'
    
    
    
if __name__ == '__main__':
    main()