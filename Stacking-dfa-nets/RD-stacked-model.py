# RD stacking

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


def forward_pass(D1, R, D2, x):
    a1 = np.matmul(D1, x) # 784 x bs = 784 x 784 x 784 x bs
    a2 = np.matmul(R, a1) # 10 x bs = 10 x 784 x 784 x bs - R not trainable
    a3 = np.matmul(D2, a2) # 10 x bs = 10 x 10 x 10 x bs
    y_hat = expit(a3) # 10 x bs
    return a1, a2, a3, y_hat

    
def backprop(R, D2, e, x, a2):
    # THIS IS WORKING...
    int1 = np.diag(D2)[:, np.newaxis]*e
    int2 = np.dot(R.T, int1)
    d1 = np.mean(x*int2, axis=1)
    d2 = (a2*e)[:, 0]
    dD1 = -np.diag(d1) #-np.matmul(np.dot(B1, e), x.T)
    dD2 = -np.diag(d2) #-np.matmul(e, a2.T)
    return dD1, dD2
    

def backward_pass(B1, e, x, a2): # DFA
    # THIS IS NOT...
    d1 = (x*np.dot(B1, e))[:, 0]
    d2 = (a2*e)[:, 0]
    dD1 = -np.diag(d1)
    dD2 = -np.diag(d2)
    return dD1, dD2
  
  
def train(x, y, n_epochs=10, lr=1e-4, batch_size=200, tol=1e-3):
    n_weak_learners=int(sys.argv[1])
    D1 = np.diag(np.random.randn(10*n_weak_learners))
    R = np.random.randn(10, 10*n_weak_learners)
    D2 = np.diag(np.random.randn(10))
    
    B1 = np.random.randn(10*n_weak_learners, 10)
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
            a1, a2, a3, y_hat = forward_pass(D1, R, D2, samples)
            error = y_hat - targets
            
            preds = np.argmax(y_hat, axis=0) 
            truth = np.argmax(targets, axis=0)
            train_error += np.sum(preds!=truth)
            loss_on_batch = log_loss(targets, y_hat)
            
            dD1, dD2 = backprop(R, D2, error, samples, a2) # backward_pass(B1, error, samples, a2)
            D1 += lr*dD1
            D2 += lr*dD2
            loss += loss_on_batch
        
        training_error = 1.*train_error/dataset_size
        
        print 'Loss at epoch', i+1, ':', loss/dataset_size
        print 'Training error:', training_error
        
        if np.abs(training_error-prev_training_error) <= tol:
            break
            
        prev_training_error = training_error
    return D1, R, D2, training_error


def test(x, y, D1, R, D2):
    outs = forward_pass(D1, R, D2, x)[-1]
    preds = np.argmax(outs, axis=0) 
    truth = np.argmax(y, axis=0)
    test_error = 1.*np.sum(preds!=truth)/preds.shape[0]
    return test_error

    
def main():
    train_linear_outputs, y_train = load_train_data()
    D1, R, D2, training_error = train(train_linear_outputs, y_train, n_epochs=200, lr=1e-3, batch_size=200, tol=1e-7)
    test_linear_outputs, y_test = load_test_data()
    test_error = test(test_linear_outputs, y_test, D1, R, D2)
    print 'Test error:', test_error*100, '%'
   
    
if __name__ == '__main__':
    main()