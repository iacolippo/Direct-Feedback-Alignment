import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.special import expit
from sklearn.metrics import log_loss
np.random.seed(1234)


def load_data(key=None):
    data = np.load('mnist.npz')
    if key == 'train':
        X_train = data['X_train']
        y_train = data['y_train']
        return X_train, y_train
    elif key == 'test':
        X_test = data['X_test']
        y_test = data['y_test']
        return X_test, y_test
    else:
        return 'Invalid'


def forward_pass(W1, W2, b1, b2, x):
    '''This is the forward pass. It is equal for any
    training algorithm. It's just one hidden layer
    with tanh activation function and sigmoid on the
    output layer. If the input is a batch, I have to 
    tile as many b1 and b2 as the batch size'''
    a1 = np.matmul(W1, x)+np.tile(b1, x.shape[1])
    h1 = np.tanh(a1)
    a2 = np.matmul(W2, h1)+np.tile(b2, x.shape[1])
    y_hat = expit(a2)
    return a1, h1, a2, y_hat
    
    
def dfa_backward_pass(e, h1, B1, a1, x):
    '''DFA backward pass as in Nokland, 2016
    for weights and biases'''
    dW2 = -np.matmul(e, np.transpose(h1))
    da1 = np.matmul(B1, e)*(1-np.tanh(a1)**2)
    dW1 = -np.matmul(da1, np.transpose(x))
    db1 = -np.sum(da1, axis=1)
    db2 = -np.sum(e, axis=1)
    return dW1, dW2, db1[:,np.newaxis], db2[:,np.newaxis]


def train_weak_learner(x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-3, n_weak_learners=10):
    
    W1, W2 = np.random.randn(800, 784), np.random.randn(10, 800)
    b1, b2 = np.random.randn(800, 1), np.random.randn(10, 1)
    
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    
    weights1 = []
    weights2 = []
    biases1 = []
    biases2 = []
    training_errors = []
    
    for idx in xrange(n_weak_learners):
        print 'Weak learner #', idx+1
        #EACH WEAK LEARNER STARTS FROM THE SAME WEIGHTS BUT HAS A DIFFERENT B1
        B1 = np.random.randn(800, 10)
        W1_idx, W2_idx = np.copy(W1), np.copy(W2)
        b1_idx, b2_idx = np.copy(b1), np.copy(b2)
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
                a1, h1, a2, y_hat = forward_pass(W1_idx, W2_idx, b1_idx, b2_idx, samples)
                error = y_hat - targets
            
                preds = np.argmax(y_hat, axis=0) 
                truth = np.argmax(targets, axis=0)
                train_error += np.sum(preds!=truth)
                loss_on_batch = log_loss(targets, y_hat)
            
                dW1, dW2, db1, db2 = dfa_backward_pass(error, h1, B1, a1, samples)
                W1_idx += lr*dW1
                W2_idx += lr*dW2
                b1_idx += lr*db1
                b2_idx += lr*db2
                loss += loss_on_batch
        
            training_error = 1.*train_error/dataset_size
        
            print 'Loss at epoch', i+1, ':', loss/dataset_size
            print 'Training error:', training_error
        
            if np.abs(training_error-prev_training_error) <= tol or i == n_epochs-1:
                # when the training is finished save Ws, bs and the training error
                weights1.append(W1_idx)
                weights2.append(W2_idx)
                biases1.append(b1_idx)
                biases2.append(b2_idx)
                training_errors.append(training_error)
                break
            
            prev_training_error = training_error
    return weights1, weights2, biases1, biases2, training_errors


def compute_linout(W1, W2, b1, b2, x, batch_size):
    dataset_size = x.shape[1]
    n_batches = dataset_size//batch_size
    linout = []
    for i in xrange(n_batches):
        samples = x[:, i*batch_size:(i+1)*batch_size]
        a2 = forward_pass(W1, W2, b1, b2, samples)[-2]
        linout.append(a2)
        
    return np.hstack(linout)


def test(W1, W2, b1, b2, test_samples, test_targets):
    outs = forward_pass(W1, W2, b1, b2, test_samples)[-1]
    preds = np.argmax(outs, axis=0) 
    truth = np.argmax(test_targets, axis=0)
    test_error = 1.*np.sum(preds!=truth)/preds.shape[0]
    return test_error  
  
    
def train(x, y, n_weak_learners, x_test, y_test):
    testing_errors = []
    batch_size=200
    train_linear_outputs = []
    test_linear_outputs = []
    
        
    # alleno tutti in train_weak_learners
    # restituisco liste di pesi e bias e per
    # quelli in zip mi calcolo gli out che mi
    # servono

    weights1, weights2, biases1, biases2, training_errors = train_weak_learner(x, y, 
                                                         n_epochs=1, lr=1e-3, 
                                                         batch_size=batch_size, tol=1e-3,
                                                         n_weak_learners=n_weak_learners)
    
    counter = 0
    for W1, W2, b1, b2 in zip(weights1, weights2, biases1, biases2):    
        train_linout = compute_linout(W1, W2, b1, b2, x, batch_size=batch_size)
        train_linear_outputs.append(train_linout)
        
        test_linout = compute_linout(W1, W2, b1, b2, x_test, batch_size=batch_size)
        test_linear_outputs.append(test_linout)
        
        test_error = test(W1, W2, b1, b2, x_test, y_test)
    
        #training_errors.append(train_error)
        testing_errors.append(test_error)
        counter +=1
        print 'Testing #:', counter
        
       
    train_linear_outputs = np.vstack(train_linear_outputs)
    test_linear_outputs = np.vstack(test_linear_outputs)
    #tofile:
    np.savez('train_linouts.npz', train_linear_outputs=train_linear_outputs)
    np.savez('test_linouts.npz', test_linear_outputs=test_linear_outputs)
    
    return training_errors, testing_errors


def main(n_weak_learners):   
    x, y = load_data(key='train')
    x_test, y_test =load_data(key='test')
    training_errors, testing_errors = train(x, y, n_weak_learners, x_test, y_test)
    
    print 'TRAINING:', training_errors
    print 'TESTING:', testing_errors
    return
    

if __name__ == '__main__':
    main(int(sys.argv[1]))