import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        self.params['w1'] = 0.0001 * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = 0.0001 * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.zeros(outputDim)
        pass


    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        X_shape = x.shape[0]
        h = np.dot(x, w1) + b1

        # LeakyReLU Activation 
        h = np.where(h > 0, h, h * 0.01)
        score = np.dot(h, w2) + b2

        score = np.where(score > 0, score, score * 0.01)
        if y is None:
            return score

        s_y = score - np.max(score, axis=1).reshape(-1, 1)

        # Calculating probability
        pb = np.exp(s_y) / np.sum(np.exp(s_y), axis=1).reshape(-1, 1)
        prob = pb[np.arange(X_shape), y]
        prob = prob[prob > 0]
        prob = np.log(prob)

        # Calculating loss
        loss = -np.sum(prob)
        loss /= X_shape        
        
        loss += reg * (np.sum(w1 * w1) + np.sum(w2 * w2))

        # Updating the Weight and Bias by using Probablility
        ds = pb.copy()
        ds[np.arange(X_shape), y] += -1
        ds /= X_shape
        grads['w2'] = h.T.dot(ds) + 2 * reg * w2
        grads['b2'] = np.sum(ds, axis=0)

        dh = np.dot(ds, w2.T)
        dh_ = (h > 0) * dh
        grads['w1'] = x.T.dot(dh_) + 2 * reg * w1
        grads['b1'] = np.sum(dh_, axis=0)

        return loss, grads


    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            new_sample = np.random.choice(x.shape[0], batchSize)
            xBatch = x[new_sample]
            yBatch = y[new_sample]

            loss, new_grads = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            self.params['w1'] += - lr * new_grads['w1']
            self.params['b1'] += - lr * new_grads['b1']
            self.params['w2'] += - lr * new_grads['w2']
            self.params['b2'] += - lr * new_grads['b2']
            
            pass

            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory



    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        z = np.maximum(0, np.dot(x, w1)) + b1
        s = np.dot(z, w2) + b2
        yPred = np.argmax(s, axis=1)
        pass
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        pass
        acc = np.mean(y == (self.predict(x))) * 100
        return acc



