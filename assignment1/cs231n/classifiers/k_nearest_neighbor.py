import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X,whatDist = 'L2'):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.
    - whatDist : select the kind of distance. (L1,L2)

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    
    print("num test shape : ",self.X_train.shape)
    print("num_train shape : ", X.shape)
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    print("num test : ",num_test)
    print("num_train : ", num_train)
    print("dists shape : ", dists.shape)
    temp = int(num_test / 10)
    
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #L2 distance
        if whatDist == 'L1':
          temp_dist = np.sum(np.abs(self.X_train[j,:]-X[i,:]))
          dists[i,j] = temp_dist
        elif whatDist == 'L2':
          temp_dist = np.sqrt(np.sum((self.X_train[j,:]-X[i,:])**2))
          dists[i,j] = temp_dist
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
      if i % temp ==0:
        print('[{}{}] {}/10'.format('#'*(int(i/temp)+1),'.'*(int(10-(i/temp+1))),str((int(i/temp)+1))))

                
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    temp = int(num_test / 10)
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # test 한번당 모든 training data 돌려주면 된다.
      temp_dist = np.sqrt(np.sum( (self.X_train - X[i])**2, axis =1))
      #print('test data i : ',X[i].shape)
      #print('test data  : ',X.shape)
      #print('train data : ',np.sum(self.X_train,axis =1).shape)
      #print(np.sum((self.X_train - X[i])**2,axis =1).shape)
      dists[i,:] = temp_dist
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
      if i % temp ==0:
        print('[{}{}] {}/10'.format('#'*(int(i/temp)+1),'.'*(int(10-(i/temp+1))),str((int(i/temp)+1))))
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    testSquared = np.sum(X**2,axis =1)
    trainSquared = np.sum(self.X_train **2,axis = 1)
    testTrain = np.dot(X, self.X_train.T)
    
    #print('sum of train : ',trainSquared.shape)
    #print('sum of test : ' ,testSquared.shape)
    #print('sum of : ', (testSquared[:,np.newaxis]).shape)
    #print('sum of : ', (testSquared[:,np.newaxis] + trainSquared).shape)
    # X^2 + Y^2 -2XY = L2
    #newaxis : https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
    dists = np.sqrt(testSquared[:,np.newaxis] + trainSquared - (2*testTrain))
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test): ## loop in number of test dataset
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #argsort = return list of sorted idx.
      # [32,2,1,87] => [2,1,0,3]
      # 얼마나 가까운 값인지 알 수 있다.
      # training set => 비교
      temp_dist = dists[i,:] # get i'th rows
      sorted_dist = np.argsort(temp_dist) # get sorted idx.
      closest_y = self.y_train[sorted_dist[0:k]] # y is label of x (mapping)
      
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = np.argmax(np.bincount(closest_y)) #max classified
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

