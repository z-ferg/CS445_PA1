"""Pure Python Decision Tree Classifier and Regressor.

Simple binary decision tree classifier and regressor.
Splits for classification are based on Gini impurity. Splits for
regression are based on variance.

Author: CS445 Instructor and Zach Ferguson!
Version:

"""
from collections import namedtuple, Counter
import numpy as np
from abc import ABC

# Named tuple is a quick way to create a simple wrapper class...
Split_ = namedtuple('Split',
                    ['dim', 'pos', 'X_left', 'y_left', 'counts_left',
                     'X_right', 'y_right', 'counts_right'])


class Split(Split_):
    """
    Represents a possible split point during the decision tree
    creation process.

    Attributes:

        dim (int): the dimension along which to split
        pos (float): the position of the split
        X_left (ndarray): all X entries that are <= to the split position
        y_left (ndarray): labels corresponding to X_left
        counts_left (Counter): label counts
        X_right (ndarray):  all X entries that are > the split position
        y_right (ndarray): labels corresponding to X_right
        counts_right (Counter): label counts
    """

    def __repr__(self):
        result = "Split(dim={}, pos={},\nX_left=\n".format(self.dim,
                                                           self.pos)
        result += repr(self.X_left) + ",\ny_left="
        result += repr(self.y_left) + ",\ncounts_left="
        result += repr(self.counts_left) + ",\nX_right=\n"
        result += repr(self.X_right) + ",\ny_right="
        result += repr(self.y_right) + ",\ncounts_right="
        result += repr(self.counts_right) + ")"

        return result


def split_generator(X, y, keep_counts=True):
    """
    Utility method for generating all possible splits of a data set
    for the decision tree construction algorithm.

    :param X: Numpy array with shape (num_samples, num_features)
    :param y: Numpy array with length num_samples
    :param keep_counts: Maintain counters (only useful for classification.)
    :return: A generator for Split objects that will yield all
            possible splits of the data
    """

    # Loop over all of the dimensions.
    for dim in range(X.shape[1]):
        if np.issubdtype(y.dtype, np.integer):
            counts_left = Counter()
            counts_right = Counter(y)
        else:
            counts_left = None
            counts_right = None

        # Get the indices in sorted order so we can sort both data and labels
        ind = np.argsort(X[:, dim])

        # Copy the data and the labels in sorted order
        X_sort = X[ind, :]
        y_sort = y[ind]

        last_split = 0
        # Loop through the midpoints between each point in the
        # current dimension
        for index in range(1, X_sort.shape[0]):

            # don't try to split between equal points.
            if X_sort[index - 1, dim] != X_sort[index, dim]:
                pos = (X_sort[index - 1, dim] + X_sort[index, dim]) / 2.0

                if np.issubdtype(y.dtype, np.integer):
                    flipped_counts = Counter(y_sort[last_split:index])
                    counts_left = counts_left + flipped_counts
                    counts_right = counts_right - flipped_counts

                last_split = index
                # Yield a possible split.  Note that the slicing here does
                # not make a copy, so this should be relatively fast.
                yield Split(dim, pos,
                            X_sort[0:index, :], y_sort[0:index], counts_left,
                            X_sort[index::, :], y_sort[index::], counts_right)


class DecisionTree(ABC):
    """
    A binary decision tree for use with real-valued attributes.

    """

    def __init__(self, max_depth=np.inf):
        """
        Decision tree constructor.

        :param max_depth: limit on the tree depth.
                          A depth 0 tree will have no splits.
        """
        self.max_depth = max_depth
        self._root = None

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy array with length num_samples
        """
        self._root = self.rec_split(X, y) # Set the root of the tree to the recursive split

    def predict(self, X):
        """
        Predict labels for a data set by finding the appropriate leaf node for
        each input and using either the the majority label or the mean value
        as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predictions.
        """
        y_predicts = []
        
        for sample in X:
            cur_node = self._root # For predictions, start at root for each sample
        
            while (cur_node.split is not None): # Go until it hits a leaf
                val = sample[cur_node.split.dim]
                
                if val <= cur_node.split.pos:   # Value should go left
                    if cur_node.left.left is None:  # Left value should be appended
                        if isinstance(self, DecisionTreeClassifier):
                            y_predicts += [cur_node.split.counts_left.most_common()[0][0]]
                        elif isinstance(self, DecisionTreeRegressor):
                            y_predicts += [np.mean(cur_node.split.y_left)]
                            
                    cur_node = cur_node.left
                else:   # Value should go right
                    if cur_node.right.right is None:  # Left value should be appended
                        if isinstance(self, DecisionTreeClassifier):
                            y_predicts += [cur_node.split.counts_right.most_common()[0][0]]
                        elif isinstance(self, DecisionTreeRegressor):
                            y_predicts += [np.mean(cur_node.split.y_right)]
                    
                    cur_node = cur_node.right
        
        return np.array(y_predicts)

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        return self.rec_depth(self._root)
    
    def rec_depth(self, node):
        if node.left is None and node.right is None:
            return 0
        return 1 + max(self.rec_depth(node.left), self.rec_depth(node.right))


class DecisionTreeClassifier(DecisionTree):
    """
    A binary decision tree classifier for use with real-valued attributes.

    """ 
    def impurity(self, y):
        label_count = Counter(y)
        summation = 1
        
        for count in label_count.values():
            summation -= (count/len(y)) ** 2
        
        return summation
    
    def weighted_impurity(self, split):
        total = len(split.y_left) + len(split.y_right)
        
        left_impurity = self.impurity(split.y_left) * (len(split.y_left)/total)
        right_impurity = self.impurity(split.y_right) * (len(split.y_right)/total)
        
        return left_impurity + right_impurity
    
    def rec_split(self, X, y, cur_depth=0):
        same_labels = True  # Check that all labels are same
        for label in y:
            if label != y[0]: 
                same_labels = False
            
        if same_labels: 
            return Node()  # All labels are same, return a leaf
        if cur_depth >= self.max_depth: 
            return Node()  # Maximum depth has been reached
        
        best_split = None
        best_gini = float('inf')
        
        for split in split_generator(X, y):
            if self.weighted_impurity(split) < best_gini:
                best_split = split
                best_gini = self.weighted_impurity(split)
        if best_split is None: 
            return Node()  # No good split was found, return leaf
        node = Node(split=best_split)
        
        node.left = self.rec_split(best_split.X_left, best_split.y_left, cur_depth + 1)
        node.right = self.rec_split(best_split.X_right, best_split.y_right, cur_depth + 1)

        return node


class DecisionTreeRegressor(DecisionTree):
    """
    A binary decision tree regressor for use with real-valued attributes.

    """ 
    def calculate_mse(self, y_left, y_right):
        total = len(y_left) + len(y_right)
        left_mse = (np.mean(y_left - np.mean(y_left)) ** 2) * len(y_left) / total
        right_mse = (np.mean(y_right - np.mean(y_right)) ** 2) * len(y_right) / total
        return left_mse + right_mse
    
    def rec_split(self, X, y, cur_depth=0):
        same_labels = True  # Check that all labels are same
        for label in y:
            if label != y[0]: 
                same_labels = False
            
        if same_labels: 
            return Node()  # All labels are same, return a leaf
        if cur_depth >= self.max_depth: 
            return Node()  # Maximum depth has been reached
        
        best_split = None
        best_mse = float('inf')

        for split in split_generator(X, y):
            if self.calculate_mse(split.y_left, split.y_right) < best_mse:
                best_split = split
                best_mse = self.calculate_mse(split.y_left, split.y_right)
                
        if best_split is None: 
            return Node()  # No good split was found, return leaf
        
        node = Node(split=best_split)
        
        node.left = self.rec_split(best_split.X_left, best_split.y_left, cur_depth + 1)
        node.right = self.rec_split(best_split.X_right, best_split.y_right, cur_depth + 1)
        
        return node


class Node:
    """
    It will probably be useful to have a Node class.  In order to use the
    visualization code in draw_trees, the node class must have three
    attributes:

    Attributes:
        left:  A Node object or Null for leaves.
        right: A Node object or Null for leaves.
        split: A Split object representing the split at this node,
                or Null for leaves
    """
    def __init__(self, left=None, right=None, split=None):
        self.left = left
        self.right = right
        self.split = split


def tree_demo():
    """Simple illustration of creating and drawing a tree classifier."""
    X = np.array([[0.88, 0.39, 0.5],
                  [0.49, 0.52, 0.5],
                  [0.68, 0.26, 0.5],
                  [0.57, 0.51, 0.5],
                  [0.61, 0.73, 0.5]])
    y = np.array([1, 0, 0, 1, 2])
    r_tree = DecisionTreeRegressor(min=4)
    r_tree.fit(X, y)
    print(r_tree._root.split)
    

if __name__ == "__main__":
    tree_demo()