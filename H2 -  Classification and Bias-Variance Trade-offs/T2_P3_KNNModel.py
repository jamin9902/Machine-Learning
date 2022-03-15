from turtle import distance
import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    def calc_distance(self, pred, x):
        return (((pred[0] - x[0]) / 3.0) ** 2.0) + ((pred[1] - x[1]) ** 2.0)

    def predict(self, X_pred):
        predictions = []
        kcount = 0
        for pred in X_pred:
            k_nearest = []
            distances = []
            for i in range(self.X.shape[0]):
                distances.append(self.calc_distance(pred, self.X[i]))
            for k in range(self.K):
                min_dist = None
                nearest_index = None
                for i in range(len(distances)):
                    if i not in k_nearest:
                        if nearest_index is None or distances[i] <= min_dist:                        
                            min_dist = distances[i]
                            nearest_index = i
                k_nearest.append(nearest_index)

            classes = []
            for i in k_nearest:
                classes.append(self.y[i])
            
            count = 0
            class_pred = classes[0]
            for i in classes:
                curr = classes.count(i)
                if curr > count:
                    count = curr
                    class_pred = i

            predictions.append(class_pred)
        
        return np.array(predictions)


        
    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y