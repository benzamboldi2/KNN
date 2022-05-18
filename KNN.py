# Import libraries
import pandas as pd
import math

# load the data
trainKNN = pd.read_csv('KNN_train.csv')
validateKNN = pd.read_csv('KNN_valid.csv')
testKNN = pd.read_csv('KNN_test.csv')

class KNN:
    
    # Initialize train data, test data, and K
    def __init__(self, trainData, testData, K):
        self.train = trainData
        self.trainLabs = self.train['Labels']
        self.train = self.train.drop(['Labels'], axis=1)
        self.test = testData
        self.K = K

    # Method to calculate euclidean distance between two vectors
    def euclidean(self, point1, point2):
        sqDistance = 0
        for point in range(len(point1)):
            diff = point2[point] - point1[point]
            sqDistance = sqDistance + (diff * diff)
        return math.sqrt(sqDistance)

    # Method to get the K nearest neighbors of a point
    def calculate_neighbors(self, point_i):
        dists = []
        for index, row in self.train.iterrows():
            row_i = [row['SepalLengthCm'], row['SepalWidthCm'], row['PetalLengthCm'], row['PetalWidthCm']]
            dists.append((index, self.euclidean(point_i, row_i)))
        dists.sort(key=lambda tup: tup[1])
        
        neighbors = []
        for i in range(self.K):
            neighbors.append(dists[i][0])
        
        return neighbors

    # Methos to classify point based on majority voting
    def classify(self, point_i):
        neighbors = self.calculate_neighbors(point_i)
        labs = [self.trainLabs[i] for i in neighbors]
        prediction = max(set(labs), key=labs.count)
        return prediction

    # Driving method to classify each point
    def run(self):
        predictions = []
        for index, row in self.test.iterrows():
            row_i = [row['SepalLengthCm'], row['SepalWidthCm'], row['PetalLengthCm'], row['PetalWidthCm']]
            output = self.classify(row_i)
            predictions.append(output)
        return(predictions)

# Main method
if __name__ == '__main__':
    trainKNN = trainKNN.drop(['Id'], axis=1)
    test20 = testKNN.drop(['Labels'], axis=1)
    test30 = testKNN.drop(['Labels'], axis=1)
    test40 = testKNN.drop(['Labels'], axis=1)
    test50 = testKNN.drop(['Labels'], axis=1)
    testKNN = testKNN.drop(['Id', 'Labels'], axis=1)

    # KNN with K=20
    KNN20 = KNN(trainKNN, testKNN, 20)
    preds20 = KNN20.run()
    test20 = pd.concat((test20, pd.Series(preds20, name='Labels')), axis=1)
    print(test20)

    # KNN with K=30
    KNN30 = KNN(trainKNN, testKNN, 30)
    preds30 = KNN30.run()
    test30 = pd.concat((test30, pd.Series(preds30, name='Labels')), axis=1)
    print(test30)

    # KNN with K=40
    KNN40 = KNN(trainKNN, testKNN, 40)
    preds40 = KNN40.run()
    test40 = pd.concat((test40, pd.Series(preds40, name='Labels')), axis=1)
    print(test40)

    # KNN with K=50
    KNN50 = KNN(trainKNN, testKNN, 50)
    preds50 = KNN50.run()
    test50 = pd.concat((test50, pd.Series(preds50, name='Labels')), axis=1)
    print(test50)