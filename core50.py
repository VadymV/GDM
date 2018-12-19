# -*- coding: utf-8 -*-
"""
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)

Modifications by Vadym Gryshchuk(vadym.gryshchuk@protonmail.com)
Last modified on 15 December 2018
"""

import numpy as np
import pandas as pd
from numba import jit


@jit
def normalize_data(data, columns):
    new_data = np.copy(data)
    for i in range(0, columns):
        max_column = max(data[:, i])
        min_column = min(data[:, i])
        for j in range(0, data.shape[0]):
            new_data[j, i] = (data[j, i] - min_column) / (max_column - min_column)
    return new_data

class CORe50:

    def read_file(self, fn) -> np.ndarray:
        data = pd.read_csv(fn, header=None, dtype=np.float64, sep=',')
        return data.values
    
    def find_category_indices(self, vector_set) -> np.ndarray:
        c = 0
        segData = 0
        segIndex = 0
        indices = np.zeros((self.numClasses, self.numLabels))

        for i in range(0, vector_set.shape[0]):
            if (vector_set[i, self.categoryLabelIndex] != segData) or (i == vector_set.shape[0] - 1):
                indices[c, 0] = segIndex
                indices[c, 1] = i
                c += 1
                segData = vector_set[i, self.categoryLabelIndex]
                segIndex = i
                
        return indices
        
    def prepare_data(self) -> None:
        # numClasses and numInstances are set here for simplicity but the
        # model does not require these values to be fixed
        self.sName = 'CORe50'
        self.numClasses = 10
        self.numInstances = 50
        self.numLabels = 2
        self.vectorIndex  = [0, 256]
        self.sessionColumnIndex = 257
        self.categoryLabelIndex = 258
        self.instanceLabelIndex = 259

        original_data = self.read_file("reduced_features_instance_based_ResNet50.csv")
        reduced_data = original_data[original_data[:, 256] < 1000]
        # prepared_data = normalize_data(reduced_data, 256)
        prepared_data = reduced_data

        sorted_data = None
        for category in range(0, self.numClasses):
            category_data = prepared_data[prepared_data[:, self.categoryLabelIndex] == category]
            if category == 0:
                sorted_data = category_data
            else:
                sorted_data = np.concatenate((sorted_data, category_data), axis=0)

        trainingSet = sorted_data[np.in1d(sorted_data[:, self.sessionColumnIndex], [1])]
        #trainingSet = sorted_data[np.in1d(sorted_data[:, self.sessionColumnIndex], [1, 2, 4, 5, 6, 8, 9, 11])]
        trainingSet = self.reduce_number_of_frames(trainingSet, 4)
        #testSet = sorted_data[np.in1d(sorted_data[:, self.sessionColumnIndex], [3, 7, 10])]
        testSet = sorted_data[np.in1d(sorted_data[:, self.sessionColumnIndex], [3])]

        self.train_category_indices = self.find_category_indices(trainingSet)
        self.test_categoty_indices = self.find_category_indices(testSet)
                
        # Pre-process samples and labels # 0 - classes (258), 1 - instances (259)
        self.trainingVectors = trainingSet[:, self.vectorIndex[0]:self.vectorIndex[1]]
        self.trainingLabels = trainingSet[:, self.categoryLabelIndex:(self.categoryLabelIndex+self.numLabels)]
        self.vecDim = self.trainingVectors.shape[1]

        self.testVectors = testSet[:, self.vectorIndex[0]:self.vectorIndex[1]]
        self.testLabels = testSet[:, self.categoryLabelIndex:(self.categoryLabelIndex+self.numLabels)]

    @staticmethod
    def reduce_number_of_frames(data, factor):
        """
        Delete frames.
        :param data:  Data.
        :param factor: An even integer value. Max is 6.
        :return:
        """
        assert factor == 2 or 4 or 6
        iterations = factor / 2
        for i in range(0, int(iterations)):
            data = np.delete(data, list(range(0, data.shape[0], 2)), axis=0)
        return data