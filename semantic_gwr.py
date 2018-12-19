# -*- coding: utf-8 -*-
"""
Semantic Gamma-GWR
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal
Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966

Modifications by Vadym Gryshchuk(vadym.gryshchuk@protonmail.com)
Last modified on 15 December 2018
"""

import numpy as np
import math
from RGWR import RGWR


class SemanticGWR(RGWR):

    def __init__(self, dimension, num_weights, num_classes, num_nodes=2) -> None:
        super().__init__(num_nodes, num_weights, num_classes, dimension)
            
    def update_label_histogram(self, bmu, label, *args) -> None:
        """
        Update a label histogram for each neuron and class.
        :param bmu: A BMU.
        :param label: A label.
        :return: An updated label histogram.
        """
        if label != -1:
            for a in range(0, self.numOfClasses):
                if a == label:
                    self.labels_category_histogram[bmu, a] += self.aIncreaseFactor

    def train(self, train_data, actual_training_labels, max_batch_epochs, insertion_threshold, beta, learning_rate_b,
              learning_rate_n, context, regulated, replay=False) -> None:
        """
        Train the model.
        :param replay:
        :param train_data: Data for training
        :param actual_training_labels: Actual labels.
        :param max_batch_epochs: Max training iterations for a batch.
        :param insertion_threshold: AN insertion threshold.
        :param beta: Beta.
        :param learning_rate_b: Learning rate of the BMU.
        :param learning_rate_n: Learning rate of the neighbours of the BMU.
        :param context: A boolean value, which is used to indicate whether the context should be used or not.
        :param regulated: A boolean value, which is used to indicate whether the network is regulated or not.
        """
        previous_bmu, previous_index, cu_qrror, cu_fcounter = \
            self.initialize_training(train_data, actual_training_labels, max_batch_epochs, insertion_threshold, beta,
                                     learning_rate_b, learning_rate_n, context, regulated)

        # Start training.
        for epoch in range(0, self.max_batch_epochs):

            for iteration in range(0, self.number_train_samples):

                self.globalContext[0] = self.train_data[iteration]
                actual_label = self.actual_labels[iteration]

                self.update_global_context(previous_bmu)

                distances = self.find_bmu_distances()
                first_index, second_index = self.find_bmu_indices(distances)
                winner_label_index = np.argmax(self.labels_category_histogram[first_index])
        
                previous_bmu[0] = self.recurrent_weights[first_index]
                self.ages += 1
                
                # Compute network activity
                cu_qrror[iteration] = distances[first_index]
                h = self.habituation[first_index]
                cu_fcounter[iteration] = h

                if ((not self.regulated) and (math.exp(-distances[first_index]) < self.insertion_threshold) and
                    (h < self.habThreshold) and (self.num_nodes < self.maxNodes)) or \
                        (self.regulated and (actual_label != winner_label_index) and
                         (h < self.habThreshold) and (self.num_nodes < self.maxNodes)):
                    # Add new neuron.
                    self.add_new_neuron(first_index, second_index, actual_label)
                else:
                    self.update_spatiotemporal_structure(first_index, second_index, actual_label, winner_label_index)
                
                # Update temporal connections    
                if (previous_index != -1) and (previous_index != first_index):
                    self.temporal[previous_index, first_index] += 1
                previous_index = first_index
            # Remove old edges
            self.remove_old_edges()
            self.compute_additional_info(epoch, cu_qrror, cu_fcounter, "Semantic", replay)

        # Remove neurons
        # if context and not replay:
        #     self.remove_weak_neurons("SM", replay)
        # elif replay:
        #     self.remove_weak_neurons("SM", replay)

    def remove_neuron_label(self, neuron_index):
        self.labels_category_histogram = np.delete(self.labels_category_histogram, neuron_index, axis=0)

    # Test GWR ################################################################
    def predict(self, data_set, use_context) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Predicts BMU data for a given data set.
        :param data_set:  Data set.
        :param use_context: A boolean, which states whether to use the temporal context or not.
        :return: BMU parameters.
        """
        distances = np.zeros(self.num_nodes)  # Initialize the distances to all nodes with 0.
        number_of_samples = data_set.shape[0]  # The number of samples.
        bmu_weights = np.zeros((number_of_samples, self.dimension))  # BMU weights for each sample.
        bmu_activation = np.zeros(number_of_samples)  # An activations of the BMU for each sample.
        bmu_label_classes = -np.ones(number_of_samples)  # Initialize BMU labels with -1, as all real labels are >= 0.
        input_context = np.zeros((self.temporal_window_size, self.dimension))  # Temporal window.
        
        if use_context:
            for sample_index in range(0, number_of_samples):
                input_context[0] = data_set[sample_index]

                distances = self.find_bmu_distances(input_context=input_context, use_gamma_weights=0)

                smallest_distance_index = np.argmin(distances)
                bmu_weights[sample_index] = self.recurrent_weights[smallest_distance_index, 0]
                bmu_activation[sample_index] = math.exp(-distances[smallest_distance_index])
                bmu_label_classes[sample_index] = np.argmax(self.labels_category_histogram[smallest_distance_index, :])

                for i in range(1, self.temporal_window_size):
                    input_context[i] = input_context[i-1]
        else:
            for sample_index in range(0, number_of_samples):
                input_sample = data_set[sample_index]
                
                for i in range(0, self.num_nodes):
                    distances[i] = np.linalg.norm(input_sample - self.recurrent_weights[i, 0])
                    
                smallest_distance_index = np.argmin(distances)
                bmu_weights[sample_index] = self.recurrent_weights[smallest_distance_index, 0]
                bmu_activation[sample_index] = math.exp(-distances[smallest_distance_index])
                bmu_label_classes[sample_index] = np.argmax(self.labels_category_histogram[smallest_distance_index, :])
            
        return bmu_weights, bmu_activation, bmu_label_classes
