# -*- coding: utf-8 -*-
"""
Episodic Gamma-GWR
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of
Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966

Modifications by Vadym Gryshchuk(vadym.gryshchuk@protonmail.com)
Last modified on 15 December 2018
"""

import numpy as np
import math

from numba import jit
from RGWR import RGWR


class EpisodicGWR(RGWR):
    """ Episodic memory. """

    def __init__(self, dimension, num_weights, num_classes, num_instances, num_nodes=2) -> None:
        """ Initializes the object. """
        self.num_of_instances = num_instances
        self.label_instances_histogram = np.zeros((num_nodes, self.num_of_instances))

        super().__init__(num_nodes, num_weights, num_classes, dimension)
            
    def update_label_histogram(self, bmu_index, label_class, label_instance) -> None:
        """
        Update the label histogram.
        :param bmu_index:
        :param label_class:
        :param label_instance:
        """
        for label in range(0, self.numOfClasses):
            if label == label_class:
                self.labels_category_histogram[bmu_index, label] += self.aIncreaseFactor
                    
        for label in range(0, self.num_of_instances):
            if label == label_instance:
                self.label_instances_histogram[bmu_index, label] += self.aIncreaseFactor

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

            for sample_index in range(0, self.number_train_samples):

                self.globalContext[0] = self.train_data[sample_index]  # Save the sample to global_context[0].
                
                category_label = self.actual_labels[sample_index, 0]
                instance_label = self.actual_labels[sample_index, 1]
                
                self.update_global_context(previous_bmu)

                distances = self.find_bmu_distances()
                first_index, second_index = self.find_bmu_indices(distances)
                winner_label_index = np.argmax(self.label_instances_histogram[first_index, :])

                previous_bmu[0] = self.recurrent_weights[first_index]
                self.ages += 1
                
                # Compute network activity
                cu_qrror[sample_index] = distances[first_index]
                h = self.habituation[first_index]
                cu_fcounter[sample_index] = h

                if (not self.regulated and math.exp(-distances[first_index]) < self.insertion_threshold and
                    h < self.habThreshold and self.num_nodes < self.maxNodes) or \
                        (self.regulated and instance_label != winner_label_index and
                         h < self.habThreshold and self.num_nodes < self.maxNodes):

                    # Add a new neuron.
                    self.add_new_neuron(first_index, second_index, category_label)
                    # Set additionally a new label for an instance.
                    self.set_new_instance_label(instance_label)

                    self.neuron_significance.resize(self.num_nodes, refcheck=False)

                else:
                    self.update_spatiotemporal_structure(first_index, second_index, category_label, winner_label_index,
                                                         instance_label)

                # Calculate a neuron's significance.
                # self.calculate_neuron_significance(first_index)

                # Update temporal connections (synaptic links), which are required for Replay.
                if (previous_index != -1) and (previous_index != first_index):
                    self.temporal[previous_index, first_index] += 1
                previous_index = first_index

            # Remove old edges
            self.remove_old_edges()

            self.compute_additional_info(epoch, cu_qrror, cu_fcounter, "Episodic", replay)
                
        # Remove neurons
        # if context and not replay:
        #     self.remove_weak_neurons("EM", replay)
        # elif replay:
        #     self.remove_weak_neurons("EM", replay)

    def calculate_neuron_significance(self, neuron_index) -> None:
        """

        :param neuron_index:
        """
        neighbours = np.nonzero(self.edges[neuron_index])
        min_neighbour_age = self.max_age

        for neighbour in neighbours[0]:
            neighbour_age = self.ages[neuron_index, neighbour]
            if neighbour_age < min_neighbour_age:
                min_neighbour_age = neighbour_age

        self.neuron_significance[neuron_index] = np.exp(-min_neighbour_age)

    def remove_neuron_label(self, neuron_index):
        self.labels_category_histogram = np.delete(self.labels_category_histogram, neuron_index, axis=0)
        self.label_instances_histogram = np.delete(self.label_instances_histogram, neuron_index, axis=0)

    # Memory replay ################################################################
    def replay_data(self, pseudo_size) -> (np.ndarray, np.ndarray):
        """

        :param pseudo_size:
        :return:
        """
        samples = np.zeros(pseudo_size)
        replay_weights = np.zeros((self.num_nodes, pseudo_size, self.dimension))
        replay_labels = np.zeros((self.num_nodes, pseudo_size, 2))

        for i in range(0, self.num_nodes):
            for r in range(0, pseudo_size):
                if r == 0:
                    samples[r] = i
                else:
                    samples[r] = np.argmax(self.temporal[int(samples[r - 1]), :])
                replay_weights[i, r] = self.recurrent_weights[int(samples[r]), 0]  # Get the neuron's weights.
                replay_labels[i, r, 0] = np.argmax(self.labels_category_histogram[int(samples[r])])
                replay_labels[i, r, 1] = np.argmax(self.label_instances_histogram[int(samples[r])])

        return replay_weights, replay_labels

    def set_new_instance_label(self, actual_label) -> None:
        """
        Set a new instance label, when a new neuron is created.
        :param actual_label: An actual label.
        """
        new_label_instance = np.zeros((1, self.num_of_instances))
        if actual_label != -1:
            new_label_instance[0, int(actual_label)] = self.aIncreaseFactor
        self.label_instances_histogram = np.concatenate((self.label_instances_histogram, new_label_instance), axis=0)
    
    # Test GWR ###################################################################
    def predict(self, data_set, use_context) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Predict on new data.
        :param data_set: Data.
        :param use_context: A boolean value, which states whether a context should be used or not.
        """
        distances = np.zeros(self.num_nodes)
        num_samples = data_set.shape[0]
        bmu_weights = np.zeros((num_samples, self.dimension))
        bmu_activation = np.zeros(num_samples)
        bmu_label_classes = -np.ones(num_samples)
        bmu_label_instances = -np.ones(num_samples)
        input_context = np.zeros((self.temporal_window_size, self.dimension))
        
        if use_context:
            for ti in range(0, num_samples):
                input_context[0] = data_set[ti]

                distances = self.find_bmu_distances(input_context)

                first_index = np.argmin(distances)
                bmu_weights[ti] = self.recurrent_weights[first_index, 0]
                bmu_activation[ti] = math.exp(-distances[first_index])
                bmu_label_classes[ti] = np.argmax(self.labels_category_histogram[first_index, :])
                bmu_label_instances[ti] = np.argmax(self.label_instances_histogram[first_index, :])

                for i in range(1, self.temporal_window_size):
                    input_context[i] = input_context[i-1]
        else:
            for ti in range(0, num_samples):
                input_sample = data_set[ti]
                
                for i in range(0, self.num_nodes):
                    distances[i] = np.linalg.norm(input_sample - self.recurrent_weights[i, 0])
                    
                first_index = np.argmin(distances)
                bmu_weights[ti] = self.recurrent_weights[first_index, 0]
                bmu_activation[ti] = math.exp(-distances[first_index])
                bmu_label_classes[ti] = np.argmax(self.labels_category_histogram[first_index, :])
                bmu_label_instances[ti] = np.argmax(self.label_instances_histogram[first_index, :])
            
        return bmu_weights, bmu_activation, bmu_label_classes, bmu_label_instances
