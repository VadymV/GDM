# -*- coding: utf-8 -*-
"""
Most of this code is based on the code from episodic_gwr.py and semantic_gwr.py.
Modifications by Vadym Gryshchuk(vadym.gryshchuk@protonmail.com)
Last modified on 15 December 2018
"""
import numpy as np
import abc
import heapq
from numba import jit


@jit
def compute_threshold_weak_neurons(habituation, replay):
    if not replay:
        return np.mean(habituation, axis=0) + np.std(habituation, axis=0)
    else:
        return np.exp(-np.mean(habituation, axis=0))


@jit
def find_neuron_distances(num_nodes, num_weights, gamma_weights, context, recurrent_weights, distances,
                          use_gamma_weights=1):
    """
    Calculates the distances between the neurons and the input.
    Numba library (https://numba.pydata.org/) is used to speed up the performance during the training.
    :param use_gamma_weights:
    :param num_nodes: The number of nodes.
    :param num_weights: The temporal windows size.
    :param gamma_weights: Gamma weights.
    :param context: A global context.
    :param recurrent_weights: Recurrent weights.
    :param distances: Distances.
    """

    for i in range(0, num_nodes):
        gamma_distance = 0.0
        for j in range(0, num_weights):
            if use_gamma_weights:
                alpha = gamma_weights[j]
            else:
                alpha = 1
            gamma_distance += (alpha * (np.sqrt(np.sum((context[j] - recurrent_weights[i, j]) ** 2))))
        distances[i] = gamma_distance


class RGWR:
    """ A super class for all classes that model the RGWR network. """
    __metaclass__ = abc.ABCMeta
    GDM_OUTPUT_PATH = "./GDM_output"

    def __init__(self, num_nodes, num_weights, num_classes, dimension):
        """ Initializes the object. """
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.temporal_window_size = num_weights
        self.numOfClasses = num_classes
        self.recurrent_weights = np.zeros((self.num_nodes, self.temporal_window_size, self.dimension))
        self.labels_category_histogram = np.zeros((num_nodes, self.numOfClasses))
        self.globalContext = np.zeros((self.temporal_window_size, self.dimension))
        self.edges = np.zeros((num_nodes, num_nodes))
        self.ages = np.zeros((num_nodes, num_nodes))
        self.habituation = np.ones(num_nodes)
        self.temporal = np.zeros((num_nodes, num_nodes))
        self.varAlpha = self.calculate_gamma_weights(self.temporal_window_size)
        self.updateRate = 0
        self.neuron_significance = np.zeros(self.num_nodes)

        self.habThreshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.maxNodes = 10000
        self.max_age = 100000
        self.distanceMax = 99999
        self.maxNeighbours = 6
        self.aIncreaseFactor = 1.0
        self.aDecreaseFactor = 0.0

        self.train_data = None
        self.number_train_samples = None
        self.actual_labels = None
        self.max_batch_epochs = None
        self.insertion_threshold = None
        self.beta = None
        self.learning_rate_b = None
        self.learning_rate_n = None
        self.context = None
        self.regulated = None
        self.neurons_per_epoch = None
        self.quantization_error = None
        self.firing_counter = None

    @staticmethod
    def compute_accuracy(bmu_label, label_set) -> float:
        """
        Compute the classification accuracy.
        :param bmu_label: Predicted labels of the BMUs.
        :param label_set: Actual labels.
        :return: The classification accuracy.
        """
        samples = len(bmu_label)  # The number of samples.
        counter_acc = 0

        for sample_index in range(0, samples):
            if bmu_label[sample_index] == label_set[sample_index]:
                counter_acc += 1

        return counter_acc / samples

    @staticmethod
    def calculate_gamma_weights(temporal_window_size) -> np.ndarray:
        """
        Calculate the exponential of all elements from 0 to temporal_window_size ny negating them at first.
        :param temporal_window_size: Temporal windows size.
        :return: Gamma weights.
        """
        exponentiated_weights = np.zeros(temporal_window_size)
        for i in range(0, len(exponentiated_weights)):
            exponentiated_weights[i] = np.exp(-i)
        exponentiated_weights[:] = exponentiated_weights[:] / sum(exponentiated_weights)
        return exponentiated_weights

    def habituate_neuron(self, index, tau) -> None:
        """
        Habituate (Decrease of the response to the stimulus) a neuron.
        :param index: The index of a neuron
        :param tau: A constant.
        """
        self.habituation[index] += (tau * 1.05 * (1. - self.habituation[index]) - tau)

    def update_neuron_weights(self, index, epsilon) -> None:
        """
        Update neuron's recurrent weights.
        :param index: The index of the neuron.
        :param epsilon: A learning rate.
        """
        delta_weights = np.zeros((self.temporal_window_size, self.dimension))
        for i in range(0, self.temporal_window_size):
            delta_weights[i] = np.array([np.dot((self.globalContext[i] -
                                                 self.recurrent_weights[index, i]), epsilon)]) * self.habituation[index]
        self.recurrent_weights[index] += delta_weights

    def update_edges(self, fi, si) -> None:
        """
        Update the edges between the neurons.
        :param fi: The index of the first BMU.
        :param si: The index of the second BMU.
        """
        neighbours_first = np.nonzero(self.edges[fi])
        if len(neighbours_first[0]) >= self.maxNeighbours:
            neuron_index = -1
            max_age_neighbour = 0
            for u in range(0, len(neighbours_first[0])):
                if self.ages[fi, neighbours_first[0][u]] > max_age_neighbour:
                    max_age_neighbour = self.ages[fi, neighbours_first[0][u]]
                    neuron_index = neighbours_first[0][u]
            self.edges[fi, neuron_index] = 0
            self.edges[neuron_index, fi] = 0
        self.edges[fi, si] = 1

    def add_new_neuron(self, first_bmu_index, second_bmu_index, actual_label) -> None:
        """
        Add a new neuron.
        :param first_bmu_index: The index of the first BMU.
        :param second_bmu_index: The index of the second BMU.
        :param actual_label: An actual label.
        """
        new_recurrent_weight = np.zeros((1, self.temporal_window_size, self.dimension))

        for i in range(0, self.temporal_window_size):
            new_recurrent_weight[0, i] = np.array([np.dot(self.recurrent_weights[first_bmu_index, i] +
                                                          self.globalContext[i], 0.5)])
        self.recurrent_weights = np.concatenate((self.recurrent_weights, new_recurrent_weight), axis=0)

        new_neuron_index = self.num_nodes
        self.num_nodes += 1
        self.habituation.resize(self.num_nodes, refcheck=False)
        self.habituation[new_neuron_index] = 1
        self.temporal.resize((self.num_nodes, self.num_nodes), refcheck=False)

        self.set_new_edges(first_bmu_index, second_bmu_index, new_neuron_index)
        self.set_new_ages(first_bmu_index, second_bmu_index, new_neuron_index)
        self.set_new_category_label(actual_label)

    def set_new_category_label(self, actual_label) -> None:
        """
        Set a new category label, when a new neuron is created.
        :param actual_label: An actual label.
        """
        new_predicted_label = np.zeros((1, self.numOfClasses))
        if actual_label != -1:
            new_predicted_label[0, int(actual_label)] = self.aIncreaseFactor
        self.labels_category_histogram = np.concatenate((self.labels_category_histogram, new_predicted_label), axis=0)

    def set_new_edges(self, first_bmu_index, second_bmu_index, new_neuron_index) -> None:
        """
         Set the ages, when a new neuron is added.
        :param first_bmu_index: First BMU index.
        :param second_bmu_index: Second BMU index.
        :param new_neuron_index: The index of the new added neuron.
        """
        self.edges.resize((self.num_nodes, self.num_nodes))
        self.edges[first_bmu_index, second_bmu_index] = 0
        self.edges[second_bmu_index, first_bmu_index] = 0
        self.edges[first_bmu_index, new_neuron_index] = 1
        self.edges[new_neuron_index, first_bmu_index] = 1
        self.edges[new_neuron_index, second_bmu_index] = 1
        self.edges[second_bmu_index, new_neuron_index] = 1

    def set_new_ages(self, first_bmu_index, second_bmu_index, new_neuron_index) -> None:
        """
         Set the ages, when a new neuron is added.
        :param first_bmu_index: First BMU index.
        :param second_bmu_index: Second BMU index.
        :param new_neuron_index: The index of the new added neuron.
        """
        self.ages.resize((self.num_nodes, self.num_nodes), refcheck=False)
        self.ages[first_bmu_index, new_neuron_index] = 0
        self.ages[new_neuron_index, first_bmu_index] = 0
        self.ages[new_neuron_index, second_bmu_index] = 0
        self.ages[second_bmu_index, new_neuron_index] = 0

    def remove_old_edges(self) -> None:
        """
        Remove the edges between neurons if their age exceeds the allowed age.
        """
        removed_edges = 0
        for i in range(0, self.num_nodes):
            neighbours = np.nonzero(self.edges[i])
            for j in neighbours[0]:
                if self.ages[i, j] > self.max_age:
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0
                    self.ages[i, j] = 0
                    self.ages[j, i] = 0

                    removed_edges += 1

    @abc.abstractmethod
    def remove_neuron_label(self, neuron_index) -> None:
        """Subclasses must override this method"""

    def remove_isolated_neurons(self, memory_name) -> None:
        """
        Remove isolated neurons.
        """
        if self.num_nodes > 2:
            neuron_index = 0
            count_removed_neurons = 0
            while neuron_index < self.num_nodes:
                neighbours = np.nonzero(self.edges[neuron_index])
                if len(neighbours[0]) < 1:
                    self.recurrent_weights = np.delete(self.recurrent_weights, neuron_index, axis=0)
                    self.remove_neuron_label(neuron_index)
                    self.edges = np.delete(self.edges, neuron_index, axis=0)
                    self.edges = np.delete(self.edges, neuron_index, axis=1)
                    self.ages = np.delete(self.ages, neuron_index, axis=0)
                    self.ages = np.delete(self.ages, neuron_index, axis=1)
                    self.habituation = np.delete(self.habituation, neuron_index)
                    self.temporal = np.delete(self.temporal, neuron_index, axis=0)
                    self.temporal = np.delete(self.temporal, neuron_index, axis=1)
                    self.num_nodes -= 1
                    count_removed_neurons += 1
                else:
                    neuron_index += 1
            if count_removed_neurons > 0:
                print("%s isolated neuron(s) are removed in %s." % (count_removed_neurons, memory_name))

    def remove_weak_neurons(self, memory_name, replay) -> None:

        habituation = self.habituation
        threshold = compute_threshold_weak_neurons(habituation, replay)

        neuron_index = 0
        count_removed_neurons = 0
        while neuron_index < self.num_nodes:
            if self.habituation[neuron_index] >= threshold:
                self.recurrent_weights = np.delete(self.recurrent_weights, neuron_index, axis=0)
                self.remove_neuron_label(neuron_index)
                self.edges = np.delete(self.edges, neuron_index, axis=0)
                self.edges = np.delete(self.edges, neuron_index, axis=1)
                self.ages = np.delete(self.ages, neuron_index, axis=0)
                self.ages = np.delete(self.ages, neuron_index, axis=1)
                self.habituation = np.delete(self.habituation, neuron_index)
                self.temporal = np.delete(self.temporal, neuron_index, axis=0)
                self.temporal = np.delete(self.temporal, neuron_index, axis=1)
                self.num_nodes -= 1
                count_removed_neurons += 1
            else:
                neuron_index += 1
        if count_removed_neurons > 0:
            print("%s weak neuron(s) are removed in %s." % (count_removed_neurons, memory_name))

    @abc.abstractmethod
    def update_label_histogram(self, bmu_index, actual_category_label, actual_instance_label) -> None:
        """Subclasses must override this method"""

    def update_spatiotemporal_structure(self, first_index, second_index, actual_category_label, winner_label_index,
                                        actual_instance_label=None) -> None:
        update_rate_b = self.learning_rate_b
        update_rate_n = self.learning_rate_n

        if self.regulated and (actual_category_label != winner_label_index):
            update_rate_b *= 0.01
            update_rate_n *= 0.01
        else:
            # Adapt label histogram
            self.update_label_histogram(first_index, actual_category_label, actual_instance_label)

        # Adapt weights and context descriptors
        self.update_neuron_weights(first_index, update_rate_b)
        self.updateRate += update_rate_b * self.habituation[first_index]

        # Habituate BMU
        self.habituate_neuron(first_index, self.tau_b)

        # Update ages
        self.ages[first_index, second_index] = 0
        self.ages[second_index, first_index] = 0

        # Update edges // Remove oldest ones
        self.update_edges(first_index, second_index)
        self.update_edges(second_index, first_index)

        # Update topological neighbours
        neighbours_first = np.nonzero(self.edges[first_index])
        for z in range(0, len(neighbours_first[0])):
            ne_index = neighbours_first[0][z]
            self.update_neuron_weights(ne_index, update_rate_n)
            self.habituate_neuron(ne_index, self.tau_n)

    def initialize_training(self, data_set, label_set, max_epochs, insertion_threshold, beta, epsilon_b, epsilon_n,
                            context, regulated) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """

        :param data_set:
        :param label_set:
        :param max_epochs:
        :param insertion_threshold:
        :param beta:
        :param epsilon_b:
        :param epsilon_n:
        :param context:
        :param regulated:
        """
        self.train_data = data_set
        self.number_train_samples = self.train_data.shape[0]
        self.actual_labels = label_set
        self.max_batch_epochs = max_epochs
        self.insertion_threshold = insertion_threshold
        self.beta = beta
        self.learning_rate_b = epsilon_b
        self.learning_rate_n = epsilon_n

        self.context = context
        if not self.context:
            self.globalContext.fill(0)

        self.regulated = regulated

        self.neurons_per_epoch = np.zeros(self.max_batch_epochs)
        self.quantization_error = np.zeros((self.max_batch_epochs, 2))
        self.firing_counter = np.zeros((self.max_batch_epochs, 2))

        self.updateRate = 0

        if self.recurrent_weights[0:2, 0].all() == 0:  # The first row of the two neurons are zero.
            self.recurrent_weights[0, 0] = self.train_data[0]  # Sets the first row of the first neuron.
            self.recurrent_weights[1, 0] = self.train_data[1]  # Sets the first row of the second neuron.
            if self.regulated:
                self.labels_category_histogram[0, int(self.actual_labels[0])] = 1
                self.labels_category_histogram[1, int(self.actual_labels[1])] = 1

        previous_bmu = np.zeros((1, self.temporal_window_size, self.dimension))
        previous_index = -1
        cu_qrror = np.zeros(self.number_train_samples)
        cu_fcounter = np.zeros(self.number_train_samples)

        return previous_bmu, previous_index, cu_qrror, cu_fcounter

    def update_global_context(self, previous_bmu) -> None:
        """

        :param previous_bmu:
        """
        # Update the global context.
        for temporal_depth in range(1, self.temporal_window_size):
            self.globalContext[temporal_depth] = (self.beta * previous_bmu[0, temporal_depth]) + \
                                                 ((1 - self.beta) * previous_bmu[0, temporal_depth - 1])

    def find_bmu_distances(self, input_context=None, use_gamma_weights=1) -> np.ndarray:
        """
        Find the distances between neurons and the input.
        Find the best and second-best matching neurons.
        :return:
        """
        distances = np.zeros(self.num_nodes)
        num_nodes = self.num_nodes
        num_weights = self.temporal_window_size
        var_alpha = self.varAlpha
        recurrent_weights = self.recurrent_weights

        if input_context is None:
            context = self.globalContext
        else:
            context = input_context

        find_neuron_distances(num_nodes, num_weights, var_alpha, context, recurrent_weights, distances,
                              use_gamma_weights)

        return distances

    @staticmethod
    def find_bmu_indices(distances) -> np.ndarray:
        """

        :param distances:
        :return:
        """
        smallest_numbers = heapq.nsmallest(2, ((k, i) for i, k in enumerate(distances)))
        return smallest_numbers[0][1], smallest_numbers[1][1]

    def compute_additional_info(self, epoch, cu_qrror, cu_fcounter, memory_name, replay) -> None:
        """

        :param memory_name:
        :param epoch:
        :param cu_qrror:
        :param cu_fcounter:
        :return:
        """
        self.neurons_per_epoch[epoch] = self.num_nodes
        self.quantization_error[epoch, 0] = np.mean(cu_qrror)
        self.quantization_error[epoch, 1] = np.std(cu_qrror)
        self.firing_counter[epoch, 0] = np.mean(cu_fcounter)
        self.firing_counter[epoch, 1] = np.std(cu_fcounter)
        self.updateRate = self.updateRate / self.number_train_samples

        if not replay:
            print(memory_name, " : ", "Epochs:", epoch + 1, ", Number of Neurons:", self.num_nodes, "Update Rate:",
                  self.updateRate, ", Quantization Error:", self.quantization_error[epoch, 0], ")")

