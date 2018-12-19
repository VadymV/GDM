# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal
Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966

Modifications by Vadym Gryshchuk(vadym.gryshchuk@protonmail.com)
Last modified on 15 December 2018
"""
from episodic_gwr import EpisodicGWR
from semantic_gwr import SemanticGWR
from core50 import CORe50
import numpy as np
from RGWR import RGWR


if __name__ == "__main__":
    
    trainFlag = 1            # Train AGWR with imported dataset
    testFlag = 1             # Compute classification accuracy

    dataset = CORe50()
    dataset.prepare_data()

    print("%s is loaded." % dataset.sName)
                
    if trainFlag:

        iRun = np.array([2, 7, 1, 8, 0, 3, 9, 6, 4, 5])
        #iRun = np.array([2, 7, 1, 8, 0])
        
        numWeights = [3, 3]                     # The size of a temporal receptive field
        batch_epochs = 1                        # Number of training epochs per mini-batch
        insertion_thresholds = [0.3, 0.001]     # Insertion thresholds
        learning_rates = [0.5, 0.005]           # Learning rates for a BMU and its neighbours
        beta = 0.7                              # Beta parameter
        trainWithReplay = 1                     # Memory replay flag
        replay_size = 5                         # The size of RNATs (reactive neural activity trajectories)

        # Initialize networks
        myEpisodicGWR = EpisodicGWR(dataset.vecDim, numWeights[0], dataset.numClasses, dataset.numInstances)
        mySemanticGWR = SemanticGWR(dataset.vecDim, numWeights[1], dataset.numClasses)

        replay_vectors = []
        replay_labels = []

        accMatrix = np.zeros((dataset.numClasses, dataset.numClasses + 1, dataset.numLabels))

        neurons_per_category = np.zeros((dataset.numClasses, 2))
        for selected_class in range(0, dataset.numClasses):
            class_index = int(iRun[selected_class])
            train_category_start_index = int(dataset.train_category_indices[class_index, 0])
            train_category_end_index = int(dataset.train_category_indices[class_index, 1])
            
            print("- Training class %s" % selected_class)
            
            regulated = (selected_class > 0)
            
            replayFlag = trainWithReplay and selected_class > 0
            if replayFlag:
                replay_vectors, replay_labels = myEpisodicGWR.replay_data(replay_size)

            myEpisodicGWR.train(dataset.trainingVectors[train_category_start_index:train_category_end_index],
                                dataset.trainingLabels[train_category_start_index:train_category_end_index],
                                batch_epochs, insertion_thresholds[0], beta, learning_rates[0], learning_rates[1], 1, 0)

            emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(
                dataset.trainingVectors[train_category_start_index:train_category_end_index], 1)
            
            mySemanticGWR.train(emBmuWeights, emBmuLabelClasses, batch_epochs, insertion_thresholds[1], beta,
                                learning_rates[0], learning_rates[1], 1, regulated)
                                
            if replayFlag:
                remove_weak_neurons = False
                for i in range(0, replay_vectors.shape[0]):
                    myEpisodicGWR.train(replay_vectors[i], replay_labels[i], 1, insertion_thresholds[0], beta,
                                        learning_rates[0], learning_rates[1], 0, 0, replay=True)
                    mySemanticGWR.train(replay_vectors[i], replay_labels[i, :, 0], 1, insertion_thresholds[1], beta,
                                        learning_rates[0], learning_rates[1], 0, 1, replay=True)
            s = 0
            for s in range(0, selected_class + 1):
                
                si = int(iRun[s])
                tti = int(dataset.test_categoty_indices[si, 0])
                tte = int(dataset.test_categoty_indices[si, 1])

                emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = \
                    myEpisodicGWR.predict(dataset.testVectors[tti:tte], 1)
                emAccuracy = myEpisodicGWR.compute_accuracy(emBmuLabelInstances, dataset.testLabels[tti:tte, 1])
                
                smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, 1)
                smAccuracy = mySemanticGWR.compute_accuracy(smBmuLabelClasses, dataset.testLabels[tti:tte, 0])

                accMatrix[selected_class, s, 0] = emAccuracy
                accMatrix[selected_class, s, 1] = smAccuracy

                # if si == 0 and class_index == 0:
                    # Compute confusion matrix for EM
                    # plot_confusion_matrix("conf_matrix_em_category_0_fifth_iteration", dataset.testLabels[tti:tte, 1],
                    #                       emBmuLabelInstances)
                    # Compute confusion matrix for SM
                    # plot_confusion_matrix("conf_matrix_sm_category_0_fifth_iteration", dataset.testLabels[tti:tte, 0],
                    #                       smBmuLabelClasses)
                
            for m in range(0, dataset.numLabels):
                aC = 0
                for u in range(0, s + 1):
                    aC += accMatrix[selected_class, u, m]
                accMatrix[selected_class, dataset.numClasses, m] = aC / (s + 1)
            neurons_per_category[selected_class, 0] = myEpisodicGWR.neurons_per_epoch[batch_epochs - 1]
            neurons_per_category[selected_class, 1] = mySemanticGWR.neurons_per_epoch[batch_epochs - 1]

        print("Test accuracy (first category seen):\n", accMatrix[0, :, :])
        print("Test accuracy (all categories seen):\n", accMatrix[dataset.numClasses - 1, :, :])

        emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(
            dataset.testVectors, 1)
        smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, 1)

        # Compute confusion matrix for EM
        # plot_confusion_matrix("conf_matrix_em_instances_all_iterations", dataset.testLabels[:, 1],
        #                       emBmuLabelInstances)
        # Compute confusion matrix for SM
        # plot_confusion_matrix("conf_matrix_sm_category_all_iterations", dataset.testLabels[:, 0],
        #                       smBmuLabelClasses)

        # Save the histograms.
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'sm_label_histogram', mySemanticGWR.labels_category_histogram)
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'em_label_histogram', myEpisodicGWR.label_instances_histogram)

        # Save the neurons per category
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'neurons_per_category', neurons_per_category)

        # Save the accuracy matrix
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'accuracy_matrix', accMatrix)

        # Save the weights.
        np.savez(RGWR.GDM_OUTPUT_PATH + "/" + 'preprocessed_inputs_weights_em_V1.npz', x=emBmuWeights,
                 y=dataset.testLabels[:, 1])
        np.savez(RGWR.GDM_OUTPUT_PATH + "/" + 'preprocessed_inputs_weights_sm_V1.npz', x=smBmuWeights,
                 y=dataset.testLabels[:, 0])
        np.savez(RGWR.GDM_OUTPUT_PATH + "/" + 'bmu_labels_instances.npz', y=emBmuLabelInstances)
        np.savez(RGWR.GDM_OUTPUT_PATH + "/" + 'bmu_labels_categories.npz', y=smBmuLabelClasses)

        # Save the recurrent weights.
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'rweights_sm', mySemanticGWR.recurrent_weights)
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'predicted_labels', mySemanticGWR.labels_category_histogram)

        # Save the edges.
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'edges_em', myEpisodicGWR.edges)
        np.save(RGWR.GDM_OUTPUT_PATH + "/" + 'edges_sm', mySemanticGWR.edges)



