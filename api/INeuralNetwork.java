package de.plath.csc.machinelearning.neural.api;

import java.util.List;

/**
 *
 * Interface for a neural network
 */
public interface INeuralNetwork {

	/**
	 * Trains the net based on the given training set
	 *
	 * @param trainingData a list of inputs
	 * @param labels       the labels of the inputs
	 */
	void learn(List<List<Double>> trainingData, List<Integer> labels);

	/**
	 *
	 * @param input a list of real numbers
	 * @return the output of the network given the input
	 */
	List<Double> getOutput(List<Double> input);
}
