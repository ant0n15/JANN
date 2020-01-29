package de.plath.csc.machinelearning.neural.api;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 *
 * Interface for neural network parameters
 */
public interface IParameters {

	/**
	 * @return the random number generator
	 */
	Random getRandom();

	/**
	 * @return the number of neurons in the input layer
	 */
	int getInputLayerSize();

	/**
	 * @return the number of hidden layers
	 */
	int getNumberOfHiddenLayers();

	/**
	 * @return the number of neurons of a hidden layer
	 */
	int getHiddenLayersSize();

	/**
	 * @return the number of neurons in the output layer
	 */
	int getOutputLayerSize();

	/**
	 * @return the momentum rate of the learning algorithm
	 */
	double getMomentum();

	/**
	 * @return the learning rate of the learning algorithm
	 */
	double getLearningRate();

	/**
	 * @return the number of iterations for training
	 */
	int getEpochs();

	/**
	 * @return the initialization of weights function
	 */
	Function<Integer, Double> getInitializationFunction();

	/**
	 * @return the activation function of neurons in the network
	 */
	Function<Double, Double> getActivationFunction();

	/**
	 * @return the derivative function
	 */
	Function<Double, Double> getDerivativeFunction();

	/**
	 * @return the error function
	 */
	BiFunction<Double, Double, Double> getErrorFunction();

	/**
	 * @return the output activation function
	 */
	Function<Double, Double> getOutputActivationFunction();

	/**
	 * @return the output derivative function
	 */
	Function<Double, Double> getOutputDerivativeFunction();

}
