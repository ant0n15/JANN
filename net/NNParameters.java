package de.plath.csc.machinelearning.neural.net;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import de.plath.csc.machinelearning.neural.api.IParameters;

/**
 *
 * Parameters for {@link NeuralNetwork}
 */
public final class NNParameters implements IParameters {

	private final Random random;

	private final int inputLayerSize;
	private final int hiddenLayers;
	private final int hiddenLayersSize;
	private final int outputLayerSize;

	private final double momentum;
	private final double learningRate;
	private final int epochs;

	private final Function<Integer, Double> initializationFunction;
	private final Function<Double, Double> activationFunction;
	private final Function<Double, Double> derivativeFunction;
	private final Function<Double, Double> outputActivationFunction;
	private final Function<Double, Double> outputDerivativeFunction;
	private final BiFunction<Double, Double, Double> errorFunction;


	//CHECKSTYLE:DISABLE:ParameterNumber will be re-factored
	private NNParameters(final Random random,
			final int inputLayerSize,
			final int hiddenLayers,
			final int hiddenLayersSize,
			final int outputLayerSize,
			final double momentum,
			final double learningRate,
			final int epochs,
			final Function<Integer, Double> initializationFunction,
			final Function<Double, Double> activationFunction,
			final Function<Double, Double> derivativeFunction,
			final Function<Double, Double> outputActivationFunction,
			final Function<Double, Double> outputDerivativeFunction,
			final BiFunction<Double, Double, Double> errorFunction) {

		this.random = random;
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayers = hiddenLayers;
		this.hiddenLayersSize = hiddenLayersSize;
		this.outputLayerSize = outputLayerSize;

		this.momentum = momentum;
		this.learningRate = learningRate;
		this.epochs = epochs;

		this.initializationFunction = initializationFunction;
		this.activationFunction = activationFunction;
		this.derivativeFunction = derivativeFunction;
		this.outputActivationFunction = outputActivationFunction;
		this.outputDerivativeFunction = outputDerivativeFunction;
		this.errorFunction = errorFunction;
	}
	//CHECKSTYLE:ENABLE:ParameterNumber

	@Override
	public Random getRandom() {
		return random;
	}

	@Override
	public int getInputLayerSize() {
		return inputLayerSize;
	}

	@Override
	public int getNumberOfHiddenLayers() {
		return hiddenLayers;
	}

	@Override
	public int getHiddenLayersSize() {
		return hiddenLayersSize;
	}

	@Override
	public int getOutputLayerSize() {
		return outputLayerSize;
	}

	@Override
	public double getMomentum() {
		return momentum;
	}

	@Override
	public double getLearningRate() {
		return learningRate;
	}

	@Override
	public int getEpochs() {
		return epochs;
	}

	@Override
	public Function<Integer, Double> getInitializationFunction() {
		return initializationFunction;
	}

	@Override
	public Function<Double, Double> getActivationFunction() {
		return activationFunction;
	}

	@Override
	public Function<Double, Double> getDerivativeFunction() {
		return derivativeFunction;
	}

	@Override
	public Function<Double, Double> getOutputActivationFunction() {
		return outputActivationFunction;
	}

	@Override
	public Function<Double, Double> getOutputDerivativeFunction() {
		return outputDerivativeFunction;
	}

	@Override
	public BiFunction<Double, Double, Double> getErrorFunction() {
		return errorFunction;
	}

	/**
	 *
	 * NNParameters builder
	 */
	public static class Builder {

		// Default values
		private static final int INPUTSIZE = 2;
		private static final int HIDDENSIZE = 2;
		private static final int OUTPUTSIZE = 2;
		private static final int EPOCHS = 5_000;

		private final long seed = 1;
		private Random random = new Random(seed);

		private int inputLayerSize = INPUTSIZE;
		private int hiddenLayers = HIDDENSIZE;
		private int hiddenLayersSize = HIDDENSIZE;
		private int outputLayerSize = OUTPUTSIZE;

		private double momentum = 0;
		private double learningRate = 1d;
		private int epochs = EPOCHS;

		//Xavier normal initializer
		private Function<Integer, Double> initializationFunction = x -> random.nextGaussian() * (1d / x);
		//Sigmoid activation
		private Function<Double, Double> activationFunction = x -> 1d / (1d + Math.exp(-x));
		//Sigmoid derivative
		private Function<Double, Double> derivativeFunction = x -> x * (1d - x);
		//Sigmoid activation
		private Function<Double, Double> outputActivationFunction = x -> 1d / (1d + Math.exp(-x));
		//Sigmoid derivative
		private Function<Double, Double> outputDerivativeFunction = x -> x * (1d - x);
		//Mean squared error
		private BiFunction<Double, Double, Double> errorFunction = (output, target) -> Math.pow((target - output), 2) / 2d;

		/**
		 *
		 * @param random number generator
		 * @return builder
		 */
		public Builder setRandom(final Random random) {
			this.random = random;
			return this;
		}

		/**
		 *
		 * @param inputLayerSize input layer size
		 * @return builder
		 */
		public Builder setInputLayerSize(final int inputLayerSize) {
			this.inputLayerSize = inputLayerSize;
			return this;
		}

		/**
		 *
		 * @param hiddenLayers number of hidden layers
		 * @return builder
		 */
		public Builder setHiddenLayers(final int hiddenLayers) {
			this.hiddenLayers = hiddenLayers;
			return this;
		}

		/**
		 *
		 * @param hiddenLayersSize size of hidden layers
		 * @return builder
		 */
		public Builder setHiddenLayersSize(final int hiddenLayersSize) {
			this.hiddenLayersSize = hiddenLayersSize;
			return this;
		}

		/**
		 *
		 * @param outputLayerSize size of output layer
		 * @return builder
		 */
		public Builder setOutputLayerSize(final int outputLayerSize) {
			this.outputLayerSize = outputLayerSize;
			return this;
		}

		/**
		 *
		 * @param momentum momentum rate
		 * @return builder
		 */
		public Builder setMomentum(final double momentum) {
			this.momentum = momentum;
			return this;
		}

		/**
		 *
		 * @param learningRate the learning rate
		 * @return builder
		 */
		public Builder setLearningRate(final double learningRate) {
			this.learningRate = learningRate;
			return this;
		}

		/**
		 *
		 * @param epochs number of epochs in training
		 * @return builder
		 */
		public Builder setEpochs(final int epochs) {
			this.epochs = epochs;
			return this;
		}

		/**
		 *
		 * @param initializationFunction the initialization function of weights
		 * @return builder
		 */
		public Builder setInitializationFunction(final Function<Integer, Double> initializationFunction) {
			this.initializationFunction = initializationFunction;
			return this;
		}

		/**
		 *
		 * @param activationFunction the activation function of neurons
		 * @return builder
		 */
		public Builder setActivationFunction(final Function<Double, Double> activationFunction) {
			this.activationFunction = activationFunction;
			return this;
		}

		/**
		 *
		 * @param derivativeFunction derivative function
		 * @return builder
		 */
		public Builder setDerivativeFunction(final Function<Double, Double> derivativeFunction) {
			this.derivativeFunction = derivativeFunction;
			return this;
		}

		/**
		 *
		 * @param outputActivationFunction the activation function of output neurons
		 * @return builder
		 */
		public Builder setOutputActivationFunction(final Function<Double, Double> outputActivationFunction) {
			this.outputActivationFunction = outputActivationFunction;
			return this;
		}

		/**
		 *
		 * @param outputDerivativeFunction output derivative function
		 * @return builder
		 */
		public Builder setOutputDerivativeFunction(final Function<Double, Double> outputDerivativeFunction) {
			this.outputDerivativeFunction = outputDerivativeFunction;
			return this;
		}

		/**
		 *
		 * @param errorFunction error function of the neural net
		 * @return builder
		 */
		public Builder setErrorFunction(final BiFunction<Double, Double, Double> errorFunction) {
			this.errorFunction = errorFunction;
			return this;
		}

		/**
		 *
		 * @return NNParameters
		 */
		public NNParameters build() {
			return new NNParameters(random,
					inputLayerSize,
					hiddenLayers,
					hiddenLayersSize,
					outputLayerSize,
					momentum,
					learningRate,
					epochs,
					initializationFunction,
					activationFunction,
					derivativeFunction,
					outputActivationFunction,
					outputDerivativeFunction,
					errorFunction);
		}
	}
}
