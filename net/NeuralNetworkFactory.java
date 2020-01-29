package de.plath.csc.machinelearning.neural.net;

import java.util.Objects;

import de.plath.csc.machinelearning.neural.api.IParameters;

/**
 * Factory class for NeuralNetwork
 */
public class NeuralNetworkFactory {

	final IParameters nnParameters;

	/**
	 *
	 * @param nnParameters the parameters of the neural network
	 */
	public NeuralNetworkFactory(final IParameters nnParameters) {

		this.nnParameters = Objects.requireNonNull(nnParameters);
	}

	/**
	 * @return NeuralNetwork
	 */
	public NeuralNetwork create() {

		final NetworkLayers networkLayers = new NetworkLayers(nnParameters);

		networkLayers.initializeLayers();
		networkLayers.connectLayers();

		final NeuralNetwork neuralNet = new NeuralNetwork(nnParameters,
				networkLayers.getInputLayer(),
				networkLayers.getHiddenLayers(),
				networkLayers.getOutputLayer());

		return neuralNet;
	}
}
