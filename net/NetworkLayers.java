package de.plath.csc.machinelearning.neural.net;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.IntStream;

import de.plath.csc.machinelearning.neural.api.IParameters;

/**
 *
 * Initializes and connects the network layers
 */
public class NetworkLayers {

	private final IParameters nnParameters;

	private final List<Neuron> inputLayer = new ArrayList<>();
	private final List<List<Neuron>> hiddenLayers = new ArrayList<>();
	private final List<Neuron> outputLayer = new ArrayList<>();

	/**
	 * @param nnParameters the parameters of the Neural Network
	 */
	public NetworkLayers(final IParameters nnParameters) {
		this.nnParameters = Objects.requireNonNull(nnParameters);
	}

	/**
	 * Initialize the layers
	 */
	public void initializeLayers() {

		addNeuronsToLayer(inputLayer, nnParameters.getInputLayerSize());
		addNeuronsToHiddenLayers(nnParameters.getHiddenLayersSize());
		addNeuronsToOutputLayer(outputLayer, nnParameters.getOutputLayerSize());
	}

	/**
	 * Fully connect the layers
	 */
	public void connectLayers() {

		connectNeuronsOfInputLayer();
		connectNeuronsOfHiddenLayers();
		connectNeuronsOfOutputLayer();
	}

	public List<Neuron> getInputLayer() {
		return inputLayer;
	}

	public List<List<Neuron>> getHiddenLayers() {
		return hiddenLayers;
	}

	public List<Neuron> getOutputLayer() {
		return outputLayer;
	}

	private void addNeuronsToLayer(final List<Neuron> layer, final int size) {

		IntStream.range(0, size)
				.forEach((i) -> layer.add(new Neuron(nnParameters.getActivationFunction())));
	}

	private void addNeuronsToOutputLayer(final List<Neuron> layer, final int size) {

		IntStream.range(0, size).forEach((i) -> layer.add(new Neuron(nnParameters.getOutputActivationFunction())));
	}

	private void addNeuronsToHiddenLayers(final int size) {

		IntStream.range(0, nnParameters.getNumberOfHiddenLayers()).forEach((h) -> {
			hiddenLayers.add(new ArrayList<>());
			addNeuronsToLayer(hiddenLayers.get(h), size);
		});
	}

	private void connectNeuronsOfInputLayer() {

		inputLayer.forEach((neuron) -> {
			final List<Synapse> inputLayerConnection = new ArrayList<>();
			inputLayerConnection.add(new Synapse(1d, Optional.empty()));
			neuron.setInputs(inputLayerConnection);
		});
	}

	private void connectNeuronsOfHiddenLayers() {

		for (int hiddenLayer = 0; hiddenLayer < hiddenLayers.size(); hiddenLayer++) {

			final List<Neuron> previousLayer = hiddenLayer == 0 ? inputLayer : hiddenLayers.get(hiddenLayer - 1);
			final List<Neuron> currentLayer = hiddenLayers.get(hiddenLayer);

			final int previousLayerSize = previousLayer.size();

			for (final Neuron neuron : currentLayer) {

				final List<Synapse> synapses = new ArrayList<>();

				previousLayer.forEach((n) -> synapses.add(
						new Synapse(nnParameters.getInitializationFunction().apply(previousLayerSize), Optional.of(n))));

				synapses.add(new Bias(nnParameters.getInitializationFunction().apply(previousLayerSize)));

				neuron.setInputs(synapses);
			}
		}
	}

	private void connectNeuronsOfOutputLayer() {

		final List<Neuron> previousLayer = hiddenLayers.get(hiddenLayers.size() - 1);
		final int previousLayerSize = previousLayer.size();

		outputLayer.forEach((neuron) -> {
			final List<Synapse> synapses = new ArrayList<>();
			previousLayer.forEach((n) -> synapses
					.add(new Synapse(nnParameters.getInitializationFunction().apply(previousLayerSize), Optional.of(n))));
			synapses.add(new Bias(nnParameters.getInitializationFunction().apply(previousLayerSize)));
			neuron.setInputs(synapses);
		});
	}
}