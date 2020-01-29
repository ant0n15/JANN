package de.plath.csc.machinelearning.neural.net;

import java.util.Iterator;
import java.util.List;


class FeedForward {

	private final List<Neuron> inputLayer;
	private final List<List<Neuron>> hiddenLayers;
	private final List<Neuron> outputLayer;

	protected FeedForward(final List<Neuron> inputLayer,
			final List<List<Neuron>> hiddenLayers,
			final List<Neuron> outputLayer) {

		this.inputLayer = inputLayer;
		this.hiddenLayers = hiddenLayers;
		this.outputLayer = outputLayer;
	}

	protected void apply(final List<Double> input) {

		feedInputLayer(input);
		feedHiddenLayers();
		feedOutputLayer();
	}

	private void feedInputLayer(final List<Double> input) {

		final Iterator<Double> iterator = input.iterator();
		inputLayer.forEach(neuron -> {
			neuron.getInputs().get(0).setWeight(iterator.next());
			neuron.calculateOutput();
		});
	}

	private void feedHiddenLayers() {
		hiddenLayers.forEach(layer -> layer.forEach(neuron -> neuron.calculateOutput()));
	}

	private void feedOutputLayer() {
		outputLayer.forEach(neuron -> neuron.calculateOutput());
	}

}
