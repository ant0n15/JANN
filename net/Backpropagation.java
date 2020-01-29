package de.plath.csc.machinelearning.neural.net;

import java.util.List;
import java.util.function.Function;

import de.plath.csc.machinelearning.neural.api.IParameters;

/**
 * Back-propagation algorithm with learning rate and momentum
 *
 */

class Backpropagation {

	private final List<Neuron> inputLayer;
	private final List<List<Neuron>> hiddenLayers;
	private final List<Neuron> outputLayer;

	private final Double momentum;
	private final Double learningRate;
	private final Function<Double, Double> derivativeFunction;
	private final Function<Double, Double> outputDerivativeFunction;

	protected Backpropagation(final List<Neuron> inputLayer,
			final List<List<Neuron>> hiddenLayers,
			final List<Neuron> outputLayer,
			final IParameters nnParameters) {

		this.inputLayer = inputLayer;
		this.hiddenLayers = hiddenLayers;
		this.outputLayer = outputLayer;

		this.momentum = nnParameters.getMomentum();
		this.learningRate = nnParameters.getLearningRate();
		derivativeFunction = nnParameters.getDerivativeFunction();
		outputDerivativeFunction = nnParameters.getOutputDerivativeFunction();
	}

	protected void apply(final List<Double> targets) {

		calculateDeltasOfAllLayers(targets);
		updateWeights();
	}

	private void updateWeights() {
		hiddenLayers.forEach(layer -> layer.forEach(neuron -> neuron.getInputs().forEach(synapse -> synapse.updateWeight())));
		outputLayer.forEach(neuron -> neuron.getInputs().forEach(synapse -> synapse.updateWeight()));
	}

	private void calculateDeltasOfAllLayers(final List<Double> targets) {

		calculateDeltasOfOutputLayer(targets);

		for (int layer = hiddenLayers.size() - 1; layer >= 0; layer--) {

			final List<Neuron> previousLayer = layer == 0 ? inputLayer : hiddenLayers.get(layer - 1);
			final List<Neuron> hiddenLayer = hiddenLayers.get(layer);
			final List<Neuron> nextLayer = layer == hiddenLayers.size() - 1 ? outputLayer : hiddenLayers.get(layer + 1);

			calculateDeltasOfOneHiddenLayer(previousLayer, hiddenLayer, nextLayer);
		}
	}

	private void calculateDeltasOfOutputLayer(final List<Double> targets) {

		final List<Neuron> previousLayer = hiddenLayers.get(hiddenLayers.size() - 1);

		for (int o = 0; o < outputLayer.size(); o++) {

			final Double target = targets.get(o);
			final Double out = outputLayer.get(o).getOutput();
			final Double errorSignal = out - target;

			for (int p = 0; p < previousLayer.size(); p++) {

				final Double previousOut = previousLayer.get(p).getOutput();
				final Double deltaWeight = outputLayer.get(o).getInputs().get(p).getDeltaWeight();
				final Double delta = learningRate * errorSignal * outputDerivativeFunction.apply(out) * previousOut;

				outputLayer.get(o).getInputs().get(p).setDelta(delta);
				outputLayer.get(o).getInputs().get(p).setDeltaWeight(delta + momentum * deltaWeight);
			}
		}
	}

	private void calculateDeltasOfOneHiddenLayer(final List<Neuron> previousLayer,
			final List<Neuron> hiddenLayer,
			final List<Neuron> nextLayer) {

		for (int h = 0; h < hiddenLayer.size(); h++) {

			final Double hiddenOut = hiddenLayer.get(h).getOutput();

			for (int p = 0; p < previousLayer.size(); p++) {

				final Double previousOut = previousLayer.get(p).getOutput();

				Double totalOutput = 0d;

				for (int n = 0; n < nextLayer.size(); n++) {

					final Double weight = nextLayer.get(n).getInputs().get(h).getWeight();
					final Double delta = nextLayer.get(n).getInputs().get(h).getDelta();

					final Double nextOut = nextLayer.get(n).getOutput();
					final Double errorSignal = weight * delta;

					totalOutput += errorSignal * derivativeFunction.apply(nextOut);
				}

				final double delta = learningRate * totalOutput * (derivativeFunction.apply(hiddenOut) * previousOut);
				final double deltaWeight = hiddenLayer.get(h).getInputs().get(p).getDeltaWeight();

				hiddenLayer.get(h).getInputs().get(p).setDelta(delta);
				hiddenLayer.get(h).getInputs().get(p).setDeltaWeight(delta + momentum * deltaWeight);
			}
		}
	}
}
