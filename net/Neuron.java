package de.plath.csc.machinelearning.neural.net;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Artificial Neuron
 *
 */
class Neuron {

	private List<Synapse> inputs = new ArrayList<>();

	private Double output;

	private Function<Double, Double> activationFunction;

	private final Function<List<Synapse>, Double> outputFunction = input -> activationFunction
			.apply(input.stream().mapToDouble(synapse -> synapse.getProduct()).sum());

	protected Neuron(final Function<Double, Double> activationFunction) {
		this.activationFunction = Objects.requireNonNull(activationFunction, "Activation function is null.");
	}

	protected List<Synapse> getInputs() {
		return inputs;
	}

	protected void setInputs(final List<Synapse> inputs) {
		this.inputs = inputs;
	}

	protected void calculateOutput() {
		this.output = outputFunction.apply(inputs);
	}

	protected Double getOutput() {
		return output;
	}

	public void setActivationFunction(final Function<Double, Double> activationFunction) {
		this.activationFunction = activationFunction;
	}
}
