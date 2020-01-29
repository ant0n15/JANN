package de.plath.csc.machinelearning.neural.net;

import java.util.Optional;

class Synapse {

	private double weight;
	private double deltaWeight;
	private double delta;
	private Optional<Neuron> input;

	protected Synapse(final double weight, final Optional<Neuron> input) {
		this.setWeight(weight);
		this.setInput(input);
		this.setDeltaWeight(0d);
	}

	protected void updateWeight() {
		weight -= deltaWeight;
	}

	protected double getDeltaWeight() {
		return deltaWeight;
	}

	protected double getProduct() {
		return input.isPresent() ? weight * input.get().getOutput() : weight;
	}

	protected double getWeight() {
		return weight;
	}

	protected double getDelta() {
		return delta;
	}

	protected void setWeight(final double weight) {
		this.weight = weight;
	}

	protected void setInput(final Optional<Neuron> input) {
		this.input = input;
	}

	protected void setDelta(final Double delta) {
		this.delta = delta;
	}

	protected void setDeltaWeight(final double deltaWeight) {
		this.deltaWeight = deltaWeight;
	}
}