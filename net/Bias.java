package de.plath.csc.machinelearning.neural.net;

import java.util.Optional;

class Bias extends Synapse {

	private double weight;
	private double deltaWeight;

	protected Bias(final double weight) {
		super(weight, Optional.empty());
		this.setWeight(weight);
		this.setDeltaWeight(0d);
	}

	@Override
	protected void updateWeight() {
		weight -= deltaWeight;
	}

	@Override
	protected double getProduct() {
		return weight * 1d;
	}

	@Override
	protected double getWeight() {
		return weight;
	}

	@Override
	protected void setWeight(final double weight) {
		this.weight = weight;
	}

	@Override
	protected void setDeltaWeight(final double deltaWeight) {
		this.deltaWeight = deltaWeight;
	}
}