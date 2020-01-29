package de.plath.csc.machinelearning.neural.net;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import de.plath.csc.machinelearning.neural.api.INeuralNetwork;
import de.plath.csc.machinelearning.neural.api.IParameters;

/**
 * A feed-forward, fully connected Artificial Neural Network for classification, with
 * back-propagation learning.
 *
 */
public class NeuralNetwork implements INeuralNetwork {

	private final IParameters nnParameters;

	private final List<Neuron> inputLayer;
	private final List<Neuron> outputLayer;

	private final FeedForward feedForward;
	private final Backpropagation backPropagation;


	/**
	 * C'tor
	 *
	 * @param nnParameters the parameters of the neural net
	 * @param inputLayer   input layer of the neural net
	 * @param hiddenLayers hidden layers of the neural net
	 * @param outputLayer  output layer of the neural net
	 *
	 */
	public NeuralNetwork(
			final IParameters nnParameters,
			final List<Neuron> inputLayer,
			final List<List<Neuron>> hiddenLayers,
			final List<Neuron> outputLayer) {

		this.nnParameters = nnParameters;
		this.inputLayer = Objects.requireNonNull(inputLayer);
		this.outputLayer = Objects.requireNonNull(outputLayer);

		feedForward = new FeedForward(inputLayer, hiddenLayers, outputLayer);
		backPropagation = new Backpropagation(inputLayer, hiddenLayers, outputLayer, nnParameters);

	}

	@Override
	public void learn(final List<List<Double>> trainingData, final List<Integer> targets) {

		Objects.requireNonNull(trainingData, "Training data is null");
		assert (!trainingData.isEmpty()) : "Training data cannot be empty";
		assert (inputLayer.size() == trainingData.get(0).size()) : "Size of inputs must match the size of the input layer.";
		assert (targets.stream().anyMatch(target -> target < 0)) : "Labels must be integers greater than or equal to 0.";

		IntStream.rangeClosed(1, nnParameters.getEpochs()).forEach(epoch -> {

			final Iterator<Integer> targetsIterator1 = targets.iterator();
			final Iterator<Integer> targetsIterator2 = targets.iterator();
			final List<Double> errors = new ArrayList<>();

			trainingData.forEach(inputData -> {

				feedForward.apply(inputData);
				backPropagation.apply(extractBinaryTarget(targetsIterator1.next()));
				calculateLearningError(targetsIterator2, errors, inputData);
			});

			//System.out.println(epoch + ", " + errors.stream().mapToDouble(e -> e).average().getAsDouble());

		});
	}

	@Override
	public List<Double> getOutput(final List<Double> input) {

		Objects.requireNonNull(input, "input is null");
		assert (inputLayer.size() == input.size()) : "Size of input must match the size of the input layer.";

		feedForward.apply(input);

		return outputLayer.stream().map(neuron -> neuron.getOutput()).collect(Collectors.toList());
	}

	private void calculateLearningError(final Iterator<Integer> targetsIterator, final List<Double> errors, final List<Double> inputData) {

		final List<Double> binaryTargets = extractBinaryTarget(targetsIterator.next());
		final Iterator<Double> binaryTargetsIterator = binaryTargets.iterator();

		errors.add(getOutput(inputData).stream()
				.mapToDouble(out -> nnParameters.getErrorFunction().apply(out, binaryTargetsIterator.next()))
				.average()
				.getAsDouble());
	}

	private List<Double> extractBinaryTarget(final Integer target) {

		return IntStream.range(0, outputLayer.size())
				.mapToDouble(i -> i == target ? 1d : 0d)
				.boxed()
				.collect(Collectors.toList());
	}
}