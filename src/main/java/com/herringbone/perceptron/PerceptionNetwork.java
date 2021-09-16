package com.herringbone.perceptron;

public class PerceptionNetwork {

    private float[] weights;
    private int numberOfWeights;
    private float[][] inputFeatures;
//    [[0,0], [1, 0], [0, 1], [1, 1]]
    private float[] inputLabels;

    public PerceptionNetwork(float[][] inputFeatures, float[] inputLabels) {
        this.inputFeatures = inputFeatures;
        this.inputLabels = inputLabels;
        this.numberOfWeights = inputFeatures[0].length;
        this.weights = new float[numberOfWeights];

        initializeWeights();
    }

    // initialize the weights randomly [-0.5, 0.5]
    private void initializeWeights() {
        for (int i=0; i < numberOfWeights; ++i) {
            weights[i] = (float)(Math.random() - 0.5);
        }
    }

    // This is how to train the model
    public void train(float learningRate) {
        float totalError = 1f;
        while (totalError != 0) {
            totalError = 0;

            // we consider all the rows of the training data set
            for (int i = 0; i < inputLabels.length; ++i) {
                float calculatedOutput = calculateOutput(inputFeatures[i]);
                float error = inputLabels[i] - calculatedOutput;
                totalError += error;

                //update the weights based on error and learning rate
                for (int j=0; j < numberOfWeights; ++j) {
                    weights[j] = weights[j] + learningRate * error * inputFeatures[i][j];
                }
            }
            System.out.println("Keep on training neural network, error is " + totalError);
        }
    }

    // This is applying the sum function and the activation function on the next input
    public float calculateOutput(float[] input) {
        float netInput = 0f;

        for (int i=0; i < input.length; i++) {
            netInput = netInput + input[i] * weights[i];
        }

        return ActivationFunction.apply(netInput);
    }

}
