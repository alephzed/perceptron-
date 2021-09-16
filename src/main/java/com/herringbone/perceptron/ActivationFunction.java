package com.herringbone.perceptron;

public class ActivationFunction {

    public static int apply(float x) {
        if (x< 1) {
            return 0;
        }
        return 1;
    }
}
