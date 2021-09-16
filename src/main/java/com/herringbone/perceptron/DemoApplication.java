package com.herringbone.perceptron;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
        float[][] inputFeatures = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float[] inputLabels = {0, 0, 0, 1}; // And operater is linearly separable so it can be solved without a hidden layer
//        float[] inputLabels = {0, 1, 1, 0}; - Xor problem cannot be solved with single layer network (no hidden layer)
//        because it is not linearly separable

        PerceptionNetwork perceptionNetwork = new PerceptionNetwork(inputFeatures, inputLabels);
        perceptionNetwork.train(0.2f);

        System.out.println("After the training, let's test the network");
        System.out.println(perceptionNetwork.calculateOutput(new float[]{0, 0}));
        System.out.println(perceptionNetwork.calculateOutput(new float[]{0, 1}));
        System.out.println(perceptionNetwork.calculateOutput(new float[]{1, 0}));
        System.out.println(perceptionNetwork.calculateOutput(new float[]{1, 1}));
    }

}
