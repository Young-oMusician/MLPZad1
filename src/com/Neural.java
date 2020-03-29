package com;

import java.util.Vector;

public class Neural {

    private Vector<Double> inputs;
    private double output;
    private double delta;
    private Vector<Double> weights;
    private double bias;
    private boolean isBias;

    public Neural(int in,boolean isBias){
        inputs = Utils.setZeros(in);
        weights = Utils.setZeros(in);
        this.isBias = isBias;
        if(isBias){
            this.bias = Utils.randomWeight();
        }
        for(int i = 0; i < weights.size(); i++){
            weights.set(i,Utils.randomWeight());
        }
    }

    public double sum(){
        double result = 0;
        if(isBias){
            result += bias;
        }
        for(int i = 0; i < inputs.size(); i++){
            result += weights.get(i) * inputs.get(i);
        }
        return result;
    }

    public void feed(){

        output = Utils.sigmoid(sum());
    }

    public void countDelta(double nextLayerDeltaSupply){
        delta = nextLayerDeltaSupply * Utils.sigmoidDerivative(sum());
    }

    public void improveWeights(double alpha){
        if(isBias){
            bias -= delta * 1;
        }
        for(int i = 0; i < weights.size(); i++){
            weights.set(i,weights.get(i) - (alpha * delta * inputs.get(i)));
        }
    }

    public void setInputs(Vector<Double> values){
        Vector<Double> copyValues = ((Vector<Double>) values.clone());
        inputs = copyValues;
    }

    public double getOutput(){
        return output;
    }

    public double getDelta(){
        return delta;
    }

    public double getWeight(int index){
        return weights.get(index);
    }

    public void setWeight(int index, double value){
        weights.set(index,value);
    }

    public double getBias(){
        return  bias;
    }

    public boolean isBias(){
        return isBias;
    }
}
