package com;

import java.util.Vector;

public class Layer {

    private Vector<Neural> neurals;
    private Vector<Double> inputs;
    private Vector<Double> outputs;
    private Vector<Double> deltasSumForPreviousLayer;
    private Vector<Double> deltas;
    private int previousLayerOutputsNumber;

    public Layer(int neuralsNumber, int previousLayerOutputsNumber, boolean isBias){
        neurals = new Vector<Neural>(neuralsNumber);
        inputs = Utils.setZeros(previousLayerOutputsNumber);
        outputs = Utils.setZeros(neuralsNumber);
        deltasSumForPreviousLayer = Utils.setZeros(previousLayerOutputsNumber);
        deltas = Utils.setZeros(neuralsNumber);
        this.previousLayerOutputsNumber = previousLayerOutputsNumber;

        for(int i = 0; i < neuralsNumber; i++){

            neurals.add(new Neural(previousLayerOutputsNumber, isBias));
        }
    }

    public void feed(){
        for(int i = 0; i < neurals.size(); i++){
            neurals.get(i).setInputs(inputs);
            neurals.get(i).feed();
            outputs.set(i,neurals.get(i).getOutput());
        }
    }

    public void countLastLayerDeltas(Vector<Double> trainingData){
        for(int i = 0; i < neurals.size(); i++){
            neurals.get(i).countDelta(outputs.get(i) - trainingData.get(i));
            deltas.set(i,neurals.get(i).getDelta());
        }
    }

    public void countDeltasSumForPreviousLayer(){
        for(int i = 0; i < previousLayerOutputsNumber; i++){
            double sum = 0;
            for(int j = 0; j < neurals.size(); j++){
                sum += deltas.get(j) * neurals.get(j).getWeight(i);
            }
            deltasSumForPreviousLayer.set(i,sum);
        }
    }

    public void countDeltas(Vector<Double> deltaSumsOfNextLayer){
        for(int i = 0; i < neurals.size(); i++){
            neurals.get(i).countDelta(deltaSumsOfNextLayer.get(i));
            deltas.set(i,neurals.get(i).getDelta());
        }
    }

    public void improveWeights(double alpha, double beta){
        for(int i = 0; i < neurals.size(); i++){
            neurals.get(i).improveWeights(alpha, beta);
        }
    }

    public void resetWeights(){
        for(int i = 0; i < neurals.size(); i++){
            neurals.get(i).resetWeights();
        }
    }
    
    public void setInputs(Vector<Double> values){
        inputs = ((Vector<Double>) values.clone());
    }
    
    public Vector<Double> getOutputs(){
        return ((Vector<Double>) outputs.clone());
    }

    public Vector<Double> getDeltasSumForPreviousLayer(){
        return ((Vector<Double>) deltasSumForPreviousLayer.clone());
    }
}
