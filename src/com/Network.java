package com;

import sun.misc.GC;

import java.util.Vector;

public class Network {

    private Vector<Layer> layers;
    private Vector<Double> trainingInputs;
    private Vector<Double> trainingOutputs;
    private Vector<Double> outputs;
    private double error;
    private double alpha;
    private int iterator;

    public Network(int[] topology, int in, boolean isBias, double alpha){
        layers = new Vector<Layer>(topology.length);
        outputs = Utils.setZeros(topology[topology.length - 1]);
        this.alpha = alpha;
        layers.add(new Layer(topology[0],in,isBias));
        for(int i = 1; i < topology.length; i++){
            layers.add(new Layer(topology[i],topology[i - 1],isBias));
        }
    }

    public void loadData(Vector<Double> trainingInputs, Vector<Double> trainingOutputs){
        this.trainingInputs = ((Vector<Double>) trainingInputs.clone());
        this.trainingOutputs = ((Vector<Double>) trainingOutputs.clone());
    }

    public void feed(){
        layers.get(0).setInputs(trainingInputs);
        layers.get(0).feed();
        for(int i = 1; i < layers.size() - 1; i++){
            layers.get(i).setInputs(layers.get(i - 1).getOutputs());
            layers.get(i).feed();
        }
        int lastLayer = layers.size() - 1;
        layers.get(lastLayer).setInputs(layers.get(lastLayer - 1).getOutputs());
        layers.get(lastLayer).feed();
        outputs = layers.get(lastLayer).getOutputs();
    }

    public void errorCheck(){
        error = 0;
        for(int i = 0; i < outputs.size(); i++) {
            error += 0.5 * (outputs.get(i) - trainingOutputs.get(i)) * (outputs.get(i) - trainingOutputs.get(i));
        }
        error /= outputs.size();
    }

    public void backPropagation(double e){

        do {
            feed();
            errorCheck();
            if(error < e){
                break;
            }
            int lastLayer = layers.size() - 1;
            layers.get(lastLayer).countLastLayerDeltas(trainingOutputs);
            for(int i = lastLayer - 1; i >= 0; i--){
                layers.get(i + 1).countDeltasSumForPreviousLayer();
                layers.get(i).countDeltas(layers.get(i + 1).getDeltasSumForPreviousLayer());
            }
            for(int i = lastLayer; i >= 0; i--){
                layers.get(i).improveWeights(alpha);
            }
        iterator++;
        }while(true);
    }

    public double getError(){
        return error;
    }

    public Vector<Double> getOutputs(){
        return ((Vector<Double>) outputs.clone());
    }

    public Vector<Double> getTrainingOutputs(){
        return ((Vector<Double>) trainingOutputs.clone());
    }

    public int getIterator() {
        return iterator;
    }
}
