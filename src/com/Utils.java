package com;

import java.util.Random;
import java.util.Vector;

public class Utils {

    public static double randomWeight(){
        Random rand = new Random();
        return (double)(((rand.nextInt()%11) - 5) / 10.0);
    }

    public static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x){
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static Vector<Double> setZeros(int x){
        Vector<Double> result = new Vector<Double>();
        for(int i = 0; i < x; i++){
            result.add(null);
        }
        return result;
    }
}
