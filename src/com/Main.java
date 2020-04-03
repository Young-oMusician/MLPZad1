package com;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;
import java.util.Vector;

public class Main {

    public static Vector<Vector<Double>> readFile(String path) {
        Scanner scan;
        Vector<Vector<Double>> skan = new Vector<Vector<Double>>();
        Vector<Double> values = new Vector<Double>();
        String line = new String();
        String[] splitedLine;
        File file = new File(path);
        try {
            scan = new Scanner(file);

            while (scan.hasNextLine()) {
                line = scan.nextLine();
                splitedLine = line.split(" ");
                for(String i : splitedLine){
                    values.add(Double.valueOf(i));
                }
                skan.add((Vector<Double>)values.clone());
                values.clear();
            }

            return  skan;
        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
            return skan;
        }

    }


    final static double EPSILON = 0.0001;

    public static void main(String[] args) throws IOException{

        Vector<Vector<Double>> inputs = readFile("../transformation.txt");
        Vector<Vector<Double>> outputs = ((Vector<Vector<Double>>) inputs.clone());
        int[] top = {2,4};
        Network net = new Network(top, 4,false, 0.01,0);
        Integer[] o = {0,1,2,3};
        String data;
        Vector<Integer> order = new Vector<Integer>(Arrays.asList(o));
        File plik = new File("../trans_1_neuron_zb.txt");
        int iterator = 0;
        try{
            plik.createNewFile();
        }catch(IOException ex){
            System.out.println(ex.getMessage());
        }
        FileWriter writer = new FileWriter(plik);
        double error = 0;
            for(int k = 0; k < 120000; k++){
                iterator++;
                Collections.shuffle(order);
                for (int i = 0; i < order.size(); i++) {
                    net.loadData(inputs.get(order.get(i)), outputs.get(order.get(i)));
                    net.backPropagation(EPSILON);
                }
                error = 0;
                for (int i = 0; i < order.size(); i++) {
                    net.loadData(inputs.get(order.get(i)), outputs.get(order.get(i)));
                    net.feed();
                    error += net.getError();
                    error /= 4;
                }
                if(k % 200 == 0){
//                    System.out.println(iterator);
//                    break;
                    data = new String(k + " " + error + "\n");
                    try{
                        writer.write(data);

                    }catch(IOException ex){
                        System.out.println(ex.getMessage());
                    }
                }
            }

            writer.close();

            Collections.shuffle(order);
            for(int j = 0; j < order.size(); j++) {

                net.loadData(inputs.get(order.get(j)), outputs.get(order.get(j)));
                net.feed();
                for (int i = 0; i < order.size(); i++) {
                    System.out.println("Network Output: " + net.getOutputs().get(i) + "    Training: " + net.getTrainingOutputs().get(i));
                }
                System.out.println("Error " + error);
            }
    }
}
