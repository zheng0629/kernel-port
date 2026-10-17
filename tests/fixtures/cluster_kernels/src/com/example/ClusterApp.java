package com.example;

/** Drives external call-frequency counts for cluster entries. */
public class ClusterApp {
    public static void main(String[] args) {
        double[] a = {1.0, 2.0, 3.0};
        // normalize is called 2x from outside its cluster, loudSum 1x
        Cluster.normalize(a);
        Cluster.normalize(a);
        System.out.println(Cluster.loudSum(a));
    }
}
