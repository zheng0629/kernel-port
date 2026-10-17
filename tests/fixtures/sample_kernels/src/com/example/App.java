package com.example;

/** Drives call-frequency counts for `mine`. */
public class App {
    public static void main(String[] args) {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        // dot is called 3x, fib 2x, the rest 0x
        System.out.println(Kernels.dot(a, b));
        System.out.println(Kernels.dot(a, a));
        System.out.println(Kernels.dot(b, b));
        System.out.println(Kernels.fib(10));
        System.out.println(Kernels.fib(20));
        System.out.println(Kernels.noisy(1));
    }
}
