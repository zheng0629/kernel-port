package com.example;

/** Overloaded pure kernels used to test method-reference resolution. */
public class Overloads {
    public static int add(int a, int b) {
        return a + b;
    }

    public static double add(double a, double b) {
        return a + b;
    }

    public static int add(int a, int b, int c) {
        return a + b + c;
    }

    public static String concat(String a, String b) {
        return a + b;
    }

    public static long only(int n) {
        return n * 2L;
    }
}
