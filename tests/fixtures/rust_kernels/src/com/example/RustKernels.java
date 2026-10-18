package com.example;

/** Fixture kernels for the kernel-port Rust glue backend tests. */
public class RustKernels {

    /** Pure: dot product (read-only array params). */
    public static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** Pure: in-place scaling (void kernel with an in/out array param). */
    public static void scale(double[] a, double f) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= f;
        }
    }

    /** Pure: String in, String out. */
    public static String greet(String name) {
        return "hello " + name;
    }

    /** Pure: boolean scalar result. */
    public static boolean isPositive(int x) {
        return x > 0;
    }

    /** Pure: read-only boolean array param. */
    public static int countTrue(boolean[] flags) {
        int n = 0;
        for (boolean b : flags) {
            if (b) {
                n++;
            }
        }
        return n;
    }

    /** Pure: 2-D add (the glue flattens row-major before the call). */
    public static double[][] matAdd(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] c = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    /** Pure: returns a freshly allocated array. */
    public static int[] iota(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }
}
