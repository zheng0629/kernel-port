package com.example;

/**
 * Fixture kernels whose parameter names collide with the FFM wrapper's
 * generated locals (``r``, ``n``, ``data``, ``outLen``, ``ptr``, ``arena``).
 * The generated ``*Ffm.java`` must still compile and verify.
 */
public class CollisionKernels {

    /** Scalar kernel: params literally named r, n, data. */
    public static double weird(double[] data, double r, long n) {
        double s = r;
        for (int i = 0; i < data.length; i++) {
            s += data[i] * n;
        }
        return s;
    }

    /** Array-returning kernel: exercises the outLen/ptr/n/data/r locals. */
    public static double[] make(int n, double r) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = r * i;
        }
        return out;
    }

    /** Void kernel with an in/out array param named data. */
    public static void fill(double[] data, double r) {
        for (int i = 0; i < data.length; i++) {
            data[i] = r;
        }
    }
}
