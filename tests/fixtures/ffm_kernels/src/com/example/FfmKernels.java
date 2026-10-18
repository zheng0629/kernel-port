package com.example;

/** Fixture kernels for the kernel-port FFM (Panama) glue backend tests. */
public class FfmKernels {

    /** Pure: dot product. */
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
}
