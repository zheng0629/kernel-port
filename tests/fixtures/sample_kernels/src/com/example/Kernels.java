package com.example;

/** Fixture kernels for kernel-port tests: pure + deliberately impure methods. */
public class Kernels {

    static final double EPS = 1e-9;

    // mutable global state -- makes noisy() impure
    static int callCount = 0;

    /** Pure: dot product. */
    public static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** Pure helper, used by norm() -- exercises the dependency closure. */
    static double square(double x) {
        return x * x;
    }

    /** Pure: Euclidean norm, calls the helper above. */
    public static double norm(double[] a) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += square(a[i]);
        }
        return Math.sqrt(s);
    }

    /** Pure: dense matrix multiply on flat row-major arrays, C = A*B. */
    public static void matMul(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    /** Pure: Sieve of Eratosthenes, returns primes below n. */
    public static int[] sieve(int n) {
        if (n < 2) {
            return new int[0];
        }
        boolean[] composite = new boolean[n];
        for (int i = 2; i * i < n; i++) {
            if (!composite[i]) {
                for (int j = i * i; j < n; j += i) {
                    composite[j] = true;
                }
            }
        }
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (!composite[i]) {
                count++;
            }
        }
        int[] out = new int[count];
        int k = 0;
        for (int i = 2; i < n; i++) {
            if (!composite[i]) {
                out[k++] = i;
            }
        }
        return out;
    }

    /** Pure: iterative Fibonacci. */
    public static long fib(int n) {
        if (n <= 1) {
            return n;
        }
        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long t = a + b;
            a = b;
            b = t;
        }
        return b;
    }

    /** Impure: touches System.out, a mutable static field, and the clock. */
    public static int noisy(int x) {
        System.out.println("computing " + x);
        callCount++;
        return x + (int) (System.currentTimeMillis() % 1000);
    }
}
