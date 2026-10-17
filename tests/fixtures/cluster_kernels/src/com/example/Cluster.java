package com.example;

/**
 * Fixture kernels for function-cluster mining tests (v0.2 workstream C1).
 *
 * <p>Disjoint helper chains, deliberately kept separate from
 * {@code sample_kernels} so call-frequency counts there are untouched:
 *
 * <ul>
 *   <li>normalize -&gt; dot, scale (all pure): one 3-member cluster
 *   <li>loudSum -&gt; trace (trace prints): cluster emitted with pure=false
 *   <li>blend -&gt; ident, plus an ambiguous call to the overloaded mix(int)/mix(int,int)
 *   <li>loneFib: pure method with no repo callees: produces no cluster
 * </ul>
 */
public class Cluster {

    /** Pure helper: dot product. */
    static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** Pure helper: scales an array in place. */
    static void scale(double[] a, double k) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= k;
        }
    }

    /** Cluster entry: normalizes an array in place via dot() and scale(). */
    public static void normalize(double[] a) {
        double n = Math.sqrt(dot(a, a));
        if (n > 0) {
            scale(a, 1.0 / n);
        }
    }

    /** Impure helper: prints its argument. */
    static double trace(double x) {
        System.out.println(x);
        return x;
    }

    /** Cluster entry whose helper is impure. */
    public static double loudSum(double[] a) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += trace(a[i]);
        }
        return s;
    }

    /** Overloaded helpers: make any call to mix(...) ambiguous by name. */
    static int mix(int x) {
        return x + 1;
    }

    static int mix(int x, int y) {
        return x + y;
    }

    /** Pure helper with a unique name. */
    static int ident(int x) {
        return x;
    }

    /** Cluster entry calling the ambiguous mix(...) plus the pure ident(...). */
    public static int blend(int x) {
        return mix(x) + ident(x);
    }

    /** Lone pure method: no repo-defined callees, so no cluster. */
    public static long loneFib(int n) {
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
}
