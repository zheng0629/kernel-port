package com.example;

/** Fixture kernels for deterministic-throw purity tests.

 *  factorial()/sqrtPos() use defensive argument validation throws and must
 *  stay pure; firstPositive() throws a non-allowlisted exception type and
 *  must stay impure; noisyFactorial() mutates static state and must stay
 *  impure.
 */
public class ThrowKernels {

    /** Pure: validation throw on an int param, then an iterative loop. */
    public static long factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be >= 0, got " + n);
        }
        long r = 1;
        for (int i = 2; i <= n; i++) {
            r *= i;
        }
        return r;
    }

    /** Pure: validation throw on a double param. */
    public static double sqrtPos(double x) {
        if (!(x >= 0.0)) {
            throw new IllegalArgumentException("x must be non-negative");
        }
        return Math.sqrt(x);
    }

    /** Pure: qualified exception name + no-arg throw. */
    public static int firstPositive(int[] a) {
        if (a == null) {
            throw new java.lang.NullPointerException("a");
        }
        for (int v : a) {
            if (v > 0) {
                return v;
            }
        }
        throw new java.util.NoSuchElementException("empty");
    }

    // mutable global state -- makes noisyFactorial() impure
    static long calls = 0;

    /** Impure control: writes to a static field. */
    public static long noisyFactorial(int n) {
        calls++;
        return factorial(n);
    }
}
