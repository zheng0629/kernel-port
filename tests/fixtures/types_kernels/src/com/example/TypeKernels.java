package com.example;

/** Fixture kernels for kernel-port v0.4: 2-D arrays, String, float tolerance.
 *
 * <p>Note on purity (v0.3 checker): {@code matMul2d}, {@code add2d},
 * {@code scale2d}, {@code greet}, {@code repeat} and {@code fsum} are pure --
 * only array creation, array/length access and string concatenation are used.
 * {@code countChar} is IMPURE per the current checker ({@code s.charAt(i)} /
 * {@code s.length()} are unknown method calls); it is included for
 * harness/glue coverage of String parameters, not for mine/extract.
 */
public class TypeKernels {

    /** Pure: C = A*B for square matrices, 2-D form. */
    public static double[][] matMul2d(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                c[i][j] = s;
            }
        }
        return c;
    }

    /** Pure: elementwise sum of two int matrices. */
    public static int[][] add2d(int[][] a, int[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        int[][] c = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    /** Pure: scale a matrix in place (exercises 2-D in/out params). */
    public static void scale2d(double[][] a, double k) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                a[i][j] *= k;
            }
        }
    }

    /** Pure: greeting via string concatenation (no String method calls). */
    public static String greet(String name) {
        return "hello " + name;
    }

    /** Pure: s repeated n times via concatenation. */
    public static String repeat(String s, int n) {
        String r = "";
        for (int i = 0; i < n; i++) {
            r = r + s;
        }
        return r;
    }

    /** Impure per the v0.3 purity checker (String method calls); kept for
     * harness/glue coverage of String parameters. */
    public static int countChar(String s, int ch) {
        int n = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ch) {
                n++;
            }
        }
        return n;
    }

    /** Pure: float accumulation -- the per-type tolerance fixture. */
    public static float fsum(float[] a) {
        float s = 0.0f;
        for (int i = 0; i < a.length; i++) {
            s += a[i];
        }
        return s;
    }
}
