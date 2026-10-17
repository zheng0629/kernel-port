package com.example;

/** Stateless utility class: only static pure methods + a static final constant. */
public class Util {

    static final double GOLDEN = 1.6180339887;

    /** Pure: linear interpolation. */
    public static double lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    /** Pure: clamp into [lo, hi]. */
    public static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /** Pure: scale by the golden-ratio constant. */
    public static double goldenScale(double v) {
        return v * GOLDEN;
    }
}
