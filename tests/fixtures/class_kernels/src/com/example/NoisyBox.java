package com.example;

/** Structurally immutable, but describe() does console I/O -- not portable. */
public class NoisyBox {

    final double value;

    public NoisyBox(double value) {
        this.value = value;
    }

    public double get() {
        return value;
    }

    /** Impure: writes to System.out. */
    public double describe() {
        System.out.println("value=" + value);
        return value;
    }
}
