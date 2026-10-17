package com.example;

/** Mutable counter: non-final field written by increment() -- not portable. */
public class Counter {

    int count;

    public Counter() {
        this.count = 0;
    }

    /** Mutates instance state. */
    public void increment() {
        count++;
    }

    public int get() {
        return count;
    }
}
