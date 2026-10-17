package com.example;

/** Immutable 2D point: final fields assigned only in the constructor. */
public class Point {

    final double x;
    final double y;

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    /** Pure: distance from the origin. */
    public double norm() {
        return Math.sqrt(x * x + y * y);
    }

    /** Pure: Euclidean distance to another point. */
    public double distance(Point other) {
        double dx = x - other.x;
        double dy = y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}
