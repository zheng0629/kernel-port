"""Shared fixtures for kernel-port tests."""

from pathlib import Path

import pytest

from kernel_port import java_ast as J

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_kernels"

# Hand-written C++ translations, standing in for the calling agent's LLM step.
TRANSLATIONS = {
    "com.example.Kernels.dot": """    double s = 0.0;
    for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];
    return s;""",
    "com.example.Kernels.fib": """    if (p_n <= 1) return p_n;
    int64_t a = 0, b = 1;
    for (int i = 2; i <= p_n; i++) { int64_t t = a + b; a = b; b = t; }
    return b;""",
    # Deliberately wrong translation (off by one) for negative testing.
    "com.example.Kernels.dot_WRONG": """    double s = 0.0;
    for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];
    return s + 1.0;""",
    "com.example.Kernels.matMul": """    for (int i = 0; i < p_n; i++) {
        for (int j = 0; j < p_n; j++) {
            double s = 0.0;
            for (int k = 0; k < p_n; k++) s += p_a[i * p_n + k] * p_b[k * p_n + j];
            const_cast<double*>(p_c)[i * p_n + j] = s;
        }
    }""",
    # Deliberately wrong void translation: output array perturbed.
    "com.example.Kernels.matMul_WRONG": """    for (int i = 0; i < p_n; i++) {
        for (int j = 0; j < p_n; j++) {
            double s = 0.0;
            for (int k = 0; k < p_n; k++) s += p_a[i * p_n + k] * p_b[k * p_n + j];
            const_cast<double*>(p_c)[i * p_n + j] = s + 1.0;
        }
    }""",
}


@pytest.fixture()
def repo():
    return J.scan_repo(FIXTURE_REPO)


def fill_translation(glue_dir: Path, qualname: str, body: str) -> None:
    """Replace the TODO(agent) block in kernels_jni.cpp with `body`."""
    cpp = glue_dir / "kernels_jni.cpp"
    text = cpp.read_text(encoding="utf-8")
    start_marker = "    // ==== BEGIN TRANSLATED BODY ===="
    end_marker = "    // ==== END TRANSLATED BODY ===="
    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)
    cpp.write_text(text[:start] + body.rstrip() + "\n" + text[end:], encoding="utf-8")
