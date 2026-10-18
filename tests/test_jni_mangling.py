"""JNI symbol correctness (v0.12): identifier escaping + overload long-form.

Background
----------
Two real JNI name-mangling gaps, both verified empirically:

1. Any ``_``/``$`` in a class/method/package name generated an export
   symbol the JVM could never resolve (``UnsatisfiedLinkError`` at load).
2. The JVM binds every same-named native declared in one class to a
   single short-form symbol. With overloads present that is *silently
   wrong* dispatch: a probe with only ``Java_Ov_add`` exported showed
   ``add(1, 2, 3)`` calling the 2-arg body and printing ``3`` instead of
   ``6`` -- no error. The JNI long form (``__<mangled-signature>``)
   disambiguates by argument signature.

Single-method glue always declares exactly one native per generated
class, so the short form stays correct there; class glue (many natives
in one class) uses the long form for entry names that collide.
"""

import re
import subprocess
from pathlib import Path

import pytest

from kernel_port import classglue, jni, overloading
from kernel_port import harness as H
from kernel_port import java_ast as J

OVERLOAD_REPO = Path(__file__).parent / "fixtures" / "overload_kernels"


def test_escape_underscore_in_method():
    assert (
        jni._jni_func_name("", "KernelsNative", "foo_bar")
        == "Java_KernelsNative_foo_1bar"
    )


def test_escape_underscore_in_package():
    assert (
        jni._jni_func_name("com.my_pkg", "K", "f") == "Java_com_my_1pkg_K_f"
    )


def test_escape_dollar():
    assert jni._jni_func_name("", "K", "f$g") == "Java_K_f_00024g"


def test_no_escapes_needed_is_stable():
    # Legacy output is byte-stable when no special chars are present.
    assert (
        jni._jni_func_name("", "KernelsNative", "dot") == "Java_KernelsNative_dot"
    )
    assert (
        jni._jni_func_name("com.example", "KernelsNative", "dot")
        == "Java_com_example_KernelsNative_dot"
    )


def test_long_suffix_primitives():
    repo = J.scan_repo(OVERLOAD_REPO)
    m = overloading.parse_method_ref(repo, "com.example.Overloads.add(int,int)")
    assert jni.jni_long_suffix(m) == "__II"


def test_long_suffix_string():
    # concat(String, String): two params, each Ljava/lang/String; escaped
    # per javac -h ground truth (slashes become plain underscores).
    repo = J.scan_repo(OVERLOAD_REPO)
    m = overloading.parse_method_ref(repo, "com.example.Overloads.concat")
    assert jni.jni_long_suffix(m) == "__Ljava_lang_String_2Ljava_lang_String_2"


def test_classglue_overloads_get_long_symbols(tmp_path):
    repo = J.scan_repo(OVERLOAD_REPO)
    files = classglue.generate_class(repo, "com.example.Overloads", tmp_path / "glue")
    cpp = files["cpp"].read_text(encoding="utf-8")
    assert "Java_OverloadsNative_add__II" in cpp
    assert "Java_OverloadsNative_add__DD" in cpp
    assert "Java_OverloadsNative_add__III" in cpp
    # No bare short symbol left for the JVM to bind every overload to.
    assert not re.search(r"Java_OverloadsNative_add(?!_)", cpp)
    # Non-overloaded methods keep the short form.
    assert "Java_OverloadsNative_concat" in cpp
    assert "Java_OverloadsNative_only" in cpp
    java = files["java"].read_text(encoding="utf-8")
    assert "native int add(int a, int b);" in java
    assert "native double add(double a, double b);" in java
    assert "native int add(int a, int b, int c);" in java


def _fill_body(text: str, helper_sig: str, body: str) -> str:
    start = text.index(helper_sig)
    b = text.index("// ==== BEGIN TRANSLATED BODY ====", start)
    e = text.index("// ==== END TRANSLATED BODY ====", b) + len(
        "// ==== END TRANSLATED BODY ===="
    )
    return text[:b] + body + "\n" + text[e:]


@pytest.mark.e2e
def test_e2e_overload_dispatch_correct(tmp_path):
    """Each overload calls its own native body (no silent cross-dispatch)."""
    repo = J.scan_repo(OVERLOAD_REPO)
    glue_dir = tmp_path / "glue"
    files = classglue.generate_class(repo, "com.example.Overloads", glue_dir)
    text = files["cpp"].read_text(encoding="utf-8")
    text = _fill_body(
        text,
        "static int32_t kernel_add(int32_t p_a, int32_t p_b)",
        "    return p_a + p_b;",
    )
    text = _fill_body(
        text,
        "static double kernel_add(double p_a, double p_b)",
        "    return p_a + p_b;",
    )
    text = _fill_body(
        text,
        "static int32_t kernel_add(int32_t p_a, int32_t p_b, int32_t p_c)",
        "    return p_a + p_b + p_c;",
    )
    files["cpp"].write_text(text, encoding="utf-8")

    driver = tmp_path / "RunOv.java"
    driver.write_text(
        "public class RunOv {\n"
        "    public static void main(String[] a) {\n"
        "        System.out.println(OverloadsNative.add(1, 2));\n"
        "        System.out.println(OverloadsNative.add(1.5, 2.5));\n"
        "        System.out.println(OverloadsNative.add(1, 2, 3));\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    classes = tmp_path / "classes"
    H.compile_java([files["java"], driver], classes)
    so = tmp_path / "libkernels.so"
    H.compile_jni(files["cpp"], so)
    proc = subprocess.run(
        [
            "java",
            "-Djava.library.path=" + str(tmp_path),
            "-cp",
            str(classes),
            "RunOv",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # Before the fix, add(1,2,3) silently printed 3 (dispatched to the
    # 2-arg body). Now each overload reaches its own implementation.
    assert proc.stdout.split() == ["3", "4.0", "6"], proc.stdout
