"""`glue --backend ffm`: generate Panama Foreign Function & Memory (FFM) bindings.

A second glue backend next to :mod:`kernel_port.jni`. Instead of a
hand-written JNI C wrapper, this generates:

- ``kernels_ffm.c``: a pure-C file with C ABI exports (``KP_API`` =
  default visibility / ``__declspec(dllexport)``), **no ``<jni.h>``**.
  The calling agent pastes the translated body into ``kernel_<name>()``
  (marked ``TODO(agent)``), exactly like the JNI helper.
- ``<Cls>Ffm.java``: a pure-Java class using only the stable
  ``java.lang.foreign`` API (``Linker``, ``SymbolLookup``,
  ``FunctionDescriptor``, ``MemorySegment``, ``Arena``) -- no JNI, no
  ``native`` methods. Its public static methods mirror the original
  kernel signatures, so the same line-protocol harness convention as the
  JNI path works: a harness runner simply calls ``<Cls>Ffm.<name>(...)``
  instead of ``<Cls>Native.<name>(...)``. Run it with
  ``-Djava.library.path=<dir containing libkernels_ffm.so>``.
- ``build_ffm.sh``: ``cc -shared -fPIC -O2`` -> ``libkernels_ffm.so``,
  then ``javac`` the binding class. Paths resolve relative to the
  script's own directory, so the glue dir can be renamed or moved.

The generated dispatcher loads the native library *lazily* (first
native-call attempt, never in a static initializer): a missing
``libkernels_ffm.so`` falls back to the original Java implementation
with a one-line stderr note instead of failing class initialization.
``-Dkernelport.native=false`` / ``setUseNative(false)`` still force the
Java path without touching the loader -- the same switch semantics as
the JNI backend (which keeps its eager static-block load).

REQUIRES JDK 22+ (25 LTS recommended) to compile and run the Java side;
the FFM API used here is stable since JDK 22. kernel-port must refuse
with a clear error on older JDKs -- see :func:`java_version_ok` (the CLI
wires the refusal; this module only reports).

C-ABI mapping choices (v0.1 type set, same as the JNI backend: ``void``,
primitives, 1-D primitive arrays):

- Fixed-width C types (``int8_t``/``int32_t``/``int64_t``/...) paired
  with the fixed-size ``ValueLayout.JAVA_*`` layouts, so layout size and
  alignment match exactly on every platform (unlike the
  platform-dependent ``C_INT``/``C_LONG``).
- ``T[]`` becomes ``const T* p, int64_t p_len``: a ``MemorySegment``
  (``ADDRESS``) plus a ``JAVA_LONG`` length. ``int64_t`` (not ``int``) is
  used for the length so the rule is uniform and huge buffers never
  overflow a 32-bit length.
- Arrays are IN/OUT: the Java wrapper allocates off-heap with
  ``arena.allocateFrom(LAYOUT, javaArray)``, invokes, then copies mutated
  elements back with ``MemorySegment.copy``. Copy-back is skipped
  automatically for array params that ``jni.written_arrays()`` proves
  read-only (conservative: it errs towards "written").
- ``boolean`` -> C ``bool`` / ``JAVA_BOOLEAN``; ``char`` -> C
  ``uint16_t`` / ``JAVA_CHAR`` (same 2-byte unsigned layout, avoids
  ``<uchar.h>`` portability issues).
- A kernel *returning* ``T[]`` becomes ``T* kernel_x(..., int64_t* out_len)``:
  the translated C body ``malloc``s the result; the Java wrapper copies
  it into a Java array and releases it via the generated ``kp_free()``
  downcall. Ownership rule: C allocates, Java frees via ``kp_free``.

The harness line protocol (``harness.render_harness``) can drive the FFM
binding by using receiver ``<Cls>Ffm`` instead of ``<Cls>Native``:
``verify`` compares the echoed arrays/return values the same way, and
``bench`` clones the arrays per iteration the same way. The parent CLI
wires a ``--backend ffm`` switch that picks the receiver; this module
only needs to keep the public static signatures identical to the kernel.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import java_ast as J
from . import jni  # read-only: reuse the v0.1 type table

#: C fixed-width type -> FFM ValueLayout constant (all fixed-size layouts).
_C_TO_LAYOUT: dict[str, str] = {
    "int8_t": "JAVA_BYTE",
    "int16_t": "JAVA_SHORT",
    "int32_t": "JAVA_INT",
    "int64_t": "JAVA_LONG",
    "float": "JAVA_FLOAT",
    "double": "JAVA_DOUBLE",
    "bool": "JAVA_BOOLEAN",
    "uint16_t": "JAVA_CHAR",  # Java char: 2-byte unsigned
}

LIB_NAME = "kernels_ffm"  # System.loadLibrary("kernels_ffm") / libkernels_ffm.so


@dataclass
class FfmPlan:
    method: J.MethodInfo
    qualname: str  # com.example.FfmKernels.dot
    class_name: str  # FfmKernelsFfm  (default package, like <Cls>Native)
    c_name: str  # kernel_dot
    lib_name: str = LIB_NAME
    package: str = ""  # v0.12: kernel's package; "" = legacy default package
    java_return: str = ""  # e.g. "double"
    # FFM ValueLayout constant for the downcall return ("", "ADDRESS" for
    # array returns, "" for void -> FunctionDescriptor.ofVoid).
    java_return_layout: str = ""
    # FFM ValueLayout constants for the downcall params, one per C param:
    # e.g. ["ADDRESS", "JAVA_LONG", "ADDRESS", "JAVA_LONG"] (arrays
    # contribute ADDRESS + JAVA_LONG each).
    java_param_layouts: list[str] = field(default_factory=list)
    c_signature: str = ""  # "double kernel_dot(const double* p_a, int64_t p_a_len, ...)"


def _map(jtype: str) -> tuple[str, str, bool, str]:
    """(jni_type, ctype, is_array, elem_ctype) -- reuses jni's v0.1 table.

    Raises ValueError with a clear message for unsupported types, same as
    the JNI backend.
    """
    return jni._map(jtype)  # read-only reuse; raises ValueError as-is


def _c_scalar(ctype: str) -> str:
    """C declaration type for a scalar: char16_t -> uint16_t (same layout)."""
    return "uint16_t" if ctype == "char16_t" else ctype


def _layout(ctype: str) -> str:
    try:
        return _C_TO_LAYOUT[_c_scalar(ctype)]
    except KeyError:
        raise ValueError(f"no FFM layout mapping for C type '{ctype}'")


def plan(method: J.MethodInfo, package: str = "") -> FfmPlan:
    if not method.is_static:
        raise ValueError("ffm glue requires a static method")
    _map(method.jtype)
    for p in method.params:
        _map(p.jtype)

    cls_simple = method.class_qualname.rsplit(".", 1)[-1]
    c_params: list[str] = []
    param_layouts: list[str] = []
    ret_jni, ret_ctype, ret_is_array, _ = _map(method.jtype)
    ret_c = "void" if method.jtype == "void" else _c_scalar(ret_ctype)

    for p in method.params:
        _jt, ctype, is_array, _e = _map(p.jtype)
        ctype = _c_scalar(ctype)
        if is_array:
            c_params.append(f"const {ctype}* p_{p.name}")
            c_params.append(f"int64_t p_{p.name}_len")
            param_layouts += ["ADDRESS", "JAVA_LONG"]
        else:
            c_params.append(f"{ctype} p_{p.name}")
            param_layouts.append(_layout(ctype))

    if ret_is_array:
        # T* kernel_x(..., int64_t* out_len): C mallocs, Java copies + kp_free.
        ret_c = f"{_c_scalar(ret_ctype)}*"
        c_params.append("int64_t* out_len")
        param_layouts.append("ADDRESS")  # the out_len pointer param
        ret_layout = "ADDRESS"
    elif method.jtype == "void":
        ret_layout = ""
    else:
        ret_layout = _layout(ret_ctype)

    c_name = f"kernel_{method.name}"
    return FfmPlan(
        method=method,
        qualname=method.qualname,
        class_name=f"{cls_simple}Ffm",
        c_name=c_name,
        java_return=method.jtype,
        java_return_layout=ret_layout,
        java_param_layouts=param_layouts,
        c_signature=f"{ret_c} {c_name}({', '.join(c_params)})",
        package=package,
    )


def render_c(fp: FfmPlan) -> str:
    """Render ``kernels_ffm.c``: pure C ABI export, no ``<jni.h>``."""
    m = fp.method
    _jt, ret_ctype, ret_is_array, _e = _map(m.jtype)
    if m.jtype == "void":
        placeholder = ""
    elif ret_is_array:
        placeholder = "    *out_len = 0; return NULL; // placeholder"
    elif m.jtype in ("float", "double"):
        placeholder = "    return 0.0; // placeholder"
    elif m.jtype == "boolean":
        placeholder = "    return false; // placeholder"
    elif m.jtype == "char":
        placeholder = "    return 0; // placeholder"
    else:
        placeholder = "    return 0; // placeholder"

    lines = [
        "// Generated by kernel-port FFM glue (Panama Foreign Function & Memory API).",
        "// Do not hand-edit: regenerate with",
        f"//   kernel-port glue --backend ffm --method {m.qualname}",
        "//",
        "// Pure-C ABI exports -- NO <jni.h>, no JNI. The calling agent pastes",
        f"// the translated body into {fp.c_name}() below (marked TODO(agent)).",
        "//",
        "// Array params arrive as (pointer, int64 length) and are IN/OUT: the",
        "// generated Java wrapper copies them back after the call. Kernels",
        "// that write through an array param must cast away const with a",
        "// plain C cast, e.g. ((double*)p_a)[i] = ... ; Kernels that return",
        "// arrays must malloc() the result and store its length in *out_len;",
        "// the Java side copies it out and releases it via kp_free().",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <stdbool.h>",
        "#include <stdlib.h>",
        "",
        "#if defined(_WIN32) || defined(__CYGWIN__)",
        "#  define KP_API __declspec(dllexport)",
        "#else",
        '#  define KP_API __attribute__((visibility("default")))',
        "#endif",
        "",
        "// ---- Translated kernel (filled in by the agent) ----",
        f"KP_API {fp.c_signature} {{",
        "    // ==== BEGIN TRANSLATED BODY ====",
        f"    // TODO(agent): paste the translated body of {m.qualname} here.",
        "    // It must compute exactly what the Java original computes.",
        placeholder if placeholder else "    // ==== END TRANSLATED BODY ====",
    ]
    if placeholder:
        lines.append("    // ==== END TRANSLATED BODY ====")
    lines += [
        "}",
        "",
        "// Releases buffers returned by array-returning kernels.",
        "KP_API void kp_free(void* p) { free(p); }",
        "",
    ]
    return "\n".join(lines)


def _java_wrapper_body(fp: FfmPlan) -> list[str]:
    """Java statements implementing the FFM downcall wrapper method.

    The IN allocation (``allocateFrom``) is always emitted; the OUT
    copy-back is skipped for array params that ``jni.written_arrays()``
    proves read-only (no repo is passed, so the analysis is
    conservative and errs towards "written" -- a skipped copy-back is
    therefore always safe). A comment in the generated code records the
    skip.

    Every generated local gets a ``kp_`` prefix and is then freshened
    against the kernel's parameter names, so a kernel whose params are
    literally named ``r``, ``n``, ``data``, ``outLen``, ``ptr`` or
    ``arena`` (or even ``kp_r``) still generates compilable code.
    References to the class-level downcall handles are qualified with the
    class name for the same reason.
    """
    m = fp.method
    cls = fp.class_name
    mh = f"{cls}.MH_{m.name.upper()}"
    mh_free = f"{cls}.MH_KP_FREE"
    taken: set[str] = {p.name for p in m.params}

    def fresh(base: str) -> str:
        name, i = base, 2
        while name in taken:
            name, i = f"{base}_{i}", i + 1
        taken.add(name)
        return name

    arena_v = fresh("kp_arena")
    # Conservative without a repo: unknown -> written -> copy-back kept.
    written = jni.written_arrays(m)
    body: list[str] = [f"        try (Arena {arena_v} = Arena.ofConfined()) {{"]
    invoke_args: list[str] = []
    copy_back: list[str] = []
    skipped: list[str] = []
    for p in m.params:
        _jt, ctype, is_array, _e = _map(p.jtype)
        layout = _layout(ctype)
        if is_array:
            seg = fresh(f"kp_s_{p.name}")
            body.append(
                f"            MemorySegment {seg} = {arena_v}.allocateFrom("
                f"ValueLayout.{layout}, {p.name});"
            )
            invoke_args += [seg, f"(long) {p.name}.length"]
            if p.name in written:
                copy_back.append(
                    f"            MemorySegment.copy({seg}, ValueLayout.{layout}, 0L,"
                    f" {p.name}, 0, {p.name}.length);"
                )
            else:
                skipped.append(p.name)
        else:
            invoke_args.append(p.name)

    # Copy-back lines for written arrays, preceded by a comment noting any
    # skipped read-only arrays (inserted after the downcall in all cases).
    post_call: list[str] = []
    if skipped:
        post_call.append(
            "            // Copy-back skipped: "
            + ", ".join(f"'{s}'" for s in skipped)
            + " read-only per jni.written_arrays (no writes through them)."
        )
    post_call += copy_back

    _jt, ret_ctype, ret_is_array, _e = _map(m.jtype)
    call = f"{mh}.invokeExact({', '.join(invoke_args)})"

    if m.jtype == "void":
        body.append(f"            {call};")
        body += post_call
    elif ret_is_array:
        layout = _layout(ret_ctype)
        out_len_v = fresh("kp_outLen")
        ptr_v = fresh("kp_ptr")
        n_v = fresh("kp_n")
        data_v = fresh("kp_data")
        r_v = fresh("kp_r")
        body += [
            f"            MemorySegment {out_len_v} = {arena_v}.allocate(ValueLayout.JAVA_LONG);",
            f"            MemorySegment {ptr_v} = (MemorySegment) {mh}.invokeExact("
            f"{', '.join(invoke_args + [out_len_v])});",
            f"            long {n_v} = {out_len_v}.get(ValueLayout.JAVA_LONG, 0L);",
            f"            MemorySegment {data_v} = {ptr_v}.reinterpret({n_v} * ValueLayout.{layout}.byteSize());",
            f"            {m.jtype} {r_v} = {data_v}.toArray(ValueLayout.{layout});",
            f"            {mh_free}.invokeExact({ptr_v});",
            *post_call,
            f"            return {r_v};",
        ]
    else:
        layout = _layout(ret_ctype)
        prim = {"JAVA_BYTE": "byte", "JAVA_SHORT": "short", "JAVA_INT": "int",
                "JAVA_LONG": "long", "JAVA_FLOAT": "float",
                "JAVA_DOUBLE": "double", "JAVA_BOOLEAN": "boolean",
                "JAVA_CHAR": "char"}[layout]
        r_v = fresh("kp_r")
        body.append(f"            {prim} {r_v} = ({prim}) {call};")
        body += post_call
        body.append(f"            return {r_v};")

    t_v = fresh("kp_t")
    body += [
        f"        }} catch (Throwable {t_v}) {{",
        f'            throw new RuntimeException("FFM call {fp.c_name} failed", {t_v});',
        "        }",
    ]
    return body


def render_java(fp: FfmPlan, fallback: bool = True) -> str:
    """Render ``<Cls>Ffm.java``: pure ``java.lang.foreign`` bindings.

    No JNI, no ``native`` methods. Requires JDK 22+ (25 LTS recommended);
    kernel-port refuses to build this on older JDKs via
    :func:`java_version_ok`.

    With ``fallback=True`` (default) the public static method is a
    dispatcher mirroring the JNI backend's: it routes to the private FFM
    downcall method while ``USE_NATIVE`` is true, and to the original Java
    implementation (``<qualname>(...)``) otherwise. Flip the switch without
    rebuilding via ``-Dkernelport.native=false`` or
    ``<Cls>Ffm.setUseNative(false)``. With ``fallback=False`` the public
    method is the bare FFM downcall (the pre-v0.5 output, byte-identical).

    The native library is loaded *lazily*: on the first native-call
    attempt, never in a static initializer. A missing
    ``lib<name>.so`` therefore cannot fail class initialization -- the
    call falls back to the Java original with a one-line stderr note,
    and the failure is sticky (no per-call retry). ``setUseNative(true)``
    re-arms the loader. The JNI backend keeps its eager static-block load;
    the switch name (``kernelport.native``), ``setUseNative``, and the
    fall-back-to-original-Java semantics are identical on both backends.
    """
    m = fp.method
    mh = f"MH_{m.name.upper()}"
    ffm_name = f"{m.name}Ffm"  # private method holding the FFM downcall
    param_ls = ", ".join(f"ValueLayout.{l}" for l in fp.java_param_layouts)
    if fp.java_return_layout:
        descriptor = f"FunctionDescriptor.of(ValueLayout.{fp.java_return_layout}"
        if param_ls:
            descriptor += f", {param_ls}"
        descriptor += ")"
    else:
        # void return: ofVoid takes only the *parameter* layouts
        descriptor = f"FunctionDescriptor.ofVoid({param_ls})"
    java_params = ", ".join(f"{p.jtype} {p.name}" for p in m.params)
    arg_names = ", ".join(p.name for p in m.params)
    ret = "" if m.jtype == "void" else "return "

    lines = [
        "// Generated by kernel-port FFM glue. Do not hand-edit: regenerate with",
        f"//   kernel-port glue --backend ffm --method {m.qualname}",
        "//",
        "// *** REQUIRES JDK 22+ (25 LTS recommended) ***",
        "// This file uses only the stable java.lang.foreign API (Linker,",
        "// SymbolLookup, FunctionDescriptor, MemorySegment, Arena): no JNI, no",
        "// `native` methods. On older JDKs kernel-port refuses to build it with",
        "// a clear error (see ffm.java_version_ok()); upgrade your JDK instead",
        "// of trying to compile this file.",
        "//",
        f"// FFM bindings for {m.qualname}.",
        "// The public static methods mirror the original kernel signatures, so",
        "// the same line-protocol harness convention as the JNI path applies:",
        "// a harness runner calls <Cls>Ffm.<name>(...) instead of",
        "// <Cls>Native.<name>(...). Run with",
        f"//   -Djava.library.path=<dir containing lib{fp.lib_name}.so>",
        "",
    ]
    if fp.package:
        # v0.12: the glue class lives in the kernel's package so it can
        # reference a non-public kernel class.
        lines.append(f"package {fp.package};")
        lines.append("")
    if fallback:
        lines += [
            "//",
            "// Fallback switch: calls go to the FFM port by default. Roll back",
            "// to the original pure-Java implementation instantly, without",
            "// rebuilding, with either of:",
            "//   java -Dkernelport.native=false ...        (JVM system property)",
            f"//   {fp.class_name}.setUseNative(false);    (programmatic override)",
            "//",
            "// Lazy native loading: the library is loaded on the FIRST",
            "// native-call attempt, never in a static initializer, so a",
            f"// MISSING lib{fp.lib_name}.so no longer breaks class",
            "// initialization. If the load fails (UnsatisfiedLinkError), a",
            "// one-line note goes to stderr and the call -- and every later",
            "// call -- falls back to the original method below instead.",
            "// (setUseNative(true) re-arms the loader.)",
            "//",
            "// With the switch off, calls delegate to the original method:",
            f"//   {m.qualname}(...)",
        ]
    lines += [
        "",
        "import java.lang.foreign.*;",
        "import java.lang.invoke.MethodHandle;",
        "",
        f"public class {fp.class_name} {{",
        "    private static final Linker LINKER = Linker.nativeLinker();",
    ]
    if fallback:
        # Lazy fields: assigned by ensureNative() on first use. The native
        # library is deliberately NOT loaded in a static initializer --
        # see the header comment above.
        lines += [
            "",
            "    // Lazily initialized by ensureNative(); see below.",
            "    private static SymbolLookup LOOKUP;",
            f"    private static MethodHandle {mh};",
            "    private static MethodHandle MH_KP_FREE;",
            "",
        ]
    else:
        lines += [
            "    private static final SymbolLookup LOOKUP;",
            "    static {",
            f'        System.loadLibrary("{fp.lib_name}");',
            "        LOOKUP = SymbolLookup.loaderLookup();",
            "    }",
            "",
            f"    private static final MethodHandle {mh} = LINKER.downcallHandle(",
            f'        LOOKUP.findOrThrow("{fp.c_name}"),',
            f"        {descriptor});",
            "",
            "    private static final MethodHandle MH_KP_FREE = LINKER.downcallHandle(",
            '        LOOKUP.findOrThrow("kp_free"),',
            "        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));",
            "",
        ]
    if fallback:
        lines += [
            "    private static boolean USE_NATIVE =",
            '        Boolean.parseBoolean(System.getProperty("kernelport.native", "true"));',
            "    // Sticky outcome of the lazy load: once loading fails we stop",
            "    // retrying and every call takes the Java path (one stderr note).",
            "    private static volatile boolean nativeReady = false;",
            "    private static volatile boolean nativeFailed = false;",
            "",
            "    /** Set false to roll back to the original Java implementation. */",
            "    public static void setUseNative(boolean useNative) {",
            "        USE_NATIVE = useNative;",
            "        if (useNative) {",
            "            // Re-arm the loader: a previously missing library may",
            "            // have been installed since.",
            "            nativeFailed = false;",
            "        }",
            "    }",
            "",
            "    /**",
            "     * Loads the native library and resolves the downcall handles",
            "     * on first use. Returns false -- and prints a one-line note",
            "     * to stderr -- when the library cannot be loaded",
            "     * (UnsatisfiedLinkError); callers then fall back to the",
            "     * original Java implementation.",
            "     */",
            "    private static synchronized boolean ensureNative() {",
            "        if (nativeReady) {",
            "            return true;",
            "        }",
            "        if (nativeFailed) {",
            "            return false;",
            "        }",
            "        try {",
            f'            System.loadLibrary("{fp.lib_name}");',
            "            LOOKUP = SymbolLookup.loaderLookup();",
            f"            {mh} = LINKER.downcallHandle(",
            f'                LOOKUP.findOrThrow("{fp.c_name}"),',
            f"                {descriptor});",
            "            MH_KP_FREE = LINKER.downcallHandle(",
            '                LOOKUP.findOrThrow("kp_free"),',
            "                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));",
            "        } catch (UnsatisfiedLinkError e) {",
            f'            System.err.println("kernel-port FFM: could not load native library \\"{fp.lib_name}\\" (" + e.getMessage() + "); falling back to the Java implementation.");',
            "            nativeFailed = true;",
            "            return false;",
            "        }",
            "        nativeReady = true;",
            "        return true;",
            "    }",
            "",
        ]
        if "private" in m.modifiers:
            # v0.12: the fallback cannot call a private original directly.
            lines.extend(jni._fallback_holder_lines(m, fp.package))
        lines += [
            "    /**",
            "     * Dispatches to the FFM downcall while USE_NATIVE is true",
            "     * and the native library loads, else to the original Java",
            "     * implementation.",
            "     */",
            f"    public static {m.jtype} {m.name}({java_params}) {{",
            f"        if ({fp.class_name}.USE_NATIVE && {fp.class_name}.ensureNative()) {{",
            f"            {ret}{ffm_name}({arg_names});",
            "        } else {",
            *jni._fallback_call_lines(m, fp.package, "            "),
            "        }",
            "    }",
            "",
            "    /** FFM downcall path; mirrors the kernel signature. */",
            f"    private static {m.jtype} {ffm_name}({java_params}) {{",
            *_java_wrapper_body(fp),
            "    }",
            "}",
            "",
        ]
    else:
        lines += [
            "    /** Pure-Java-callable entry point; mirrors the kernel signature. */",
            f"    public static {m.jtype} {m.name}({java_params}) {{",
            *_java_wrapper_body(fp),
            "    }",
            "}",
            "",
        ]
    return "\n".join(lines)


def render_build_script(fp: FfmPlan, out_dir: str | Path) -> str:
    """Render a POSIX shell snippet: ``cc -shared -fPIC -O2`` the C file,
    then ``javac`` the binding class. ``CC``/``JAVAC`` env vars override
    the tools; ``JAVAC`` must point at a JDK >= 22 (see
    :func:`java_version_ok`).

    All paths resolve relative to the script's own directory (not the
    invoking cwd, and not a baked-in directory name), so the script keeps
    working after the glue directory is renamed or moved -- the old form
    baked ``OUT_DIR="<name>"`` in and broke as soon as the directory name
    no longer matched.
    """
    java_rel = (
        "/".join(fp.package.split(".")) + "/" if fp.package else ""
    ) + f"{fp.class_name}.java"
    return f"""#!/bin/sh
# Generated by kernel-port FFM glue for {fp.qualname}.
# Builds lib{fp.lib_name}.so and compiles {fp.class_name}.java.
# Requires JDK 22+ (25 LTS recommended) for the Java side.
# Run it from any directory: paths resolve relative to this script's own
# location, so renaming or moving the glue directory keeps working.
#   CC         - C compiler (default: cc)
#   JAVAC      - javac from a JDK >= 22 (default: javac on PATH)
#   JAVACFLAGS - extra javac flags, e.g. "-cp /path/to/kernel-classes"
#                (needed for the default fallback=True form, whose dispatcher
#                references the original kernel class)
set -eu
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
CC_BIN="${{CC:-cc}}"
JAVAC_BIN="${{JAVAC:-javac}}"
"$CC_BIN" -shared -fPIC -O2 -o "$OUT_DIR/lib{fp.lib_name}.so" "$OUT_DIR/kernels_ffm.c"
# shellcheck disable=SC2086
"$JAVAC_BIN" ${{JAVACFLAGS:-}} -d "$OUT_DIR/classes" "$OUT_DIR/{java_rel}"
"""


def generate(
    method: J.MethodInfo, out_dir: Path, fallback: bool = True, package: str = ""
) -> dict[str, Path]:
    """Write the FFM glue files; return {kind: path}.

    ``fallback=True`` (default) renders the public wrapper as a dispatcher
    with a ``USE_NATIVE`` switch (``-Dkernelport.native=false`` or
    ``<Cls>Ffm.setUseNative(false)``) that rolls back to the original Java
    implementation; see :func:`render_java`. ``fallback=False`` renders the
    bare FFM downcall form. The 2-arg positional call shape used by
    pipeline.py keeps working unchanged.

    ``package`` (v0.12): like :func:`kernel_port.jni.generate` -- when
    non-empty the ``<Cls>Ffm.java`` class is generated in the kernel's
    package so it can reference a non-public kernel class. Use
    :func:`kernel_port.jni.glue_package` to compute it.
    """
    fp = plan(method, package=package)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    java_dir = out_dir
    if package:
        java_dir = out_dir / Path(*package.split("."))
        java_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "java": java_dir / f"{fp.class_name}.java",
        "c": out_dir / "kernels_ffm.c",
        "build": out_dir / "build_ffm.sh",
    }
    files["java"].write_text(render_java(fp, fallback=fallback), encoding="utf-8")
    files["c"].write_text(render_c(fp), encoding="utf-8")
    files["build"].write_text(
        render_build_script(fp, out_dir), encoding="utf-8"
    )
    files["build"].chmod(0o755)
    return files


# ---------------------------------------------------------------------------
# JDK detection and build helpers (used by tests; the CLI wires --backend ffm
# on top of these).
# ---------------------------------------------------------------------------


def _parse_major(version_text: str) -> int | None:
    m = re.search(r"(\d+)(?:\.(\d+))?", version_text)
    if not m:
        return None
    major = int(m.group(1))
    if major == 1:  # old-style "1.8.0_xxx"
        return int(m.group(2) or 0)
    return major


def java_version_ok(binary: str = "javac") -> tuple[bool, str]:
    """Check that ``binary`` is a JDK >= 22 (FFM API is stable since 22).

    Returns ``(ok, message)``; on failure the message explains the
    requirement and suggests JDK 25 LTS. Never raises.
    """
    try:
        proc = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return (
            False,
            f"could not run '{binary} -version': {e}. "
            "kernel-port FFM glue requires JDK 22+ (25 LTS recommended).",
        )
    text = (proc.stdout + "\n" + proc.stderr).strip()
    major = _parse_major(text)
    if major is None:
        return (
            False,
            f"could not parse a JDK version from '{binary} -version' output "
            f"({text!r}). kernel-port FFM glue requires JDK 22+ (25 LTS recommended).",
        )
    if major >= 22:
        return True, f"{binary} reports JDK {major}: OK for FFM glue (>= 22)."
    return (
        False,
        f"{binary} reports JDK {major}, but kernel-port FFM glue requires "
        "JDK 22+ (25 LTS recommended): the stable java.lang.foreign API does "
        "not exist on older JDKs. Install Temurin 25 (or any JDK 22+) and "
        "point KERNEL_PORT_JDK or PATH at it.",
    )


def find_jdk(min_version: int = 22) -> Path | None:
    """Locate a JDK >= ``min_version`` for tests.

    Search order: ``$KERNEL_PORT_JDK`` env var, then
    ``~/workspace/tools/jdk-*/bin/javac``, then ``javac`` on PATH.
    Returns the JDK home directory, or None.
    """
    candidates: list[Path] = []
    env = os.environ.get("KERNEL_PORT_JDK")
    if env:
        candidates.append(Path(env))
    tools = Path.home() / "workspace" / "tools"
    if tools.is_dir():
        candidates += sorted(
            (p for p in tools.glob("jdk-*") if (p / "bin" / "javac").exists()),
            reverse=True,
        )
    for home in candidates:
        javac = home / "bin" / "javac"
        if javac.exists():
            ok, _msg = java_version_ok(str(javac))
            if ok:
                return home
    which = shutil.which("javac")
    if which:
        ok, _msg = java_version_ok(which)
        if ok:
            # JDK home is two levels up from bin/javac
            return Path(which).resolve().parent.parent
    return None


def compile_c(
    src: Path, lib_path: Path, cc: str = "cc", extra_flags: Sequence[str] = ()
) -> None:
    """Compile ``kernels_ffm.c`` -> shared library. Raises on failure."""
    cmd = [cc, "-shared", "-fPIC", "-O2", *extra_flags, "-o", str(lib_path), str(src)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"C compile failed ({' '.join(cmd)}):\n{proc.stderr}")


def compile_java(
    sources: list[Path],
    classes_dir: Path,
    javac: str = "javac",
    sourcepath: list[Path] | None = None,
) -> None:
    """Compile Java sources with the given ``javac``. Raises on failure."""
    import os

    classes_dir.mkdir(parents=True, exist_ok=True)
    cmd = [javac, "-d", str(classes_dir)]
    if sourcepath:
        cmd += ["-sourcepath", os.pathsep.join(str(p) for p in sourcepath)]
    cmd += [str(s) for s in sources]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"javac failed ({' '.join(cmd)}):\n{proc.stderr}")


def run_java(
    java: str,
    classpath: Path,
    main_class: str,
    args: Sequence[str] = (),
    lib_dir: Path | None = None,
    extra_jvm_args: Sequence[str] = (),
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    """Run a Java class; sets ``-Djava.library.path`` when ``lib_dir`` given.

    ``extra_jvm_args`` are inserted before the main class (e.g. for
    ``-Dkernelport.native=false`` in fallback tests).
    """
    cmd = [java, "-cp", str(classpath), *extra_jvm_args]
    if lib_dir is not None:
        cmd.append(f"-Djava.library.path={lib_dir}")
    cmd += [main_class, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def build_and_load_paths(
    out_dir: Path,
    *,
    cc: str = "cc",
    javac: str = "javac",
) -> dict[str, Path]:
    """Compile ``kernels_ffm.c`` -> ``lib<name>.so`` and ``*.java`` ->
    ``classes/`` under ``out_dir``.

    Returns ``{"c_src", "java_src", "lib", "classes", "lib_dir"}``. Raises
    RuntimeError with the tool output on failure. ``javac`` must be from a
    JDK >= 22 -- check with :func:`java_version_ok` first.
    """
    out_dir = Path(out_dir)
    lib_path = out_dir / f"lib{LIB_NAME}.so"
    classes_dir = out_dir / "classes"
    compile_c(out_dir / "kernels_ffm.c", lib_path, cc=cc)
    compile_java(
        [p for p in out_dir.glob("*.java")], classes_dir, javac=javac
    )
    return {
        "c_src": out_dir / "kernels_ffm.c",
        "java_src": next(out_dir.glob("*Ffm.java")),
        "lib": lib_path,
        "classes": classes_dir,
        "lib_dir": out_dir,
    }
