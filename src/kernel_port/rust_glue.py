"""`glue --backend rust`: generate Rust (cdylib) bindings for a kernel method.

A third glue backend next to :mod:`kernel_port.jni` and
:mod:`kernel_port.ffm`. Given a pure static method, generates into ``--out``:

- ``<Class>Native.java`` (default package): the same dispatcher the JNI
  backend renders -- ``static native`` declarations (or the v0.3 dispatcher
  with the ``USE_NATIVE`` fallback switch) -- except
  ``System.loadLibrary("<lib_name>")`` carries the per-kernel cdylib name
  (see :func:`lib_name_for`), so two kernels' ``.so`` artifacts can never
  be confused. The harness line protocol therefore needs zero changes:
  ``verify``/``bench`` drive ``<Cls>Native.<name>`` exactly as with the
  C++ backend.
- ``src/lib.rs``: a Rust crate whose ``kernel_<name>()`` helper carries a
  marked ``TODO(agent)`` region where the calling agent pastes the
  translated Rust body, plus a ``#[no_mangle] pub extern "system"``
  entry point implementing the JNI symbol ``Java_<mangled>`` directly.
- ``Cargo.toml``: ``crate-type = ["cdylib"]``, ``name = "<lib_name>"``
  (per kernel, see :func:`lib_name_for`), with the ``jni`` crate (0.21)
  as the only dependency.
- ``build_rust.sh``: ``cargo build --release`` then ``javac`` the
  binding class.

DESIGN DECISION (jni crate vs C shim): the recommended approach is the
``jni`` crate: Rust exports the JNI entry points itself, so the compiled
``libkernels.so`` loads via the *existing* JNI harness with no Java-side
changes at all -- only the build step differs (``cargo`` instead of
``g++``). The fallback considered was a plain ``extern "C"`` Rust cdylib
plus a C++ JNI wrapper calling into it; it was rejected because it would
duplicate the entire JNI marshaling layer (array pinning, string
conversion, copy-back) in C++ for no gain -- the ``jni`` crate already
provides safe wrappers for every JNI call the C++ backend uses
(``Get*ArrayElements`` -> ``get_*_array_region``,
``Release*ArrayElements`` -> ``set_*_array_region`` (copy-back) or
nothing (read-only), ``NewStringUTF`` -> ``new_string``). The generated
``lib.rs`` uses a tiny ``jni_try!`` macro so a failed JNI call returns a
sensible default (0 / null) instead of panicking across the FFI boundary.

TYPE MAPPING (same flattened conventions as the JNI backend, v0.4):

- ``void`` / primitives -> ``()`` / ``i8/i16/i32/i64/f32/f64/bool/u16``.
- ``T[]`` -> ``&[T]`` when :func:`jni.written_arrays` proves the param
  read-only, else ``&mut [T]`` (copied back after the call, like the C++
  backend's release mode 0). The slice carries its length, so the Rust
  helper has no ``p_<name>_len`` parameter (unlike the C++ helper).
- ``double[][]`` / ``int[][]`` -> ``&[T]`` / ``&mut [T]`` plus
  ``i32`` rows/cols; the Java glue flattens row-major before the call,
  exactly like the JNI backend. A 2-D return is ``Vec<T>`` plus
  ``&mut i32`` out-params for the dimensions (mirrors the C++ helper).
- ``String`` -> ``&str`` in (via ``get_string``, i.e. Modified UTF-8 --
  the same caveat as the JNI backend: only ASCII input is guaranteed to
  round-trip), ``String`` out (via ``new_string``).

Unsupported types are rejected with the same taxonomy as
``harness.check_harness_types`` (``void``, primitives, 1-D primitive
arrays, ``double[][]``/``int[][]``, ``String``); the validation reuses
``jni._classify`` so the two backends cannot drift apart.

Fallback dispatcher: the v0.3/v0.5 pattern is mirrored, not reimplemented:
the Java file is rendered by ``jni.render_java_dispatcher`` and the Rust
entry point takes its name from ``jni._dispatcher_plan`` (``<name>Native``,
or ``<name>Flat`` for 2-D methods), so Java and Rust can never disagree
on the symbol.

Limitations shared with the JNI backend: JNI's underscore-escaping rule
(``_`` -> ``_1`` in class/method names) is not applied -- a kernel whose
class or method name contains ``_`` would fail to link under *both*
backends identically.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import java_ast as J
from . import jni  # read-only: type taxonomy, written_arrays, Java rendering

#: Max length of a generated cdylib/library name (cargo-friendly).
_MAX_LIB_NAME_LEN = 64

#: ``[a-z0-9_]`` runs of anything else collapse to a single underscore.
_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def lib_name_for(qualname: str) -> str:
    """cdylib/library name unique per kernel, derived from its qualified name.

    Stale-artifact foot-gun fix: the old fixed crate name ``kernels`` meant
    two different glue dirs (or a shared ``CARGO_TARGET_DIR``) collided on
    the same ``libkernels.so`` build key, producing ``UnsatisfiedLinkError``
    or silently wrong results. A per-kernel name makes every crate distinct,
    so a shared target dir is safe.

    The name is ``kp_<sanitized qualname>_<sha1[:8]>``: sanitized to
    ``[a-z0-9_]`` (valid for cargo package/lib names and
    ``System.loadLibrary``), prefixed so it can never start with a digit or
    collide with a system library, and suffixed with 8 hex chars of the
    qualname's sha1 so distinct kernels can never collide even when they
    sanitize identically (``com.Foo.bar`` vs ``com.Foo_bar``). Deterministic:
    the same kernel always gets the same name, so rebuilds are reproducible.
    """
    base = _SANITIZE_RE.sub("_", qualname.lower()).strip("_") or "k"
    digest = hashlib.sha1(qualname.encode("utf-8")).hexdigest()[:8]
    base = base[: _MAX_LIB_NAME_LEN - len("kp_") - 1 - len(digest)]
    return f"kp_{base}_{digest}"


_PACKAGE_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"')


def crate_name_from_toml(toml_path: str | Path) -> str:
    """Read the ``[package] name`` from a glue ``Cargo.toml``.

    Glue dirs generated before per-kernel names say ``name = "kernels"`` --
    returned as-is, so they keep building and loading exactly as before.
    Returns ``"kernels"`` when the file is missing or has no package name.
    """
    in_package = False
    try:
        for line in Path(toml_path).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("["):
                in_package = s == "[package]"
                continue
            if in_package:
                m = _PACKAGE_NAME_RE.match(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return "kernels"

#: java scalar -> Rust type.
_RUST_SCALAR: dict[str, str] = {
    "byte": "i8",
    "short": "i16",
    "int": "i32",
    "long": "i64",
    "float": "f32",
    "double": "f64",
    "boolean": "bool",
    "char": "u16",
}

#: java scalar -> jni::sys type (the C-ABI aliases the jni crate uses).
_SYS: dict[str, str] = {
    "byte": "jbyte",
    "short": "jshort",
    "int": "jint",
    "long": "jlong",
    "float": "jfloat",
    "double": "jdouble",
    "boolean": "jboolean",
    "char": "jchar",
}

#: 1-D java array -> (JObject wrapper, Rust elem, jni region stem, sys elem).
#: Mirrors jni._TYPE_MAP's supported set exactly (no char[] there either).
_ARRAY: dict[str, tuple[str, str, str, str]] = {
    "byte[]": ("JByteArray", "i8", "byte", "jbyte"),
    "short[]": ("JShortArray", "i16", "short", "jshort"),
    "int[]": ("JIntArray", "i32", "int", "jint"),
    "long[]": ("JLongArray", "i64", "long", "jlong"),
    "float[]": ("JFloatArray", "f32", "float", "jfloat"),
    "double[]": ("JDoubleArray", "f64", "double", "jdouble"),
    "boolean[]": ("JBooleanArray", "bool", "boolean", "jboolean"),
}

#: Rust elem -> zero-value literal for generated buffer initialization.
_ZERO: dict[str, str] = {
    "i8": "0i8",
    "i16": "0i16",
    "i32": "0",
    "i64": "0i64",
    "f32": "0.0f32",
    "f64": "0.0",
    "bool": "false",
    "u16": "0u16",
}


@dataclass
class RustPlan:
    method: J.MethodInfo
    native_class: str  # <Cls>Native (same Java file as the JNI backend)
    jni_func: str  # Java_<Cls>Native_<name>[Native|Flat] (the exported symbol)
    helper_name: str  # kernel_<name>
    lib_name: str  # cdylib name, unique per kernel (see lib_name_for)
    flat: bool = False  # 2-D params/return -> Java flattens row-major
    package: str = ""  # v0.12: kernel's package; "" = legacy default package


def _classify(jtype: str) -> str:
    """'scalar' | 'array1d' | 'array2d' | 'string'; same taxonomy as the
    JNI backend, with the error message retargeted at the Rust backend."""
    try:
        return jni._classify(jtype)
    except ValueError as e:
        raise ValueError(str(e).replace("JNI glue", "Rust glue")) from None


def _2d_elem(jtype: str) -> tuple[str, str, str, str]:
    """_ARRAY entry for the element type of a 2-D array type."""
    elem_java = jni._2D_INFO[jtype]["java_elem"]  # "double" | "int"
    return _ARRAY[elem_java + "[]"]


def plan(method: J.MethodInfo, fallback: bool = True, package: str = "") -> RustPlan:
    """Validate the method and compute names.

    Naming (including the ``Native``/``Flat`` entry-point suffixes) is
    delegated to :mod:`kernel_port.jni` so the Java file and the Rust
    symbol can never disagree. ``package`` (v0.12) is threaded into the
    JNI plan so the Java class lands in the kernel's package when the
    kernel is not public; see :func:`kernel_port.jni.glue_package`.
    """
    if not method.is_static:
        raise ValueError("rust glue requires a static method")
    _classify(method.jtype)
    for p in method.params:
        _classify(p.jtype)
    gp = jni.plan(method, package=package)
    if fallback:
        gp = jni._dispatcher_plan(gp)
    return RustPlan(
        method=method,
        native_class=gp.native_class,
        jni_func=gp.jni_func,
        helper_name=gp.helper_name,
        lib_name=lib_name_for(method.qualname),
        flat=gp.flat,
        package=package,
    )


def _helper_params(rp: RustPlan) -> str:
    parts: list[str] = []
    written = jni.written_arrays(rp.method)
    for p in rp.method.params:
        kind = _classify(p.jtype)
        if kind == "array2d":
            _job, elem, _stem, _sy = _2d_elem(p.jtype)
            mut = "mut " if p.name in written else ""
            parts.append(f"p_{p.name}: &{mut}[{elem}]")
            parts.append(f"p_{p.name}_rows: i32")
            parts.append(f"p_{p.name}_cols: i32")
        elif kind == "string":
            parts.append(f"p_{p.name}: &str")
        elif kind == "array1d":
            _job, elem, _stem, _sy = _ARRAY[p.jtype]
            mut = "mut " if p.name in written else ""
            parts.append(f"p_{p.name}: &{mut}[{elem}]")
        else:
            parts.append(f"p_{p.name}: {_RUST_SCALAR[p.jtype]}")
    if _classify(rp.method.jtype) == "array2d":
        parts.append("p_out_rows: &mut i32")
        parts.append("p_out_cols: &mut i32")
    return ", ".join(parts)


def _helper_ret(rp: RustPlan) -> str:
    kind = _classify(rp.method.jtype)
    if rp.method.jtype == "void":
        return ""
    if kind == "array2d":
        return f"Vec<{_2d_elem(rp.method.jtype)[1]}>"
    if kind == "string":
        return "String"
    if kind == "array1d":
        return f"Vec<{_ARRAY[rp.method.jtype][1]}>"
    return _RUST_SCALAR[rp.method.jtype]


def _placeholder(rp: RustPlan) -> list[str]:
    """Placeholder body lines between the BEGIN/END markers."""
    kind = _classify(rp.method.jtype)
    if rp.method.jtype == "void":
        return []
    lines: list[str] = []
    if kind == "array2d":
        lines += ["    *p_out_rows = 0;", "    *p_out_cols = 0;", "    Vec::new() // placeholder"]
    elif kind == "string":
        lines.append("    String::new() // placeholder")
    elif kind == "array1d":
        lines.append("    Vec::new() // placeholder")
    elif rp.method.jtype in ("float", "double"):
        lines.append("    0.0 // placeholder")
    elif rp.method.jtype == "boolean":
        lines.append("    false // placeholder")
    else:
        lines.append("    0 // placeholder")
    return lines


def _err_default(rp: RustPlan) -> str:
    """jni_try! error default: what the entry point returns when a JNI
    call fails (OOM on pinning, basically).

    Returned as a bare expression fragment (no trailing semicolon): the
    ``jni_try!`` macro's ``$d:expr`` slot expands to ``Err(_) => $d,``,
    so ``return 0;`` would not match the macro pattern.
    """
    kind = _classify(rp.method.jtype)
    if rp.method.jtype == "void":
        return "return"
    if kind in ("array1d", "array2d", "string"):
        return "return std::ptr::null_mut()"
    if rp.method.jtype in ("float", "double"):
        # the entry point returns jfloat/f32 or jdouble/f64: `return 0`
        # would be a type error there.
        return "return 0.0"
    return "return 0"


def _wrapper_param_decls(rp: RustPlan) -> tuple[
    list[str], list[str], list[str], list[str], set[str], set[str]
]:
    """Generate the entry-point body around the helper call.

    Returns ``(decls, call_args, releases, ext_params, obj_imports,
    sys_imports)``: lines before the call, helper call arguments, lines
    after the call (copy-back), the ``extern "system"`` parameter list,
    and the jni object/sys type names the signature needs.
    """
    m = rp.method
    dflt = _err_default(rp)
    written = jni.written_arrays(m)
    decls: list[str] = []
    call_args: list[str] = []
    releases: list[str] = []
    ext_params: list[str] = []
    obj_imports = {"JClass"}
    sys_imports: set[str] = set()

    def array_param(pname: str, arr: tuple[str, str, str, str],
                    ext_name: str, is_2d: bool) -> None:
        job, elem, stem, _sy = arr
        obj_imports.add(job)
        ro = pname not in written
        ext_params.append(f"{ext_name}: {job}<'local>")
        get = f"get_{stem}_array_region"
        set_ = f"set_{stem}_array_region"
        if elem == "bool":
            # jboolean is u8 on the wire; the helper sees idiomatic bool.
            decls.append(
                f"    let p_{pname}_len = jni_try!(env.get_array_length(&{ext_name}), {dflt}) as usize;"
            )
            decls.append(f"    let mut p_{pname}_u8 = vec![0u8; p_{pname}_len];")
            decls.append(
                f"    jni_try!(env.{get}(&{ext_name}, 0, &mut p_{pname}_u8), {dflt});"
            )
            decls.append(
                f"    let mut p_{pname}: Vec<bool> = p_{pname}_u8.iter().map(|&b| b != 0).collect();"
            )
            call_args.append(f"&{'mut ' if not ro else ''}p_{pname}")
            if not ro:
                releases.append(
                    f"    let p_{pname}_u8: Vec<u8> = p_{pname}.iter().map(|&b| b as u8).collect();"
                )
                releases.append(
                    f"    jni_try!(env.{set_}(&{ext_name}, 0, &p_{pname}_u8), {dflt});"
                )
        else:
            decls.append(
                f"    let p_{pname}_len = jni_try!(env.get_array_length(&{ext_name}), {dflt}) as usize;"
            )
            decls.append(f"    let mut p_{pname} = vec![{_ZERO[elem]}; p_{pname}_len];")
            decls.append(
                f"    jni_try!(env.{get}(&{ext_name}, 0, &mut p_{pname}), {dflt});"
            )
            call_args.append(f"&{'mut ' if not ro else ''}p_{pname}")
            if not ro:
                releases.append(
                    f"    jni_try!(env.{set_}(&{ext_name}, 0, &p_{pname}), {dflt});"
                )
        if is_2d:
            # rows/cols are plain jint scalars passed straight through.
            sys_imports.add("jint")
            ext_params.append(f"{pname}_rows: jint")
            ext_params.append(f"{pname}_cols: jint")
            call_args.append(f"{pname}_rows")
            call_args.append(f"{pname}_cols")

    for p in m.params:
        kind = _classify(p.jtype)
        if kind == "array2d":
            array_param(p.name, _2d_elem(p.jtype), f"{p.name}_flat", True)
        elif kind == "array1d":
            array_param(p.name, _ARRAY[p.jtype], p.name, False)
        elif kind == "string":
            obj_imports.add("JString")
            ext_params.append(f"{p.name}: JString<'local>")
            decls.append(
                f"    let p_{p.name}_js = jni_try!(env.get_string(&{p.name}), {dflt});"
            )
            decls.append(
                f"    let p_{p.name}: &str = jni_try!(p_{p.name}_js.to_str(), {dflt});"
            )
            call_args.append(f"p_{p.name}")
        else:
            sys_imports.add(_SYS[p.jtype])
            ext_params.append(f"{p.name}: {_SYS[p.jtype]}")
            if p.jtype == "boolean":
                decls.append(f"    let p_{p.name}: bool = {p.name} != 0;")
                call_args.append(f"p_{p.name}")
            else:
                call_args.append(p.name)  # jbyte..jdouble/jchar are ABI-identical
    return decls, call_args, releases, ext_params, obj_imports, sys_imports


def _wrapper_tail(rp: RustPlan, releases: list[str],
                  obj_imports: set[str], sys_imports: set[str]) -> tuple[str, list[str]]:
    """Return ``(ret_annotation, tail_lines)``: the entry point's return
    type annotation ("" for void) and the lines from the helper call
    through the return statement."""
    m = rp.method
    kind = _classify(m.jtype)
    dflt = _err_default(rp)
    # Re-derive call args (same order as _wrapper_param_decls produced).
    _decls, call_args, _rel, _ext, _o, _s = _wrapper_param_decls(rp)
    helper = rp.helper_name
    args = ", ".join(call_args)

    if m.jtype == "void":
        return "", [f"    {helper}({args});", *releases]

    if kind == "array2d":
        job, elem, stem, _sy = _2d_elem(m.jtype)
        obj_imports.update({"JObject", "JObjectArray", job})
        sys_imports.add("jobjectArray")
        desc = jni._2D_INFO[m.jtype]["array_desc"]
        elem_new = f"new_{stem}_array"
        elem_set = f"set_{stem}_array_region"
        return " -> jobjectArray", [
            "    let mut out_rows: i32 = 0;",
            "    let mut out_cols: i32 = 0;",
            f"    let result: Vec<{elem}> = {helper}({args}, &mut out_rows, &mut out_cols);",
            *releases,
            f'    let elem_cls = jni_try!(env.find_class("{desc}"), {dflt});',
            "    let out = jni_try!(env.new_object_array(out_rows, &elem_cls, JObject::null()),"
            f" {dflt});",
            "    for i in 0..out_rows {",
            f"        let row = jni_try!(env.{elem_new}(out_cols), {dflt});",
            "        let st = (i as usize) * (out_cols as usize);",
            f"        jni_try!(env.{elem_set}(&row, 0, &result[st..st + out_cols as usize]), {dflt});",
            # JPrimitiveArray has no AsRef<JObject> impl; convert explicitly.
            f"        jni_try!(env.set_object_array_element(&out, i, JObject::from(row)), {dflt});",
            "    }",
            "    out.into_raw()",
        ]

    if kind == "string":
        sys_imports.add("jstring")
        return " -> jstring", [
            f"    let result: String = {helper}({args});",
            *releases,
            f"    jni_try!(env.new_string(&result), {dflt}).into_raw()",
        ]

    if kind == "array1d":
        job, elem, stem, sy = _ARRAY[m.jtype]
        obj_imports.add(job)
        sys_imports.add(sy + "Array")
        new = f"new_{stem}_array"
        set_ = f"set_{stem}_array_region"
        if elem == "bool":
            build = [
                f"    let result: Vec<bool> = {helper}({args});",
                *releases,
                "    let result_raw: Vec<u8> = result.iter().map(|&b| b as u8).collect();",
                f"    let out = jni_try!(env.{new}(result.len() as i32), {dflt});",
                f"    jni_try!(env.{set_}(&out, 0, &result_raw), {dflt});",
                "    out.into_raw()",
            ]
        else:
            build = [
                f"    let result: Vec<{elem}> = {helper}({args});",
                *releases,
                f"    let out = jni_try!(env.{new}(result.len() as i32), {dflt});",
                f"    jni_try!(env.{set_}(&out, 0, &result), {dflt});",
                "    out.into_raw()",
            ]
        return f" -> {sy}Array", build

    # scalar return
    sys_imports.add(_SYS[m.jtype])
    if m.jtype == "boolean":
        tail = [
            f"    let result: bool = {helper}({args});",
            *releases,
            "    result as jboolean",
        ]
    else:
        tail = [
            f"    let result: {_RUST_SCALAR[m.jtype]} = {helper}({args});",
            *releases,
            "    result",
        ]
    return f" -> {_SYS[m.jtype]}", tail


def render_lib_rs(rp: RustPlan) -> str:
    m = rp.method
    hp = _helper_params(rp)
    hr = _helper_ret(rp)
    decls, call_args, releases, ext_params, obj_imports, sys_imports = (
        _wrapper_param_decls(rp)
    )
    ret_ann, tail = _wrapper_tail(rp, releases, obj_imports, sys_imports)

    obj_use = ", ".join(sorted(obj_imports))
    sys_use = ", ".join(sorted(sys_imports))
    # A method with no scalar params/return (e.g. `void m()`) needs no
    # jni::sys imports; `use jni::sys::{};` would be a syntax error.
    sys_line = (
        f"use jni::sys::{{{sys_use}}};"
        if sys_use
        else "// (no jni::sys imports needed)"
    )
    # `mut env`: several JNIEnv methods we call (get_string, find_class,
    # new_object_array) take &mut self in jni 0.21.
    sig_params = ",\n    ".join(
        ["mut env: JNIEnv<'local>", "_class: JClass<'local>"] + ext_params
    )

    helper_sig = f"fn {rp.helper_name}({hp})"
    if hr:
        helper_sig += f" -> {hr}"
    body = [
        "// Generated by kernel-port Rust glue (--backend rust).",
        "// Do not hand-edit: regenerate with",
        f"//   kernel-port glue --backend rust --method {m.qualname}",
        "//",
        "// The `jni` crate exports the JNI entry points directly, so the",
        "// compiled cdylib loads via the EXISTING JNI harness",
        f"// (System.loadLibrary(\"{rp.lib_name}\") in {rp.native_class}.java) with",
        "// zero harness-protocol changes -- only the build step differs",
        "// (`cargo build --release` instead of g++).",
        "//",
        "// Conventions (same flattened mapping as the JNI backend):",
        "// - T[] params are &[T] when jni.written_arrays proves them read-only,",
        "//   &mut [T] otherwise (copied back after the call). Slices carry",
        "//   their length: no p_<name>_len parameter.",
        "// - double[][]/int[][] arrive flattened row-major (the Java glue",
        "//   flattens before the call) as &[T]/&mut [T] plus i32 rows/cols.",
        "// - String params are &str (Modified UTF-8, same caveat as JNI).",
        "",
        f"use jni::objects::{{{obj_use}}};",
        sys_line,
        "use jni::JNIEnv;",
        "",
        "/// Unwrap a JNI call; on failure return the entry point's default",
        "/// (0 / null) instead of panicking across the FFI boundary.",
        "macro_rules! jni_try {",
        "    ($e:expr, $d:expr) => {",
        "        match $e {",
        "            Ok(v) => v,",
        "            Err(_) => $d,",
        "        }",
        "    };",
        "}",
        "",
        "// ---- Translated kernel (filled in by the agent) ----",
        f"{helper_sig} {{",
        "    // ==== BEGIN TRANSLATED BODY ====",
        f"    // TODO(agent): paste the translated body of {m.qualname} here.",
        "    // It must compute exactly what the Java original computes.",
        *_placeholder(rp),
        "    // ==== END TRANSLATED BODY ====",
        "}",
        "",
        "#[no_mangle]",
        f"pub extern \"system\" fn {rp.jni_func}<'local>(",
        f"    {sig_params},",
        f"){ret_ann} {{",
        *decls,
        *tail,
        "}",
        "",
    ]
    return "\n".join(body)


def render_cargo_toml(rp: RustPlan) -> str:
    """Render ``Cargo.toml``: a ``cdylib`` named per kernel (see
    :func:`lib_name_for`, so ``System.loadLibrary("<name>")`` resolves the
    built ``lib<name>.so``) with the ``jni`` crate as the only dependency.

    ``jni`` is pinned to the 0.21 series: the generated ``lib.rs`` relies
    on 0.21's ``jboolean = u8`` wire type for its boolean conversions
    (the 0.22 series changed ``jboolean`` to ``bool``).
    """
    return f"""# Generated by kernel-port Rust glue for {rp.method.qualname}.
# Do not hand-edit: regenerate with
#   kernel-port glue --backend rust --method {rp.method.qualname}
[package]
name = "{rp.lib_name}"
version = "0.1.0"
edition = "2021"

[lib]
name = "{rp.lib_name}"
crate-type = ["cdylib"]

[dependencies]
jni = "0.21"
"""


def render_build_script(rp: RustPlan, out_dir: str | Path) -> str:
    """Render a POSIX shell snippet: ``cargo build --release``, copy the
    cdylib next to the binding class, then ``javac`` the binding class.

    ``CARGO``/``JAVAC`` env vars override the tools. ``CARGO_TARGET_DIR``
    defaults to ``$OUT_DIR/target`` (same default as
    :func:`compile_rust`), so cargo artifacts live next to the build and
    are cleanable; export it explicitly for a shared target dir across
    crates. Note the javac step compiles only ``<Cls>Native.java``: with
    the default dispatcher (``fallback=True``) the original kernel class
    must be compiled in the same javac invocation (or already be on the
    classpath), exactly as in the JNI backend's build flow.
    """
    out = str(out_dir)
    # v0.12: the binding class may live in the kernel's package directory.
    java_rel = (
        "/".join(rp.package.split(".")) + "/" if rp.package else ""
    ) + f"{rp.native_class}.java"
    return f"""#!/bin/sh
# Generated by kernel-port Rust glue for {rp.method.qualname}.
# Builds lib{rp.lib_name}.so (cargo) and compiles {rp.native_class}.java.
#   CARGO - cargo binary (default: cargo on PATH)
#   JAVAC - javac (default: javac on PATH)
#   CARGO_TARGET_DIR - cargo target dir (default: $OUT_DIR/target, so
#     artifacts live next to the build and are cleanable with it; an
#     explicit setting is honored -- safe to share across kernels because
#     the crate name is unique per kernel, see lib_name_for)
# Tip: once the jni crate is cached, export CARGO_NET_OFFLINE=true to
# avoid network stalls during the build.
set -eu
OUT_DIR="{out}"
: "${{CARGO_TARGET_DIR:=$OUT_DIR/target}}"
CARGO_BIN="${{CARGO:-cargo}}"
JAVAC_BIN="${{JAVAC:-javac}}"
"$CARGO_BIN" build --release --manifest-path "$OUT_DIR/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/lib{rp.lib_name}.so" "$OUT_DIR/lib{rp.lib_name}.so"
"$JAVAC_BIN" -d "$OUT_DIR/classes" "$OUT_DIR/{java_rel}"
"""


def generate(
    method: J.MethodInfo, out_dir: Path, fallback: bool = True, package: str = ""
) -> dict[str, Path]:
    """Write the Rust glue files; return {kind: path}.

    ``fallback=True`` (default) renders ``<Cls>Native.java`` as a
    dispatcher with a ``USE_NATIVE`` switch (``-Dkernelport.native=false``
    or ``<Cls>Native.setUseNative(false)``) that rolls back to the original
    Java implementation; the Rust entry point is then ``<name>Native``
    (``<name>Flat`` for 2-D methods). ``fallback=False`` renders the bare
    ``static native`` declaration with the entry point ``<name>``. The
    2-arg positional call shape used by pipeline.py keeps working
    unchanged.

    The Java file is rendered by :mod:`kernel_port.jni` with the Rust
    backend's per-kernel ``System.loadLibrary("<lib_name>")`` (see
    :func:`lib_name_for`); it is otherwise the same dispatcher the JNI
    backend emits. Only the native implementation differs (``src/lib.rs``
    + ``Cargo.toml`` instead of ``kernels_jni.cpp``).

    ``package`` (v0.12): like :func:`kernel_port.jni.generate` -- when
    non-empty the ``<Cls>Native.java`` class is generated in the kernel's
    package. Use :func:`kernel_port.jni.glue_package` to compute it.
    """
    rp = plan(method, fallback=fallback, package=package)
    gp = jni.plan(method, package=package)
    if fallback:
        gp = jni._dispatcher_plan(gp)
    out_dir = Path(out_dir)
    (out_dir / "src").mkdir(parents=True, exist_ok=True)
    java_dir = out_dir
    if package:
        java_dir = out_dir / Path(*package.split("."))
        java_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "java": java_dir / f"{rp.native_class}.java",
        "rust": out_dir / "src" / "lib.rs",
        "cargo": out_dir / "Cargo.toml",
        "build": out_dir / "build_rust.sh",
    }
    if fallback:
        java_text = jni.render_java_dispatcher(gp, lib_name=rp.lib_name)
    else:
        java_text = jni.render_java(gp, lib_name=rp.lib_name)
    files["java"].write_text(java_text, encoding="utf-8")
    files["rust"].write_text(render_lib_rs(rp), encoding="utf-8")
    files["cargo"].write_text(render_cargo_toml(rp), encoding="utf-8")
    files["build"].write_text(
        render_build_script(rp, out_dir), encoding="utf-8"
    )
    files["build"].chmod(0o755)
    return files


# ---------------------------------------------------------------------------
# Build helpers (used by tests; the CLI wires --backend rust on top of these).
# ---------------------------------------------------------------------------


def _find_cargo() -> Path | None:
    """Locate the cargo binary: PATH first, then ~/.cargo/bin/cargo."""
    which = shutil.which("cargo")
    if which:
        return Path(which)
    home_cargo = Path.home() / ".cargo" / "bin" / "cargo"
    return home_cargo if home_cargo.exists() else None


def _cdylib_name(lib_name: str = "kernels") -> str:
    """Platform cdylib file name for a crate named ``lib_name``."""
    if sys.platform == "win32":
        return f"{lib_name}.dll"
    if sys.platform == "darwin":
        return f"lib{lib_name}.dylib"
    return f"lib{lib_name}.so"


def _target_dir(out_dir: Path) -> Path:
    """Directory cargo writes release artifacts to: ``$CARGO_TARGET_DIR``
    when set, else ``out_dir/target`` (cargo's default)."""
    env_target = os.environ.get("CARGO_TARGET_DIR")
    if env_target:
        return Path(env_target)
    return out_dir / "target"


def _cargo_home() -> Path:
    """Cargo home dir: ``$CARGO_HOME`` when set, else ``~/.cargo``."""
    return Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo")


def _jni_crate_cached() -> bool:
    """True when the pinned ``jni`` 0.21 crate is already in the local
    cargo registry cache -- either as a ``.crate`` file under
    ``registry/cache`` or extracted under ``registry/src``.

    The prefix match is ``jni-0.21`` (not ``jni-``), so a cached
    ``jni-0.22`` does not count: the generator is pinned to 0.21 because
    0.22 changed the ``jboolean`` wire type.
    """
    home = _cargo_home()
    for kind in ("cache", "src"):
        base = home / "registry" / kind
        if not base.is_dir():
            continue
        try:
            registries = [p for p in base.iterdir() if p.is_dir()]
        except OSError:
            continue
        for reg in registries:
            try:
                entries = list(reg.iterdir())
            except OSError:
                continue
            for p in entries:
                if not p.name.startswith("jni-0.21"):
                    continue
                if kind == "cache" and p.is_file() and p.suffix == ".crate":
                    return True
                if kind == "src" and p.is_dir():
                    return True
    return False


#: Lines of cargo stderr kept when a build fails (cargo prints the
#: actionable error last; the full log is usually pages of warnings).
_CARGO_ERROR_TAIL = 30


def _tail(text: str, lines: int = _CARGO_ERROR_TAIL) -> str:
    """Last ``lines`` lines of ``text``."""
    return "\n".join(text.splitlines()[-lines:])


def compile_rust(out_dir: str | Path, timeout: float = 900) -> Path:
    """Run ``cargo build --release`` in ``out_dir`` (which must contain the
    ``Cargo.toml``/``src/lib.rs`` written by :func:`generate`).

    The crate name is read from the manifest's ``[package] name`` (see
    :func:`crate_name_from_toml`): old glue dirs say ``kernels`` and keep
    building exactly as before, while new glue dirs carry a per-kernel
    name (see :func:`lib_name_for`) so a shared ``CARGO_TARGET_DIR`` can no
    longer mix up two kernels' artifacts.

    Build hygiene: ``CARGO_TARGET_DIR`` is pinned to ``<out_dir>/target``
    for the cargo invocation unless already set in the environment (an
    explicit non-empty setting is respected -- safe to share across
    kernels now that crate names are unique; keep it on a big filesystem,
    e.g. under ``~/workspace``, not a small tmpfs). If you need the
    build kept off a small tmpfs entirely, pass an ``out_dir`` outside the
    temp tree.

    Network: ``--offline`` is passed when the pinned ``jni`` 0.21 crate is
    already in the local cargo registry cache (see
    :func:`_jni_crate_cached`), so a warm machine never stalls on the
    network; when the cache is cold, cargo runs online so it can download
    the dependency. Set ``CARGO_NET_OFFLINE`` explicitly to override this
    choice either way.

    Returns the built cdylib path
    (``<target>/release/lib<name>.so`` on Linux). Raises
    RuntimeError with the tail of cargo's stderr on failure (not a raw
    dump), on timeout, or when no cargo binary is found (see
    :func:`_find_cargo`).
    """
    out_dir = Path(out_dir)
    cargo = _find_cargo()
    if cargo is None:
        raise RuntimeError(
            "cargo not found on PATH or at ~/.cargo/bin/cargo; "
            "install the Rust toolchain (https://rustup.rs) to build the "
            "Rust glue backend."
        )
    env = dict(os.environ)
    if not env.get("CARGO_TARGET_DIR"):
        # Pin artifacts next to the build (cargo also treats an empty
        # value as unset, so normalize it the same way).
        env["CARGO_TARGET_DIR"] = str(out_dir / "target")
    manifest = out_dir / "Cargo.toml"
    lib_name = crate_name_from_toml(manifest)
    cmd = [str(cargo), "build", "--release", "--manifest-path", str(manifest)]
    if "CARGO_NET_OFFLINE" not in env and _jni_crate_cached():
        cmd.append("--offline")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"cargo build timed out after {timeout}s "
            f"({' '.join(cmd)}); pass a larger timeout= to compile_rust()"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"cargo build failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"--- cargo stderr (last {_CARGO_ERROR_TAIL} lines) ---\n"
            f"{_tail(proc.stderr)}"
        )
    lib = _target_dir(out_dir) / "release" / _cdylib_name(lib_name)
    if not lib.exists():
        raise RuntimeError(
            f"cargo build succeeded but {lib} is missing; "
            f"cargo stdout:\n{proc.stdout}"
        )
    return lib


# ---------------------------------------------------------------------------
# Cheap Rust FFI lint: canonical implementation lives in lint.lint_rust;
# lint_lib_rs below is kept as a legacy alias.
# ---------------------------------------------------------------------------


def lint_lib_rs(text: str) -> list:
    """Cheap FFI lint over a glue ``src/lib.rs`` (generated wrapper plus the
    agent-pasted translated body). Returns a list of
    :class:`lint.LintWarning` (kind, 1-based line, message).

    Legacy entry point: delegates to :func:`lint.lint_rust`, the canonical
    Rust lint (``ffi_panic`` / ``missing_no_mangle`` / ``wrong_abi`` /
    ``missing_jni_use``). Kept so existing callers and tests keep working.
    """
    from . import lint as _lint

    return _lint.lint_rust(text)
