"""`glue`: generate JNI boilerplate for a kernel method.

Given a pure static method, generates into ``--out``:

- ``<Class>Native.java`` (default package): ``static native`` declarations
  mirroring the kernel signature, plus ``System.loadLibrary("kernels")``.
  The default package keeps JNI symbol names short (``Java_KernelsNative_dot``).
  v0.12: when the kernel class or method is not public, the class is instead
  generated in the kernel's package (see :func:`glue_package`) so the
  fallback dispatcher can reference the original; the JNI symbol then
  carries the mangled package (``Java_com_example_KernelsNative_dot``).
- ``kernels_jni.cpp``: a pure-C++ helper ``kernel_<name>(...)`` per method
  with a marked ``TODO(agent)`` spot where the calling agent pastes the
  translated body, plus the ``JNIEXPORT`` wrapper that marshals
  ``jintArray``/``jdouble``/... to raw pointers. Array parameters the kernel
  may write through are copied back after the call (release mode 0);
  read-only arrays are released with ``JNI_ABORT`` to skip the copy-back.
  The written/read-only split is computed by ``written_arrays()`` in this
  module (conservative: unknown callees count as written) -- safe for
  differential testing, and the agent no longer needs to narrow this by hand.
- ``CMakeLists.txt``: builds ``libkernels.so`` with ``find_package(JNI)``.

v0.1 supported types: ``void``, primitives
(``byte/short/int/long/float/double/boolean/char``) and 1-D primitive
arrays. v0.4 adds ``double[][]``, ``int[][]`` and ``String``.

DESIGN DECISION (2-D arrays): a ``double[][]`` parameter is flattened
row-major by the generated Java glue into one contiguous ``double[]``
plus ``int <name>_rows, <name>_cols`` *before* the native call, so the JNI
wrapper pins exactly one array (``GetDoubleArrayElements``) instead of
pinning every row via ``GetObjectArrayElement``. The C++ helper therefore
sees ``const double* p_a, int p_a_rows, int p_a_cols``. A ``double[][]``
return goes the other way: the helper returns the data as a flat
row-major ``std::vector<double>`` and reports the result dimensions
through ``int* p_out_rows, int* p_out_cols``; the JNI wrapper then builds
the ``jobjectArray`` (per-row ``NewDoubleArray`` + ``SetDoubleArrayRegion``
-- construction, not pinning). 2-D parameters are copied back into the
caller's ``double[][]`` after the call (in/out, like 1-D arrays). 2-D
arrays must be rectangular; ragged input is out of domain. Only
``double[][]`` and ``int[][]`` are supported; any other 2-D element type
(``float[][]``, ``long[][]``, ``String[][]``, ...) is rejected with a
clear error.

DESIGN DECISION (String): a ``String`` parameter arrives as
``const char*`` via ``GetStringUTFChars`` (Modified UTF-8); a ``String``
return is a ``std::string`` wrapped with ``NewStringUTF``. Modified-UTF-8
caveat: NUL becomes 0xC0 0x80 and supplementary characters become CESU-8
surrogate pairs, so byte-oriented translations only match Java semantics
on ASCII input. ``null`` is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import java_ast as J

# java type -> (jni type, c++ type, is_array, element c++ type)
# NOTE: this is the v0.1 table, kept minimal on purpose. Generic consumers
# of jni._map (e.g. ffm.py, which reuses it read-only and unpacks the
# 4-tuple) must keep *rejecting* the v0.4 types loudly instead of silently
# mistreating a double[][] as a 1-D array, so the new types live in
# _2D_INFO / the String branch below and are validated via _classify().
_TYPE_MAP: dict[str, tuple[str, str, bool, str]] = {
    "byte": ("jbyte", "int8_t", False, ""),
    "short": ("jshort", "int16_t", False, ""),
    "int": ("jint", "int32_t", False, ""),
    "long": ("jlong", "int64_t", False, ""),
    "float": ("jfloat", "float", False, ""),
    "double": ("jdouble", "double", False, ""),
    "boolean": ("jboolean", "bool", False, ""),
    "char": ("jchar", "char16_t", False, ""),
    "void": ("void", "void", False, ""),
    "byte[]": ("jbyteArray", "int8_t", True, "int8_t"),
    "short[]": ("jshortArray", "int16_t", True, "int16_t"),
    "int[]": ("jintArray", "int32_t", True, "int32_t"),
    "long[]": ("jlongArray", "int64_t", True, "int64_t"),
    "float[]": ("jfloatArray", "float", True, "float"),
    "double[]": ("jdoubleArray", "double", True, "double"),
    "boolean[]": ("jbooleanArray", "bool", True, "bool"),
}

#: v0.4 2-D array support: java type -> marshaling info. Deliberately NOT
#: in _TYPE_MAP (see the note above).
_2D_INFO: dict[str, dict[str, str]] = {
    "double[][]": {
        "elem_c": "double",  # C++ element type
        "java_elem": "double",  # Java element type (for generated helpers)
        "jni_arr": "jdoubleArray",
        "jtype_elem": "jdouble",
        "get": "GetDoubleArrayElements",
        "release": "ReleaseDoubleArrayElements",
        "new": "NewDoubleArray",
        "set": "SetDoubleArrayRegion",
        "array_desc": "[D",  # for FindClass when building the return
    },
    "int[][]": {
        "elem_c": "int32_t",
        "java_elem": "int",
        "jni_arr": "jintArray",
        "jtype_elem": "jint",
        "get": "GetIntArrayElements",
        "release": "ReleaseIntArrayElements",
        "new": "NewIntArray",
        "set": "SetIntArrayRegion",
        "array_desc": "[I",
    },
}

_JNI_GET = {
    "int8_t": ("GetByteArrayElements", "ReleaseByteArrayElements", "jbyte"),
    "int16_t": ("GetShortArrayElements", "ReleaseShortArrayElements", "jshort"),
    "int32_t": ("GetIntArrayElements", "ReleaseIntArrayElements", "jint"),
    "int64_t": ("GetLongArrayElements", "ReleaseLongArrayElements", "jlong"),
    "float": ("GetFloatArrayElements", "ReleaseFloatArrayElements", "jfloat"),
    "double": ("GetDoubleArrayElements", "ReleaseDoubleArrayElements", "jdouble"),
    "bool": ("GetBooleanArrayElements", "ReleaseBooleanArrayElements", "jboolean"),
}
_JNI_NEW = {
    "int8_t": ("NewByteArray", "SetByteArrayRegion"),
    "int16_t": ("NewShortArray", "SetShortArrayRegion"),
    "int32_t": ("NewIntArray", "SetIntArrayRegion"),
    "int64_t": ("NewLongArray", "SetLongArrayRegion"),
    "float": ("NewFloatArray", "SetFloatArrayRegion"),
    "double": ("NewDoubleArray", "SetDoubleArrayRegion"),
    "bool": ("NewBooleanArray", "SetBooleanArrayRegion"),
}


@dataclass
class GluePlan:
    method: J.MethodInfo
    native_class: str  # KernelsNative
    jni_func: str  # Java_KernelsNative_dot
    helper_name: str  # kernel_dot
    flat: bool = False  # v0.4: method uses 2-D params/return -> flatten in Java
    package: str = ""  # v0.12: kernel's package; "" = legacy default package


def glue_package(repo, method: J.MethodInfo) -> str:
    """Package for the generated Java glue class, or ``""`` for the legacy
    default package.

    The default-package glue cannot reference a kernel whose class or
    method is not ``public`` (v0.12 dogfood: commons-numbers'
    package-private ``BoostErf`` broke ``verify``). In that case the glue
    class is generated in the kernel's own package; otherwise ``""`` is
    returned and the historical default-package output is preserved
    byte-for-byte.
    """
    cls = repo.classes.get(method.class_qualname) if repo is not None else None
    pkg = cls.package if cls is not None else ""
    if not pkg:
        return ""
    class_public = cls is not None and "public" in cls.modifiers
    method_public = "public" in method.modifiers or (
        cls is not None
        and cls.kind == "interface"
        and "private" not in method.modifiers
    )
    return "" if (class_public and method_public) else pkg


def _jni_escape(name: str) -> str:
    """Escape one Java identifier per the JNI name-mangling rules.

    ``_`` -> ``_1``, ``;`` -> ``_2``, ``[`` -> ``_3``, ``$`` -> ``_00024``.
    Without this, any kernel whose class/method/package name contains an
    underscore (or ``$``) generates an export symbol the JVM can never
    resolve -- ``UnsatisfiedLinkError`` at load time.
    """
    return (
        name.replace("_", "_1")
        .replace(";", "_2")
        .replace("[", "_3")
        .replace("$", "_00024")
    )


def _jni_escape_sig(descriptors: str) -> str:
    """Escape a concatenated JNI argument descriptor for long-form symbols.

    Same rules as :func:`_jni_escape`, plus ``/`` -> ``_``. Order matters:
    the identifier escapes run *first* and ``/`` -> ``_`` last, so the
    underscores produced from ``/`` are not re-escaped -- e.g.
    ``Ljava/lang/String;`` becomes ``Ljava_lang_String_2`` (verified
    against ``javac -h`` output), not ``Ljava_1lang_1String_2``.
    """
    return _jni_escape(descriptors).replace("/", "_")


_PRIM_DESC = {
    "byte": "B",
    "short": "S",
    "int": "I",
    "long": "J",
    "float": "F",
    "double": "D",
    "boolean": "Z",
    "char": "C",
}


def _jni_descriptor(jtype: str) -> str:
    """JNI field descriptor for a glue-supported Java type."""
    if jtype in _PRIM_DESC:
        return _PRIM_DESC[jtype]
    if jtype == "String":
        return "Ljava/lang/String;"
    if jtype in _2D_INFO:
        return "[" + _jni_descriptor(_2D_INFO[jtype]["java_elem"] + "[]")
    if jtype.endswith("[]"):
        return "[" + _jni_descriptor(jtype[:-2])
    raise ValueError(f"no JNI descriptor for unsupported type {jtype!r}")


def _native_param_descriptors(method: J.MethodInfo) -> list[str]:
    """JNI descriptors of the *native* method's parameters.

    For 2-D methods the native entry point takes the flattened form
    (``elem[]`` + ``int rows`` + ``int cols`` per 2-D parameter), so the
    descriptors reflect the flattened signature, not the Java one.
    """
    out: list[str] = []
    for p in method.params:
        if _classify(p.jtype) == "array2d":
            elem = _2D_INFO[p.jtype]["java_elem"]
            out += [_jni_descriptor(elem + "[]"), "I", "I"]
        else:
            out.append(_jni_descriptor(p.jtype))
    return out


def jni_long_suffix(method: J.MethodInfo) -> str:
    """The ``__<mangled-signature>`` suffix making a JNI symbol overload-proof.

    The JVM binds every same-named native in one class to a single
    short-form symbol -- with overloads present that is *silently wrong*
    dispatch (verified empirically: the 3-arg overload called the 2-arg
    body and returned a wrong result with no error). The long form
    disambiguates by argument signature.
    """
    return "__" + _jni_escape_sig("".join(_native_param_descriptors(method)))


def _jni_func_name(
    package: str, native_class: str, entry: str, long_sig: str = ""
) -> str:
    """JNI export symbol, with the package mangled in when packaged.

    Components are escaped per the JNI mangling rules (v0.12: underscores
    in names used to generate unresolvable symbols). ``long_sig`` is the
    already-escaped ``__<signature>`` suffix for overloaded natives
    (see :func:`jni_long_suffix`); empty for the normal short form.
    """
    if package:
        head = (
            "Java_"
            + _jni_escape(package).replace(".", "_")
            + "_"
            + _jni_escape(native_class)
            + "_"
            + _jni_escape(entry)
        )
    else:
        head = f"Java_{_jni_escape(native_class)}_{_jni_escape(entry)}"
    return head + long_sig


# ---------------------------------------------------------------------------
# Reflective fallback (v0.12)
# ---------------------------------------------------------------------------
# When the kernel method is private, the dispatcher's fallback cannot call
# the original directly -- even from the same package. The glue then
# carries a setAccessible reflective handle instead.

_REFLECT_CLASS = {
    "byte": "byte.class",
    "short": "short.class",
    "int": "int.class",
    "long": "long.class",
    "float": "float.class",
    "double": "double.class",
    "boolean": "boolean.class",
    "char": "char.class",
    "String": "String.class",
}
_REFLECT_BOX = {
    "byte": "(Byte)",
    "short": "(Short)",
    "int": "(Integer)",
    "long": "(Long)",
    "float": "(Float)",
    "double": "(Double)",
    "boolean": "(Boolean)",
    "char": "(Character)",
}


def _reflect_class_literal(jtype: str) -> str:
    """``'double' -> 'double.class'``, ``'double[]' -> 'double[].class'``."""
    dims = 0
    while jtype.endswith("[]"):
        jtype = jtype[:-2]
        dims += 1
    base = _REFLECT_CLASS.get(jtype)
    if base is None:
        raise ValueError(f"no reflective class literal for type '{jtype}'")
    return base[: -len(".class")] + "[]" * dims + ".class"


def _reflect_cast(jtype: str) -> str:
    """Boxing cast for a reflective ``Method.invoke`` result (``""`` for void)."""
    if jtype == "void":
        return ""
    if jtype in _REFLECT_BOX:
        return _REFLECT_BOX[jtype]
    return f"({jtype})"  # arrays, String


def _kernel_class_ref(method: J.MethodInfo, package: str) -> str:
    """Source-level reference to the kernel class from the glue class.

    The glue class lives in ``package`` (or the default package when
    ``""``), so the kernel class is referenced by its simple (possibly
    nested, e.g. ``Outer.Inner``) name when packaged, fully qualified
    otherwise.
    """
    if package and method.class_qualname.startswith(package + "."):
        return method.class_qualname[len(package) + 1 :]
    return method.class_qualname


def _fallback_holder_lines(method: J.MethodInfo, package: str) -> list[str]:
    """Static ``FALLBACK_M`` handle for a private kernel method."""
    params = ", ".join(_reflect_class_literal(p.jtype) for p in method.params)
    return [
        "",
        "    // Reflective handle for the private original (fallback path).",
        "    private static final java.lang.reflect.Method FALLBACK_M;",
        "    static {",
        "        try {",
        f"            FALLBACK_M = {_kernel_class_ref(method, package)}.class",
        f'                .getDeclaredMethod("{method.name}"'
        + (f", {params}" if params else "")
        + ");",
        "            FALLBACK_M.setAccessible(true);",
        "        } catch (ReflectiveOperationException e) {",
        "            throw new ExceptionInInitializerError(e);",
        "        }",
        "    }",
        "",
    ]


def _fallback_call_lines(
    method: J.MethodInfo, package: str, indent: str
) -> list[str]:
    """Statements delegating to the original kernel implementation.

    A direct call when the method is accessible; setAccessible reflection
    when it is private (v0.12 dogfood: ``BoostErf.erfInvImp`` is private).
    """
    m = method
    arg_names = ", ".join(p.name for p in m.params)
    ret = "" if m.jtype == "void" else "return "
    if "private" not in m.modifiers:
        return [f"{indent}{ret}{m.qualname}({arg_names});"]
    cast = _reflect_cast(m.jtype)
    invoke = f"FALLBACK_M.invoke(null{', ' + arg_names if arg_names else ''})"
    call = f"{cast} {invoke};" if cast else f"{invoke};"
    return [
        f"{indent}try {{",
        f"{indent}    {ret}{call}",
        f"{indent}}} catch (java.lang.reflect.InvocationTargetException e) {{",
        f"{indent}    throw new RuntimeException(e.getCause());",
        f"{indent}}} catch (ReflectiveOperationException e) {{",
        f"{indent}    throw new RuntimeException(e);",
        f"{indent}}}",
    ]


def _map(jtype: str) -> tuple[str, str, bool, str]:
    # Unchanged v0.1 contract (ffm.py depends on the 4-tuple + ValueError).
    try:
        return _TYPE_MAP[jtype]
    except KeyError:
        raise ValueError(
            f"unsupported type '{jtype}' for JNI glue in v0.1 "
            f"(supported: void, primitives, 1-D primitive arrays)"
        )


def _classify(jtype: str) -> str:
    """'scalar' | 'array1d' | 'array2d' | 'string'; raises ValueError listing
    every supported type (this is the validation entry point for glue)."""
    if jtype in _TYPE_MAP:
        return "array1d" if _TYPE_MAP[jtype][2] else "scalar"
    if jtype in _2D_INFO:
        return "array2d"
    if jtype == "String":
        return "string"
    raise ValueError(
        f"unsupported type '{jtype}' for JNI glue "
        f"(supported: void, primitives, 1-D primitive arrays, "
        f"double[][], int[][], String)"
    )


def _needs_flat(method: J.MethodInfo) -> bool:
    """True when any parameter or the return type is a 2-D array: the
    generated Java glue must flatten before the native call (see the module
    docstring for the design decision)."""
    return any(_classify(p.jtype) == "array2d" for p in method.params) or (
        _classify(method.jtype) == "array2d"
    )


def plan(method: J.MethodInfo, package: str = "") -> GluePlan:
    if not method.is_static:
        raise ValueError("glue in v0.1 requires a static method")
    _classify(method.jtype)
    for p in method.params:
        _classify(p.jtype)
    flat = _needs_flat(method)
    cls_simple = method.class_qualname.rsplit(".", 1)[-1]
    native_class = f"{cls_simple}Native"
    # For 2-D methods the JNI entry point implements the private
    # "<name>Flat" native (flat signature); see render_java/render_cpp.
    entry = method.name + ("Flat" if flat else "")
    return GluePlan(
        method=method,
        native_class=native_class,
        jni_func=_jni_func_name(package, native_class, entry),
        helper_name=f"kernel_{method.name}",
        flat=flat,
        package=package,
    )


def _helper_params(method: J.MethodInfo) -> str:
    parts = []
    for p in method.params:
        kind = _classify(p.jtype)
        if kind == "array2d":
            elem_c = _2D_INFO[p.jtype]["elem_c"]
            parts.append(
                f"const {elem_c}* p_{p.name}, int p_{p.name}_rows, int p_{p.name}_cols"
            )
        elif kind == "string":
            parts.append(f"const char* p_{p.name}")
        elif kind == "array1d":
            _jni, ctype, _a, _e = _map(p.jtype)
            parts.append(f"const {ctype}* p_{p.name}, int p_{p.name}_len")
        else:
            _jni, ctype, _a, _e = _map(p.jtype)
            parts.append(f"{ctype} p_{p.name}")
    if _classify(method.jtype) == "array2d":
        # 2-D return: the helper reports the result dimensions through these.
        parts.append("int* p_out_rows, int* p_out_cols")
    return ", ".join(parts)


def _helper_call_args(method: J.MethodInfo) -> str:
    parts = []
    for p in method.params:
        kind = _classify(p.jtype)
        if kind == "array2d":
            parts.append(f"p_{p.name}, p_{p.name}_rows, p_{p.name}_cols")
        elif kind == "string":
            parts.append(f"p_{p.name}")
        elif kind == "array1d":
            parts.append(f"p_{p.name}, p_{p.name}_len")
        else:
            parts.append(f"p_{p.name}")
    return ", ".join(parts)


def _helper_ret(method: J.MethodInfo) -> str:
    kind = _classify(method.jtype)
    if method.jtype == "void":
        return "void"
    if kind == "array2d":
        # Flat row-major data; dimensions come back via p_out_rows/p_out_cols.
        return f"std::vector<{_2D_INFO[method.jtype]['elem_c']}>"
    if kind == "string":
        return "std::string"
    _jni, ctype, is_array, _elem = _map(method.jtype)
    if is_array:
        return f"std::vector<{ctype}>"
    return ctype


# ---------------------------------------------------------------------------
# written_arrays: which array params the kernel may write through
# ---------------------------------------------------------------------------


def _is_array_param(jtype: str) -> bool:
    """True for 1-D and 2-D array types. Never raises (unlike _classify)."""
    if jtype in _TYPE_MAP:
        return _TYPE_MAP[jtype][2]
    return jtype in _2D_INFO


def _array_root(node, src: bytes) -> str | None:
    """Name of the bare identifier an array_access is rooted at.

    ``a[i]`` -> ``"a"``, ``a[i][j]`` -> ``"a"``; anything else -> None.
    A bare identifier is returned as-is (covers reassignment ``a = ...``).
    """
    while node is not None and node.type == "array_access":
        kids = J.named(node)
        node = kids[0] if kids else None
    if node is not None and node.type == "identifier":
        return J.text(node, src)
    return None


def _body_and_src(method: J.MethodInfo, repo) -> tuple:
    """Return ``(body_node, src_bytes)`` for analysis.

    With a repo the body node carries file-relative offsets, so the source
    is read from the repo file. Without one (e.g. inside render_cpp, which
    only sees the MethodInfo) the declaration source is re-parsed and the
    block of the re-parsed declaration is used instead.
    """
    if method.body is None:
        return None, None
    if repo is not None:
        try:
            src = J.source_bytes(repo.root, method.file)
        except OSError:
            return None, None
        return method.body, src
    if not method.source:
        return None, None
    tree = J.parser().parse(method.source.encode("utf-8"))
    for top in J.named(tree.root_node):
        if top.type == "method_declaration":
            blk = J.child(top, "block")
            if blk is not None:
                return blk, method.source.encode("utf-8")
    return None, None


def _simple_class(repo, simple: str) -> str | None:
    """Repo-local class qualname for a simple name; None when ambiguous."""
    if simple in repo.classes:
        return simple
    matches = [qn for qn in repo.classes if qn.rsplit(".", 1)[-1] == simple]
    return matches[0] if len(matches) == 1 else None


def _resolve_callee(method: J.MethodInfo, repo, name: str, receiver, src: bytes):
    """Resolve a call to a repo-local MethodInfo, else None.

    Same resolution shape as the purity walk: bare / ``this.`` calls stay in
    the caller's class, ``ClassName.`` calls resolve by simple class name;
    anything else (``System.out.println``, ``obj.m()``, overloads) is
    unresolvable.
    """
    if receiver is None or receiver.type == "this":
        cls_qn = method.class_qualname
    elif receiver.type == "identifier":
        cls_qn = _simple_class(repo, J.text(receiver, src))
        if cls_qn is None:
            return None
    else:
        return None
    cands = repo.methods.get((cls_qn, name), [])
    return cands[0] if len(cands) == 1 else None


def written_arrays(method: J.MethodInfo, repo=None) -> set[str]:
    """Names of array-typed params the method body may write through.

    A param counts as written when the body
    - assigns through it: ``a[i] = x`` / ``a[i] += x`` (an
      ``assignment_expression`` whose LHS is an ``array_access`` rooted at
      the param) or ``a[i]++`` / ``--a[i]`` (``update_expression``), or
    - passes the param itself to a call. When the callee resolves to a
      repo-local method the analysis recurses into it, mapping the argument
      position to the callee's formal; a recursive call edge is assumed
      written. An unresolvable callee (JDK code such as ``Arrays.sort(a)``,
      or no ``repo`` given at all) is assumed written.

    Conservative direction: a false "written" only costs a copy-back, while
    a false "read-only" would wrongly skip one -- so the analysis errs
    towards "written". When the body cannot be analyzed at all (no body
    node, unreadable file, unparseable declaration source), every array
    param is conservatively reported as written -- absence of evidence is
    not evidence of read-only. Exposed at module level for reuse (e.g.
    the FFM backend) via ``from . import jni``.
    """
    result = _written_arrays(method, repo, ())
    if result is None:
        # body unavailable: conservatively assume every array param written
        return {p.name for p in method.params if _is_array_param(p.jtype)}
    return result


def _written_arrays(
    method: J.MethodInfo, repo, stack: tuple[str, ...]
) -> set[str] | None:
    """Inner analysis; None means the body could not be analyzed."""
    body, src = _body_and_src(method, repo)
    if body is None:
        return None
    params = {p.name for p in method.params if _is_array_param(p.jtype)}
    if not params:
        return set()
    written: set[str] = set()

    def check_call(node) -> None:
        kids = J.named(node)
        arg_list = J.child(node, "argument_list")
        if arg_list is None:
            return
        args = J.named(arg_list)
        escaped = [
            i
            for i, a in enumerate(args)
            if a.type == "identifier" and J.text(a, src) in params
        ]
        if not escaped:
            return
        # callee name / receiver (same shape as the purity walk)
        name_node, receiver = None, None
        if arg_list in kids:
            idx = kids.index(arg_list)
            for j in range(idx - 1, -1, -1):
                if kids[j].type == "identifier":
                    name_node = kids[j]
                    receiver = kids[j - 1] if j > 0 else None
                    break
        target = (
            _resolve_callee(method, repo, J.text(name_node, src), receiver, src)
            if repo is not None and name_node is not None
            else None
        )
        if (
            target is None
            or target.qualname == method.qualname
            or target.qualname in stack
        ):
            # unknown/JDK callee, or a recursive call edge: assume written
            for i in escaped:
                written.add(J.text(args[i], src))
            return
        sub = _written_arrays(target, repo, stack + (method.qualname,))
        for i in escaped:
            pname = J.text(args[i], src)
            if sub is None:
                written.add(pname)  # callee body unanalyzable: be conservative
            elif i < len(target.params) and target.params[i].name in sub:
                written.add(pname)
            elif i >= len(target.params):
                written.add(pname)  # cannot map the argument: be conservative

    def visit(node) -> None:
        t = node.type
        if t == "assignment_expression":
            kids = J.named(node)
            if kids:
                root = _array_root(kids[0], src)
                if root in params:
                    written.add(root)
        elif t == "update_expression":
            for c in J.named(node):
                if c.type == "array_access":
                    root = _array_root(c, src)
                    if root in params:
                        written.add(root)
        elif t == "method_invocation":
            check_call(node)
        for c in node.children:
            if c.is_named:
                visit(c)

    visit(body)
    return written


# ---------------------------------------------------------------------------
# Java side: flattening wrappers for 2-D methods
# ---------------------------------------------------------------------------


def _flat_native_name(method: J.MethodInfo) -> str:
    return f"{method.name}Flat"


def _flat_java_params(method: J.MethodInfo) -> list[str]:
    """Parameter declarations for the private flat native method."""
    parts = []
    for p in method.params:
        if _classify(p.jtype) == "array2d":
            elem = _2D_INFO[p.jtype]["java_elem"]
            parts.append(f"{elem}[] {p.name}_flat")
            parts.append(f"int {p.name}_rows")
            parts.append(f"int {p.name}_cols")
        else:
            parts.append(f"{p.jtype} {p.name}")
    return parts


def _flat_wrapper_lines(plan: GluePlan, native_name: str, indent: str) -> list[str]:
    """Body lines (at `indent`) for the public 2-D wrapper: flatten each 2-D
    argument, call the flat native, copy 2-D params back (in/out)."""
    m = plan.method
    setup: list[str] = []
    call_args: list[str] = []
    teardown: list[str] = []
    for p in m.params:
        if _classify(p.jtype) == "array2d":
            setup.append(
                f"int {p.name}_rows = {p.name}.length, "
                f"{p.name}_cols = {p.name}_rows > 0 ? {p.name}[0].length : 0;"
            )
            setup.append(
                f"{_2D_INFO[p.jtype]['java_elem']}[] {p.name}_flat = flatten({p.name});"
            )
            call_args += [f"{p.name}_flat", f"{p.name}_rows", f"{p.name}_cols"]
            teardown.append(f"unflattenInto({p.name}_flat, {p.name});")
        else:
            call_args.append(p.name)
    lines = [indent + s for s in setup]
    if m.jtype == "void":
        lines.append(f"{indent}{native_name}({', '.join(call_args)});")
    else:
        lines.append(f"{indent}{m.jtype} r = {native_name}({', '.join(call_args)});")
    lines.extend(indent + t for t in teardown)
    if m.jtype != "void":
        lines.append(f"{indent}return r;")
    return lines


def _flat_java_helpers(plan: GluePlan) -> list[str]:
    """flatten/unflattenInto overloads for the 2-D element types in use."""
    need = {p.jtype for p in plan.method.params if _classify(p.jtype) == "array2d"}
    lines: list[str] = []
    for jt in sorted(need):
        elem = _2D_INFO[jt]["java_elem"]
        lines += [
            "",
            f"    private static {elem}[] flatten({elem}[][] m) {{",
            "        int rows = m.length, cols = rows > 0 ? m[0].length : 0;",
            f"        {elem}[] f = new {elem}[rows * cols];",
            "        for (int i = 0; i < rows; i++)",
            "            System.arraycopy(m[i], 0, f, i * cols, cols);",
            "        return f;",
            "    }",
            "",
            f"    private static void unflattenInto({elem}[] f, {elem}[][] m) {{",
            "        int rows = m.length, cols = rows > 0 ? m[0].length : 0;",
            "        for (int i = 0; i < rows; i++)",
            "            System.arraycopy(f, i * cols, m[i], 0, cols);",
            "    }",
        ]
    return lines


def render_java(plan: GluePlan, lib_name: str = "kernels") -> str:
    """Render ``<Cls>Native.java``.

    ``lib_name`` is the ``System.loadLibrary`` argument; the Rust backend
    passes its per-kernel cdylib name here (see
    :func:`kernel_port.rust_glue.lib_name_for`).
    """
    m = plan.method
    lines = [
        "// Generated by kernel-port. Do not hand-edit: regenerate with",
        f"//   kernel-port glue --method {m.qualname}",
        f"// JNI bindings for {m.qualname}",
        "",
    ]
    if plan.package:
        # v0.12: the glue class lives in the kernel's package so it can
        # reference a non-public kernel class.
        lines.append(f"package {plan.package};")
        lines.append("")
    lines += [
        f"public class {plan.native_class} {{",
        "    static {",
        f'        System.loadLibrary("{lib_name}");',
        "    }",
        "",
    ]
    if plan.flat:
        # 2-D methods: the public method keeps the natural signature and
        # flattens before calling the private flat native.
        native_name = _flat_native_name(m)
        java_params = ", ".join(f"{p.jtype} {p.name}" for p in m.params)
        lines.append(f"    public static {m.jtype} {m.name}({java_params}) {{")
        lines.extend(_flat_wrapper_lines(plan, native_name, "        "))
        lines.append("    }")
        lines.append("")
        lines.append(
            f"    private static native {m.jtype} {native_name}"
            f"({', '.join(_flat_java_params(m))});"
        )
        lines.extend(_flat_java_helpers(plan))
    else:
        java_params = ", ".join(f"{p.jtype} {p.name}" for p in m.params)
        lines.append(f"    public static native {m.jtype} {m.name}({java_params});")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _dispatcher_plan(gp: GluePlan) -> GluePlan:
    """Variant of a plan for dispatcher mode: the JNI entry point is
    ``<name>Native`` (``Java_<Cls>Native_<name>Native``), while the helper
    name is unchanged so the translated C++ body is shared. For 2-D
    methods the entry point is already ``<name>Flat`` (set by plan())."""
    return GluePlan(
        method=gp.method,
        native_class=gp.native_class,
        jni_func=gp.jni_func + ("" if gp.flat else "Native"),
        helper_name=gp.helper_name,
        flat=gp.flat,
        package=gp.package,
    )


def render_java_dispatcher(plan: GluePlan, lib_name: str = "kernels") -> str:
    """Render ``<Cls>Native.java`` as a dispatcher with a fallback switch.

    The public method has the same signature as the kernel but is *not*
    native: it routes to the private native ``<name>Native`` entry point
    while ``USE_NATIVE`` is true, and to the original Java implementation
    (``<qualname>(...)``) otherwise. Flip the switch without rebuilding via
    ``-Dkernelport.native=false`` or ``<Cls>Native.setUseNative(false)``.

    For 2-D methods the private native is the flat ``<name>Flat`` variant
    and the public method flattens before the call (same contract as
    :func:`render_java`).

    ``lib_name`` is the ``System.loadLibrary`` argument; the Rust backend
    passes its per-kernel cdylib name here (see
    :func:`kernel_port.rust_glue.lib_name_for`).
    """
    m = plan.method
    native_name = _flat_native_name(m) if plan.flat else f"{m.name}Native"
    java_params = ", ".join(f"{p.jtype} {p.name}" for p in m.params)
    arg_names = ", ".join(p.name for p in m.params)
    ret = "" if m.jtype == "void" else "return "
    lines = [
        "// Generated by kernel-port. Do not hand-edit: regenerate with",
        f"//   kernel-port glue --method {m.qualname} --fallback",
        f"// JNI bindings for {m.qualname} (dispatcher with fallback switch).",
        "//",
        "// Fallback switch: calls go to the native port by default. Roll back",
        "// to the original pure-Java implementation instantly, without",
        "// rebuilding, with either of:",
        "//   java -Dkernelport.native=false ...        (JVM system property)",
        f"//   {plan.native_class}.setUseNative(false);    (programmatic override)",
        "//",
        "// With the switch off, calls delegate to the original method:",
        f"//   {m.qualname}(...)",
        "",
    ]
    if plan.package:
        # v0.12: the glue class lives in the kernel's package so it can
        # reference a non-public kernel class.
        lines.append(f"package {plan.package};")
        lines.append("")
    lines += [
        f"public class {plan.native_class} {{",
        "    static {",
        f'        System.loadLibrary("{lib_name}");',
        "    }",
        "",
        "    private static boolean USE_NATIVE =",
        '        Boolean.parseBoolean(System.getProperty("kernelport.native", "true"));',
        "",
        "    /** Set false to roll back to the original Java implementation. */",
        "    public static void setUseNative(boolean useNative) {",
        "        USE_NATIVE = useNative;",
        "    }",
        "",
    ]
    if "private" in m.modifiers:
        lines.extend(_fallback_holder_lines(m, plan.package))
    lines += [
        f"    public static {m.jtype} {m.name}({java_params}) {{",
        "        if (USE_NATIVE) {",
    ]
    if plan.flat:
        lines.extend(_flat_wrapper_lines(plan, native_name, "            "))
    else:
        lines.append(f"            {ret}{native_name}({arg_names});")
    lines += [
        "        } else {",
        *_fallback_call_lines(m, plan.package, "            "),
        "        }",
        "    }",
        "",
    ]
    if plan.flat:
        lines.append(
            f"    private static native {m.jtype} {native_name}"
            f"({', '.join(_flat_java_params(m))});"
        )
        lines.extend(_flat_java_helpers(plan))
    else:
        lines.append(f"    private static native {m.jtype} {native_name}({java_params});")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _null_return(jni_ret: str) -> str:
    """Early-return statement when a Get* pin fails (returns null)."""
    if jni_ret == "void":
        return "return;"
    if jni_ret == "jboolean":
        return "return JNI_FALSE;"
    if jni_ret in ("jobjectArray", "jstring"):
        return "return nullptr;"
    return "return 0;"


def _release_mode_line(pname: str, written: set[str]) -> tuple[str, str]:
    """(mode, comment) for a Release*ArrayElements call.

    Written params copy back (mode 0); read-only params skip the copy-back
    with JNI_ABORT -- see written_arrays().
    """
    if pname in written:
        return "0", "in/out: copy back written elements"
    return "JNI_ABORT", "read-only: skip copy-back"


def render_cpp(plan: GluePlan) -> str:
    m = plan.method
    hp = _helper_params(m)
    hr = _helper_ret(m)
    rkind = _classify(m.jtype)
    # Conservative without a repo: params passed to unresolvable calls count
    # as written (see written_arrays).
    written = written_arrays(m)

    # JNI return type, needed up-front for the pin-failure early return.
    if m.jtype == "void":
        jni_ret = "void"
    elif rkind == "array2d":
        jni_ret = "jobjectArray"
    elif rkind == "string":
        jni_ret = "jstring"
    else:
        jni_ret = _map(m.jtype)[0]
    null_ret = _null_return(jni_ret)

    get_lines: list[str] = []
    release_lines: list[str] = []
    fwd_args: list[str] = []
    jni_params: list[str] = []
    for p in m.params:
        kind = _classify(p.jtype)
        if kind == "array2d":
            info = _2D_INFO[p.jtype]
            ctype, jelem = info["elem_c"], info["jtype_elem"]
            # One pin for the whole flattened array (see module docstring).
            get_lines.append(
                f"    {ctype}* p_{p.name} = reinterpret_cast<{ctype}*>(\n"
                f"        env->{info['get']}({p.name}_flat, nullptr));\n"
                f"    if (p_{p.name} == nullptr) {null_ret}"
            )
            mode, why = _release_mode_line(p.name, written)
            release_lines.append(
                f"    env->{info['release']}({p.name}_flat, reinterpret_cast<{jelem}*>(\n"
                f"        p_{p.name}), {mode}); // {why}"
            )
            fwd_args.append(f"p_{p.name}, {p.name}_rows, {p.name}_cols")
            jni_params.append(f"{info['jni_arr']} {p.name}_flat")
            jni_params.append(f"jint {p.name}_rows")
            jni_params.append(f"jint {p.name}_cols")
        elif kind == "string":
            get_lines.append(
                f"    const char* p_{p.name} = env->GetStringUTFChars({p.name}, nullptr);\n"
                f"    if (p_{p.name} == nullptr) {null_ret}"
            )
            release_lines.append(
                f"    env->ReleaseStringUTFChars({p.name}, p_{p.name});"
            )
            fwd_args.append(f"p_{p.name}")
            jni_params.append(f"jstring {p.name}")
        elif kind == "array1d":
            jni_t, ctype, _a, _e = _map(p.jtype)
            get_fn, rel_fn, jelem = _JNI_GET[ctype]
            get_lines.append(
                f"    jsize p_{p.name}_len = env->GetArrayLength({p.name});\n"
                f"    {ctype}* p_{p.name} = reinterpret_cast<{ctype}*>(\n"
                f"        env->{get_fn}({p.name}, nullptr));\n"
                f"    if (p_{p.name} == nullptr) {null_ret}"
            )
            mode, why = _release_mode_line(p.name, written)
            release_lines.append(
                f"    env->{rel_fn}({p.name}, reinterpret_cast<{jelem}*>(\n"
                f"        p_{p.name}), {mode}); // {why}"
            )
            fwd_args.append(f"p_{p.name}, p_{p.name}_len")
            jni_params.append(f"{jni_t} {p.name}")
        else:
            fwd_args.append(p.name)
            jni_t, _c, _a, _e = _map(p.jtype)
            jni_params.append(f"{jni_t} {p.name}")

    jni_sig = ", ".join(["JNIEnv* env", "jclass"] + jni_params)

    # return-value marshaling
    call_lines: list[str] = []
    ret_marshal: list[str] = []
    if m.jtype == "void":
        call_lines.append(f"    {plan.helper_name}({', '.join(fwd_args)});")
    elif rkind == "array2d":
        info = _2D_INFO[m.jtype]
        call_lines.append("    int p_out_rows = 0, p_out_cols = 0;")
        call_lines.append(
            f"    std::vector<{info['elem_c']}> result = {plan.helper_name}("
            f"{', '.join(fwd_args + ['&p_out_rows', '&p_out_cols'])});"
        )
        ret_marshal = [
            f"    jclass rowCls = env->FindClass(\"{info['array_desc']}\");",
            "    jobjectArray out = env->NewObjectArray(p_out_rows, rowCls, nullptr);",
            "    for (int i = 0; i < p_out_rows; i++) {",
            f"        {info['jni_arr']} row = env->{info['new']}(p_out_cols);",
            f"        env->{info['set']}(row, 0, p_out_cols,",
            f"            reinterpret_cast<const {info['jtype_elem']}*>(result.data() + (size_t)i * p_out_cols));",
            "        env->SetObjectArrayElement(out, i, row);",
            "        env->DeleteLocalRef(row);",
            "    }",
            "    return out;",
        ]
    elif rkind == "string":
        call_lines.append(
            f"    std::string result = {plan.helper_name}({', '.join(fwd_args)});"
        )
        ret_marshal = ["    return env->NewStringUTF(result.c_str());"]
    elif rkind == "array1d":
        _jni, ctype, _a, _e = _map(m.jtype)
        new_fn, set_fn = _JNI_NEW[ctype]
        call_lines.append(
            f"    std::vector<{ctype}> result = {plan.helper_name}({', '.join(fwd_args)});"
        )
        ret_marshal = [
            f"    {jni_ret} out = env->{new_fn}((jsize)result.size());",
            f"    env->{set_fn}(out, 0, (jsize)result.size(),",
            f"                 reinterpret_cast<const {_JNI_GET[ctype][2]}*>(result.data()));",
            "    return out;",
        ]
    else:
        _jni, ctype, _a, _e = _map(m.jtype)
        call_lines.append(
            f"    {ctype} result = {plan.helper_name}({', '.join(fwd_args)});"
        )
        ret_marshal = ["    return result;"]

    body = [
        "// Generated by kernel-port JNI glue.",
        "// The calling agent pastes the translated C++ body into the",
        f"// {plan.helper_name}() function below (marked TODO(agent)).",
        "//",
        "// Array parameters the kernel may write are copied back after the",
        "// call (release mode 0); read-only arrays use JNI_ABORT to skip the",
        "// copy-back (see written_arrays() in kernel_port/jni.py).",
        "",
        "#include <jni.h>",
        "#include <cstdint>",
        "#include <string>",
        "#include <vector>",
        "",
        "// ---- Translated kernel (filled in by the agent) ----",
        f"static {hr} {plan.helper_name}({hp}) {{",
        "    // ==== BEGIN TRANSLATED BODY ====",
        f"    // TODO(agent): paste the translated body of {m.qualname} here.",
        "    // It must compute exactly what the Java original computes.",
    ]
    if rkind == "array2d":
        body += [
            "    // For a 2-D return: return the data as a flat row-major vector",
            "    // and set *p_out_rows / *p_out_cols to the result dimensions.",
        ]
    elif rkind == "string" and m.jtype != "void":
        body += [
            "    // For a String return: return a std::string (wrapped with",
            "    // NewStringUTF by the JNI wrapper below).",
        ]
    if hr == "void":
        body.append("    // ==== END TRANSLATED BODY ====")
        body.append("}")
    elif hr.startswith("std::vector") or hr == "std::string":
        body.append("    return {}; // placeholder")
        body.append("    // ==== END TRANSLATED BODY ====")
        body.append("}")
    else:
        body.append("    return 0; // placeholder")
        body.append("    // ==== END TRANSLATED BODY ====")
        body.append("}")
    body += [
        "",
        "extern \"C\" {",
        "",
        f"JNIEXPORT {jni_ret} JNICALL {plan.jni_func}({jni_sig}) {{",
        *get_lines,
        *call_lines,
        *release_lines,
        *ret_marshal,
        "}",
        "",
        '} // extern "C"',
        "",
    ]
    return "\n".join(body)


def render_cmake() -> str:
    return """# Generated by kernel-port.
cmake_minimum_required(VERSION 3.16)
project(kernels_jni)
find_package(JNI REQUIRED)
add_library(kernels SHARED kernels_jni.cpp)
target_include_directories(kernels PRIVATE ${JNI_INCLUDE_DIRS})
set_target_properties(kernels PROPERTIES CXX_STANDARD 17)
"""


def generate(
    method: J.MethodInfo, out_dir: Path, fallback: bool = True, package: str = ""
) -> dict[str, Path]:
    """Write glue files; return {kind: path}.

    ``fallback=False`` (default) reproduces the exact v0.1 output: a bare
    ``static native`` declaration. ``fallback=True`` turns
    ``<Cls>Native.java`` into a dispatcher that routes to the native port
    while ``USE_NATIVE`` is true and back to the original Java
    implementation otherwise (see :func:`render_java_dispatcher`); the C++
    ``JNIEXPORT`` entry point is renamed to ``<name>Native`` accordingly.

    For 2-D methods (``plan.flat``) both modes emit a public Java wrapper
    that flattens before calling the private flat ``<name>Flat`` native;
    the C++ ``JNIEXPORT`` entry point is ``<name>Flat`` in both modes.

    ``package`` (v0.12): when non-empty, the ``<Cls>Native.java`` class is
    generated in the kernel's package (``package <pkg>;`` + matching
    directory layout + package-mangled JNI symbol) so it can reference a
    non-public kernel class. ``""`` preserves the historical default-package
    output byte-for-byte. Use :func:`glue_package` to compute it.
    """
    gp = plan(method, package=package)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    java_dir = out_dir
    if package:
        java_dir = out_dir / Path(*package.split("."))
        java_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "java": java_dir / f"{gp.native_class}.java",
        "cpp": out_dir / "kernels_jni.cpp",
        "cmake": out_dir / "CMakeLists.txt",
    }
    if fallback:
        files["java"].write_text(render_java_dispatcher(gp), encoding="utf-8")
        files["cpp"].write_text(render_cpp(_dispatcher_plan(gp)), encoding="utf-8")
    else:
        files["java"].write_text(render_java(gp), encoding="utf-8")
        files["cpp"].write_text(render_cpp(gp), encoding="utf-8")
    files["cmake"].write_text(render_cmake(), encoding="utf-8")
    return files
