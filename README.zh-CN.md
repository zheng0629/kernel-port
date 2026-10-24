# kernel-port

[English](README.md) | **中文**

**从 Java 仓库中挖掘纯计算"算子"，转成 C++ 经 JNI 调用——全程可验证。**

`kernel-port` 是一套 **agent 可调用的工具链**，做跨语言性能移植。它自己不调 LLM，而是把翻译之外的*确定性*环节全部包办，让 coding agent（如 Muse）跑通整条流水线：

```
mine → extract →（agent 用自己的 LLM 翻译）→ glue → verify → bench
```

| 环节 | 命令 | 作用 |
|------|------|------|
| 1. 挖 | `kernel-port mine <repo>` | 扫描 Java 源码找纯计算方法、**函数簇**（入口 + 纯 helper，打包成一个候选）、**可整体移植的类**；按调用次数 × 计算强度排序（`--profile` 可按实测 CPU 时间重排） |
| 2. 抽 | `kernel-port extract --method <qualname>` | 抽出方法 + 依赖闭包，输出纯度报告和干净的 prompt 素材给 LLM（重载：`--method com.example.K.foo(int,double)`） |
| 3. 翻译 | *（调用方的 agent 用 LLM 做）* | 工具刻意止步于此——agent 负责把函数体翻成 C++/C/Rust（按所选后端） |
| 4. 接 | `kernel-port glue --method … --out <dir>` | 生成 FFI 样板：`*Native.java`（**自带 fallback 开关**）、`kernels_jni.cpp` 骨架（标好填空位置）、`CMakeLists.txt`；`--backend ffm` 生成 Panama FFM 绑定 + 纯 C 导出（需 JDK 22+）；`--backend rust` 生成 Rust cdylib（jni crate，需 cargo） |
| 5. 验 | `kernel-port verify --glue-dir <dir> …` | 差分测试：随机输入 **+ 边界值用例**，两边跑、比对输出；浮点容差按类型区分（`--tol-float` 1e-6、`--tol-double` 1e-12） |
| 6. 测 | `kernel-port bench --glue-dir <dir> …` | wall-clock benchmark，**拆分调用开销** + **盈亏平衡点分析**；只有超出 ±20% 噪声带才下"更快/更慢"结论，带内报 inconclusive；`--rigorous` 在有 JMH/Google Benchmark 时启用 |
| 一键 | `kernel-port pipeline …` | mine→extract→glue→verify→bench 一次跑完，输出统一报告（终端 + JSON + HTML） |

CodeMorph、SwapCode 之类只是"粘贴代码→吐翻译"的网页：没有仓库挖掘、没有纯度证明、没有 FFI 胶水、没有差分测试、没有 benchmark。kernel-port 是翻译*之外*的整条流水线——这就是壁垒。

## v0.12 新特性

- **真实仓库 dogfood（apache/commons-numbers）。** 第二轮真实仓库验证——下面每个修复都来自非 fixture 代码，都有回归测试（`tests/test_dogfood_v12.py`）。
- **示例/benchmark 目录不再参与挖掘。** `java_ast.is_skipped_source` 现在也排除 example/benchmark 目录（`examples`、`samples`、`jmh`、`benchmark` 等精确名，以及 `-examples`、`-jmh` 等连字符后缀——`commons-numbers-examples` 这类 Maven 模块会被跳过，但 `com.example` 包目录不会），叠加在 v0.8 的测试目录排除之上。
- **包感知胶水 + 反射回退。** 类或方法不是 public 时，胶水生成在 kernel 所在的包里（JNI 导出符号带包名混淆）；private 的 kernel 方法在 fallback 路径走 `setAccessible` 反射句柄，而不是直接调用。
- **最小合成文件放宽可见性。** 合成的 harness 副本现在把成员放宽为 `public`，private 的 kernel 方法也能从 harness 的异包直接调用，不再静默回退到编译原文件。
- **`extract` 输出不再重排。** 预格式化的抽取行（如 `Location:`）用 `soft_wrap` 打印，重定向到文件时不会在路径中间换行。
- **`--lang python` 的 flag 告警。** `mine --lang python` 配 `--clusters`/`--no-clusters`、`glue --lang python` 配 `--class` 现在会在 stderr 告警并忽略该 flag，不再静默丢弃。
- **诚实的 Python 崩溃报告。** verify/bench 的 driver 子进程死在 Python import 上（`ModuleNotFoundError`/`ImportError`）时，会明确报告为 driver import 错误（环境问题），不再被误标为 C 翻译的 segfault。真正的崩溃（信号 kill）依然按崩溃报告。
- **Python bench：`--sizes` 与 `--list-len`。** `bench --lang python` 支持 `--list-len`（默认 64：没给 `--sizes` 时的 list 长度）和 `--sizes`（逐 size 表格 + 盈亏平衡分析，与 Java 语义一致）。

## v0.11 新特性

- **repair 循环加固。** 停滞检测改为拿失败签名和*所有*历史回合比较，A/B/A/B 交替振荡也能抓住（以前只比相邻两轮）。`stalled`/`gave_up` 报告带上最后一次尝试的 body（`last_body`）和人话停因（`"failure signature repeated from round 2 (oscillation)"`），CLI 直接打印。回合行补上 boundary 计数（`round 1: 3 passed / 5 failed (boundary 2/6)`）。
- **`repair --lang python`。** 自动修复循环也支持 Python→C（cffi）链路了：失败签名来自 Python verify 报告，prompt 素材是 C 视角，停止条件不变（成功/停滞含振荡/达上限）。Java 行为不变。
- **Rust FFI lint 统一。** `lint.lint_rust` 成为唯一的 Rust lint，四个检查：`missing_no_mangle`（入口缺 `#[no_mangle]`）、`ffi_panic`（代码里的 `panic!`/`.unwrap()`/`.expect(`，`//` 注释已剥离）、`wrong_abi`（非 `"system"` ABI）、`missing_jni_use`。`glue --backend rust` 全量报告；`rust_glue.lint_lib_rs` 保留为兼容别名。
- **终端输出不再吞 `[...]`。** 用户侧内容（`Sequence[float]` 这种签名、路径、原因）全部做 Rich markup 转义；以前被静默吃掉的 `[random]`/`[boundary]` 用例标签现在正常显示。
- **Python 纯度：属性式仓库内调用。** `import pkg.mod` + `pkg.mod.func(...)`（含 `as` 别名）现在像 v0.10 的 from-import 修复一样解析仓库内被调函数；真正的外部模块保持保守判定。
- **Python 覆盖率制导 fuzz**（`verify --lang python --coverage`，opt-in）。`sys.monitoring` 录制 Python 原版分支、`ast`+`dis` 静态分支清单、字面量收割的制导输入、ddmin 最小化——和 Java `--coverage` 同样的优雅降级约定。

## v0.10 新特性

- **Python 纯度误拒修复。** `from pkg.mod import func` 中 `pkg.mod` 属于被扫仓库时，现在会解析被调函数并正常做纯度检查（以前直接判"外部模块"→不纯）。相对导入（`from .mod import func`）正确解析为绝对路径（以前记错了，是个潜在的 soundness 洞）。`*rest: int` 解析出参数名 `rest`；变长参数直接给出清晰的拒绝原因，不再报莫名其妙的 "unknown name"。Allowlist 补齐：单参数 `type(x)`、`str.isdigit/isalpha/isspace/isdecimal/isnumeric/isalnum`、非变异集合运算；`set.add`/`discard` 只允许在可证明本地创建的集合上调用。
- **诚实的 benchmark 结论。** ±20% 噪声带：只有超出噪声带才敢说"更快"/"更慢"，带内一律报 "inconclusive（太接近，无法下结论）"。预热改为实测 JIT 收敛（有上限）而非固定次数。适用于全部后端（JNI、FFM、Rust、Python/cffi），终端/JSON/HTML 口径一致。
- **容忍 JVM 启动噪声。** JVM launcher 在 stdout 上的警告（如 stale `hsperfdata` 锁）会被跳过并记一条 stderr，不再被误判为协议错误导致验证失败。真正的协议违规依然会 loud fail。
- **Rust 后端对齐。** `bench --backend rust` 也有调用开销拆分和盈亏平衡分析了，和 JNI 一致。修了共享 `CARGO_TARGET_DIR` + 固定 crate 名的坑：crate 名按 glue 目录唯一化，不同 kernel 的 `.so` 不会串。文案按后端区分（"Rust is X faster"、"fix the Rust body"）。
- **FFM 代码生成修复。** 生成的 wrapper 局部变量名防碰撞（`kp_` 前缀 + 按参数名 freshen——参数就叫 `r` 也不会再编译失败）。`build_ffm.sh` 从脚本自身位置解析输出目录，不再写死相对路径名。fallback dispatcher 改为懒加载 native 库：`.so` 缺失时回退到 Java 原实现（一条 stderr 提示），不再让类初始化直接炸掉。

## v0.9 新特性

- **三大后端全部经过真实仓库验证。** v0.8 只在 JOML 上验证了 JNI；v0.9 把另外两条后端也跑了真实全链：JOML 的 `org.joml.Intersectiond.testRayTriangle` 走 `--backend ffm` 和 `--backend rust`（verify 都通过；bench 都老实报告"不值得转"），`--lang python` 跑了 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) 的 `maths`/`sorts`（mine→手写 C→glue→verify→bench）。下面每个 bug 都是这样抓出来的，都有回归测试。
- **解析缓存不再怕扫描根目录变化。** 缓存里存的文件路径是相对于写入时根目录的；换个根目录重扫同一批文件时路径会叠成两截直接崩溃。现在加载时按当前扫描根重写路径。
- **pipeline 不再丢掉重载后缀。** `pipeline --method 'K.foo(int,double)'` 解析完重载后、extract 前把 `(types)` 后缀弄丢了，导致 `AmbiguityError`。现在透传原始引用。
- **反例文件名对齐。** `verify` 把失败用例存在裸方法名对应的文件里，但 `extract --with-counterexamples` 按 CLI 原始字符串（含后缀）去找，带后缀时就默默错过了。现在两边都用解析后的方法名。
- **Python 纯度：补 soundness 大洞 + 与 Java 对齐。** (1) **严重**——tree-sitter 里单独的生成器表达式直接挂在 `call` 节点下（没有 `argument_list` 包裹），以前根本没被检查，`open()`/`print()` 写在生成器里也能判纯；现在会检查了。(2) 防御性的 `raise ValueError(...)` 不再判不纯——Java 在 v0.4 得到的确定性 throw 豁免，Python 也有了（仅限内置异常+纯参数；自定义异常、预构造对象、裸 `raise` 仍拒绝）。
- **`repair` 支持 `--backend rust`。** 以前 repair 直接拒绝 Rust 后端，现在可以走 Rust 工具链跑"验证失败→重翻→再验"循环。
- **extract 的提示按后端定制。** 新增 `extract --backend ffm|rust`，LLM 提示里的目标语言说明按后端生成（纯 C ABI / Rust 约定），不再永远写 C++17/JNI；`pipeline` 和 `repair` 自动透传。
- **MCP `pipeline` 的 Python 缓存参数对齐。** MCP 之前拒绝 `lang='python'` 配 `candidates_dir`/`no_cache`/`cache_dir`，但 CLI 早就支持；现在校验逻辑与 CLI 一致。
- **修复 `.kp-work` 目录叠床架屋。** work dir 本来就已经是 `.kp-work` 时不再往里再套一层。

## v0.8 新特性

- **真实仓库 dogfood（JOML）。** 首次在非 fixture 代码上验证：挖掘 [JOML](https://github.com/JOML-CI/JOML)（121 文件、662 方法），对 `org.joml.Intersectiond.testRayTriangle`（Möller–Trumbore）跑完 agent 全流程：extract → 手写 C++ → glue → verify **随机 100/100、边界 24/24** → bench。bench 给了诚实的结论：**慢 2.47 倍，不值得转**——开销拆分正确指出是 C++ 计算本身慢，不是 JNI 边界的问题。详见 [DOGFOOD.md](DOGFOOD.md)。
- **`build_env` 不再编译整个仓库。** 以前对仓库下所有 `*.java` 跑 `javac`——真实仓库会直接炸（test 源码要 JUnit、有的文件要 annotation-only 依赖）。现在编译**合成的最小 kernel 文件**（kernel + 依赖闭包 + 仅保留被引用的 import；`extractor.render_minimal_sources`），合成不安全时回退到 `-sourcepath` 编译原文件。附带好处：harness 现在测的正是被抽出来的 kernel——也就是 agent 要翻译的那份代码。
- **`mine` 快约 2–3 倍。** 以前每个方法都重读一次文件（121 个文件读了 11541 次）；现在 `java_ast.source_bytes`（mtime+size 校验的缓存）被 miner/purity/extractor/jni 共用。JOML 全量 `mine`：84–165 秒 → 约 53 秒。
- **测试源码不再参与挖掘。** `src/test`/`test`/`tests` 目录永远不是转译候选（修之前 JOML 的 `TestUtil` 会被排进去）。
- **枚举的静态方法现在能解析了。** tree-sitter 把它们放在 `enum_body_declarations` 下，以前 mine/extract 完全看不见。
- **Python pipeline 对齐。** 指纹缓存（`--no-cache`/`--cache-dir`，`cache_hit` 短路）和多候选择优（`--candidates-dir`，全通过中最快的胜出）在 `pipeline --lang python` 下也能用了，语义与 Java 一致。

## v0.7 新特性

- **Python→C 后端切片：cffi 胶水 + verify/bench**（`glue/verify/bench --lang python`）。生成 `kernels_cffi.c`（plain-C-ABI 的 `kp_<name>` 导出，带标记的翻译区）+ `kernels_cffi.py` cffi 绑定；`gcc -shared -fPIC -O2` 编译。保守类型映射——`int`→`int64_t`、`float`→`double`、`bool`→`int`、`str`→`const char*`、`list[int]`/`list[float]`→ 指针+长度——映射不了的（无注解参数、`*args`、`list` 返回值）直接报清晰错误，绝不猜。verify 跑在 driver 子进程里，翻译 segfault 只会记为失败用例，不会把调用方带崩；边界用例（空 list、int64 极限、inf/nan）与随机用例分开报告，被修改的 list 参数回显后逐元素比对。bench 是老实的 wall-clock，小 kernel 会如实告诉你转 C 反而更慢（cffi 调用开销主导）——和 JNI bench 一样的诚实。
- **`pipeline --lang python`：Python→C 一键全链。** mine → extract（prompt 素材里带胶水呈现的精确 C 签名）→ glue → verify → bench，一份报告（console + JSON + HTML）；没翻译时停在 glue 并给出精确下一步。MCP 的 `pipeline` 工具也支持 `lang="python"`。默认候选选择会跳过"纯但签名映射不了"的函数，选第一个可映射的。指纹缓存同样生效：验证通过的翻译存在 `~/.cache/kernel-port`（`--cache-dir` 改位置、`--no-cache` 跳过），同一 kernel 下次直接 `cache_hit` 返回——不重建、不重测。`--candidates-dir` 接受一个放 `*.c` 候选翻译的目录，每个候选在全新胶水副本里 verify + bench，最快且全过的记为 winner。
- **Rust 后端加固。** `--backend rust` 的 Java 侧也有了和 JNI/FFM 一样的 fallback 开关（`-Dkernelport.native=false` / `setUseNative(false)` 秒切回 Java 原实现，已 e2e 验证）。`CARGO_TARGET_DIR` 默认落在 work dir 下（不再碰 `/tmp`）；本地已有 `jni` 0.21 缓存时用 `--offline` 编译；cargo 失败只吐 exit code + 最后 30 行 stderr，不再 raw dump。
- **Coverage 保持 opt-in——用数据说话。** 评估过把 `verify --coverage` 做成默认：所有实测 kernel 开销都是普通 verify 的 3–27 倍（<2x 的默认化前提不成立），所以保持可选，数字写进文档。顺手加固：覆盖率轮失败优雅降级为普通 verify 结果（只告警不崩）；ddmin 在调用次数预算之外加了 wall-clock 预算；一轮内只编译一次、多个用例 batch 复用同一个 JVM session（`CoveredSession`）。

## v0.6 新特性

- **Rust glue 后端**（`glue/verify/bench --backend rust`）。走 `jni` crate 生成 Rust cdylib——Rust 代码直接导出 JNI 入口点，经由现有 JNI harness 加载，完全不需要 C++ wrapper。需要 `cargo` 在 `PATH` 上；agent 往 `src/lib.rs` 的 `TODO(agent)` 处填翻译，流程与 C++ 骨架一致。
- **`kernel-port repair`：自动修复循环。** `verify` 失败 → 失败用例回灌 prompt → 调用方 hook（`--propose-cmd`，后面接你的 LLM）出新版翻译 → 再验，直到通过、停滞（连续两轮相同失败）或达到 `--max-rounds`。工具本身不直连 LLM，循环是确定性的。
- **覆盖率制导 fuzz**（`verify --coverage`，可选开启）。对 Java 原版做源码级分支插桩（if/循环/switch/三元/catch 计数器）；随机 + 边界两轮之后，再跑一轮针对未覆盖分支的定向输入（比较附近的字面量/边界值注入）。报告给出被测方法的分支覆盖率，失败的定向用例会先做 ddmin 最小化（调用次数 +  wall-clock 双预算）再进入反例回灌。刻意保持可选而非默认：实测开销是普通 `verify` 的 3–27 倍（示例 fixture 上 dot x12.2、fib x3.1、sieve x26.7），因为该轮要重新编译插桩后的源码并把所有用例重跑一遍新的 JVM——光是 javac + JVM 启动的 ~2–4 秒固定成本就超过了一次普通 cheap verify。覆盖率收益是真实的（sieve 60% → 100%；随机测试永远打不中的魔法常量分支能被定向轮找到），但默认开启所需的 <2x 前提在所有实测 kernel 上都不成立。覆盖率轮内部失败会优雅降级为普通 verify 结果（只告警，不崩命令）。
- **增量解析缓存。** `mine`/`extract`/`glue` 复用 tree-sitter 解析结果（按路径 + mtime + size 存 `~/.cache/kernel-port`，可用 `$KERNEL_PORT_PARSE_CACHE_DIR` 改位置）。`--no-parse-cache` / `--refresh-parse-cache` 控制；hit/miss 信息走 stderr，`--format json` 的 stdout 保持干净。
- **Python 算子挖掘（Python→C epic 第一片）。** `mine --lang python` 用保守纯度检查找纯计算 Python 函数（禁 IO/`random`/全局可变/`eval`/框架装饰器，`math.*` 白名单）。cffi 胶水 + verify/bench 在后续切片。

## v0.5 新特性

- **JNI 安全 lint。** `glue` 生成完 `kernels_jni.cpp` 后自动做 lint：每个 `Get*` pin 必须有配对的 `Release*`、pin 后必须判空、循环内 `NewObjectArray`/`NewObject` 没有 `DeleteLocalRef` 会告警。告警直接打印；kernel-port 自己生成的胶水是 lint-clean 的。
- **JNI 生成代码加固。** pin 数组/字符串后加判空；只读数组参数用 `JNI_ABORT` 释放（跳过回拷），读写性由 `jni.written_arrays()` 静态判定；FFM 后端同样跳过只读数组的 copy-back。
- **`pipeline --backend ffm`。** 一键 pipeline（以及 MCP 的 `pipeline` 工具）支持 `--backend ffm`，全链走 Panama 后端。
- **`String` 方法纯度白名单。** 28 个可证明纯的 `String` 方法（`length`、`charAt`、`substring`、`equals`、`hashCode` 等）不再导致方法被拒；正则/字符集/locale 相关的（`split`、`getBytes`、`intern`、`matches` 等）仍会被拒绝并说明原因。
- **FFM fallback 开关。** `--backend ffm` 的胶水也带回滚开关了，和 JNI 一样：`-Dkernelport.native=false` 或 `setUseNative(false)` 秒切回 Java 原实现。

## v0.4 新特性

- **确定性 `throw` 不再判不纯。** `if (n < 0) throw new IllegalArgumentException(...)` 这类参数校验不再导致方法被拒。规则：`throw new <unchecked类型>(纯参数)` 可接受；throw 预先构造好的异常对象、rethrow 捕获的异常、checked 异常仍会被拒绝并说明原因。
- **支持 `double[][]` / `int[][]` 和 `String`。** 二维数组在 Java 侧按行优先展平成一维连续数组（+ rows/cols）再进 native——一次 JNI pin，不用逐行封送；展平约定会写进给 LLM 的 prompt 素材。
- **重载解析。** `--method` 支持签名后缀，如 `--method com.example.K.foo(int,double)`；不带后缀遇到重载时，会列出所有候选而不是直接报错。
- **verify 浮点容差按类型区分。** `float` 按 1e-6、`double` 按 1e-12 相对误差比对（`--tol-float` / `--tol-double` 可调）。float32 累积误差不再导致偶发误报——verify 是信任锚。
- **Panama FFM glue 后端**（`glue --backend ffm`，`verify`/`bench` 同样支持）。生成纯 C ABI 导出 + `java.lang.foreign` 绑定——不用手写 JNI wrapper，没有本地引用表，没有手动异常检查。需要 JDK 22+（推荐 25 LTS），低版本会明确报错拒绝。

## 环境要求

- Python ≥ 3.10
- JDK 17+ 和 `g++` 在 `PATH` 中 —— `verify` / `bench` 需要编译并运行两边代码
- `--backend ffm` 需要 JDK 22+（推荐 25 LTS），低版本会明确报错
- `mcp` Python 包 —— 仅 `kernel-port mcp` 需要（`pip install 'kernel-port-agent[mcp]'`）
- `cargo` —— 仅 `--backend rust` 需要

## 安装

```bash
pip install kernel-port-agent
# 或从源码：
pip install -e .
```

**给 agent 看的：** 仓库根目录带有一份 [SKILL.md](SKILL.md)（Agent Skills 格式），支持 skill 的 agent 可直接发现并调用本工具链。

## 快速上手

```bash
# 1. 挖候选（方法 + 函数簇 + 类）
kernel-port mine ./my-java-project

# 2. 抽一个（纯度报告 + 给 LLM 的 prompt 素材）
kernel-port extract ./my-java-project --method com.example.Kernels.matMul

# 3. 生成 JNI 胶水（默认带 fallback 开关）
kernel-port glue ./my-java-project --method com.example.Kernels.matMul --out ./port/matmul

# 4.（agent）把函数体翻成 C++；翻译里的 #include 会被自动提到生成文件头部

# 5. 证明翻得对（随机 + 边界值）
kernel-port verify ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul --cases 200

# 6. 证明翻得值（开销拆分 + 盈亏平衡）
kernel-port bench ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul --sizes 16,64,256,1024 --html bench.html
```

或者一键跑完（翻译仍由 agent 提供）：

```bash
kernel-port pipeline ./my-java-project --method com.example.Kernels.matMul \
    --translated-body ./matmul.cpp --html report.html --json report.json
```

重载用后缀消歧；FFM 后端完全跳过 JNI：

```bash
# 消歧重载
kernel-port extract ./my-java-project --method 'com.example.K.foo(int,double)'

# Panama FFM 后端（JDK 22+，推荐 25 LTS）
kernel-port glue ./my-java-project --method com.example.Kernels.dot \
    --backend ffm --out ./port/dot-ffm
# ……agent 填好 kernels_ffm.c……
kernel-port verify ./my-java-project --method com.example.Kernels.dot \
    --backend ffm --glue-dir ./port/dot-ffm --cases 100
```

## agent 工作流

1. `mine` 按价值排序候选，agent 先翻最值的。除了单个方法，还能发现**函数簇**（入口方法 + 它调用的纯 helper，整体翻译）和**可移植类**（无状态工具类，或字段全 `final` 的不可变类；有可变状态的类会被拒绝并说明原因）。`--no-clusters` 回到纯方法级列表；`--profile <collapsed.txt>` 用 async-profiler 的 collapsed-stack 文件按实测 CPU 时间重排，而不是静态启发式。（`--lang python` 没有函数簇：`--clusters`/`--no-clusters` 会在 stderr 告警并被忽略。）
2. `extract` 给 LLM 恰好需要的东西：类型签名、纯度结论、方法 + 依赖闭包（helper、常量）的完整源码。不纯的方法直接拒绝并说明原因。`--class` 整类抽取。上一次 `verify` 失败后，`--with-counterexamples` 会把失败的输入输出拼回 prompt 素材，让下一轮翻译对着修。验证通过的翻译按算子指纹（归一化 AST hash）缓存在 `~/.cache/kernel-port`，`extract` 命中缓存直接复用（`--no-cache` 跳过）。
3. `glue` 让 LLM 只翻*算法*，不用碰 FFI 样板。可能被写入的数组参数调完拷回，只读数组跳过拷回（`JNI_ABORT`）——默认安全。生成的 `*Native.java` 是个**分发器**：Java 原实现保留在 `-Dkernelport.native=false`（或 `setUseNative(false)`）开关后面，线上翻车秒回滚。`--no-fallback` 回到朴素 native 声明。`--class` 给可移植类的每个方法生成胶水（`--lang python` 的胶水是函数级的：`--class` 会在 stderr 告警并被忽略）。
4. `verify` 生成随机输入**加边界值用例**（空数组、0、NaN/±Infinity、`Integer.MIN_VALUE/MAX_VALUE`……——LLM 翻译最容易翻车的地方），两边跑、按浮点容差比对。Java 原实现自己拒绝的输入会被跳过（超出定义域）；native 崩溃或结果不一致则大声失败。失败用例会落盘，供 `--with-counterexamples` 回灌。
5. `bench` 报告两边 median 单次耗时，**把调用开销**（用空 native 调用实测）和纯 native 计算时间**分开**，并告诉你慢的原因：开销主导（函数太小）还是计算主导（翻译得不好）。`--sizes 16,64,256,1024` 做**盈亏平衡分析**：输入规模多大时转译才开始划算（`--lang python` 同样支持：逐 size 表格 + 盈亏平衡分析；没给 `--sizes` 时用 `--list-len`（默认 64）决定 list 长度）。`--rigorous` 在装了 JMH / Google Benchmark 时启用，否则用增强 wall-clock（IQR 去离群 + 95% 置信区间）并如实提示（`--rigorous` 和 `--html` 在 `--lang python` 路径下仅支持 Java）。`--html` 输出自包含报告（内联 SVG 图表，零外部依赖）。
6. `pipeline` 全链一次跑完：mine → extract → glue →（agent 的 `--translated-body`，`#include` 自动提升）→ verify → bench；`--candidates-dir` 可一次试 N 版翻译，自动选出"全通过且最快"的优胜者；`--backend jni|ffm|rust` 选择 glue 后端；`--html`/`--json` 输出报告。没给 `--translated-body` 时停在 glue 之后，打印 agent 的下一步精确指令。
7. `kernel-port mcp` 启动 **stdio MCP server**，把 `mine` / `extract` / `glue` / `verify` / `bench` / `pipeline` 暴露成 MCP 工具，任何支持 MCP 的 agent 都能原生调用。

## 设计说明

- **纯度检查偏保守。** 误拒（纯函数被拒但给出明确原因）可以接受；误放行（不纯的代码被判纯）是 bug。白名单很小：`Math.*`/`StrictMath.*`（除 `random()`）、数组创建、`static final` 常量、不可变类上下文中对 `final` 实例字段的读。v0.4 起**确定性 throw** 可接受：`throw new IllegalArgumentException(...)`（纯参数）不破坏纯度；throw 预构造异常对象、rethrow 捕获变量、checked 异常仍然拒绝。
- **范围：** 静态方法；`void`、基本类型、一维基本类型数组，以及 v0.4 新增的 `double[][]`/`int[][]`（行优先展平）和 `String`（经 `GetStringUTFChars`/`NewStringUTF` 按 UTF-8 处理，Modified-UTF-8 注意事项已文档化）。对象、其他二维类型、泛型会明确报错拒绝。类级别只支持无状态/不可变类。
- 重载方法用 `--method` 后缀消歧，如 `com.example.K.foo(int,double)`（先按参数个数过滤，再精确匹配类型）。
- 解析基于 tree-sitter（与 `deadcode-hunter` 同一技术路线）。

## License

MIT
