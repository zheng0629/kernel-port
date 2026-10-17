# kernel-port

**从 Java 仓库中挖掘纯计算"算子"，转成 C++ 经 JNI 调用——全程可验证。**

`kernel-port` 是一套 **agent 可调用的工具链**，做跨语言性能移植。它自己不调 LLM，而是把翻译之外的*确定性*环节全部包办，让 coding agent（如 Muse）跑通整条流水线：

```
mine → extract →（agent 用自己的 LLM 翻译）→ glue → verify → bench
```

| 环节 | 命令 | 作用 |
|------|------|------|
| 1. 挖 | `kernel-port mine <repo>` | 扫描 Java 源码找纯计算方法、**函数簇**（入口 + 纯 helper，打包成一个候选）、**可整体移植的类**；按调用次数 × 计算强度排序（`--profile` 可按实测 CPU 时间重排） |
| 2. 抽 | `kernel-port extract --method <qualname>` | 抽出方法 + 依赖闭包，输出纯度报告和干净的 prompt 素材给 LLM |
| 3. 翻译 | *（调用方的 agent 用 LLM 做）* | 工具刻意止步于此——agent 负责把函数体翻成 C++ |
| 4. 接 | `kernel-port glue --method … --out <dir>` | 生成 JNI 样板：`*Native.java`（**自带 fallback 开关**）、`kernels_jni.cpp` 骨架（标好填空位置）、`CMakeLists.txt` |
| 5. 验 | `kernel-port verify --glue-dir <dir> …` | 差分测试：随机输入 **+ 边界值用例**，两边跑、比对输出 |
| 6. 测 | `kernel-port bench --glue-dir <dir> …` | wall-clock benchmark，**拆分 JNI 调用开销** + **盈亏平衡点分析**；`--rigorous` 在有 JMH/Google Benchmark 时启用 |
| 一键 | `kernel-port pipeline …` | mine→extract→glue→verify→bench 一次跑完，输出统一报告（终端 + JSON + HTML） |

CodeMorph、SwapCode 之类只是"粘贴代码→吐翻译"的网页：没有仓库挖掘、没有纯度证明、没有 FFI 胶水、没有差分测试、没有 benchmark。kernel-port 是翻译*之外*的整条流水线——这就是壁垒。

## 环境要求

- Python ≥ 3.10
- JDK 17+ 和 `g++` 在 `PATH` 中 —— `verify` / `bench` 需要编译并运行两边代码
- `mcp` Python 包 —— 仅 `kernel-port mcp` 需要（`pip install 'kernel-port[mcp]'`）

## 安装

```bash
pip install kernel-port
# 或从源码：
pip install -e .
```

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

## agent 工作流

1. `mine` 按价值排序候选，agent 先翻最值的。除了单个方法，还能发现**函数簇**（入口方法 + 它调用的纯 helper，整体翻译）和**可移植类**（无状态工具类，或字段全 `final` 的不可变类；有可变状态的类会被拒绝并说明原因）。`--no-clusters` 回到纯方法级列表；`--profile <collapsed.txt>` 用 async-profiler 的 collapsed-stack 文件按实测 CPU 时间重排，而不是静态启发式。
2. `extract` 给 LLM 恰好需要的东西：类型签名、纯度结论、方法 + 依赖闭包（helper、常量）的完整源码。不纯的方法直接拒绝并说明原因。`--class` 整类抽取。上一次 `verify` 失败后，`--with-counterexamples` 会把失败的输入输出拼回 prompt 素材，让下一轮翻译对着修。验证通过的翻译按算子指纹（归一化 AST hash）缓存在 `~/.cache/kernel-port`，`extract` 命中缓存直接复用（`--no-cache` 跳过）。
3. `glue` 让 LLM 只翻*算法*，不用碰 JNI 样板。数组参数按 in/out 处理（调完拷回）——默认安全。生成的 `*Native.java` 是个**分发器**：Java 原实现保留在 `-Dkernelport.native=false`（或 `setUseNative(false)`）开关后面，线上翻车秒回滚。`--no-fallback` 回到朴素 native 声明。`--class` 给可移植类的每个方法生成胶水。
4. `verify` 生成随机输入**加边界值用例**（空数组、0、NaN/±Infinity、`Integer.MIN_VALUE/MAX_VALUE`……——LLM 翻译最容易翻车的地方），两边跑、按浮点容差比对。Java 原实现自己拒绝的输入会被跳过（超出定义域）；JNI 崩溃或结果不一致则大声失败。失败用例会落盘，供 `--with-counterexamples` 回灌。
5. `bench` 报告两边 median 单次耗时，**把 JNI 调用开销**（用空 native 调用实测）和纯 C++ 计算时间**分开**，并告诉你慢的原因：开销主导（函数太小）还是计算主导（翻译得不好）。`--sizes 16,64,256,1024` 做**盈亏平衡分析**：输入规模多大时转译才开始划算。`--rigorous` 在装了 JMH / Google Benchmark 时启用，否则用增强 wall-clock（IQR 去离群 + 95% 置信区间）并如实提示。`--html` 输出自包含报告（内联 SVG 图表，零外部依赖）。
6. `pipeline` 全链一次跑完：mine → extract → glue →（agent 的 `--translated-body`，`#include` 自动提升）→ verify → bench；`--candidates-dir` 可一次试 N 版翻译，自动选出"全通过且最快"的优胜者；`--html`/`--json` 输出报告。没给 `--translated-body` 时停在 glue 之后，打印 agent 的下一步精确指令。
7. `kernel-port mcp` 启动 **stdio MCP server**，把 `mine` / `extract` / `glue` / `verify` / `bench` / `pipeline` 暴露成 MCP 工具，任何支持 MCP 的 agent 都能原生调用。

## 设计说明

- **纯度检查偏保守。** 误拒（纯函数被拒但给出明确原因）可以接受；误放行（不纯的代码被判纯）是 bug。白名单很小：`Math.*`/`StrictMath.*`（除 `random()`）、数组创建、`static final` 常量、不可变类上下文中对 `final` 实例字段的读。
- **范围：** 静态方法；`void`、基本类型、一维基本类型数组。`String`、对象、二维数组、泛型会明确报错拒绝。类级别只支持无状态/不可变类。
- 解析基于 tree-sitter（与 `deadcode-hunter` 同一技术路线）。

## License

MIT
