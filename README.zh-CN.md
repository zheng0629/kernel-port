# kernel-port

**从 Java 仓库中挖掘纯计算"算子"，并经 JNI 移植到 C++ —— 全程可验证。**

`kernel-port` 是一套 **agent 可调用的工具链**，做跨语言性能移植。它自己不接 LLM，而是把翻译之外的所有*确定性*环节包下来，让 coding agent（如 Muse）跑通整条流水线：

```
mine → extract →（agent 用自己的 LLM 翻译）→ glue → verify → bench
```

| 步骤 | 命令 | 作用 |
|------|------|------|
| 1. 挖 | `kernel-port mine <repo>` | 扫描 Java 源码，找出纯计算函数（无 IO、无网络、无可变全局状态），按 调用次数 × 计算密集度 排序 |
| 2. 抽 | `kernel-port extract --method <全限定名>` | 抽出函数及其依赖闭包，输出纯度报告 + 干净的 prompt 素材给 LLM |
| 3. 翻译 | *（调用方的 agent/LLM 做）* | 工具到此止步 —— agent 只翻算法本体 |
| 4. 接 | `kernel-port glue --method … --out <dir>` | 生成 JNI 样板：`*Native.java`、`kernels_jni.cpp` 骨架（含待填标记）、`CMakeLists.txt` |
| 5. 验 | `kernel-port verify --glue-dir <dir> …` | 差分测试：随机输入，两边跑，比对输出 |
| 6. 测 | `kernel-port bench --glue-dir <dir> …` | wall-clock 对比（预热 + N 次计时取中位数），给出加速比结论 |

现有工具（CodeMorph、SwapCode 等）只是"粘贴代码→吐翻译结果"的网页：没有仓库挖掘、没有纯度证明、没有 FFI 胶水、没有差分测试、没有 benchmark。kernel-port 是翻译*周围*的那条流水线 —— 这就是壁垒。

## 环境要求

- Python ≥ 3.10
- `PATH` 上有 JDK（17+）和 `g++` —— `verify` / `bench` 需要编译并运行两边代码

## 安装

```bash
pip install kernel-port
# 或从源码：
pip install -e .
```

## 快速上手

```bash
# 1. 找候选
kernel-port mine ./my-java-project

# 2. 抽取（纯度报告 + 给 LLM 的 prompt 素材）
kernel-port extract ./my-java-project --method com.example.Kernels.matMul

# 3. 生成 JNI 胶水
kernel-port glue ./my-java-project --method com.example.Kernels.matMul --out ./port/matmul

# 4.（agent）把翻译好的 C++ 本体粘贴到 ./port/matmul/kernels_jni.cpp 的 TODO(agent) 处

# 5. 证明翻得对
kernel-port verify ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul --cases 200

# 6. 证明翻得值
kernel-port bench ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul
```

`mine` 输出示例：

```
com.example.Kernels.dot    src/com/example/Kernels.java:12   yes   3    17.6
com.example.Kernels.sieve  src/com/example/Kernels.java:48   yes   0    13.1
com.example.Kernels.noisy  src/com/example/Kernels.java:91   no    1     0.0
```

不纯的函数不会被悄悄丢掉 —— 会列出来并给出具体原因（`calls System.out.println(...)`、`mutates field 'callCount'` ……）。

## Agent 工作流

1. `mine` 把候选按价值排序，agent 先翻最值的那个。
2. `extract` 给 LLM 恰好需要的东西：类型签名、纯度结论、函数本体 + 依赖闭包（helper、常量）的完整源码。不纯的直接拒绝并说明原因。
3. `glue` 让 LLM 只翻译*算法*，不用碰 JNI 样板。数组参数默认按 in/out 处理（调用后拷回）—— 先保证正确。
4. `verify` 生成随机输入跑两边，浮点数按容差比对。Java 原版自己都拒绝的输入会被跳过（超出定义域）；JNI 崩溃或结果不一致则明确报错。
5. `bench` 报告两边每次调用的中位数耗时，给出结论，如 `JNI is 1.88x faster` —— 或者诚实的 `1.49x SLOWER -- not worth porting`。

## 设计说明

- **纯度检查偏保守。** 误拒（把纯函数判为不纯，但给出原因）在 v0.1 可接受；误放（把不纯代码判为纯）视为 bug。白名单故意很小：`Math.*`/`StrictMath.*`（不含 `random()`）、数组创建、`static final` 常量。
- **v0.1 范围：** 只支持静态方法；类型只支持 `void`、基本类型、一维基本类型数组。`String`、对象、二维数组、泛型会直接报错说明。
- 解析基于 tree-sitter（与 `deadcode-hunter` 同一技术路线）。

## 路线图（v0.2）

- `--rigorous` benchmark 模式（Java 侧 JMH，C++ 侧 Google Benchmark）
- 重载函数消歧（`--signature`）
- 实例方法与更多类型（`String`、二维数组）

## 许可证

MIT
