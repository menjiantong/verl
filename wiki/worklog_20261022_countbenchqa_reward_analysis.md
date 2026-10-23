# CountBenchQA 评分器分析：短回答与格式/推理奖励（2026-10-22）

## 1. 结论摘要

- `countbenchqa.py` **没有长度项**，不直接奖励短回答、也不惩罚长回答。
- 但它的**有效奖励只有 accuracy**：`score = 0.9*acc + 0.1*format`，而日志里 `critic/score/max` 每一步都是 **0.9**，说明没有任何“答对且格式正确”的样本拿到 1.0；格式分 0.1 实际上没有给到正确回答。
- 数据集 prompt 明确要求 `<think>...</think>` 和 `\boxed{}`，但评分器：
  - `valid_format` 只检查“整段回复以 `\boxed{整数}` 结尾”，**完全不检查 `<think>`**；
  - 没有 boxed 时，fallback 取 `</think>` 之后（或整段）最后一个独立整数，因此一个裸的正确答案也能拿 0.9。
- 结果是：**推理和格式都没有边际奖励**。一个正确但很短的答案拿 0.9，一个正确且带完整推理的答案也拿 0.9（只要格式分仍然拿不到）。RL 会把不产生奖励的推理剪掉，短回答是稳定最优解。
- 所以“评分标准导致模型学会短回答”**部分成立，但表述要精确**：不是评分器奖励短，而是评分器**不奖励推理/格式**；短回答是模型在唯一有效信号（最终整数）下的最短路径。评分器是**结构性使能因素**，不是直接的长度奖励。
- 另一个配置层面的矛盾：当前命令有 `+data.apply_chat_template_kwargs.enable_thinking=false`，而数据集 prompt 要求 `<think>`；模型即使不输出 thinking，评分器也不扣分。prompt/chat template/scorer 三方不一致。

## 2. 代码证据

`verl/verl/utils/reward_score/countbenchqa.py`：

- `:19-27`：`_BOXED_RE`、`_STRICT_FINAL_RE`、`_STANDALONE_INT_RE`。
- `:30-36`：`_last_boxed_int` 取最后一个 boxed 整数。
- `:39-43`：`_last_int_after_think` 在没有 `</think>` 时从整段文本取最后一个独立整数。
- `:81-83`：

```python
accuracy = float(predicted == expected)
valid_format = float(bool(_STRICT_FINAL_RE.fullmatch(response.strip())))
score = (1.0 - format_score) * accuracy + format_score * valid_format
```

- `_STRICT_FINAL_RE`（`:23`）是 `.*\\boxed\s*\{\s*[+-]?\d+\s*\}\s*$`，只要求“以 boxed 整数结尾”，没有 `<think>`。
- `default_compute_score`（`verl/utils/reward_score/__init__.py:97-103`）调用 `countbenchqa.compute_score(solution_str, ground_truth)`，使用默认 `format_score=0.1`。

数据集 prompt（`countbenchqa_lite_train.parquet` 第 0 行）：

```text
How many dogs are there in the image? You FIRST think about the reasoning process
as an internal monologue and then provide the final answer. The reasoning process
MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.
```

对比：`geo3k.py:19-22` 的 `format_reward` 是

```python
re.fullmatch(r"<think>.*</think>.*\\boxed\{.*\}", predict_str, re.DOTALL)
```

即 geo3k 同时检查 `<think>` 和 `\boxed{}`；countbenchqa 只检查结尾 boxed。

## 3. 日志证据：格式分实际为 0

当前 run（`W2_longrun_20261022_132541.outer.log`）和上次 W2（`long_W2_20261021_000307.nohup.log`）的 `critic/score/max`：

| step | PREV score max | CUR score max | PREV score min | CUR score min |
|---|---|---|---|---|
| 1 | 0.9 | 0.9 | 0.0 | 0.0 |
| 2 | 0.9 | 0.9 | 0.0 | 0.0 |
| 3 | 0.9 | 0.9 | 0.0 | 0.0 |
| 4 | 0.9 | 0.9 | 0.0 | 0.0 |
| 5 | 0.9 | 0.9 | 0.0 | 0.0 |

如果存在一条“答对且以 `\boxed{N}` 结尾”的回复，分数应为 `0.9*1 + 0.1*1 = 1.0`；实际 max 始终是 0.9，说明**这批 64 条回复里没有任何正确回答拿到格式分**。格式项没有起到奖励格式/推理的作用。

均值也符合“只有 0 和 0.9 两档”：

| step | CUR score mean | mean / 0.9 | 等价正确数（/64） | CUR resp |
|---|---|---|---|---|
| 1 | 0.168750 | 0.1875 | 12 | 433.5 |
| 2 | 0.337500 | 0.3750 | 24 | 389.7 |
| 3 | 0.464062 | 0.5156 | 33 | 338.9 |
| 4 | 0.576562 | 0.6406 | 41 | 260.2 |
| 5 | 0.562500 | 0.6250 | 40 | 274.8 |

`critic/score/mean == critic/rewards/mean`，reward manager 没有额外的长度惩罚或 overlong 惩罚；分数就是 scorer 的原始输出。

## 4. 影响机制

1. **正确裸答案 = 0.9**：没有 boxed 时，`_last_int_after_think` 从整段取最后一个整数；一个裸的 `2` 也会 `accuracy=1`，`format=0`，score=0.9。
2. **正确且带推理 = 0.9**：如果推理后没有严格以 `\boxed{N}` 结尾，format 仍是 0；即使以 boxed 结尾，实测也没有样本拿到 1.0。推理的边际收益是 0。
3. **错误答案 = 0**：除非它恰好以 boxed 结尾拿 0.1；当前 max=0.9 说明正确回答没有拿到这个 0.1。
4. **最短正确路径最优**：在“只看最终整数”的奖励下，模型没有理由保留推理；如果推理偶尔算错，反而降低期望奖励。RL 会剪掉推理，response 变短。
5. **不是直接长度奖励**：scorer 没有 token/length 项；长度塌缩是“推理无奖励 + 任务简单 + 模型能直接答对”的间接结果。

## 5. 与其他 scorer 对比

| scorer | 答案提取 | 分数 | 是否检查 `<think>`/推理 | 长度处理 |
|---|---|---|---|---|
| `countbenchqa.py` | 最后一个 boxed 整数；无 boxed 时取 `</think>` 后/整段最后一个独立整数 | `0.9*acc + 0.1*format`；format 只查结尾 boxed | **否**；实测 format 分为 0 | 无长度项；fallback 接受裸整数 |
| `geo3k.py` | `extract_boxed_content` + `grade_answer` | `0.9*acc + 0.1*format`；format `fullmatch(<think>.*</think>.*\boxed{...})` | **是**，同时要求 thinking 和 boxed | 无长度项 |
| `gsm8k.py` | strict `#### N` 或 flexible 最后一个数字 | 1/0（`format_score=0.0`） | 否 | 无长度项 |
| `math_reward.py` | 最后一个 `\boxed{}` + `is_equiv` | 1/0 | 否 | 无长度项 |
| `math_dapo.py` | Minerva `Answer:`，失败后 boxed | 1/-1 | 否 | 只截取最后 300 字符做效率优化，不是奖励 |
| `search_r1_like_qa_em.py` | `<answer>` 标签 + EM/F1 | 1/0 或 F1 | 否；对 >10 个 answer 标签做降权 | 无长度项 |

结论：**大多数 scorer 都是 answer-only**，countbenchqa 并不是唯一不奖励推理的。差别在于：

- countbenchqa 的 prompt **明确要求推理**，而 scorer 不检查；
- 它的 0.1 format 项本来可以奖励 boxed，但实测没有生效；
- geo3k 是唯一同时检查 `<think>` 和 `\boxed{}` 的参考实现。

## 6. 批判性评价

**“评分标准导致短回答”是否成立？**

- **成立的部分**：评分器只给最终整数奖励，推理/格式没有边际收益；模型在 RL 下会剪掉不产生奖励的推理，短回答是稳定最优解。当前 5 步 response 434→260、reward 0.169→0.5625，和这个机制一致。
- **不成立/需要修正的部分**：
  - 评分器**没有直接奖励短**，也没有长度惩罚；它不是“short answer reward”。
  - 评分器**不是两次 run 差异的唯一原因**：PREV 用同一个 scorer、format 同样为 0，但 response 保持 474-495；当前 run 变短更快，是 RL 轨迹/advantage/采样不同。scorer 定义目标，轨迹决定到达速度。
  - 评分器**不直接参与** `training/rollout_probs_diff_mean` 的计算；它通过改变 token 分布/advantage 方向间接影响一致性指标。不要把 diff_mean 超标直接归因于 scorer。
- **是否“有害”取决于目标**：
  - 如果目标是**答案准确率**：短回答可能是好事（减少 512 截断；当前正确数从 12/64 升到 40/64），scorer 可以保留。
  - 如果目标是**推理能力**：scorer 与 prompt 错位，推理不会被奖励，必须改；否则 RL 会持续剪掉推理。

## 7. 建议的验证/修改（未执行）

1. **先量化**：scorer 返回 `acc`/`format`/`predicted`，但当前 step 日志只记录 score。建议在 reward manager 或 debug 路径把 `acc`、`format` 作为 `reward_extra/*` 记录；这样能直接看到 format 率和预测值，而不是只靠 max=0.9 推断。
2. **如果目标是推理**，做 5-10 step 单变量 A/B：
   - A：保持当前 scorer。
   - B：把 `valid_format` 改成 geo3k 风格，要求 `<think>...</think>` 且以 `\boxed{N}` 结尾；format_score 仍 0.1。
   - C（更强）：没有 `<think>`/`\boxed{}` 时 accuracy 也不给分（score=0），或用更高 format 权重；但要警惕格式/长度 reward hacking。
   - D：如果答案准确率才是目标，直接改 prompt 为 answer-only，删掉 `<think>`/`\boxed{}` 要求，让 prompt 和 scorer 一致。
3. **不要在当前 run 中途改**：当前 step5，`save_freq=-1`，重启会从 step0 开始；先让当前 run 跑完或到稳定点，再做 A/B。
4. 所有新环境变量用 `MY_ENV_` 前缀，代码修改旁加中文注释（工作准则 6.1/6.2）。

## 8. 证据路径

- scorer：`verl/verl/utils/reward_score/countbenchqa.py`
- 调度：`verl/verl/utils/reward_score/__init__.py:97-103`
- 对比：`geo3k.py:19-35`、`gsm8k.py:20-76`、`math_reward.py:17-28`、`math_dapo.py:249-279`、`search_r1_like_qa_em.py:96-128`
- 数据 prompt：`/mnt/share/m00899630/dsv41/rl_data/countbenchqa_lite_train.parquet` 第 0 行
- 日志：`verl/logs/W2_longrun_20261022_132541.outer.log`、`verl/logs/long_W2_20261021_000307.nohup.log`
