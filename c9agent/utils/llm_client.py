"""
c9agent/utils/llm_client.py — LLM 通信客户端

从 newagent/llm_tools.py 的核心函数扩展而来。主要改进：
1. 改用 httpx 异步（原版用 asyncio.run 同步包装，效率低）
2. 增加 structured_output() —— 要求 LLM 返回 Pydantic 模型
3. 保留原版的 JSON 自动重试机制
4. 保留原版的代码生成 + 自动调试机制
"""

import httpx
import json
from typing import Optional, TypeVar, Type
from pydantic import BaseModel
from c9agent.config import LLM_CONFIG

T = TypeVar("T", bound=BaseModel)

# ============================================================================
# 基础 LLM 调用（复用 newagent 的模式）
# ============================================================================

def _get_client():
    """创建 HTTP 客户端，使用 config 中的配置"""
    primary = LLM_CONFIG["primary"]
    return httpx.Client(timeout=primary.get("timeout", 120.0))


def run_llm(prompt: str, system_prompt: str = "",
            max_tokens: int = None) -> str:
    """
    发送提示词到 LLM，返回原始文本响应。

    这是最基础的调用。复用 newagent/llm_tools.py 的 run_LLM 逻辑，
    但改用 httpx 同步 Client（而非 asyncio.run 包裹的 AsyncClient）。

    参数:
        prompt: 用户提示词
        system_prompt: 系统提示词（可选，用于设定 LLM 角色）
    """
    cfg = LLM_CONFIG["primary"]
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        with _get_client() as client:
            response = client.post(
                cfg["api_url"],
                json={
                    "model": cfg["model"],
                    "messages": messages,
                    "max_tokens": max_tokens or cfg.get("max_tokens", 2000),
                    "temperature": cfg.get("temperature", 0.2),
                },
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content if content else "[LLM returned empty response]"
    except httpx.RequestError as e:
        return f"[请求错误] {e}"
    except Exception as e:
        return f"[系统错误] {e}"


def answer_to_json(text: str) -> dict:
    """
    从 LLM 响应中提取 JSON 对象。

    复用 newagent/llm_tools.py 的 answer2json，
    LLM 经常在 JSON 外面包一层解释文字，需要剥离。
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"响应中未找到 JSON 对象: {text[:200]}...")
    return json.loads(text[start:end])


def run_llm_json(prompt: str, system_prompt: str = "", max_retries: int = 3) -> dict:
    """
    调用 LLM 并解析 JSON 响应。失败时自动重试。

    复用 newagent/llm_tools.py 的 run_LLM_json_auto_retry 逻辑。
    """
    for attempt in range(max_retries):
        answer = run_llm(prompt, system_prompt)
        try:
            return answer_to_json(answer)
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"LLM JSON 解析失败，已重试 {max_retries} 次: {e}"
                )
    return {}


def structured_output(
    prompt: str,
    output_model: Type[T],
    system_prompt: str = "",
    max_retries: int = 3,
) -> T:
    """
    要求 LLM 返回结构化的 Pydantic 模型。这是新增的功能。

    使用方式:
        result = structured_output(
            "提取以下文本中的 ALS 临床特征...",
            output_model=ALSPatientData,
        )
        # result 是 ALSPatientData 实例，可直接 .age_at_onset 等

    原理:
        把 Pydantic 模型的 JSON Schema 注入提示词，
        LLM 按 Schema 输出 JSON，然后解析为模型实例。
    """
    schema = output_model.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    full_prompt = f"""{prompt}

请严格按照以下 JSON Schema 返回结果（只返回 JSON，不要其他文字）:

```json
{schema_str}
```

重要规则:
- 只返回符合上述 Schema 的有效 JSON
- 不要包含任何解释、推理过程或 Markdown 标记
- 如果某个字段没有信息，使用 null（不要编造数据）
"""

    for attempt in range(max_retries):
        try:
            data = run_llm_json(full_prompt, system_prompt)
            return output_model.model_validate(data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"结构化输出解析失败，已重试 {max_retries} 次: {e}"
                )
    # unreachable, but type checker needs this
    raise RuntimeError("structured_output failed")


# ============================================================================
# 判断类函数（复用 newagent 的 judge 模式）
# ============================================================================

def judge_yes_no(question: str, context: str = "") -> bool:
    """
    让 LLM 判断一个问题是 YES 还是 NO。

    复用 newagent/llm_tools.py 的 judge_user_ready / judge_user_satisfied 模式。

    用于:
    - 判断提取的临床信息是否完整
    - 判断文献是否与患者相关
    - 判断推理链是否自洽
    """
    prompt = f"""你是一个 ALS 临床决策支持系统的判断模块。

上下文: {context}

问题: {question}

请只回答 YES 或 NO。不要加任何解释。"""
    reply = run_llm(prompt)
    return "YES" in reply.upper()


# ============================================================================
# 代码生成与调试（保留 newagent 的 absolute_exec 模式，用于模型训练脚本）
# ============================================================================

def generate_code(task_description: str) -> str:
    """让 LLM 生成 Python 代码（用于模型训练/评估脚本）"""
    prompt = f"""你是一个 Python 数据科学专家。请生成代码完成以下任务。

任务: {task_description}

要求:
- 只返回 Python 代码，不要 Markdown 标记
- 使用 import 声明所需的库
- 代码应该可以直接运行
"""
    reply = run_llm(prompt)
    # 剥离可能的 ```python ... ``` 包裹
    if "```" in reply:
        reply = reply.split("```python")[-1].split("```")[0]
    return reply.strip()


def debug_code(code: str, error_message: str) -> str:
    """让 LLM 调试出错的代码。复用 newagent 的 debug_agent 模式。"""
    prompt = f"""以下 Python 代码运行时出错，请修复它。

代码:
```python
{code}
```

错误信息:
{error_message}

请返回修复后的完整代码（只返回代码，不要解释）。"""
    return generate_code(f"修复以下代码的错误: {error_message}\n\n{code}")
