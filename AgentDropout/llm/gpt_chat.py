import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import async_timeout
import contextvars
from transformers import AutoTokenizer

from AgentDropout.llm.format import Message
from AgentDropout.llm.price import cost_count, cost_count_llama3, cost_count_deepseek
from AgentDropout.llm.llm import LLM
from AgentDropout.llm.llm_registry import LLMRegistry


load_dotenv()
MINE_BASE_URL = os.getenv("BASE_URL")
MINE_API_KEYS = os.getenv("API_KEY")

# Per-call token usage from the API's own `usage` field (canonical location for this repo).
# Each entry: {"tag", "model", "prompt_tokens", "completion_tokens", "cost"}.
# `USAGE_TAG` is a ContextVar so concurrent asyncio tasks (one per question) attribute
# their calls to the right question; callers set it before awaiting graph.arun().
USAGE_LOG: List[Dict[str, Any]] = []
USAGE_TAG: contextvars.ContextVar = contextvars.ContextVar("USAGE_TAG", default=None)

# print(MINE_BASE_URL)


# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             if isinstance(response_data['data'],str):
#                 prompt = "".join([item['content'] for item in msg])
#                 cost_count(prompt,response_data['data'],model)
#                 return response_data['data']
#             else:
#                 raise Exception("api error")

_ACLIENT = None  # lazily-created, process-wide AsyncOpenAI client, reused across every call.
# Creating a fresh client (and its underlying httpx connection pool) on every single achat() call --
# hundreds of times per cell -- left many connection pools never explicitly closed, which can make
# asyncio's shutdown/cleanup hang for a very long time after all real work is done (observed: a cell
# whose eval loop finished cleanly in 13 minutes with correct output left the process alive and silent
# for 2+ hours afterward). Reusing one client avoids piling up unclosed pools.
def _get_client() -> AsyncOpenAI:
    global _ACLIENT
    if _ACLIENT is None:
        # Explicit generous timeout: the SDK's own default (600s) was firing as
        # openai.APITimeoutError on dense, unpruned, code-heavy prompts (e.g. HumanEval
        # MAS_roundT: no edge pruning means every agent's full code+reasoning output gets
        # included for every edge, compounding across 2 rounds into very large prompts) --
        # a real per-request timeout, not just overall cell slowness, and no amount of
        # retrying at the same limit fixes it.
        _ACLIENT = AsyncOpenAI(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL, timeout=1800)
    return _ACLIENT


# Team 8 (Oct 2026): pin one OpenRouter provider per model. Unpinned, a single cell was served by
# Novita, DeepInfra and Groq in turn, which may run different quantizations (and Novita caps context
# at 16K tokens, too short for dense multi-agent prompts). No fallbacks: a pinned provider outage
# fails loudly (the call is retried, then counted as an execution error) instead of silently
# switching hardware mid-cell. Set OPENROUTER_PIN=0 to disable.
PROVIDER_PINS = {
    "meta-llama/llama-3.1-8b-instruct": "DeepInfra",   # fp8
    "qwen/qwen-2.5-72b-instruct": "DeepInfra",         # fp8
    "deepseek/deepseek-chat-v3-0324": "SiliconFlow",   # fp8 (DeepSeek-V3's native precision)
}


def _provider_routing(model):
    pin = PROVIDER_PINS.get(model)
    if not pin or os.getenv("OPENROUTER_PIN", "1") == "0":
        return None
    return {"provider": {"order": [pin], "allow_fallbacks": False}}


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None,):
    aclient = _get_client()
    try:
        async with async_timeout.timeout(1900):
            completion = await aclient.chat.completions.create(model=model,messages=msg,max_tokens=max_tokens,temperature=temperature,
                                                                extra_body=_provider_routing(model))
        response_message = completion.choices[0].message.content

        usage = getattr(completion, "usage", None)
        if usage is not None:
            USAGE_LOG.append({
                "tag": USAGE_TAG.get(),
                "model": model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "cost": getattr(usage, "cost", None),
                "provider": getattr(completion, "provider", None),  # OpenRouter returns the serving provider
            })

        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(6))
async def achat_deepseek(model: str, msg: List[Dict],):
    model = ''
    # print(1111111)
    api_kwargs = dict(api_key = deepseek_api, base_url = deepseek_url)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        # print(completion)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_deepseek(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def achat_llama(model: str, msg: List[Dict]):
    # print(111111111111)
    api_kwargs = dict(api_key = "API-KEY", base_url = "http://localhost:6789/v1")
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_llama3(prompt, response_message, model)
            return response_message

    except Exception as e:
        print(f"Error in achat_llama: {e}")
        # raise
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages,max_tokens,temperature)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_deepseek(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('llama')
class LlamaChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        # print(11111111111111111111)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_llama(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass