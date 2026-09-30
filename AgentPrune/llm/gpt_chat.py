import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
import contextvars
from openai import OpenAI, AsyncOpenAI

from AgentPrune.llm.format import Message
from AgentPrune.llm.price import cost_count
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEY = os.getenv('API_KEY')

# Per-call token usage log (mirrors AgentDropout/llm/gpt_chat.py; not used by run_gsm8k.py).
USAGE_LOG: List[Dict[str, Any]] = []
USAGE_TAG: contextvars.ContextVar = contextvars.ContextVar("USAGE_TAG", default=None)


@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,):
    client = AsyncOpenAI(base_url = MINE_BASE_URL, api_key = MINE_API_KEY,)
    chat_completion = await client.chat.completions.create(messages = msg,model = model,max_tokens = max_tokens,temperature = temperature,)
    response = chat_completion.choices[0].message.content
    usage = getattr(chat_completion, "usage", None)
    if usage is not None:
        USAGE_LOG.append({
            "tag": USAGE_TAG.get(),
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cost": getattr(usage, "cost", None),
        })
    return response
    

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
            messages = [{'role':'user', 'content':'messages'}]
        return await achat(self.model_name,messages,max_tokens,temperature)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass