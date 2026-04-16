#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any
 
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
    apply_hf_chat_template,
    parse_chat_messages_futures,
    resolve_chat_template_content_format,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
)
from vllm.entrypoints.openai.serving_engine import ChatLikeRequest
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.deepseek_v32 import DeepseekV32Tokenizer
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tool_parsers import ToolParser
 
logger = init_logger(__name__)
 
 
async def _preprocess_chat(
        self,
        request: ChatLikeRequest | ResponsesRequest,
        tokenizer: TokenizerLike | None,
        messages: list[ChatCompletionMessageParam],
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: list[dict[str, Any]] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
        add_special_tokens: bool = False,
) -> tuple[list[ConversationMessage], list[TokensPrompt]]:
    model_config = self.model_config
 
    resolved_content_format = resolve_chat_template_content_format(
        chat_template,
        tool_dicts,
        chat_template_content_format,
        tokenizer,
        model_config=model_config,
    )
    conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
        messages,
        model_config,
        content_format=resolved_content_format,
    )
 
    _chat_template_kwargs: dict[str, Any] = dict(
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tools=tool_dicts,
        documents=documents,
    )
    _chat_template_kwargs.update(chat_template_kwargs or {})
 
    request_prompt: str | list[int]
 
    if tokenizer is None:
        request_prompt = "placeholder"
    elif isinstance(tokenizer, MistralTokenizer):
        request_prompt = await self._apply_mistral_chat_template_async(
            tokenizer,
            messages=messages,
            **_chat_template_kwargs,
        )
    elif isinstance(tokenizer, DeepseekV32Tokenizer):
        request_prompt = tokenizer.apply_chat_template(
            conversation=conversation,
            messages=messages,
            model_config=model_config,
            **_chat_template_kwargs,
        )
    else:
        request_prompt = apply_hf_chat_template(
            tokenizer=tokenizer,
            conversation=conversation,
            model_config=model_config,
            **_chat_template_kwargs,
        )
 
    mm_data = await mm_data_future
 
    # tool parsing is done only if a tool_parser has been set and if
    # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
    # is set, we want to prevent parsing a tool_call hallucinated by the LLM
    should_parse_tools = tool_parser is not None and (
            hasattr(request, "tool_choice") and request.tool_choice != "none"
    )
 
    if should_parse_tools:
        if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
            msg = (
                "Tool usage is only supported for Chat Completions API "
                "or Responses API requests."
            )
            raise NotImplementedError(msg)
        request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore
 
    # adaptor begin: prefill token skip tokenize
    if request.kv_transfer_params and "prompt_token_ids" in request.kv_transfer_params:
        engine_prompt = TokensPrompt(
            prompt_token_ids = request.kv_transfer_params["prompt_token_ids"])
    # adaptor end
    else:
        if tokenizer is None:
            prompt_inputs = TokensPrompt(prompt=request_prompt, prompt_token_ids=[1])
        elif isinstance(request_prompt, str):
            prompt_inputs = await self._tokenize_prompt_input_async(
                request,
                tokenizer,
                request_prompt,
                add_special_tokens=add_special_tokens,
            )
        else:
            # For MistralTokenizer
            prompt_inputs = TokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt,
            )
 
        engine_prompt = TokensPrompt(prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if "prompt" in prompt_inputs:
            engine_prompt["prompt"] = prompt_inputs["prompt"]
 
    if mm_data is not None:
        engine_prompt["multi_modal_data"] = mm_data
 
    if mm_uuids is not None:
        engine_prompt["multi_modal_uuids"] = mm_uuids
 
    if request.mm_processor_kwargs is not None:
        engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs
 
    if hasattr(request, "cache_salt") and request.cache_salt is not None:
        engine_prompt["cache_salt"] = request.cache_salt
 
    return conversation, [engine_prompt]
 
 
from vllm.entrypoints.openai.serving_engine import OpenAIServing
OpenAIServing._preprocess_chat = _preprocess_chat
