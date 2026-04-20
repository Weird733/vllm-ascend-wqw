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
        request: RendererChatRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[ProcessorInputs]]:
        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=is_mistral_tokenizer(renderer.tokenizer),
            ),
        )

        mm_config = self.model_config.multimodal_config

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        (conversation,), (engine_prompt,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        if tool_parser is not None:
            tool_choice = getattr(request, "tool_choice", "none")
            if tool_choice != "none":
                if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                    msg = (
                        "Tool usage is only supported for Chat Completions API "
                        "or Responses API requests."
                    )
                    raise NotImplementedError(msg)

                # TODO: Update adjust_request to accept ResponsesRequest
                tokenizer = renderer.get_tokenizer()
                request = tool_parser(tokenizer).adjust_request(request=request)  # type: ignore[arg-type]

        # adaptor begin: prefill token skip tokenize
        if request.kv_transfer_params and "prompt_token_ids" in request.kv_transfer_params:
            engine_prompt = TokensPrompt(
                prompt_token_ids = request.kv_transfer_params["prompt_token_ids"])
        # adaptor end

        return conversation, [engine_prompt]


#  vllm-project/vllm/vllm/entrypoints/openai/engine/serving.py
from vllm.entrypoints.openai.engine.serving import OpenAIServing
OpenAIServing._preprocess_chat = _preprocess_chat
