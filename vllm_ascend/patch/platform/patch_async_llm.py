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
import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any
 
from vllm.config import VllmConfig
# from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm_ascend import envs as envs_ascend
 
logger = init_logger(__name__)

def _validate_truncation_size(
    max_model_len: int,
    truncate_prompt_tokens: int | None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> int | None:
    if truncate_prompt_tokens is not None:
        if truncate_prompt_tokens <= -1:
            truncate_prompt_tokens = max_model_len

        if truncate_prompt_tokens > max_model_len:
            raise ValueError(
                f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                f"is greater than max_model_len ({max_model_len})."
                f" Please, select a smaller truncation size."
            )

        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

    else:
        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = False

    return truncate_prompt_tokens
 
async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        engine_prompt: PromptType | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """
    Main function called by the API server to kick off a request
        * 1) Making an AsyncStream corresponding to the Request.
        * 2) Processing the Input.
        * 3) Adding the Request to the Detokenizer.
        * 4) Adding the Request to the EngineCore (separate process).
 
    A separate output_handler loop runs in a background AsyncIO task,
    pulling outputs from EngineCore and putting them into the
    per-request AsyncStream.
 
    The caller of generate() iterates the returned AsyncGenerator,
    returning the RequestOutput back to the caller.
    """
 
    if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and sampling_params.prompt_logprobs
    ):
        raise ValueError(
            "--kv-sharing-fast-prefill produces incorrect logprobs for "
            "prompt tokens, please disable it when the requests need "
            "prompt logprobs"
        )
 
    try:
        # adaptor begin: reuse prefilled tokens
        if envs_ascend.PD_DECODE_SKIP_PREPROCESS and engine_prompt is not None:
            if "prefilled_token_ids" in engine_prompt and engine_prompt["prefilled_token_ids"] != []:
                if sampling_params.n == 1:
                    output = RequestOutput(request_id=request_id,
                                           prompt=None, finished=False, prompt_logprobs=None,
                                           prompt_token_ids=engine_prompt["prompt_token_ids"],
                                           outputs=[CompletionOutput(index=0,
                                                                     cumulative_logprob=None, logprobs=None,
                                                                     text=engine_prompt["prefilled_texts"],
                                                                     token_ids=engine_prompt["prefilled_token_ids"])])
                else:
                    # Fan out child requests (for n>1).
                    parent_request = ParentRequest(request_id, sampling_params)
                    for idx in range(sampling_params.n):
                        request_id_child, params = parent_request.get_child_info(idx)
                        output = RequestOutput(request_id=request_id_child,
                                               prompt=None, finished=False, prompt_logprobs=None,
                                               prompt_token_ids=engine_prompt["prompt_token_ids"],
                                               outputs=[CompletionOutput(index=idx,
                                                                         cumulative_logprob=None, logprobs=None,
                                                                         text=engine_prompt["prefilled_texts"],
                                                                         token_ids=engine_prompt["prefilled_token_ids"])])
                engine_prompt["prefilled_token_ids"] = []
                yield output
        # adaptor end
 
        # We start the output_handler on the first call to generate() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        self._run_output_handler()
 
        # Wait until generation is resumed if the engine is paused.
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)
 
        if tokenization_kwargs is None:
            tokenization_kwargs = {}
            truncate_prompt_tokens = sampling_params.truncate_prompt_tokens
 
            _validate_truncation_size(
                self.model_config.max_model_len,
                truncate_prompt_tokens,
                tokenization_kwargs,
            )
 
        q = await self.add_request(
            request_id,
            prompt,
            sampling_params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            prompt_text=prompt_text,
        )
 
        # The output_handler task pushes items into the queue.
        # This task pulls from the queue and yields to caller.
        finished = False
        while not finished:
            # Note: drain queue without await if possible (avoids
            # task switching under load which helps performance).
            out = q.get_nowait() or await q.get()
 
            # Note: both OutputProcessor and EngineCore handle their
            # own request cleanup based on finished.
            finished = out.finished
            yield out
 
    # If the request is disconnected by the client, generate()
    # is cancelled or the generator is garbage collected. So,
    # we abort the request if we end up here.
    except (asyncio.CancelledError, GeneratorExit):
        await self.abort(request_id)
        if self.log_requests:
            logger.info("Request %s aborted.", request_id)
        raise
 
    # Engine is dead. Do not abort since we shut down.
    except EngineDeadError:
        if self.log_requests:
            logger.info("Request %s failed (engine dead).", request_id)
        raise
 
    # Request validation error.
    except ValueError:
        if self.log_requests:
            logger.info("Request %s failed (bad request).", request_id)
        raise
 
    # Unexpected error in the generate() task (possibly recoverable).
    except Exception as e:
        await self.abort(request_id)
        if self.log_requests:
            print(e)
            logger.info("Request %s failed.", request_id)
        raise EngineGenerateError() from e
 
 
async def get_vllm_config(self) -> VllmConfig:
    return self.vllm_config
 
 
from vllm.v1.engine.async_llm import AsyncLLM
AsyncLLM.generate = generate
AsyncLLM.get_vllm_config = get_vllm_config
