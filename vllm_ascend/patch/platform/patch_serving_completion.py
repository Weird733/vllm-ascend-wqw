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
import time
from collections.abc import AsyncGenerator
from collections.abc import Sequence as GenericSequence
from typing import cast
 
import jinja2
from fastapi import Request
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import (
    GenerationError,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import EmbedsPrompt, TokensPrompt, is_embeds_prompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list
from vllm.v1.sample.logits_processor import validate_logits_processors_parameters
from vllm_ascend import envs as envs_ascend
 
logger = init_logger(__name__)
 
 
async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request | None = None,
) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
    """Completion API similar to OpenAI's API.
 
    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.
 
    NOTE: Currently we do not support the following feature:
        - suffix (the language models we currently support do not support
        suffix)
    """
    error_check_ret = await self._check_model(request)
    if error_check_ret is not None:
        return error_check_ret
 
    # If the engine is dead, raise the engine's DEAD_ERROR.
    # This is required for the streaming case, where we return a
    # success status before we actually start generating text :).
    if self.engine_client.errored:
        raise self.engine_client.dead_error
 
    # adaptor begin: reuse prefilled tokens
    if request.kv_transfer_params and "prompt_token_ids" in request.kv_transfer_params:
        request.prompt = request.kv_transfer_params["prompt_token_ids"]
    # adaptor end
 
    # Return error for unsupported features.
    if request.suffix is not None:
        return self.create_error_response("suffix is not currently supported")
 
    if request.echo and request.prompt_embeds is not None:
        return self.create_error_response("Echo is unsupported with prompt embeds.")
 
    if request.prompt_logprobs is not None and request.prompt_embeds is not None:
        return self.create_error_response(
            "prompt_logprobs is not compatible with prompt embeds."
        )
 
    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    created_time = int(time.time())
 
    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata
 
    try:
        lora_request = self._maybe_get_adapters(request)
 
        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = await self.engine_client.get_tokenizer()
        renderer = self._get_renderer(tokenizer)
 
        engine_prompts = await renderer.render_prompt_and_embeds(
            prompt_or_prompts=request.prompt,
            prompt_embeds=request.prompt_embeds,
            config=self._build_render_config(request),
        )
    except ValueError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except TypeError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except RuntimeError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
    except jinja2.TemplateError as e:
        logger.exception("Error in preprocessing prompt inputs")
        return self.create_error_response(str(e))
 
    # Extract data_parallel_rank from header (router can inject it)
    data_parallel_rank = self._get_data_parallel_rank(raw_request)
 
    # Schedule the request and get the result generator.
    generators: list[AsyncGenerator[RequestOutput, None]] = []
    try:
        for i, engine_prompt in enumerate(engine_prompts):
            prompt_text, prompt_token_ids, prompt_embeds = (
                self._get_prompt_components(engine_prompt)
            )
 
            input_length = None
            if prompt_token_ids is not None:
                input_length = len(prompt_token_ids)
            elif prompt_embeds is not None:
                input_length = len(prompt_embeds)
            else:
                raise NotImplementedError
 
            # adaptor begin: reuse prefilled tokens
            if envs_ascend.PD_DECODE_SKIP_PREPROCESS:
                if request.kv_transfer_params and "prefilled_token" in request.kv_transfer_params:
                    new_tokens = tokenizer.convert_ids_to_tokens(request.kv_transfer_params["prefilled_token"][0])
                    delta_text = tokenizer.convert_tokens_to_string([new_tokens])
                    # If the decoded text ends with '�', it indicates an incomplete UTF-8 sequence — abandon reuse.
                    # If the prefilled side has already triggered a stop reason, we also fall back to normal generation.
                    if delta_text.endswith("�") or any(s is not None for s in request.kv_transfer_params["stop_reasons"]) \
                            or request.kv_transfer_params["prefilled_token"][0] == tokenizer.eos_token_id:
                        request.kv_transfer_params.pop("prefilled_token", None)
                    else:
                        engine_prompt["prefilled_token_ids"] = request.kv_transfer_params["prefilled_token"]
                        engine_prompt["prefilled_texts"] = delta_text
            # adapt end
 
            if self.default_sampling_params is None:
                self.default_sampling_params = {}
 
            max_tokens = get_max_tokens(
                max_model_len=self.max_model_len,
                request=request,
                input_length=input_length,
                default_sampling_params=self.default_sampling_params,
            )
 
            sampling_params: SamplingParams | BeamSearchParams
            
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.model_config.logits_processor_pattern,
                    self.default_sampling_params,
                )
                validate_logits_processors_parameters(
                    self.logits_processors,
                    sampling_params,
                )
            print(f"-----patch serving 191 {sampling_params=}")
            request_id_item = f"{request_id}-{i}"
 
            self._log_inputs(
                request_id_item,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )
 
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
 
            # Mypy inconsistently requires this second cast in different
            # environments. It shouldn't be necessary (redundant from above)
            # but pre-commit in CI fails without it.
            engine_prompt = cast(EmbedsPrompt | TokensPrompt, engine_prompt)
            print(f"---patch serving completion---{TokensPrompt=}")
            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_prompt,
                    request_id=request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                engine_request, tokenization_kwargs = await self._process_inputs(
                    request_id_item,
                    engine_prompt,
                    sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
 
                generator = self.engine_client.generate(
                    engine_request,
                    sampling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    prompt_text=prompt_text,
                    tokenization_kwargs=tokenization_kwargs,
                    data_parallel_rank=data_parallel_rank,
                    engine_prompt=engine_prompt,
                )
 
            generators.append(generator)
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))
 
    result_generator = merge_async_iterators(*generators)
 
    model_name = self.models.model_name(lora_request)
    num_prompts = len(engine_prompts)
 
    # We do not stream the results when using beam search.
    stream = request.stream and not request.use_beam_search
 
    # Streaming response
    if stream:
        return self.completion_stream_generator(
            request,
            engine_prompts,
            result_generator,
            request_id,
            created_time,
            model_name,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            request_metadata=request_metadata,
        )
 
    # Non-streaming response
    final_res_batch: list[RequestOutput | None] = [None] * num_prompts
    try:
        async for i, res in result_generator:
            final_res_batch[i] = res
 
        for i, final_res in enumerate(final_res_batch):
 
            # The output should contain the input text
            # We did not pass it into vLLM engine to avoid being redundant
            # with the inputs token IDs
            if final_res.prompt is None:
                engine_prompt = engine_prompts[i]
                final_res.prompt = (
                    None
                    if is_embeds_prompt(engine_prompt)
                    else engine_prompt.get("prompt")
                )
 
        final_res_batch_checked = cast(list[RequestOutput], final_res_batch)
 
        # adaptor begin: reuse prefilled tokens
        prompt_token_ids = []
        for req_output in final_res_batch_checked:
            prompt_token_ids.append(req_output.prompt_token_ids)
        if final_res_batch_checked[0].kv_transfer_params and envs_ascend.PD_DECODE_SKIP_PREPROCESS:
            ## In Prefill node, the response will carry prompt_token_ids with kv_transfer_params
            final_res_batch_checked[0].kv_transfer_params["prompt_token_ids"] = prompt_token_ids
        # adaptor end
 
        response = self.request_output_to_completion_response(
            final_res_batch_checked,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")
    except GenerationError as e:
        return self._convert_generation_error_to_response(e)
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))
 
    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    if request.stream:
        response_json = response.model_dump_json()
 
        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"
 
        return fake_stream_generator()
 
    return response
 
 
def request_output_to_completion_response(
    self,
    final_res_batch: list[RequestOutput],
    request: CompletionRequest,
    request_id: str,
    created_time: int,
    model_name: str,
    tokenizer: TokenizerLike | None,
    request_metadata: RequestResponseMetadata,
) -> CompletionResponse:
    choices: list[CompletionResponseChoice] = []
    num_prompt_tokens = 0
    num_generated_tokens = 0
    kv_transfer_params = None
    last_final_res = None
    for final_res in final_res_batch:
        last_final_res = final_res
        prompt_token_ids = final_res.prompt_token_ids
        prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
        prompt_text = final_res.prompt
 
        # adaptor begin: reuse and skip tokenize
        if envs_ascend.PD_DECODE_SKIP_PREPROCESS:
            if request.kv_transfer_params and "prefilled_token" in request.kv_transfer_params:
                prompt_token_ids = request.kv_transfer_params["prefilled_token"]
                new_tokens = tokenizer.convert_ids_to_tokens(prompt_token_ids[0])
                prompt_text = tokenizer.convert_tokens_to_string([new_tokens])
                final_res.outputs[0].text = prompt_text + final_res.outputs[0].text
 
            ## In Prefill node, the response will carry prompt_token_ids with kv_transfer_params
            if final_res.kv_transfer_params:
                final_res.kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids
            if final_res.kv_transfer_params:
                final_res.kv_transfer_params["prefilled_token"] = [final_res.outputs[0].token_ids[0]]
        # adaptor end
 
        token_ids: GenericSequence[int]
        out_logprobs: GenericSequence[dict[int, Logprob] | None] | None
 
        for output in final_res.outputs:
            self._raise_if_error(output.finish_reason, request_id)
 
            if request.echo:
                if request.return_token_ids:
                    prompt_text = ""
                if request.max_tokens == 0:
                    token_ids = prompt_token_ids
                    out_logprobs = prompt_logprobs
                    output_text = prompt_text
                else:
                    token_ids = [*prompt_token_ids, *output.token_ids]
 
                    if request.logprobs is None:
                        out_logprobs = None
                    else:
                        out_logprobs = [
                            *prompt_logprobs,
                            *output.logprobs,
                        ]
 
                    output_text = prompt_text + output.text
            else:
                token_ids = output.token_ids
                out_logprobs = output.logprobs
                output_text = output.text
 
            if request.logprobs is not None:
                logprobs = self._create_completion_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    tokenizer=tokenizer,
                    num_output_top_logprobs=request.logprobs,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None
 
            choice_data = CompletionResponseChoice(
                index=len(choices),
                text=output_text,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
                prompt_logprobs=final_res.prompt_logprobs,
                prompt_token_ids=(
                    prompt_token_ids if request.return_token_ids else None
                ),
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
            )
            choices.append(choice_data)
 
            num_generated_tokens += len(output.token_ids)
 
        num_prompt_tokens += len(prompt_token_ids)
 
    # adaptor begin: reuse prefilled tokens
    if envs_ascend.PD_DECODE_SKIP_PREPROCESS:
        if request.kv_transfer_params and "prefilled_token" in request.kv_transfer_params:
            num_generated_tokens += 1
        if final_res_batch and final_res_batch[0].kv_transfer_params:
            final_res_batch[0].kv_transfer_params["stop_reasons"] = [output.stop_reason for output in final_res_batch[0].outputs]
    # adapt end
 
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
 
    if (
        self.enable_prompt_tokens_details
        and last_final_res
        and last_final_res.num_cached_tokens
    ):
        usage.prompt_tokens_details = PromptTokenUsageInfo(
            cached_tokens=last_final_res.num_cached_tokens
        )
 
    request_metadata.final_usage_info = usage
    if final_res_batch:
        kv_transfer_params = final_res_batch[0].kv_transfer_params
    return CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
        kv_transfer_params=kv_transfer_params,
    )
 
 
OpenAIServingCompletion.create_completion = create_completion
OpenAIServingCompletion.request_output_to_completion_response = request_output_to_completion_response
