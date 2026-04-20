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
from vllm.v1.request import Request
from vllm_ascend import envs as envs_ascend
 
 
def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        # assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens

        # adaptor begin: reuse prfilled tokens
        if envs_ascend.PD_DECODE_SKIP_PREPROCESS:
            if request.sampling_params.extra_args['kv_transfer_params'] \
                    and "prefilled_token" in request.sampling_params.extra_args['kv_transfer_params']:
                request.prompt_token_ids.extend(request.sampling_params.extra_args['kv_transfer_params']['prefilled_token'])
                request.append_output_token_ids(request.sampling_params.extra_args['kv_transfer_params']['prefilled_token'])
    
        # Update the request state for scheduling.
        request.num_computed_tokens = request.num_tokens - 1
        # adaptor end   
        self.finished_recving_kv_req_ids.remove(request.request_id)
 
 
from vllm.v1.core.sched.scheduler import Scheduler
Scheduler._update_waiting_for_remote_kv = _update_waiting_for_remote_kv
