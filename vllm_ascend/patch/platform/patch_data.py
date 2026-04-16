# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a patched file of the vllm-ascend project.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import NotRequired, TypedDict, TypeIs, TypeVar

if TYPE_CHECKING:
    from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalInputs,
                                        MultiModalUUIDDict)
from vllm_ascend import envs as envs_ascend


class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    token_type_ids: NotRequired[list[int]]
    """A list of token type IDs to pass to the cross encoder model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

    multi_modal_uuids: NotRequired["MultiModalUUIDDict"]
    """
    Optional user-specified UUIDs for multimodal items, mapped by modality.
    Lists must match the number of items per modality and may contain `None`.
    For `None` entries, the hasher will compute IDs automatically; non-None
    entries override the default hashes for caching.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """
    # adapt for reuse_prefilled_tokens
    """
    This is used when the model supports reusing prefilled tokens.
    """
    prefilled_token_ids: Optional[list[int]] = []
    prefilled_texts: Optional[str] = ""

    # adapt end


from vllm.inputs import data
data.TokensPrompt = TokensPrompt