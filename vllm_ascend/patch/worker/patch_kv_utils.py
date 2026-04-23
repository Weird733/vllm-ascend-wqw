from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

from vllm.v1.outputs import _combine_non_none
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypeVar

import numpy as np
import vllm

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    KVConnectorStats = object
    KVConnectorWorkerMetadata = object
    KVConnectorKVEvents = object



@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0
    first_tokens: dict[str, int] | None = None

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )

    @classmethod
    def merge(cls, *outputs: "KVConnectorOutput"):
        assert len(outputs) > 0, "Cannot merge empty outputs"
        finished_sending = _combine_non_none(
            set.union, [output.finished_sending for output in outputs]
        )
        finished_recving = _combine_non_none(
            set.union, [output.finished_recving for output in outputs]
        )
        kv_connector_stats = _combine_non_none(
            lambda x, y: x.aggregate(y),
            [output.kv_connector_stats for output in outputs],
        )
        kv_cache_events = _combine_non_none(
            lambda x, y: x.merge(y),
            [output.kv_cache_events for output in outputs],
        )
        invalid_block_ids = _combine_non_none(
            set.union, [output.invalid_block_ids for output in outputs]
        )
        assert invalid_block_ids is not None

        assert all(
            output.expected_finished_count == outputs[0].expected_finished_count
            for output in outputs
        )
        expected_finished_count = outputs[0].expected_finished_count

        # Merge first_token dictionaries from all outputs
        first_token_dicts = [output.first_tokens for output in outputs if output.first_tokens]
        first_tokens = {}
        for d in first_token_dicts:
            first_tokens.update(d)

        return cls(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            kv_connector_stats=kv_connector_stats,
            kv_cache_events=kv_cache_events,
            invalid_block_ids=invalid_block_ids,
            expected_finished_count=expected_finished_count,
            first_tokens=first_tokens if first_tokens else None,
        )




def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
) -> ModelRunnerOutput | None:
    if not outputs[output_rank]:
        return None

    # Aggregate kv_connector_output from all workers

    def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
    ) -> None:
        for req_id in req_ids or ():
            remaining_count = remaining_count_dict.get(
                req_id, self._expected_finished_count
            )
            remaining_count_dict[req_id] = remaining_count - 1
            if remaining_count_dict[req_id] == 0:
                finished_set.add(req_id)
                del remaining_count_dict[req_id]

    finished_sending = set[str]()
    finished_recving = set[str]()
    aggregated_kv_connector_stats = None
    aggregated_kv_connector_worker_meta = None
    combined_kv_cache_events = None
    invalid_block_ids = set[int]()
    first_tokens = None
    for model_runner_output in outputs:
        assert model_runner_output is not None
        kv_output = model_runner_output.kv_connector_output
        if kv_output is not None and kv_output.first_tokens:
            first_tokens = kv_output.first_tokens
        if not kv_output:
            continue
        # Allow the worker to dynamically update the expected number of
        # finished sending/recving for new requests.
        if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
        ):
            logger.debug(
                "Expected finished requests updated from %d to %d",
                self._expected_finished_count,
                kv_output.expected_finished_count,
            )
            self._expected_finished_count = kv_output.expected_finished_count

        update_finished_set(
            kv_output.finished_sending, self._send_remaining_count, finished_sending
        )
        update_finished_set(
            kv_output.finished_recving, self._recv_remaining_count, finished_recving
        )

        # Aggregate kv_connector_stats from all workers.
        if aggregated_kv_connector_stats is None:
            # Use the first worker's kv_connector_stats as accumulator.
            aggregated_kv_connector_stats = kv_output.kv_connector_stats
        elif kv_connector_stats := kv_output.kv_connector_stats:
            if aggregated_kv_connector_stats is None:
                aggregated_kv_connector_stats = kv_connector_stats
            else:
                assert isinstance(
                    aggregated_kv_connector_stats, type(kv_connector_stats)
                )
                aggregated_kv_connector_stats = (
                    aggregated_kv_connector_stats.aggregate(kv_connector_stats)
                )

        # Aggregate kv_connector_worker_meta from all workers.
        if aggregated_kv_connector_worker_meta is None:
            # Use the first worker's kv_connector_worker_meta as accumulator.
            aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
        elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
            aggregated_kv_connector_worker_meta = (
                aggregated_kv_connector_worker_meta.aggregate(
                    kv_connector_worker_meta
                )
            )

        # Combine kv_cache_events from all workers.
        if combined_kv_cache_events is None:
            # Use the first worker's kv_cache events as start event list.
            combined_kv_cache_events = kv_output.kv_cache_events
        elif kv_cache_events := kv_output.kv_cache_events:
            assert isinstance(
                combined_kv_cache_events,
                type(kv_cache_events),
            )
            worker_kv_cache_events = kv_cache_events.get_all_events()
            combined_kv_cache_events.add_events(worker_kv_cache_events)
            combined_kv_cache_events.increment_workers(1)
        invalid_block_ids |= kv_output.invalid_block_ids

    # select output of the worker specified by output_rank
    output = outputs[output_rank]

    assert output is not None
    output.kv_connector_output = KVConnectorOutput(
        finished_sending=finished_sending or None,
        finished_recving=finished_recving or None,
        kv_connector_stats=aggregated_kv_connector_stats or None,
        kv_cache_events=combined_kv_cache_events or None,
        kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
        invalid_block_ids=invalid_block_ids,
        expected_finished_count=self._expected_finished_count,
        first_tokens=first_tokens,
    )

    return output

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
KVOutputAggregator.aggregate = aggregate