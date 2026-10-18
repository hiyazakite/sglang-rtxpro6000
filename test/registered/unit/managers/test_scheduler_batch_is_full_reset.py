"""Regression tests for the batch_is_full reset across hybrid pools.

``batch_is_full`` latches when the PrefillAdder fills *any* admission budget.
For single-pool models that verdict is durable: the batch is only un-full when a
running request finishes. Hybrid pools (SWA, SSM/Mamba) size admission in
several dimensions at once, so the verdict goes stale on its own -- a mamba slot
frees, or an evictable state becomes reclaimable -- with no new request arriving
to re-trigger the adder. Latching it there starves a queued prefill until an
unrelated event rebuilds the running batch; for Mamba that is the decode of the
whole running batch, i.e. seconds of TTFT for one request.

These tests pin the reset condition: hybrid SSM must reset, plain and
single-pool models must keep the latch.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _scheduler_for_prefill_raw(
    *,
    is_hybrid_swa: bool,
    is_hybrid_ssm: bool,
    enable_priority_preemption: bool,
    batch_is_full: bool,
) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_hierarchical_cache = False
    s.enable_priority_preemption = enable_priority_preemption
    s.is_hybrid_swa = is_hybrid_swa
    s.is_hybrid_ssm = is_hybrid_ssm
    # An empty waiting queue makes _get_new_batch_prefill_raw return at its
    # first gate, so the reset above it is the only state change under test.
    s.waiting_queue = []
    s.chunked_req = None
    s.tree_cache = MagicMock()
    s.running_batch = MagicMock()
    s.running_batch.batch_is_full = batch_is_full
    return s


class TestBatchIsFullReset(CustomTestCase):
    def _run(self, **kwargs) -> bool:
        s = _scheduler_for_prefill_raw(**kwargs)
        # The reset is above the only get_memory() call on this path; stub it so
        # the test does not need a published runtime context.
        with patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=MagicMock(enable_flexkv=False),
        ):
            s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=s.running_batch
            )
        return s.running_batch.batch_is_full

    def test_hybrid_ssm_resets_latched_batch_is_full(self):
        # The bug: a Mamba server kept the latch and never retried the adder.
        self.assertFalse(
            self._run(
                is_hybrid_swa=False,
                is_hybrid_ssm=True,
                enable_priority_preemption=False,
                batch_is_full=True,
            )
        )

    def test_hybrid_swa_and_ssm_both_reset(self):
        self.assertFalse(
            self._run(
                is_hybrid_swa=True,
                is_hybrid_ssm=True,
                enable_priority_preemption=False,
                batch_is_full=True,
            )
        )

    def test_plain_model_keeps_the_latch(self):
        # No hybrid pool: the verdict is durable, so the latch must survive and
        # spare the adder rebuild on every iteration.
        self.assertTrue(
            self._run(
                is_hybrid_swa=False,
                is_hybrid_ssm=False,
                enable_priority_preemption=False,
                batch_is_full=True,
            )
        )

    def test_priority_preemption_still_resets(self):
        self.assertFalse(
            self._run(
                is_hybrid_swa=False,
                is_hybrid_ssm=False,
                enable_priority_preemption=True,
                batch_is_full=True,
            )
        )

    def test_reset_only_clears_a_set_latch(self):
        self.assertFalse(
            self._run(
                is_hybrid_swa=False,
                is_hybrid_ssm=True,
                enable_priority_preemption=False,
                batch_is_full=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
