import unittest
from unittest import mock

from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


class TestQSACompressedKeyConsumerFence(unittest.TestCase):
    """The QSA indexer must wait for a layerwise HiCache load before reading
    compressed index keys, exactly like get_key_buffer() (issue #74)."""

    def _pool(self, counter):
        pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
        pool.layer_transfer_counter = counter
        pool.start_layer = 0
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
        pool.qsa_compressed_k_buffer_pool = ["layer3", "layer7"]
        return pool

    def test_waits_for_the_model_layer_before_returning_the_buffer(self):
        counter = mock.Mock()
        pool = self._pool(counter)

        self.assertEqual(pool.get_qsa_compressed_k_buffer(7), "layer7")
        counter.wait_until.assert_called_once_with(7)

    def test_no_counter_means_no_wait(self):
        pool = self._pool(None)
        self.assertEqual(pool.get_qsa_compressed_k_buffer(3), "layer3")


if __name__ == "__main__":
    unittest.main()
