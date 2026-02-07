"""Tests for EtherscanCollector block-range pagination logic."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collect_onchain import EtherscanCollector


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    """Return an EtherscanCollector with a fake key and no rate-limit delay."""
    return EtherscanCollector(api_key="test", rate_limit_delay=0)


def make_transfer(n, block_number=19000000):
    """Build a realistic Etherscan transfer dict."""
    return {
        "blockNumber": str(block_number),
        "timeStamp": str(1700000000 + n),
        "hash": f"0x{'a' * 63}{n:01x}",
        "logIndex": str(n),
        "from": f"0x{'b' * 40}",
        "to": f"0x{'c' * 40}",
        "value": str(1_000_000 * (n + 1)),  # raw integer value
        "gasPrice": str(30_000_000_000),     # 30 Gwei in wei
        "gasUsed": str(60_000),
    }


def make_transfers(count, block_start=19000000):
    """Build a list of N transfer dicts with unique hashes/logIndexes."""
    return [make_transfer(i, block_number=block_start + i) for i in range(count)]


# ---------------------------------------------------------------------------
# 1. _get_token_decimals
# ---------------------------------------------------------------------------

class TestGetTokenDecimals:
    def test_usdt_has_6_decimals(self, collector):
        assert collector._get_token_decimals("usdt") == 6

    def test_usdc_has_6_decimals(self, collector):
        assert collector._get_token_decimals("usdc") == 6

    def test_dai_has_18_decimals(self, collector):
        assert collector._get_token_decimals("dai") == 18

    def test_unknown_defaults_to_18(self, collector):
        assert collector._get_token_decimals("unknown_token") == 18


# ---------------------------------------------------------------------------
# 2. _format_transfer_df
# ---------------------------------------------------------------------------

class TestFormatTransferDf:
    def test_usdt_values_divided_by_1e6(self, collector):
        transfers = [make_transfer(0)]
        df = collector._format_transfer_df(transfers, "usdt")
        # make_transfer(0) value = 1_000_000 raw → 1_000_000 / 1e6 = 1.0
        assert df["value"].iloc[0] == pytest.approx(1.0)

    def test_dai_values_divided_by_1e18(self, collector):
        transfers = [make_transfer(0)]
        df = collector._format_transfer_df(transfers, "dai")
        # make_transfer(0) value = 1_000_000 raw → 1_000_000 / 1e18
        assert df["value"].iloc[0] == pytest.approx(1_000_000 / 1e18)

    def test_expected_columns(self, collector):
        transfers = [make_transfer(0)]
        df = collector._format_transfer_df(transfers, "usdt")
        expected_cols = [
            "timestamp", "coin", "value", "gas_price", "gas_used",
            "from", "to", "hash", "blockNumber",
        ]
        assert list(df.columns) == expected_cols

    def test_gas_price_in_gwei(self, collector):
        transfers = [make_transfer(0)]
        df = collector._format_transfer_df(transfers, "usdt")
        # 30_000_000_000 wei → 30.0 Gwei
        assert df["gas_price"].iloc[0] == pytest.approx(30.0)

    def test_coin_column_set(self, collector):
        transfers = [make_transfer(0)]
        df = collector._format_transfer_df(transfers, "usdc")
        assert df["coin"].iloc[0] == "usdc"

    def test_empty_list_returns_empty_dataframe(self, collector):
        df = collector._format_transfer_df([], "usdt")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# 3. get_block_by_timestamp
# ---------------------------------------------------------------------------

class TestGetBlockByTimestamp:
    def test_returns_block_number_on_valid_response(self, collector):
        collector._request = MagicMock(return_value={"result": "19000000"})
        block = collector.get_block_by_timestamp(1700000000)
        assert block == 19000000

    def test_raises_on_non_digit_result(self, collector):
        collector._request = MagicMock(
            return_value={"result": "Error! Invalid timestamp"}
        )
        with pytest.raises(ValueError, match="Could not resolve block"):
            collector.get_block_by_timestamp(1700000000)

    def test_raises_on_none_result(self, collector):
        collector._request = MagicMock(return_value={"result": None})
        with pytest.raises(ValueError, match="Could not resolve block"):
            collector.get_block_by_timestamp(1700000000)

    def test_passes_closest_param(self, collector):
        collector._request = MagicMock(return_value={"result": "19000000"})
        collector.get_block_by_timestamp(1700000000, closest="after")
        args = collector._request.call_args[0][0]
        assert args["closest"] == "after"


# ---------------------------------------------------------------------------
# 4. _fetch_transfers_in_block_range
# ---------------------------------------------------------------------------

class TestFetchTransfersInBlockRange:
    def test_under_cap_no_splitting(self, collector):
        """When results < MAX, return them directly without recursion."""
        transfers_100 = make_transfers(100)
        collector.get_token_transfers = MagicMock(return_value=transfers_100)

        result = collector._fetch_transfers_in_block_range("0xFAKE", 1000, 2000)
        assert len(result) == 100
        # Should have been called exactly once (no splits)
        collector.get_token_transfers.assert_called_once()

    def test_hits_cap_triggers_split(self, collector):
        """When first call returns exactly 10k, the range is split in two."""
        cap = collector.MAX_RESULTS_PER_QUERY
        big_batch = make_transfers(cap)
        small_left = make_transfers(50, block_start=1000)
        small_right = make_transfers(60, block_start=2000)

        # First call (full range): returns cap → split
        # Second call (left half): under cap
        # Third call (right half): under cap
        collector.get_token_transfers = MagicMock(
            side_effect=[big_batch, small_left, small_right]
        )

        result = collector._fetch_transfers_in_block_range("0xFAKE", 1000, 2000)
        assert len(result) == 50 + 60
        assert collector.get_token_transfers.call_count == 3

    def test_max_depth_guard(self, collector):
        """At MAX_SPLIT_DEPTH, stop recursing and return truncated results."""
        cap = collector.MAX_RESULTS_PER_QUERY
        big_batch = make_transfers(cap)
        collector.get_token_transfers = MagicMock(return_value=big_batch)

        # Start at max depth — should not recurse
        result = collector._fetch_transfers_in_block_range(
            "0xFAKE", 1000, 2000, depth=collector.MAX_SPLIT_DEPTH
        )
        assert len(result) == cap
        # Only one call, no recursion
        collector.get_token_transfers.assert_called_once()

    def test_single_block_guard(self, collector):
        """Same start and end block → no recursion even if cap is hit."""
        cap = collector.MAX_RESULTS_PER_QUERY
        big_batch = make_transfers(cap)
        collector.get_token_transfers = MagicMock(return_value=big_batch)

        result = collector._fetch_transfers_in_block_range("0xFAKE", 5000, 5000)
        assert len(result) == cap
        collector.get_token_transfers.assert_called_once()

    def test_non_list_response_returns_empty(self, collector):
        """If API returns a string/dict instead of a list, return []."""
        collector.get_token_transfers = MagicMock(
            return_value="Max rate limit reached"
        )
        result = collector._fetch_transfers_in_block_range("0xFAKE", 1000, 2000)
        assert result == []


# ---------------------------------------------------------------------------
# 5. _collect_by_block_range
# ---------------------------------------------------------------------------

class TestCollectByBlockRange:
    def test_returns_formatted_dataframe(self, collector):
        transfers = make_transfers(5)
        collector.get_block_by_timestamp = MagicMock(side_effect=[1000, 2000])
        collector._fetch_transfers_in_block_range = MagicMock(return_value=transfers)

        df = collector._collect_by_block_range(
            "usdt", datetime(2023, 1, 1), datetime(2023, 6, 1)
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "timestamp" in df.columns

    def test_deduplication(self, collector):
        """Duplicate (hash, logIndex) entries are removed."""
        t1 = make_transfer(0)
        t2 = make_transfer(1)
        t1_dup = make_transfer(0)  # same hash and logIndex as t1

        collector.get_block_by_timestamp = MagicMock(side_effect=[1000, 2000])
        collector._fetch_transfers_in_block_range = MagicMock(
            return_value=[t1, t2, t1_dup]
        )

        df = collector._collect_by_block_range(
            "usdt", datetime(2023, 1, 1), datetime(2023, 6, 1)
        )
        assert len(df) == 2  # t1_dup removed

    def test_start_block_gt_end_block_returns_empty(self, collector):
        """If start_block > end_block, return empty DataFrame."""
        collector.get_block_by_timestamp = MagicMock(side_effect=[5000, 1000])

        df = collector._collect_by_block_range(
            "usdt", datetime(2023, 1, 1), datetime(2023, 6, 1)
        )
        assert df.empty

    def test_unknown_coin_raises_valueerror(self, collector):
        with pytest.raises(ValueError, match="Unknown contract"):
            collector._collect_by_block_range(
                "fakecoin", datetime(2023, 1, 1), datetime(2023, 6, 1)
            )


# ---------------------------------------------------------------------------
# 6. collect_transfer_metrics dispatch
# ---------------------------------------------------------------------------

class TestCollectTransferMetricsDispatch:
    def test_with_start_date_calls_block_range(self, collector):
        collector._collect_by_block_range = MagicMock(return_value=pd.DataFrame())

        collector.collect_transfer_metrics(
            "usdt", start_date=datetime(2023, 1, 1), end_date=datetime(2023, 6, 1)
        )
        collector._collect_by_block_range.assert_called_once()

    def test_without_start_date_calls_pages(self, collector):
        collector._collect_by_pages = MagicMock(return_value=pd.DataFrame())

        collector.collect_transfer_metrics("usdt")
        collector._collect_by_pages.assert_called_once()

    def test_end_date_defaults_to_now_when_omitted(self, collector):
        collector._collect_by_block_range = MagicMock(return_value=pd.DataFrame())

        before = datetime.now(timezone.utc)
        collector.collect_transfer_metrics("usdt", start_date=datetime(2023, 1, 1))
        after = datetime.now(timezone.utc)

        _, kwargs = collector._collect_by_block_range.call_args
        # The method is called positionally: coin_key, start_date, end_date
        args = collector._collect_by_block_range.call_args[0]
        end_date_used = args[2]
        assert before <= end_date_used <= after


# ---------------------------------------------------------------------------
# 7. _collect_by_pages (legacy)
# ---------------------------------------------------------------------------

class TestCollectByPages:
    def test_returns_combined_data_from_multiple_pages(self, collector):
        page1 = make_transfers(10, block_start=1000)
        page2 = make_transfers(5, block_start=2000)

        collector.get_token_transfers = MagicMock(side_effect=[page1, page2, []])

        df = collector._collect_by_pages("usdt", num_pages=3)
        assert len(df) == 15

    def test_empty_first_page_returns_empty(self, collector):
        collector.get_token_transfers = MagicMock(return_value=[])

        df = collector._collect_by_pages("usdt", num_pages=3)
        assert df.empty

    def test_unknown_coin_raises_valueerror(self, collector):
        with pytest.raises(ValueError, match="Unknown contract"):
            collector._collect_by_pages("fakecoin")
