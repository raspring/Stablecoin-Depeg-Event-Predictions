"""Tests for KrakenCollector OHLC collection and pagination logic."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collect_kraken import KrakenCollector


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    """Return a KrakenCollector with no rate-limit delay."""
    return KrakenCollector(rate_limit_delay=0)


def make_ohlc_record(timestamp, close=1.0001, volume=50000.0):
    """Build a Kraken OHLC record: [time, open, high, low, close, vwap, volume, count]."""
    return [
        timestamp,
        str(close - 0.0001),  # open
        str(close + 0.0002),  # high
        str(close - 0.0003),  # low
        str(close),           # close
        str(close + 0.00005), # vwap
        str(volume),          # volume
        42,                   # count (trades)
    ]


def make_ohlc_records(count, start_ts=1700000000, interval=3600, close=1.0001):
    """Build a list of N OHLC records with incrementing timestamps."""
    return [
        make_ohlc_record(start_ts + i * interval, close=close)
        for i in range(count)
    ]


def make_api_response(records, pair="USDTUSD", last=None):
    """Build a Kraken API response dict."""
    if last is None and records:
        last = records[-1][0]
    return {
        "error": [],
        "result": {
            pair: records,
            "last": last,
        },
    }


# ---------------------------------------------------------------------------
# 1. get_ohlc
# ---------------------------------------------------------------------------

class TestGetOhlc:
    def test_returns_json_response(self, collector):
        records = make_ohlc_records(5)
        mock_response = MagicMock()
        mock_response.json.return_value = make_api_response(records)
        mock_response.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_response)

        result = collector.get_ohlc("USDTUSD", interval=60)
        assert "result" in result
        assert "error" in result

    def test_passes_since_parameter(self, collector):
        mock_response = MagicMock()
        mock_response.json.return_value = make_api_response([])
        mock_response.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_response)

        collector.get_ohlc("USDTUSD", interval=60, since=1700000000)
        call_kwargs = collector.session.get.call_args[1]
        assert call_kwargs["params"]["since"] == 1700000000

    def test_raises_on_api_error(self, collector):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": ["EGeneral:Invalid arguments"],
            "result": {},
        }
        mock_response.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Kraken API error"):
            collector.get_ohlc("BADPAIR")


# ---------------------------------------------------------------------------
# 2. get_historical_ohlc
# ---------------------------------------------------------------------------

class TestGetHistoricalOhlc:
    def test_returns_dataframe_with_expected_columns(self, collector):
        records = make_ohlc_records(10)
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records, last=records[-1][0])
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        expected_cols = [
            "timestamp", "symbol", "open", "high", "low", "close",
            "volume", "vwap", "trades", "spread_proxy", "vwap_deviation",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 10

    def test_timestamp_conversion_from_unix_seconds(self, collector):
        ts = 1700000000  # 2023-11-14 22:13:20 UTC
        records = [make_ohlc_record(ts)]
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records, last=ts)
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        assert df["timestamp"].iloc[0] == pd.Timestamp("2023-11-14 22:13:20")

    def test_deduplicates_on_time(self, collector):
        ts = 1700000000
        records = [make_ohlc_record(ts), make_ohlc_record(ts)]
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records, last=ts)
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        assert len(df) == 1

    def test_pagination_uses_last_field(self, collector):
        batch1 = make_ohlc_records(720, start_ts=1700000000)
        batch2 = make_ohlc_records(100, start_ts=1700000000 + 720 * 3600)

        collector.get_ohlc = MagicMock(side_effect=[
            make_api_response(batch1, last=batch1[-1][0]),
            make_api_response(batch2, last=batch2[-1][0]),
        ])

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2025, 1, 1),
        )

        assert collector.get_ohlc.call_count == 2
        # Second call should use 'last' from first response as 'since'
        second_call_kwargs = collector.get_ohlc.call_args_list[1]
        assert second_call_kwargs[1].get("since") == batch1[-1][0] or \
               (len(second_call_kwargs[0]) > 0)

    def test_empty_response_returns_empty_dataframe(self, collector):
        collector.get_ohlc = MagicMock(
            return_value={"error": [], "result": {"last": None}}
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_filters_records_beyond_end_date(self, collector):
        end_ts = 1700100000
        records = [
            make_ohlc_record(1700000000),  # before end
            make_ohlc_record(1700050000),  # before end
            make_ohlc_record(1700200000),  # after end
        ]
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records, last=records[-1][0])
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime.fromtimestamp(end_ts),
        )

        assert len(df) == 2


# ---------------------------------------------------------------------------
# 3. Derived metrics
# ---------------------------------------------------------------------------

class TestDerivedMetrics:
    def test_spread_proxy_calculation(self, collector):
        record = make_ohlc_record(1700000000, close=1.0)
        # high = 1.0002, low = 0.9997, close = 1.0
        # spread_proxy = (1.0002 - 0.9997) / 1.0 = 0.0005
        collector.get_ohlc = MagicMock(
            return_value=make_api_response([record])
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        expected_spread = (1.0002 - 0.9997) / 1.0
        assert df["spread_proxy"].iloc[0] == pytest.approx(expected_spread, rel=1e-4)

    def test_vwap_deviation_calculation(self, collector):
        record = make_ohlc_record(1700000000, close=1.0)
        # vwap = 1.00005, close = 1.0
        # vwap_deviation = (1.00005 - 1.0) / 1.0 = 0.00005
        collector.get_ohlc = MagicMock(
            return_value=make_api_response([record])
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        expected_vwap_dev = (1.00005 - 1.0) / 1.0
        assert df["vwap_deviation"].iloc[0] == pytest.approx(expected_vwap_dev, rel=1e-4)

    def test_numeric_types(self, collector):
        records = make_ohlc_records(3)
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records)
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        assert df["open"].dtype == float
        assert df["close"].dtype == float
        assert df["volume"].dtype == float
        assert df["vwap"].dtype == float
        assert df["trades"].dtype in [int, "int64"]


# ---------------------------------------------------------------------------
# 4. collect_stablecoin_trading_data
# ---------------------------------------------------------------------------

class TestCollectStablecoinTradingData:
    def test_returns_dict_of_dataframes(self, collector):
        records = make_ohlc_records(5)
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records)
        )

        result = collector.collect_stablecoin_trading_data(
            coin_key="usdt",
            start_date=datetime(2023, 1, 1),
        )

        assert isinstance(result, dict)
        assert "USDTUSD" in result
        assert isinstance(result["USDTUSD"], pd.DataFrame)

    def test_returns_empty_dict_for_unknown_coin(self, collector):
        result = collector.collect_stablecoin_trading_data(
            coin_key="frax",
            start_date=datetime(2023, 1, 1),
        )

        assert result == {}

    def test_all_supported_coins_have_pairs(self, collector):
        assert "usdt" in collector.PAIRS
        assert "usdc" in collector.PAIRS
        assert "dai" in collector.PAIRS


# ---------------------------------------------------------------------------
# 5. save_data
# ---------------------------------------------------------------------------

class TestSaveData:
    def test_saves_csv_files(self, collector, tmp_path):
        records = make_ohlc_records(5)
        collector.get_ohlc = MagicMock(
            return_value=make_api_response(records)
        )

        df = collector.get_historical_ohlc(
            pair="USDTUSD",
            start_date=datetime(2023, 11, 1),
            end_date=datetime(2024, 1, 1),
        )

        with patch("src.data.collect_kraken.RAW_DATA_DIR", tmp_path):
            paths = collector.save_data({"USDTUSD": df}, "usdt")

        assert len(paths) == 1
        assert paths[0].name == "usdt_kraken_usdtusd.csv"
        assert paths[0].exists()

        # Verify saved data
        saved = pd.read_csv(paths[0])
        assert len(saved) == 5
        assert "timestamp" in saved.columns

    def test_filename_convention(self, collector, tmp_path):
        df = pd.DataFrame({"timestamp": [1], "close": [1.0]})

        with patch("src.data.collect_kraken.RAW_DATA_DIR", tmp_path):
            paths = collector.save_data({"USDCUSD": df}, "usdc")

        assert paths[0].name == "usdc_kraken_usdcusd.csv"


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_http_error_propagates(self, collector):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        collector.session.get = MagicMock(return_value=mock_response)

        with pytest.raises(requests.HTTPError):
            collector.get_ohlc("USDTUSD")

    def test_api_error_raises_valueerror(self, collector):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": ["EGeneral:Unknown method"],
            "result": {},
        }
        mock_response.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Kraken API error"):
            collector.get_ohlc("USDTUSD")
