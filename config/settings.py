from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Stablecoin configuration
STABLECOINS = {
    "usdt": {
        "coingecko_id": "tether",
        "defillama_id": 1,
        "name": "Tether",
        "peg": 1.0,
        "type": "fiat-backed",
        "ethereum_contract": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "binance_pairs": ["BTCUSDT", "ETHUSDT"],
    },
    "usdc": {
        "coingecko_id": "usd-coin",
        "defillama_id": 2,
        "name": "USD Coin",
        "peg": 1.0,
        "type": "fiat-backed",
        "ethereum_contract": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "binance_pairs": ["BTCUSDC", "ETHUSDC"],
    },
    "dai": {
        "coingecko_id": "dai",
        "defillama_id": 5,
        "name": "Dai",
        "peg": 1.0,
        "type": "crypto-collateralized",
        "ethereum_contract": "0x6B175474E89094C44Da98b954EesdfeeB131e",
        "binance_pairs": ["DAIUSDT"],
    },
    "frax": {
        "coingecko_id": "frax",
        "defillama_id": 6,
        "name": "Frax",
        "peg": 1.0,
        "type": "hybrid",
        "ethereum_contract": "0x853d955aCEf822Db058eb8505911ED77F175b99e",
        "binance_pairs": ["FRAXUSDT"],
    },
}

# Depeg threshold
DEPEG_THRESHOLD = 0.01  # 1% deviation from peg
