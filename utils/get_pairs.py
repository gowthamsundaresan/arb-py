import os
from web3 import Web3
from itertools import combinations
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv('NETWORK_CHAIN_RPC_URL')

w3 = Web3(Web3.HTTPProvider(database_url))

# Token addresses and names
TOKENS = {
    '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 'WETH',
    '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599': 'WBTC',
    '0x6B175474E89094C44Da98b954EedeAC495271d0F': 'DAI',
    '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': 'USDC',
    '0xdAC17F958D2ee523a2206206994597C13D831ec7': 'USDT'
}

# DEX factory addresses
DEX_FACTORIES = {
    '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f': 'Uniswap V2',
    '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac': 'Sushiswap'
}

def get_pair_addresses(token_pairs: List[Tuple[str, str]], dex_factories: dict) -> dict:
    # ABI for the 'getPair' function
    factory_abi = [{
        "constant": True,
        "inputs": [{"name": "tokenA", "type": "address"}, {"name": "tokenB", "type": "address"}],
        "name": "getPair",
        "outputs": [{"name": "pair", "type": "address"}],
        "type": "function"
    }]

    results = {}

    for dex_address, dex_name in dex_factories.items():
        factory_contract = w3.eth.contract(address=dex_address, abi=factory_abi)
        dex_results = {}

        for token_a, token_b in token_pairs:
            try:
                pair_address = factory_contract.functions.getPair(token_a, token_b).call()
                dex_results[(token_a, token_b)] = pair_address
            except Exception as e:
                dex_results[(token_a, token_b)] = f"Error: {str(e)}"

        results[dex_name] = dex_results

    return results

# Generate all combinations of token pairs
token_pairs = list(combinations(TOKENS.keys(), 2))

result = get_pair_addresses(token_pairs, DEX_FACTORIES)

for dex, pairs in result.items():
    print(f"DEX: {dex}")
    for pair, address in pairs.items():
        token_a_name = TOKENS[pair[0]]
        token_b_name = TOKENS[pair[1]]
        print(f"  {token_a_name} - {token_b_name}: {address}")
    print()