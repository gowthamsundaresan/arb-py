import os
import numpy as np
import cvxpy as cp
import asyncio
from collections import deque
from typing import List, Tuple, Dict
from web3 import Web3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.optimize import minimize_scalar
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv('NETWORK_CHAIN_RPC_URL')

w3 = Web3(Web3.HTTPProvider(database_url))

# ABI for the getReserves function
PAIR_ABI = [{
    "constant": True,
    "inputs": [],
    "name": "getReserves",
    "outputs": [
        {"name": "_reserve0", "type": "uint112"},
        {"name": "_reserve1", "type": "uint112"},
        {"name": "_blockTimestampLast", "type": "uint32"}
    ],
    "type": "function"
}]

def get_reserves(pair_address: str) -> Tuple[int, int]:
    """Query the reserves for a given Uniswap-like pair."""
    pair_contract = w3.eth.contract(address=pair_address, abi=PAIR_ABI)
    reserves = pair_contract.functions.getReserves().call()
    return reserves[0], reserves[1]

def update_reserves(pair_addresses: List[str]) -> List[List[float]]:
    """Update reserves for all pairs."""
    return [list(get_reserves(addr)) for addr in pair_addresses if addr != '0x0000000000000000000000000000000000000000']

def optimal_arbitrage(reserves: List[List[float]], fees: List[float], A: List[np.ndarray], max_input: float, input_asset_index: int):
    """Compute the optimal arbitrage given current reserves and find ideal input amount."""
    num_tokens = len(A[0])
    
    def solve_for_input(t):
        # Build variables
        deltas = [cp.Variable(2, nonneg=True) for _ in range(len(reserves))]
        lambdas = [cp.Variable(2, nonneg=True) for _ in range(len(reserves))]
        psi = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas, lambdas)])
        
        # Objective is to maximize the return of the input asset
        obj = cp.Maximize(psi[input_asset_index])
        
        # Reserves after trade
        new_reserves = [cp.hstack([R[0] + gamma_i*D[0] - L[0], R[1] + gamma_i*D[1] - L[1]]) 
                        for R, gamma_i, D, L in zip(reserves, fees, deltas, lambdas)]
        
        # Trading function constraints (assuming all are Uniswap v2 pools for simplicity)
        cons = [
            cp.geo_mean(new_r) >= cp.geo_mean(r) for new_r, r in zip(new_reserves, reserves)
        ]
        
        # Allow all assets at hand to be traded
        current_assets = np.zeros(num_tokens)
        current_assets[input_asset_index] = t
        cons.append(psi + current_assets >= 0)
        
        # Set up and solve problem
        prob = cp.Problem(obj, cons)
        try:
            prob.solve()
        except cp.error.SolverError:
            return None, None

        if prob.status != cp.OPTIMAL:
            return None, None
        
        return psi.value[input_asset_index] - t, [(d.value, l.value) for d, l in zip(deltas, lambdas)]

    result = minimize_scalar(
        lambda t: -solve_for_input(t)[0] if solve_for_input(t)[0] is not None else 0,  # Negative because we want to maximize
        bounds=(0, max_input),
        method='bounded'
    )
    
    ideal_input = result.x
    profit, trades = solve_for_input(ideal_input)
    
    return ideal_input, profit, trades
def is_profitable(profit: float, initial_asset_value: float) -> bool:
    """Check if the arbitrage is profitable"""
    gas_fees = 0.25 * profit
    net_profit = profit - gas_fees
    return net_profit > 0.001 * initial_asset_value

def check_arbitrage(input_asset_index: int, token: str, reserves: List[List[float]], fees: List[float], A: List[np.ndarray], max_input: float) -> Dict:
    """Check for multi-hop arbitrage opportunities for a single input asset."""
    try:
        ideal_input, profit, trades = optimal_arbitrage(reserves, fees, A, max_input, input_asset_index)
        
        if profit is None or trades is None:
            return None
        
        if is_profitable(profit, ideal_input):
            return {
                "token": token,
                "profit": profit,
                "trades": trades,
                "input_asset_index": input_asset_index,
                "ideal_input": ideal_input
            }
    except Exception as e:
        print(f"Error in check_arbitrage for {token}: {str(e)}")
    return None

def process_arbitrage_result(result: Dict, pair_addresses: List[str], tokens: List[str], A: List[np.ndarray]) -> bool:
    """Process and potentially execute a profitable arbitrage opportunity."""
    if result:
        print(f"Profitable arbitrage found starting with {result['token']}!")
        print(f"Ideal input amount: {result['ideal_input']:.6f} {result['token']}")
        print(f"Gross profit: {result['profit']:.6f} {result['token']}")
        print(f"Net profit: {0.75 * result['profit']:.6f} {result['token']}")
        for i, ((delta, lambda_), addr) in enumerate(zip(result['trades'], pair_addresses)):
            print(f"Pair {addr}: Trade {delta[0]:.6f} {tokens[A[i][0].nonzero()[0][0]]} for {lambda_[1]:.6f} {tokens[A[i][1].nonzero()[0][0]]}")
        
        # TODO: Simulate the trade
        return True
    return False

def handle_event(block_number: int, executor):
    """Handle new block events."""
    print(f"New block mined: {block_number}")
    
    reserves = update_reserves(pair_addresses)
    
    # Use ProcessPoolExecutor for parallel arbitrage checking
    futures = [executor.submit(check_arbitrage, i, token, reserves, fees, A, max_inputs[i]) 
               for i, token in enumerate(tokens)]
    
    # Process results as they complete
    for future in as_completed(futures):
        result = future.result()
        if result:
            # Submit result processing to ThreadPoolExecutor
            executor.submit(process_arbitrage_result, result, pair_addresses, tokens, A)
            break  # Exit after submitting the first profitable opportunity for processing

async def process_block(block_number, executor):
    print(f"Processing block: {block_number}")
    reserves = update_reserves(pair_addresses)
    
    futures = [executor.submit(check_arbitrage, i, token, reserves, fees, A, max_inputs[i]) 
               for i, token in enumerate(tokens)]
    
    any_profitable = False
    for future in as_completed(futures):
        result = future.result()
        profitable = process_arbitrage_result(result, pair_addresses, tokens, A)
        if profitable:
            any_profitable = True
            break  # Exit after finding the first profitable opportunity
    
    if not any_profitable:
        print(f"No profitable arbitrage opportunities found in block {block_number}")
    
    return any_profitable

async def main_loop(process_executor, thread_executor):
    block_queue = deque(maxlen=5)  # Store up to 5 recent blocks
    block_filter = w3.eth.filter('latest')

    while True:
        for event in block_filter.get_new_entries():
            latest_block = w3.eth.get_block('latest')
            block_number = latest_block['number']
            
            if block_number not in block_queue:
                block_queue.append(block_number)
        
        # Process any blocks in the queue
        while block_queue:
            block_to_process = block_queue.popleft()
            profitable = await process_block(block_to_process, thread_executor)
            if profitable:
                # Clear the queue if a profitable opportunity was found
                block_queue.clear()
                break
        
        await asyncio.sleep(1)

# Main execution
if __name__ == "__main__":
    # Define token addresses
    token_addresses = {
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    }

    # Define pair addresses and their corresponding tokens
    pair_info = {
        "0xBb2b8038a1640196FbE3e38816F3e67Cba72D940": ("WETH", "WBTC"),
        "0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11": ("WETH", "DAI"),
        "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc": ("WETH", "USDC"),
        "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852": ("WETH", "USDT"),
        "0x231B7589426Ffe1b75405526fC32aC09D44364c4": ("WBTC", "DAI"),
        "0x004375Dff511095CC5A197A54140a24eFEF3A416": ("WBTC", "USDC"),
        "0x0DE0Fa91b6DbaB8c8503aAA2D1DFa91a192cB149": ("WBTC", "USDT"),
        "0xAE461cA67B15dc8dc81CE7615e0320dA1A9aB8D5": ("DAI", "USDC"),
        "0xB20bd5D04BE54f870D5C0d3cA85d82b34B836405": ("DAI", "USDT"),
        "0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f": ("USDC", "USDT"),
        "0xCEfF51756c56CeFFCA006cD410B03FFC46dd3a58": ("WETH", "WBTC"),
        "0xC3D03e4F041Fd4cD388c549Ee2A29a9E5075882f": ("WETH", "DAI"),
        "0x397FF1542f962076d0BFE58eA045FfA2d347ACa0": ("WETH", "USDC"),
        "0x06da0fd433C1A5d7a4faa01111c044910A184553": ("WETH", "USDT"),
        "0x622D4a772B72f56602546559c95d7Ca214EbB24F": ("WBTC", "DAI"),
        "0x784178D58b641a4FebF8D477a6ABd28504273132": ("WBTC", "USDT"),
        "0xAaF5110db6e744ff70fB339DE037B990A20bdace": ("DAI", "USDC"),
        "0x055CEDfe14BCE33F985C41d9A1934B7654611AAC": ("DAI", "USDT"),
        "0xD86A120a06255Df8D4e2248aB04d4267E23aDfaA": ("USDC", "USDT"),
    }

    pair_addresses = list(pair_info.keys())
    tokens = list(token_addresses.keys())
    
    # Create A matrices for each pair
    A = []
    for pair_address in pair_addresses:
        if pair_address != '0x0000000000000000000000000000000000000000':
            token0, token1 = pair_info[pair_address]
            pair_matrix = np.zeros((len(tokens), 2))
            pair_matrix[tokens.index(token0), 0] = 1
            pair_matrix[tokens.index(token1), 1] = 1
            A.append(pair_matrix)
    
    fees = [0.997] * len(pair_addresses)
    max_inputs = [1000, 100, 10000, 10000, 10000]  # Maximum input amounts for each token

    # Set up a filter for new blocks
    block_filter = w3.eth.filter('latest')

    # Create a ProcessPoolExecutor for arbitrage checking
    with ProcessPoolExecutor() as process_executor:
        # Create a ThreadPoolExecutor for result processing and trade execution
        with ThreadPoolExecutor() as thread_executor:
            asyncio.run(main_loop(process_executor, thread_executor))
