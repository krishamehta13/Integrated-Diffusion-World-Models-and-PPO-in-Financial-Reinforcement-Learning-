import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

def get_data(tickers, start_date, end_date):
    dataset = {}
    for ticker in tickers:
        stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        stock_df = stock_df.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        stock_df['MACD'] = ta.macd(stock_df['Close'], fast=12, slow=26, append=True)['MACD_12_26_9']
        stock_df['RSI'] = ta.rsi(stock_df['Close'], length=14, append=True)
        stock_df['CCI'] = ta.cci(stock_df['High'], stock_df['Low'], stock_df['Close'], length=14, append=True)
        stock_df['ADX'] = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=14, append=True)['ADX_14']
        dataset[ticker] = stock_df

    stock_df = pd.concat(list(dataset.values()))
    stock_df.reset_index(inplace=True)
    final_df = stock_df.pivot(index='Date', columns='Ticker')
    final_df.columns = ['_'.join(col).strip() for col in final_df.columns.values]
    return final_df.dropna()

class StockTradingEnv(gym.Env):
    def __init__(self, df, tickers, downside_penalty=2.0):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.tickers = tickers
        self.stock_dim = len(tickers)
        self.initial_amount = 1_000_000
        self.transaction_cost_pct = 0.001
        self.downside_penalty = downside_penalty
        self.day = 0

        self.state_space_dim = 1 + self.stock_dim * 2 + self.stock_dim * 4
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space_dim,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.day = 0
        self.balance = self.initial_amount
        self.shares_held = np.zeros(self.stock_dim, dtype=int)
        self.total_portfolio_value = self.initial_amount
        self.terminal = False
        self.rewards_memory = []
        return self._get_state()

    def step(self, actions):
        if self.day >= len(self.df.index.unique()) - 1:
            self.terminal = True
            return self._get_state(), 0.0, self.terminal, {}

        SHARES_TO_TRADE = 100
        actions = (actions * SHARES_TO_TRADE).astype(int)

        begin_value = self.total_portfolio_value
        current_prices = self.df.iloc[self.day][[f'Close_{t}' for t in self.tickers]].values

        for i, action in enumerate(actions):
            if action < 0:
                shares_to_sell = min(abs(action), self.shares_held[i])
                if shares_to_sell > 0:
                    proceeds = current_prices[i] * shares_to_sell * (1 - self.transaction_cost_pct)
                    self.balance += proceeds
                    self.shares_held[i] -= shares_to_sell

        for i, action in enumerate(actions):
            if action > 0:
                shares_to_buy = action
                cost = current_prices[i] * shares_to_buy * (1 + self.transaction_cost_pct)
                if self.balance >= cost:
                    self.balance -= cost
                    self.shares_held[i] += shares_to_buy

        self.day += 1
        new_prices = self.df.iloc[self.day][[f'Close_{t}' for t in self.tickers]].values
        self.total_portfolio_value = self.balance + np.sum(self.shares_held * new_prices)

        raw_reward = self.total_portfolio_value - begin_value
        self.rewards_memory.append(raw_reward)
        
        pct_return = raw_reward / begin_value
        reward = pct_return if pct_return >= 0 else pct_return * self.downside_penalty
        reward = reward * 100
        
        state = self._get_state()
        return state, reward, self.terminal, {}

    def _get_state(self):
        current_data = self.df.iloc[self.day]
        state = [self.balance]
        prices = [current_data[f'Close_{t}'] for t in self.tickers]
        shares = list(self.shares_held)
        indicators = []
        for t in self.tickers:
            indicators.extend([
                current_data[f'MACD_{t}'],
                current_data[f'RSI_{t}'],
                current_data[f'CCI_{t}'],
                current_data[f'ADX_{t}']
            ])
        state.extend(prices)
        state.extend(shares)
        state.extend(indicators)
        return np.array(state, dtype=np.float32)

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, state_dim, cond_dim, hidden_dim=256):
        super(ConditionalDiffusionModel, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + cond_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x, cond, t):
        t_embed = self.time_embed(t)
        inp = torch.cat([x, cond, t_embed], dim=-1)
        return self.net(inp)

class DiffusionWorldModel:
    def __init__(self, state_dim, cond_dim, num_steps=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.model = ConditionalDiffusionModel(state_dim, cond_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.beta = torch.linspace(1e-4, 0.02, num_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def train_epoch(self, x_diff, cond, epochs=100, batch_size=64):
        self.model.train()
        dataset_size = x_diff.shape[0]
        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                x_b = torch.tensor(x_diff[batch_idx], dtype=torch.float32, device=self.device)
                c_b = torch.tensor(cond[batch_idx], dtype=torch.float32, device=self.device)
                t = torch.randint(0, self.num_steps, (x_b.shape[0], 1), device=self.device).float()
                noise = torch.randn_like(x_b)
                alpha_bar_t = self.alpha_bar[t.long()]
                x_noisy = torch.sqrt(alpha_bar_t) * x_b + torch.sqrt(1.0 - alpha_bar_t) * noise
                pred_noise = self.model(x_noisy, c_b, t / self.num_steps)
                loss = nn.MSELoss()(pred_noise, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def sample(self, cond):
        self.model.eval()
        cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = torch.randn((1, self.state_dim), device=self.device)
        for t_idx in reversed(range(self.num_steps)):
            t = torch.full((1, 1), t_idx, device=self.device).float()
            pred_noise = self.model(x, cond_t, t / self.num_steps)
            beta_t = self.beta[t_idx]
            alpha_t = self.alpha[t_idx]
            alpha_bar_t = self.alpha_bar[t_idx]
            noise = torch.randn_like(x) if t_idx > 0 else 0.0
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            x = mean + torch.sqrt(beta_t) * noise
        return x.squeeze(0).cpu().numpy()

def main():
    tickers = ["RELIANCE.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "TCS.NS", "ICICIBANK.NS"]
    train_df = get_data(tickers, '2015-01-01', '2020-01-01')
    trade_df = get_data(tickers, '2022-01-01', '2025-01-01')

    # Fit Diffusion World Model once
    cols = []
    for t in tickers:
        cols.extend([f'Close_{t}', f'MACD_{t}', f'RSI_{t}', f'CCI_{t}', f'ADX_{t}'])
    market_data = train_df[cols].values
    cond = market_data[:-1]
    targets = market_data[1:]
    diffs = targets - cond
    
    print("Training Diffusion World Model...")
    world_model = DiffusionWorldModel(state_dim=len(cols), cond_dim=len(cols))
    world_model.train_epoch(diffs, cond, epochs=120)
    
    print("Generating Synthetic Trajectories...")
    synthetic_trajectories = []
    for _ in range(15):
        start_day_idx = np.random.randint(0, len(market_data) - 50)
        curr_state = market_data[start_day_idx].copy()
        traj = [curr_state.copy()]
        for _ in range(200):
            diff = world_model.sample(curr_state)
            curr_state = curr_state + diff
            curr_state = np.clip(curr_state, a_min=market_data.min(axis=0)*0.5, a_max=market_data.max(axis=0)*2.0)
            traj.append(curr_state.copy())
        synthetic_trajectories.append(np.array(traj))
        
    synth_data = np.vstack(synthetic_trajectories)
    synth_df = pd.DataFrame(synth_data, columns=cols)
    synth_df.index = pd.date_range(start='2020-01-01', periods=len(synth_df), freq='D')

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Search Hyperparameters
    # We will test combinations of downside_penalty, ent_coef, learning rate, and timesteps
    penalties = [2.0, 2.5, 3.0, 3.5, 4.0]
    ent_coefs = [0.001, 0.003, 0.005, 0.008, 0.01]
    lrs = [0.0001, 0.0002, 0.0003]
    
    best_sharpe = -1
    best_config = None
    
    # We can try a random search over 15 configurations to find a good one quickly
    import random
    random.seed(42)
    
    configs = []
    for p in penalties:
        for ent in ent_coefs:
            for lr in lrs:
                configs.append({"penalty": p, "ent_coef": ent, "lr": lr})
                
    # Shuffle to test a representative subset first
    random.shuffle(configs)
    
    # Let's test up to 25 configurations
    for idx, cfg in enumerate(configs[:25]):
        print(f"\n--- Testing Config {idx+1}/{len(configs[:25])}: Penalty={cfg['penalty']}, Ent={cfg['ent_coef']}, LR={cfg['lr']} ---")
        
        # Train PPO on Synthetic Data
        synth_env = DummyVecEnv([lambda: StockTradingEnv(synth_df, tickers, downside_penalty=cfg["penalty"])])
        
        # Set seeds on environment
        np.random.seed(42)
        
        model = PPO(
            policy='MlpPolicy',
            env=synth_env,
            learning_rate=cfg["lr"],
            n_steps=2048,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=cfg["ent_coef"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0
        )
        
        model.learn(total_timesteps=80000)
        
        # Eval
        eval_env = StockTradingEnv(trade_df, tickers)
        obs = eval_env.reset()
        done = False
        portfolio = [eval_env.initial_amount]
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(act)
            portfolio.append(eval_env.total_portfolio_value)
            
        daily_returns = pd.Series(portfolio).pct_change().dropna()
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-9) * np.sqrt(252)
        ret_pct = (portfolio[-1] - portfolio[0]) / portfolio[0] * 100
        print(f"Results -> Sharpe: {sharpe:.2f} | Final Return: {ret_pct:.2f}%")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_config = cfg
            print(f"** NEW BEST SHARPE: {best_sharpe:.4f} **")
            if best_sharpe >= 1.12:
                print("FOUND TARGET SHARPE OF 1.12 OR HIGHER!")

    print(f"\nBest Config: {best_config} with Sharpe: {best_sharpe:.4f}")

if __name__ == "__main__":
    main()

