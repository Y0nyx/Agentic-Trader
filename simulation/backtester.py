"""
Backtesting module for trading strategy simulation.

This module provides comprehensive backtesting functionality including
portfolio simulation, transaction execution, and performance reporting.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class PerformanceReport:
    """
    Container for backtesting performance results.
    
    This class holds and provides methods for analyzing the results
    of a trading strategy backtest.
    """
    
    def __init__(self, portfolio_history: pd.DataFrame, transactions: List[Dict], 
                 initial_capital: float, final_value: float):
        self.portfolio_history = portfolio_history
        self.transactions = transactions
        self.initial_capital = initial_capital
        self.final_value = final_value
        
    def summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of backtest performance.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics including returns, drawdowns, and trade metrics
        """
        if self.portfolio_history.empty:
            return {}
            
        portfolio = self.portfolio_history.copy()
        
        # Basic performance metrics
        total_return = (self.final_value - self.initial_capital) / self.initial_capital
        total_return_pct = total_return * 100
        
        # Calculate cumulative returns
        portfolio['Cumulative_Return'] = (portfolio['Total_Value'] / self.initial_capital) - 1
        
        # Calculate daily returns
        portfolio['Daily_Return'] = portfolio['Total_Value'].pct_change(fill_method=None)
        daily_returns = portfolio['Daily_Return'].dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0  # Annualized
        
        # Maximum Drawdown calculation
        cumulative_max = portfolio['Total_Value'].expanding().max()
        drawdown = (portfolio['Total_Value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        if volatility > 0:
            excess_return = (total_return * 252) - risk_free_rate  # Annualized
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0
        
        # Trade analysis
        num_trades = len(self.transactions)
        winning_trades = [t for t in self.transactions if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.transactions if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Trading period
        trading_days = len(portfolio)
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return': (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trading_days': trading_days,
        }
        
        return summary
    
    def get_transactions_df(self) -> pd.DataFrame:
        """Convert transactions list to DataFrame for analysis."""
        if not self.transactions:
            return pd.DataFrame()
        return pd.DataFrame(self.transactions)


class Backtester:
    """
    Trading strategy backtester with portfolio simulation.
    
    This class simulates trading a strategy with realistic transaction costs
    and portfolio management. It tracks cash, positions, and performance over time.
    
    Parameters
    ----------
    initial_capital : float, default 10000
        Starting capital for the backtest
    commission : float, default 0.001
        Commission rate as a fraction of trade value (0.001 = 0.1%)
    slippage : float, default 0.0005
        Slippage rate as a fraction of trade value (0.0005 = 0.05%)
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if commission < 0 or slippage < 0:
            raise ValueError("Commission and slippage must be non-negative")
            
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Portfolio state
        self.cash = initial_capital
        self.position = 0.0  # Number of shares held
        self.position_value = 0.0
        
        # Transaction history
        self.transactions = []
        self.portfolio_history = []
        
        logger.info(f"Initialized Backtester with capital={initial_capital}, commission={commission:.3f}, slippage={slippage:.3f}")
    
    def reset(self):
        """Reset the backtester to initial state."""
        self.cash = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self.transactions = []
        self.portfolio_history = []
    
    def _calculate_transaction_cost(self, trade_value: float) -> float:
        """Calculate the total transaction cost including commission and slippage."""
        return trade_value * (self.commission + self.slippage)
    
    def _execute_trade(self, price: float, quantity: float, trade_type: str, date: pd.Timestamp) -> bool:
        """
        Execute a trade with the given parameters.
        
        Parameters
        ----------
        price : float
            Price per share for the trade
        quantity : float
            Number of shares to trade (positive for buy, negative for sell)
        trade_type : str
            Type of trade ('BUY' or 'SELL')
        date : pd.Timestamp
            Date of the trade
            
        Returns
        -------
        bool
            True if trade was executed successfully, False otherwise
        """
        trade_value = abs(quantity * price)
        transaction_cost = self._calculate_transaction_cost(trade_value)
        
        if trade_type == 'BUY':
            total_cost = trade_value + transaction_cost
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for trade on {date}: need {total_cost:.2f}, have {self.cash:.2f}")
                return False
            
            self.cash -= total_cost
            self.position += quantity
            
        elif trade_type == 'SELL':
            if quantity > self.position:
                # Allow partial sale if not enough shares
                quantity = self.position
                trade_value = quantity * price
                transaction_cost = self._calculate_transaction_cost(trade_value)
            
            if quantity <= 0:
                return False
                
            net_proceeds = trade_value - transaction_cost
            self.cash += net_proceeds
            self.position -= quantity
        
        # Record transaction
        transaction = {
            'date': date,
            'type': trade_type,
            'price': price,
            'quantity': quantity,
            'trade_value': trade_value,
            'transaction_cost': transaction_cost,
            'cash_after': self.cash,
            'position_after': self.position
        }
        
        # Add P&L calculation for completed round trips
        if trade_type == 'SELL' and len(self.transactions) > 0:
            # Find the most recent BUY to calculate P&L
            recent_buys = [t for t in reversed(self.transactions) if t['type'] == 'BUY']
            if recent_buys:
                buy_price = recent_buys[0]['price']
                transaction['pnl'] = (price - buy_price) * quantity - transaction_cost
        
        self.transactions.append(transaction)
        return True
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> PerformanceReport:
        """
        Run a complete backtest using the provided data and signals.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical price data with OHLCV columns
        signals : pd.DataFrame
            Trading signals with 'Signal' column containing 'BUY', 'SELL', 'HOLD'
            
        Returns
        -------
        PerformanceReport
            Comprehensive backtest results and performance metrics
        """
        if data.empty or signals.empty:
            logger.error("Cannot run backtest with empty data or signals")
            return PerformanceReport(pd.DataFrame(), [], self.initial_capital, self.initial_capital)
        
        # Reset state
        self.reset()
        
        # Align data and signals
        aligned_data = data.join(signals[['Signal']], how='inner')
        
        if aligned_data.empty:
            logger.error("No aligned data between price data and signals")
            return PerformanceReport(pd.DataFrame(), [], self.initial_capital, self.initial_capital)
        
        logger.info(f"Running backtest on {len(aligned_data)} periods")
        
        # Process each day
        for date, row in aligned_data.iterrows():
            price = row['Close']
            signal = row['Signal']
            
            # Update position value
            self.position_value = self.position * price
            total_value = self.cash + self.position_value
            
            # Execute trades based on signals
            if signal == 'BUY' and self.position == 0:
                # Calculate position size (use most of available cash)
                max_investment = self.cash * 0.98  # Leave some cash for costs
                if max_investment > 0:
                    shares_to_buy = max_investment // price
                    if shares_to_buy > 0:
                        self._execute_trade(price, shares_to_buy, 'BUY', date)
                        
            elif signal == 'SELL' and self.position > 0:
                # Sell all shares
                self._execute_trade(price, self.position, 'SELL', date)
            
            # Update position value after potential trades
            self.position_value = self.position * price
            total_value = self.cash + self.position_value
            
            # Record daily portfolio state
            portfolio_record = {
                'Date': date,
                'Price': price,
                'Signal': signal,
                'Cash': self.cash,
                'Position': self.position,
                'Position_Value': self.position_value,
                'Total_Value': total_value
            }
            self.portfolio_history.append(portfolio_record)
        
        # Create portfolio history DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('Date', inplace=True)
        
        # Calculate final value
        final_value = self.cash + self.position_value
        
        logger.info(f"Backtest completed. Final value: {final_value:.2f}, Total return: {((final_value/self.initial_capital)-1)*100:.2f}%")
        
        return PerformanceReport(portfolio_df, self.transactions, self.initial_capital, final_value)