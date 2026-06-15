import numpy as np
import pandas as pd
import scipy.optimize as sco

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calcola rendimento e rischio atteso di un portafoglio dati i pesi."""
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calcola l'indice di Sharpe negativo (per minimizzazione)."""
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

def minimize_volatility(weights, mean_returns, cov_matrix):
    """Calcola la volatilità (per minimizzazione)."""
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return p_std

def optimize_analytical(mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Ottimizzazione analitica usando scipy.optimize.
    Trova i pesi del portafoglio con Max Sharpe e Min Volatilità.
    """
    num_assets = len(mean_returns)
    args_sharpe = (mean_returns, cov_matrix, risk_free_rate)
    args_vol = (mean_returns, cov_matrix)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # Portafoglio di tangenza (Max Sharpe)
    opt_sharpe = sco.minimize(negative_sharpe, num_assets*[1./num_assets,], args=args_sharpe,
                        method='SLSQP', bounds=bounds, constraints=constraints)
                        
    # Portafoglio a varianza minima (Min Volatility)
    opt_vol = sco.minimize(minimize_volatility, num_assets*[1./num_assets,], args=args_vol,
                        method='SLSQP', bounds=bounds, constraints=constraints)
                        
    perf_sharpe = portfolio_performance(opt_sharpe.x, mean_returns, cov_matrix)
    perf_vol = portfolio_performance(opt_vol.x, mean_returns, cov_matrix)
    
    return {
        'max_sharpe': {
            'weights': opt_sharpe.x,
            'return': perf_sharpe[0],
            'risk': perf_sharpe[1],
            'sharpe': -opt_sharpe.fun
        },
        'min_vol': {
            'weights': opt_vol.x,
            'return': perf_vol[0],
            'risk': perf_vol[1],
            'sharpe': (perf_vol[0] - risk_free_rate) / perf_vol[1]
        }
    }

def optimize_montecarlo(mean_returns, cov_matrix, num_portfolios=40000, risk_free_rate=0.01):
    """
    Simulazione Monte Carlo per la stima della frontiera efficiente.
    Genera num_portfolios portafogli con pesi casuali normalizzati a 1.
    """
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
    return results, weights_record

def get_mc_optimal_portfolios(results, weights_record):
    """Estrae dal risultato Monte Carlo i portafogli max_sharpe e min_vol"""
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])
    
    return {
        'max_sharpe': {
            'weights': weights_record[max_sharpe_idx],
            'return': results[0, max_sharpe_idx],
            'risk': results[1, max_sharpe_idx],
            'sharpe': results[2, max_sharpe_idx]
        },
        'min_vol': {
            'weights': weights_record[min_vol_idx],
            'return': results[0, min_vol_idx],
            'risk': results[1, min_vol_idx],
            'sharpe': results[2, min_vol_idx]
        }
    }

def calculate_portfolio_beta(weights, asset_betas):
    """Calcola il Beta del portafoglio come combinazione lineare dei beta."""
    return np.dot(weights, asset_betas)
