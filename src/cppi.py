import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

def fit_garch(returns):
    """
    Stima i parametri di un modello GARCH(1,1) sui rendimenti.
    Ritorna i parametri mu, omega, alpha, beta e la volatilità annualizzata stimata.

    Args:
        returns (pd.Series): Rendimenti giornalieri.

    Returns:
        dict: Parametri stimati.  
            mu: media dei rendimenti
            omega: parametro omega del GARCH(1,1)
            alpha: parametro alpha del GARCH(1,1)
            beta: parametro beta del GARCH(1,1)
            ann_vol: volatilità annualizzata stimata
        res: oggetto del modello GARCH(1,1) stimato
    """
    # Rimuovi eventuali NaN
    rets = returns.dropna()
    
    # Stima modello GARCH(1,1)
    am = arch_model(rets, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(disp='off')
    
    # Estrai parametri
    mu = res.params['mu']
    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    
    # Calcola volatilità (condizionale) dell'ultimo giorno annualizzata
    last_var = res.conditional_volatility.iloc[-1] ** 2
    ann_vol = np.sqrt(last_var * 252)
    
    params = {
        'mu': mu,
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'ann_vol': ann_vol
    }
    
    return params, res

def simulate_garch_paths(params, s0, N_sim=5000, periodi=252):
    """
    Simula percorsi di prezzo futuri usando il modello GARCH(1,1) stimato.

    Args:
        params (dict): Parametri stimati.
        s0 (float): Valore iniziale del portafoglio.
        N_sim (int): Numero di simulazioni.
        periodi (int): Numero di periodi.

    Returns:
        np.ndarray: Array di rendimenti simulati.
    """
    mu = params['mu']
    omega = params['omega']
    alpha = params['alpha']
    beta = params['beta']
    
    # Inizializza array
    returns_sim = np.zeros((N_sim, periodi))
    variances_sim = np.zeros((N_sim, periodi))
    
    # Valori iniziali 
    uncond_var = omega / (1 - alpha - beta) if (alpha + beta < 1) else omega
    
    # Genera shock casuali
    z = np.random.normal(0, 1, (N_sim, periodi))
    
    for t in range(periodi):
        if t == 0:
            variances_sim[:, t] = uncond_var
        else:
            variances_sim[:, t] = omega + alpha * (returns_sim[:, t-1] - mu)**2 + beta * variances_sim[:, t-1]
            
        returns_sim[:, t] = mu + np.sqrt(variances_sim[:, t]) * z[:, t]
        
    return returns_sim

def cppi_historical(returns, S0=100, mdo=85, mult=3, trading_filter=0.05, costo_equity=0.003):
    """
    Esegue la simulazione CPPI sui rendimenti storici.
    Assume rendimenti giornalieri e calcola la strategia passo per passo.

    Args:
        returns (pd.Series): Rendimenti giornalieri.
        S0 (float): Valore iniziale del portafoglio.
        mdo (int): Minimum acceptable drawdown in percent.
        mult (int): Multiplier for the cushion.
        trading_filter (float): Trading filter percentage.
        costo_equity (float): Cost of equity transactions.

    Returns:
        pd.DataFrame: Risultati della simulazione.
    """
    T = len(returns)
    
    portf_val = np.zeros(T)
    equity_alloc = np.zeros(T)
    cash_alloc = np.zeros(T)
    cushion_arr = np.zeros(T)
    floor_arr = np.zeros(T)
    
    # Inizializzazione t=0
    portf_val[0] = S0
    floor_arr[0] = S0 * (mdo / 100)
    cushion_arr[0] = portf_val[0] - floor_arr[0]
    
    # Allocazione iniziale
    e_target = mult * cushion_arr[0]
    e_target = max(0, min(e_target, portf_val[0])) # limitato tra 0 e il valore del portafoglio
    
    equity_alloc[0] = e_target * (1 - costo_equity)
    cash_alloc[0] = portf_val[0] - e_target
    
    # Loop su ogni giorno
    for t in range(1, T):
        # 1. Aggiorna il valore degli asset con i rendimenti di t
        r_eq = returns.iloc[t]
        r_cash = 0.0
        
        # Aggiorna posizioni prima del rebalancing
        eq_val_pre = equity_alloc[t-1] * (1 + r_eq)
        cash_val_pre = cash_alloc[t-1] * (1 + r_cash)
        portf_val[t] = eq_val_pre + cash_val_pre
        
        # 2. Ricalcola Floor e Cushion
        floor_arr[t] = floor_arr[t-1] * (1 + r_cash)         
        cushion_arr[t] = max(0, portf_val[t] - floor_arr[t])
        
        # 3. Calcola nuova allocazione target
        e_target = mult * cushion_arr[t]
        e_target = max(0, min(e_target, portf_val[t]))
        
        # 4. Applica trading filter
        peso_attuale = eq_val_pre / portf_val[t] if portf_val[t] > 0 else 0
        peso_target = e_target / portf_val[t] if portf_val[t] > 0 else 0
        
        if abs(peso_target - peso_attuale) > trading_filter:
            # Rebalance
            delta_eq = e_target - eq_val_pre
            costi = abs(delta_eq) * costo_equity
            
            portf_val[t] -= costi
            
            # Ricalcola target dopo costi
            e_target = mult * max(0, portf_val[t] - floor_arr[t])
            e_target = max(0, min(e_target, portf_val[t]))
            
            equity_alloc[t] = e_target
            cash_alloc[t] = portf_val[t] - e_target
        else:
            # No rebalance
            equity_alloc[t] = eq_val_pre
            cash_alloc[t] = cash_val_pre
            
    # Crea un DataFrame per i risultati
    df_res = pd.DataFrame({
        'Date': returns.index,
        'Portfolio_Value': portf_val,
        'Equity': equity_alloc,
        'Cash': cash_alloc,
        'Floor': floor_arr,
        'Cushion': cushion_arr
    })
    df_res.set_index('Date', inplace=True)
    
    return df_res

def cppi_montecarlo(returns_matrix, S0=100, mdo=85, mult=3, trading_filter=0.05, costo_equity=0.003):
    """
    Esegue la simulazione CPPI su multipli percorsi Monte Carlo in modo vettorializzato o veloce.

    Args:
        returns_matrix (np.ndarray): Array di rendimenti simulati. 
            Formato: (N_sim, periodi)
        S0 (float): Valore iniziale del portafoglio.
        mdo (int): Minimum acceptable drawdown in percent.
        mult (int): Multiplier for the cushion.
        trading_filter (float): Trading filter percentage.
        costo_equity (float): Cost of equity transactions.

    Returns:
        np.ndarray: Array di valori finali del portafoglio.
    """
    N_sim, T = returns_matrix.shape
    
    portf_vals = np.zeros((N_sim, T))
    equity_alloc = np.zeros((N_sim, T))
    cash_alloc = np.zeros((N_sim, T))
    floors = np.zeros((N_sim, T))
    
    # Inizializzazione t=0
    portf_vals[:, 0] = S0
    floors[:, 0] = S0 * (mdo / 100)
    cushion = portf_vals[:, 0] - floors[:, 0]
    
    e_target = np.clip(mult * cushion, 0, portf_vals[:, 0])
    equity_alloc[:, 0] = e_target * (1 - costo_equity)
    cash_alloc[:, 0] = portf_vals[:, 0] - e_target
    
    r_cash = 0.0
    
    for t in range(1, T):
        r_eq = returns_matrix[:, t]
        
        eq_val_pre = equity_alloc[:, t-1] * (1 + r_eq)
        cash_val_pre = cash_alloc[:, t-1] * (1 + r_cash)
        portf_vals[:, t] = eq_val_pre + cash_val_pre
        
        floors[:, t] = floors[:, t-1] * (1 + r_cash)
        cushion = np.maximum(0, portf_vals[:, t] - floors[:, t])
        
        e_target = np.clip(mult * cushion, 0, portf_vals[:, t])
        
        peso_attuale = np.where(portf_vals[:, t] > 0, eq_val_pre / portf_vals[:, t], 0)
        peso_target = np.where(portf_vals[:, t] > 0, e_target / portf_vals[:, t], 0)
        
        rebalance_mask = np.abs(peso_target - peso_attuale) > trading_filter
        
        # Calcola costi e aggiusta portafoglio dove si fa rebalancing
        delta_eq = e_target - eq_val_pre
        costi = np.abs(delta_eq) * costo_equity * rebalance_mask
        
        portf_vals[:, t] -= costi
        
        # Ricalcola target post-costi per chi rebilancia
        e_target_new = np.clip(mult * np.maximum(0, portf_vals[:, t] - floors[:, t]), 0, portf_vals[:, t])
        
        equity_alloc[:, t] = np.where(rebalance_mask, e_target_new, eq_val_pre)
        cash_alloc[:, t] = np.where(rebalance_mask, portf_vals[:, t] - e_target_new, cash_val_pre)
        
    return portf_vals

def buy_and_hold(returns, S0=100):
    """
    Calcola il valore del portafoglio per una strategia Buy&Hold semplice

    Args:
        returns (pd.Series): Rendimenti giornalieri.
        S0 (float): Valore iniziale del portafoglio.

    Returns:
        pd.DataFrame: Valore cumulativo del portafoglio.
    """
    # Valore cumulativo = S0 * prod(1+r)
    if isinstance(returns, pd.Series):
        cum_ret = (1 + returns).cumprod()
    else:
        cum_ret = np.cumprod(1 + returns, axis=1)
    
    return S0 * cum_ret

def compare_strategies(cppi_historical_val, bh_historical_val, cppi_mc_vals, bh_mc_vals, rf_daily=0.0):
    """
    Calcola e confronta le metriche tra CPPI e B&H (storico e MC)

    Args:
        cppi_historical_val (pd.Series): Valore del portafoglio con CPPI (storico).
        bh_historical_val (pd.Series): Valore del portafoglio con B&H (storico).
        cppi_mc_vals (np.ndarray): Valori del portafoglio con CPPI (Monte Carlo).
        bh_mc_vals (np.ndarray): Valori del portafoglio con B&H (Monte Carlo).
        rf_daily (float): Tasso di interesse risk-free giornaliero.

    Returns:    
        dict: Dizionario contenente le metriche comparate.
    """
    metrics = {}
    
    # 1. Metriche storiche
    cppi_hist_ret = (cppi_historical_val.iloc[-1] / cppi_historical_val.iloc[0]) - 1
    bh_hist_ret = (bh_historical_val.iloc[-1] / bh_historical_val.iloc[0]) - 1
    
    cppi_hist_daily_ret = cppi_historical_val.pct_change().dropna()
    bh_hist_daily_ret = bh_historical_val.pct_change().dropna()
    
    cppi_hist_vol = cppi_hist_daily_ret.std() * np.sqrt(252)
    bh_hist_vol = bh_hist_daily_ret.std() * np.sqrt(252)
    
    cppi_sharpe = ((cppi_hist_daily_ret.mean() - rf_daily) / cppi_hist_daily_ret.std()) * np.sqrt(252)
    bh_sharpe = ((bh_hist_daily_ret.mean() - rf_daily) / bh_hist_daily_ret.std()) * np.sqrt(252)
    
    # 2. Metriche Monte Carlo
    cppi_mc_ret = (cppi_mc_vals[:, -1] / cppi_mc_vals[:, 0]) - 1
    bh_mc_ret = (bh_mc_vals[:, -1] / bh_mc_vals[:, 0]) - 1
    
    metrics = {
        ('Historical', 'Return'): {'CPPI': cppi_hist_ret, 'B&H': bh_hist_ret},
        ('Historical', 'Volatility'): {'CPPI': cppi_hist_vol, 'B&H': bh_hist_vol},
        ('Historical', 'Sharpe Ratio'): {'CPPI': cppi_sharpe, 'B&H': bh_sharpe},
        ('MonteCarlo (Mean)', 'Return'): {'CPPI': np.mean(cppi_mc_ret), 'B&H': np.mean(bh_mc_ret)},
        ('MonteCarlo (Mean)', 'Volatility'): {'CPPI': np.std(cppi_mc_ret), 'B&H': np.std(bh_mc_ret)},
        ('MonteCarlo (Mean)', 'Min Return'): {'CPPI': np.min(cppi_mc_ret), 'B&H': np.min(bh_mc_ret)},
        ('MonteCarlo (Mean)', 'Max Return'): {'CPPI': np.max(cppi_mc_ret), 'B&H': np.max(bh_mc_ret)}
    }
    
    return metrics
