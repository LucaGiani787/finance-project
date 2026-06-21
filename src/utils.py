import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


# Creazione scatter plot con retta di regressione
def plot_scatter(x, y, color_scatter, title, xlabel, ylabel): 
    """Crea uno scatter plot con retta di regressione. 
    
    Args:
        x (np.ndarray): coordinate x
        y (np.ndarray): coordinate y
        color_scatter (str): colore dei punti
        title (str): titolo del grafico
        xlabel (str): etichetta dell'asse x
        ylabel (str): etichetta dell'asse y

    Returns:
        None
    """   
    # Scatter plot
    plt.scatter(x, y, color=color_scatter, alpha=0.4, s=15, edgecolors='none')
    
    # Retta di regressione
    ## Calcolo dei coefficienti della retta (y = mx + q)
    m, q = np.polyfit(x, y, 1)
    ## Tracciamento della retta di tendenza lineare
    x_range = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_range, m * x_range + q, color='red', linestyle='-', lw=2)
    
    plt.title(title, fontsize=11, fontweight='bold')
    plt.xlabel(xlabel, fontsize=9)
    plt.ylabel(ylabel, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)

    x_lims = plt.xlim()
    y_lims = plt.ylim()
    
    # 2. Trova il valore massimo assoluto tra tutti i limiti estratti
    max_val = max(abs(x_lims[0]), abs(x_lims[1]), abs(y_lims[0]), abs(y_lims[1]))
    
    # 3. Applica il valore massimo in modo simmetrico
    plt.xlim([-max_val, max_val])
    plt.ylim([-max_val, max_val])
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.savefig(f'./relazione/images/{xlabel}_{ylabel}.png')
    plt.show()
    plt.close()


def calcola_esposizione_rolling_multi_titoli(
    df, asset_cols, factor_cols, window=60
):
    """Calcola l'esposizione rolling ai fattori per più titoli singoli.

    Args:
        df (pd.DataFrame): dataframe con DatetimeIndex contenente sia i rendimenti in eccesso
        asset_cols (list): lista con i nomi delle colonne dei 6 titoli 
        factor_cols (list): lista con i nomi delle colonne dei fattori 
        window (int): finestra mobile in numero di periodi (mesi o giorni).

    Returns:
        pd.DataFrame:
    """
    tutti_i_risultati = []
    formula_base = 'Q("Mkt-RF") + SMB + HML + RMW + CMA'

    for asset in asset_cols:
        risultati_asset = []
        indici_temporali = []

        for i in range(window, len(df) + 1):
            finestra_dati = df.iloc[i - window : i]

            # formula specifica per l'asset corrente
            formula = f"{asset} ~ {formula_base}"

            # smf.ols vuole la formula e l'intero DataFrame della finestra
            modello = smf.ols(formula=formula, data=finestra_dati).fit()

            metriche = modello.params.to_dict()
            metriche["Titolo"] = asset

            risultati_asset.append(metriche)
            indici_temporali.append(finestra_dati.index[-1])

        if risultati_asset:
            df_asset = pd.DataFrame(risultati_asset, index=indici_temporali)
            tutti_i_risultati.append(df_asset)

    df_finale = pd.concat(tutti_i_risultati)
    df_finale.index.name = "Data"
    df_finale = df_finale.reset_index().set_index(["Data", "Titolo"])

    return df_finale


def plot_fama_french_rolling_exposure(df_exposure, stocks, factors):
    """Genera i grafici delle esposizioni rolling ai fattori Fama-French per una lista di titoli.

    Args:
        df_exposure (pd.DataFrame): dataframe con MultiIndex contenente il livello 'Titolo'
        stocks (list): lista dei titoli da ciclare e plottare
        factors (list): lista dei fattori Fama-French da includere nel grafico

    Returns:
        None
    """
    for titolo in stocks:
        # Estrazione e conversione indice
        dati_singolo_titolo = df_exposure.xs(titolo, level="Titolo")
        dati_singolo_titolo.index = pd.to_datetime(dati_singolo_titolo.index)

        # Creazione della figura
        _, ax = plt.subplots(figsize=(12, 5))

        # Plot dei fattori
        for fattore in factors:
            ax.plot(
                dati_singolo_titolo.index,
                dati_singolo_titolo[fattore],
                label=fattore,
                linewidth=2,
            )

        # Personalizzazione del grafico
        ax.set_title(
            f"Esposizioni Rolling ai Fattori Fama-French per {titolo}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Data", fontsize=11)
        ax.set_ylabel("Valore del Beta (Coefficiente)", fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        # plt.savefig(f"./relazione/images/{titolo}_rolling_exposure.png", bbox_inches="tight")
        plt.show()


def create_multivariate_dataset(dataset, look_back, target_index):
    """Crea un dataset per il training di una rete LSTM multivariata.
    
    Args:
        dataset (np.ndarray): array contenente le features (incluso il target)
        look_back (int): numero di periodi da usare come input
        target_index (int): indice della colonna da prevedere (target)
    
    Returns:
        X (np.ndarray):
            Array di input per la rete LSTM.
        Y (np.ndarray):
            Array dei target.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, target_index])
    return np.array(X), np.array(Y)


def implement_macd_strategy(prices, data):    
    """Implementazione della strategia MACD per il backtesting vettoriale.
    
    Args:
        prices (np.ndarray): array contenente i prezzi dei titoli
        data (np.ndarray): array contenente i dati del MACD
    
    Returns:
        buy_price (np.ndarray):
            Array contenente i prezzi di acquisto.
        sell_price (np.ndarray):
            Array contenente i prezzi di vendita.
        macd_signal (np.ndarray):
            Array contenente i segnali MACD.
    """    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(data)):
        if data['MACD'].iloc[i] > data['Signal line'].iloc[i]:
            if signal != 1:
                buy_price.append(prices.iloc[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['MACD'].iloc[i] < data['Signal line'].iloc[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices.iloc[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal