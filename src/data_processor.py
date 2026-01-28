import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_and_process_campaigns(csv_path: str = None) -> pd.DataFrame:
    """
    Carrega o dataset de campanhas e calcula métricas derivadas:

    - CPA (Cost per Acquisition): spend_BRL / conversions
    - ROAS (Return on Ad Spend): revenue_BRL / spend_BRL
    - CTR (Click Through Rate): clicks / impressions

    Também trata divisões por zero com np.nan.

    Parameters
    ----------
    csv_path : str, optional
        Caminho para o CSV. Se None, usa data/campaigns.csv a partir do root do projeto.

    Returns
    -------
    pd.DataFrame
        DataFrame original com colunas adicionais: cpa_BRL, roas, ctr.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "campaigns.csv")

    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Converte para float de forma explícita
    df["spend_BRL"] = df["spend_BRL"].astype(float)
    df["revenue_BRL"] = df["revenue_BRL"].astype(float)
    df["impressions"] = df["impressions"].astype(float)
    df["clicks"] = df["clicks"].astype(float)
    df["conversions"] = df["conversions"].astype(float)

    # Evita divisão por zero usando np.where
    df["cpa_BRL"] = np.where(
        df["conversions"] > 0,
        df["spend_BRL"] / df["conversions"],
        np.nan,
    )

    df["roas"] = np.where(
        df["spend_BRL"] > 0,
        df["revenue_BRL"] / df["spend_BRL"],
        np.nan,
    )

    df["ctr"] = np.where(
        df["impressions"] > 0,
        df["clicks"] / df["impressions"],
        np.nan,
    )

    return df


def aggregate_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por canal, somando spend, revenue, impressions, clicks, conversions
    e recalculando métricas derivadas para análise de mídia.

    Returns
    -------
    pd.DataFrame
        DataFrame agregado com colunas: spend_BRL, revenue_BRL, impressions,
        clicks, conversions, cpa_BRL, roas, ctr.
    """
    grouped = (
        df.groupby("channel")
        .agg(
            spend_BRL=("spend_BRL", "sum"),
            revenue_BRL=("revenue_BRL", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )

    grouped["cpa_BRL"] = np.where(
        grouped["conversions"] > 0,
        grouped["spend_BRL"] / grouped["conversions"],
        np.nan,
    )

    grouped["roas"] = np.where(
        grouped["spend_BRL"] > 0,
        grouped["revenue_BRL"] / grouped["spend_BRL"],
        np.nan,
    )

    grouped["ctr"] = np.where(
        grouped["impressions"] > 0,
        grouped["clicks"] / grouped["impressions"],
        np.nan,
    )

    return grouped


def get_best_channel_by_roas(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Retorna o canal com maior ROAS e o valor correspondente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame já agregado por canal.

    Returns
    -------
    (str, float)
        Nome do canal e valor de ROAS.
    """
    if "channel" not in df.columns or "roas" not in df.columns:
        raise ValueError("DataFrame precisa conter colunas 'channel' e 'roas'.")

    best_row = df.loc[df["roas"].idxmax()]
    return best_row["channel"], float(best_row["roas"])
