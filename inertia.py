import collections
from datetime import datetime
import json
from functools import lru_cache
from math import sqrt, isnan
import re

import pandas as pd
import polars as pl
import plotly.io as pio
import plotly.graph_objects as go
import requests
from sqlalchemy import Integer, Table, select, func, create_engine, MetaData, and_, Column, Float
from sqlalchemy.orm import Session

import py_bsm_lib as pbl


pio.renderers.default = 'browser'


def calculate_vanna_inertia(underlying, data):
    data = data.reset_index()
    data = data[["strike", "vanna"]]
    data = data.groupby("strike").sum()
    agg_vanna = data.sum()

    inertias = {}
    strikes = list(data.index)
    strikes = [strike for strike in strikes if underlying*0.95 <= strike <= underlying*1.05]
    if not strikes:
        temp = {abs(s - underlying): s for s in list(data.index)}
        temp = collections.OrderedDict(sorted(temp.items()))
        bound1 = list(temp.values())[0]
        bound2 = list(temp.values())[1]
        if bound1 < bound2:
            strikes = [strike for strike in list(data.index) if bound1 <= strike <= bound2]
        else:
            strikes = [strike for strike in list(data.index) if bound2 <= strike <= bound1]
    for strike in strikes:
        if strike < underlying:
            subset = data[strike:underlying].__deepcopy__()
            mass = subset["vanna"].sum()/agg_vanna
            if underlying - strike != 0:
                inertia = mass * ((-data.loc[strike, "vanna"])/(underlying - strike))
            else:
                inertia = 0
            inertias[strike] = inertia
        else:
            subset = data[underlying:strike].__deepcopy__()
            mass = -1 * subset["vanna"].sum()/agg_vanna
            if underlying - strike != 0:
                inertia = mass * ((data.loc[strike, "vanna"])/(underlying - strike))
            else:
                inertia = 0
            inertias[strike] = inertia
    try:
        inertias = pd.DataFrame(list(inertias.values()), index=list(inertias.keys()))
    except TypeError:
        inertias = pd.DataFrame([i["vanna"] if not isinstance(i, int) else i for i in list(inertias.values())], index=list(inertias.keys()))
    inertias = pd.concat([inertias[:underlying].sort_index(ascending=False).cumsum().sort_index(ascending=True), inertias[underlying:].cumsum()])
    inertial = inertias.loc[:underlying].sum() - inertias.loc[underlying:].sum()
    inertial = (inertias.loc[:underlying].sum() + inertias.loc[underlying:].sum())/inertial
    return inertial[0]


def execute_query(query, engine):
    with Session(bind=engine) as session:
        with session.begin():
            try:
                results = session.execute(query).fetchall()
                session.commit()

                return results
            except Exception as e:
                print(f"ERROR:: failed to query on, rolling back transaction. {e}")
                session.rollback()

                return None


def rust_update(row, vol, u, rfr, dividend, columns):
    size = row[columns.index('size')]
    ticker = row[columns.index('ticker')]
    strike = row[columns.index('strike')]
    owner = row[columns.index('owner')]
    expiration = row[columns.index('expiration')]
    kind = row[columns.index('kind')]
    symbol = row[columns.index('symbol')]
    t = row[columns.index('t')]
    ex = False

    try:
        try:
            bsm_iv = None
            while bsm_iv is None:
                bsm = pbl.bsm_greeks(float(u), float(strike), t, rfr, vol, dividend, kind)
                bsm_iv = bsm['iv']
                if isnan(bsm_iv):
                    vol = vol / sqrt(252)
        except Exception as e:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}::update_options::rust_update:: {ticker}, {e}")
            print(f"Variable check: vol-{vol}, underlying-{u}, rfr-{rfr}, dividend-{dividend}")

        if float(strike) >= u:
            dag = -bsm['gamma']
        else:
            dag = bsm['gamma']

        # reconstruct row
        reconstructed_row = {'charm': bsm['charm'], 'dag': dag,
                             'delta': bsm['delta'], 'expiration': expiration,
                             'gamma': bsm['gamma'], 'kind': kind, 'owner': owner,
                             'rho': bsm['rho'], 'size': size,
                             'strike': strike, 'symbol': symbol, 'theta': bsm['theta'], 'ticker': ticker,
                             'vanna': bsm['vanna'], 'vega': bsm['vega'], 'vomma': bsm['vomma'], "iv": bsm["iv"],
                             "value": bsm["value"]}
        return tuple(reconstructed_row.values())
    except Exception as e:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}::update_options::rust_update::Error occured with {ticker} update: {row}, {e}")
        reconstructed_row = {'charm': 0, 'dag': 0, 'delta': 0,
                             'expiration': expiration, 'gamma': 0, 'kind': kind,
                             'owner': owner, 'rho': 0, 'size': size,
                             'strike': strike, 'symbol': symbol, 'theta': 0, 'ticker': ticker,
                             'vanna': 0, 'vega': 0, 'vomma': 0, "iv": 0, "value": 0}
        return tuple(reconstructed_row.values())


def get_underlying():
    response = requests.get(f"https://ejqxjp513k.execute-api.us-east-1.amazonaws.com/beta/volland_funds/price?symbol=SPX",
                            headers={
                                'x-api-key': "IWDehwwoBw8fWHnnkyfIv7Uut3BrqmWt3hHfRBrz",
                            })
    underlying = json.loads(response.text)['price']
    vol = json.loads(response.text)['volatility']
    dividend = json.loads(response.text)["dividend"]
    return underlying, vol, dividend


@lru_cache
def parse_option_string(row):
    return tuple(re.findall(r"[^\W\d_]+|\d+", row[0]))


@lru_cache
def handle_time(ticker, expiration, adjusted_time=None):
    if adjusted_time is None:
        now = datetime.now()
    else:
        now = adjusted_time

    if 'SPX' == ticker:
        expiration = datetime(expiration.year, expiration.month, expiration.day, hour=(9 + int(4)), minute=30, second=0)
    else:
        expiration = datetime(expiration.year, expiration.month, expiration.day, hour=(16 + int(4)), minute=0, second=0)

    t = pbl.time_to_expire(now.strftime("%y%m%d %H:%M:%S"), expiration.strftime("%y%m%d %H:%M:%S"))
    return t


user = "postgres"
password = "cd2f21ce9a281b1dde3034dfd7ea8d46a453394451962d8426145f4e02c42f4939dbd9ab3d28ae37ed22ef0c4ecc66621998cee5099d53858e27519fcd787502bedb1643a3e3eb4f3140e576ec75ccd3724ce5593167a9e0763559222fb1d56d3f3365771fb7ddef4994aa44e5164eb98cd0069155e5e2ae6ee0689250c96f80"
host = "100.26.140.20"
db = "trades"

production_engine = create_engine(f'postgresql://{user}:{password}@{host}/{db}')

day = datetime.now()
ticker = "SPX"
rfr = 0.0055
reconstructed_columns = ['charm', 'dag', 'delta', 'expiration', 'gamma', 'kind', 'owner',
                                      'rho', 'size', 'strike', 'symbol', 'theta', 'ticker', 'vanna',
                                      'vega', 'vomma', "iv", "value"]
underlying, vol, dividend = get_underlying()
table = Table(f'full_options_{ticker.lower()}', MetaData(), autoload_with=production_engine)
query = (select(table.c.symbol, table.c.owner, func.sum(func.cast(table.c.size, Integer)))
         .where(func.cast(table.c.expiration, Integer) >= int(day.strftime('%y%m%d'))).group_by(table.c.owner,
                                                                                                 table.c.symbol))

production_trades = execute_query(query, production_engine)

pl_df = pl.DataFrame([{"symbol": item[0], "owner": item[1],
                       "size": item[2]} for item in production_trades])
pl_df = pl_df.with_columns(pl.col("symbol").map_elements(lambda s: s.split(":")[-1] if ":" in s else s, return_dtype=pl.String))
pl_df = pl_df.with_columns(pl_df.map_rows(lambda s: parse_option_string(s)))
pl_df = pl_df.rename({"column_0": "ticker", "column_1": "expiration", "column_2": "kind", "column_3": "strike"})
pl_df = pl_df[["symbol", "owner", "size", "ticker", "expiration", "kind", "strike"]]
pl_df = pl_df.with_columns(
    pl.when(pl.col('expiration').str.len_chars() > 6).then(pl.col("expiration").str.slice(1)).otherwise(
        pl.col("expiration")).alias('expiration'))
pl_df = pl_df.with_columns(pl.col("expiration").cast(pl.Int64, strict=False).alias("expiration"))
pl_df = pl_df.with_columns(pl.col("strike").cast(pl.Int64) / 1000)
pl_df = pl_df.with_columns(
    pl.when(pl.col("kind") == 'C').then(pl.lit("call")).otherwise(pl.lit("put")).alias('kind'))

columns = pl_df.columns
pl_df = pl_df.with_columns(
    pl_df['expiration'].map_elements(lambda x: datetime.strptime(str(x), "%y%m%d"), return_dtype=pl.Datetime))
pl_df = pl_df.with_columns(
    pl_df.map_rows(
        lambda row: handle_time(row[columns.index("ticker")], row[columns.index("expiration")])))
pl_df = pl_df.rename({"map": "t"})
pl_df = pl_df.filter(pl.col('t') > 0)

columns = pl_df.columns
updated_data = pl_df.map_rows(
    lambda row: rust_update(row, vol, underlying, rfr, dividend, columns))
updated_data = updated_data.rename(
    {k: v for k, v in zip(updated_data.columns, reconstructed_columns)})

updated_data = updated_data.with_columns(
    pl.when(pl.col('delta').abs() <= 0.5).then(pl.col('delta').abs() * pl.col('vanna')).otherwise(
        (1 - pl.col('delta').abs()) * pl.col('vanna')).alias('vanna'))

options_institution = updated_data.filter(pl.col("owner") == "institution")
options_consumer = updated_data.filter(pl.col("owner") == "consumer")

options_institution = options_institution.with_columns(pl.col("vanna") * pl.col("size"))
options_consumer = options_consumer.with_columns(pl.col("vanna") * pl.col("size"))

options_institution = options_institution.drop(
    ['owner', 'size', 'symbol', 'ticker', 'dag', 'delta',
     'rho', 'theta', 'charm', 'vomma', "gamma", "iv", "value"])
options_consumer = options_consumer.drop(
    ['owner', 'size', 'symbol', 'ticker', 'dag', 'delta', 'rho',
     'theta', 'charm', 'vomma', "gamma", "iv", "value"])

strike_group1 = options_institution.to_pandas().groupby(['expiration', 'strike', 'kind']).sum()
strike_group2 = options_consumer.to_pandas().groupby(['expiration', 'strike', 'kind']).sum()
strike_groups = strike_group1.subtract(strike_group2, fill_value=0)
for col in strike_groups.columns:
    strike_groups[col] = strike_groups[col] * 100 * underlying

strike_groups.reset_index(inplace=True)
expirations = list(set(strike_groups['expiration']))

overall_vanna_inertia = calculate_vanna_inertia(underlying, strike_groups)
inertias = {}
for expiration in expirations:
    subset = strike_groups[strike_groups['expiration'] == expiration]
    inertias[expiration] = calculate_vanna_inertia(underlying, subset)

fig = go.Figure()
fig.add_trace(go.Bar(x=list(inertias.keys()), y=list(inertias.values()), name="Inertia By Expiration"))
fig.add_hline(y=overall_vanna_inertia)
fig.update_layout(title="Inertia By Expiration")
fig.show()