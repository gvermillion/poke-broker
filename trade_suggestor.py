# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: abbvie_consumer_mls
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trader

# %%
from pokebroker import config, utils, logger
from pokebroker.data.base import DataHandlerBase
from pokebroker.data import PriceData, TraderInventory

# %%
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from pokemontcgsdk import Card, RestClient
from itertools import chain, product
from functools import reduce
from ortools.linear_solver import pywraplp
import logging


# %%
def validate_suggested_trades(
    suggested_trades: pl.DataFrame,
    inventory: pl.DataFrame
) -> bool:
    
    logger.info('Validating suggested trades')
    assert (
        suggested_trades
        .join(
            inventory,
            left_on=[
                'from',
                'price_id',
                'id',
                'name'
            ],
            right_on=[
                'trader',
                'price_id',
                'id',
                'name'
            ],
            how='left'
        )
        .filter(
            pl.col('quantity') < 2
        )
        .is_empty()
    ), (
        'Suggested trades include cards that the giving trader does not have.'
    )
    assert (
        suggested_trades
        .group_by(
            'price_id'
        )
        .agg(*[
            pl.count('from').alias('n_trades')
        ])
        .filter(
            pl.col('n_trades') > 1
        )
        .is_empty()
    ), (
        'Suggested trades include cards that are traded more than once.'
    )
    assert (
        suggested_trades
        .group_by(
            [
                'to',
                'price_id'
            ]
        )
        .agg(
            pl.count('from').alias('n_receives')
        )
        .filter(
            pl.col('n_receives') > 1
        )
        .is_empty()
    ), (
        'Suggested trades include cards that are received more than once.'
    )

    assert (
        suggested_trades
        .group_by(
            [
                'from',
                'price_id'
            ]
        )
        .agg(
            pl.count('to').alias('n_gives')
        )
        .filter(
            pl.col('n_gives') > 1
        )
        .is_empty()
    ), (
        'Suggested trades include cards that are given more than once.'
    )
    return True


# %%
def get_suggested_trades(
    decision_variables: dict,
    price_data: pl.DataFrame,
    inventory: pl.DataFrame
) -> pl.DataFrame:
    
    logger.debug('Extracting suggested trades from decision variables.')
    out: pl.DataFrame = (
        pl.from_dict({
            str(_trade): _decision.solution_value()
            for _trade, _decision in decision_variables.items()
        })
        .with_columns(
            pl.lit(1).alias('id'),
        )
        .melt(
            id_vars=[
                'id',
            ],
            value_vars=[
                str(_trade)
                for _trade in decision_variables.keys()
            ],
            value_name='trade',
        )
        .drop([
            'id'
        ])
        .with_columns(
            pl.col('variable')
            .str.replace_all('\(', '')
            .str.replace_all('\)', '')
            .str.replace_all(' ', '')
            .str.replace_all("\'", '')
            .str.split_exact(',', 2)
            .alias('variable')
        )
        .unnest('variable')
        .rename({
            'field_0': 'from',
            'field_1': 'to',
            'field_2': 'price_id'
        })
        .filter(
            pl.col('trade') > 0
        )
        .join(
            price_data,
            on=[
                'price_id'
            ],
            how='left'
        )
        .select(
            'set_id',
            'id',
            'name',
            'from',
            'to',
            'price_id',
            'price',
        )
    )
    assert validate_suggested_trades(
        suggested_trades=out,
        inventory=inventory
    )

    return out


# %%
def maximize_card_distribution(
    cards: list,
    prices: dict,
    inventory: dict,
    price_tolerance: float = 0.10,
    default_price: float = 0.0,
):
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return "No solver available."
    
    traders = list(set(map(
        lambda _inventory: _inventory[0],
        inventory.keys()
    )))

    # Decision variables: decisions[i][j][c] where i, j are traders and c is card type
    # i -> giving trader, j -> receiving trader
    decisions = (
        {
            _trade: solver.BoolVar(
                'decisions[{},{},{}]'
                .format(*_trade))
            for _trade in filter(
                lambda _traders: _traders[0] != _traders[1],
                product(
                    traders,
                    traders,
                    cards
                )
            )
        }
    )
    logger.info('Finding optimial trades amongst {:,} possible trades'.format(len(decisions)))
    logger.debug('Traders: %s', traders)
    logger.debug('Traders: %s', traders[::-1])

    # Objective: Maximize value traded
    solver.Maximize(solver.Sum(
        decisions[i, j, c] * prices.get(c, default_price) 
        for i in traders for j in traders if i != j for c in cards
    ))

    # # Constraints: Try to balance the value received by each trader
    # for i in traders:
    #     other = (
    #         {
    #             1: 2,
    #             2: 1
    #         }
    #         [i]
    #     )
    #     i_value = solver.Sum(decisions[i, other, c] * prices.get(c, default_price) for c in cards)
    #     other_value = solver.Sum(decisions[other, i, c] * prices.get(c, default_price) for c in cards)
    #     i_value_greater = i_value >= other_value
    #     solver.Add(
    #         (
    #             i_value - other_value
    #             if i_value_greater else 
    #             other_value - i_value
    #         )  <= price_tolerance / 2 * (i_value + other_value)
    #     )
    #     # solver.Add(
    #     #     solver.Sum(decisions[i, other, c] * prices.get(c, default_price) for c in cards) >= 
    #     #     solver.Sum(decisions[other, i, c] * prices.get(c, default_price) for c in cards) * 1 - (price_tolerance / 2)
    #     # )

    # Contraint: The percent difference in value of traded cards between traders is less than the price tolerance
    solver.Add(
        solver.Sum(decisions[(*traders, c)] * prices.get(c, default_price) for c in cards) >= 
        solver.Sum(decisions[(*traders[::-1], c)] * prices.get(c, default_price) for c in cards) * (1 - (price_tolerance / 2))
    )
    solver.Add(
        solver.Sum(decisions[(*traders[::-1], c)] * prices.get(c, default_price) for c in cards) >= 
        solver.Sum(decisions[(*traders, c)] * prices.get(c, default_price) for c in cards) * (1 - (price_tolerance / 2))
    )

    # Constraints: Each trader can only trade up to the number of cards they have minus 1 and
    # each card can only be traded once
    for c in cards:
        solver.Add(
            solver.Sum(
                decisions[(*traders, c)] 
                for c in [c]
            ) <= max(
                inventory.get(
                    (traders[0],c),
                    0
                ) - 1,
                0
            )
        )
        solver.Add(
            solver.Sum(
                decisions[(*traders[::-1], c)] 
                for c in [c]
            ) <= max(
                inventory.get(
                    (traders[-1],c),
                    0
                ) - 1,
                0
            )
        )
        solver.Add(solver.Sum(
            decisions[(*traders, c)] + decisions[(*traders[::-1], c)] for c in [c]) <= 1
        )
    # solver.Add(solver.Sum(decisions[1, 0, c] for c in cards) >= 1)
        
 
    # Contraint: Each trader only receives one copy of each card
    for j in traders:
        for c in cards:
            solver.Add(
                solver.Sum(decisions[i, j, c] for i in traders if i != j) <= 1
            )

    

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        logger.debug('Objective value = ${:2f}'.format(
            solver.Objective().Value()
        ))
        for _combo in (
            traders,
            traders[::-1]
        ):
            logger.debug('Trader {} traded {:.0f} cards with a trade value of: ${:.2f}'.format(
                _combo[0],
                sum(decisions[(*_combo, c)].solution_value() for c in cards),
                sum(decisions[(*_combo, c)].solution_value() * prices.get(c, default_price) for c in cards)
            ))
            logger.debug(
                'Trader %s gives: %s', 
                _combo[0],
                [
                    c for c in cards if decisions[(*_combo, c)].solution_value() > 0
                ]
            )
        logger.debug('Total cards traded: {:.0f}'.format(
           sum(decisions[(*traders, c)].solution_value() for c in cards) + sum(decisions[(*traders[::-1], c)].solution_value() for c in cards)
        ))

    else:
        print('The problem does not have an optimal solution.')

    return solver, decisions


# %%
trader_collection_files = {
    'pickle_boy': '/Users/grantvermillion/Downloads/orion.csv',
    'mr_vermillion': '/Users/grantvermillion/Downloads/grant.csv',
}

# %%
price_data: pl.DataFrame = PriceData('data/price').load()


# %%

inventory: pl.DataFrame = pl.concat(map(
    lambda _trader_id: (
        TraderInventory(
            trader_id=_trader_id,
            path=trader_collection_files[_trader_id]
        )
        .load()
    ),
    trader_collection_files,
))
(
    inventory
    .join(
        price_data,
        on=[
            'price_id'
        ],
        how='left'
    )
    .with_columns(
        (
            pl.col('price') * pl.col('quantity')
        )
        .alias('ownership_value')
    )
    .group_by(
        'trader'
    )
    .agg(*[
        pl.count('price_id').alias('quantity'),
        pl.n_unique('price_id').alias('unique_cards'),
        pl.sum('ownership_value').alias('total_value')
    ])
)

# %%
(
    inventory
    .unique(
        subset=[
            'trader',
            'price_id'
        ]
    )
    .join(
        price_data,
        on=[
            'price_id'
        ],
        how='left'
    )
    .with_columns(
        (
            pl.col('price') * pl.col('quantity')
        )
        .alias('ownership_value')
    )
    .group_by(
        'trader'
    )
    .agg(*[
        pl.count('price_id').alias('quantity'),
        pl.n_unique('price_id').alias('unique_cards'),
        pl.sum('ownership_value').alias('total_value')
    ])
)

# %%

# %%
(
    inventory
            .select([
                'price_id',
                'quantity'
            ])
            .filter(
                pl.col('quantity') > 0
            )
            # .unique()
            .height
)

# %%

# %%
args = {
    'cards': (
        price_data
        .join(
            inventory
            .select([
                'price_id',
                'quantity'
            ])
            .filter(
                pl.col('quantity') > 0
            )
            .unique(
                subset=[
                    'price_id'
                ]
            ),
            how='inner',
            on=[
                'price_id'
            ]
        )
        .unique(
            subset=[
                'price_id'
            ]
        )
        ['price_id']
        .to_list()
    ),
    'prices': dict(
        price_data
        .select([
            'price_id',
            'price'
        ])
        .rows_by_key(
            key='price_id',
            unique=True
        )
    ),
    'inventory': dict(
        inventory
        # .filter(pl.col('trader') == tra)
        .select([
            'trader',
            'price_id',
            'quantity'
        ])
        .rows_by_key(
            key=[
                'trader',
                'price_id'
            ],
            unique=True
        )
    ),
    'price_tolerance': 0.01,
}
args.keys()

# %%

# Run the function
solver, decision_variables = maximize_card_distribution(
    **args
)

# %%
suggested_trades = get_suggested_trades(
    decision_variables=decision_variables,
    price_data=price_data,
    inventory=inventory
)
(
    suggested_trades
    .group_by(
        ['from', 'to']
    )
    .agg([
        pl.sum('price').alias('total_value'),
        pl.count('price').alias('total_cards'),
    ])
)

# %%
(
    trader_inventory
    .assign(**{
        'from': lambda df: (
            df
            ['trader']
            .map({
                'orion': '1',
                'grant': '2'
            })
        )
    })
    .merge(
        suggested_trades
        .filter([
            'from',
            'price_id'
        ]),
        how='inner',
        on=[
            'from',
            'price_id'
        ]
    )
    # .groupby(['from', 'set'])
    # .agg(**{
    #     # 'total_value': ('trend_price', 'sum'),
    #     'total_cards': ('price_id', 'nunique'),
    # })
)

# %%
(
    inventory
    .with_columns(
        pl.col('id')
        .str.split_exact('-', 1)
        .alias('set_id')
    )
    .unnest('set_id')
    .with_columns(
        pl.col('field_0')
        .is_in(
            price_data
            ['set_id']
        )
        .alias('good_set_id')
    )
    .filter(~pl.col('good_set_id'))
    ['field_0']
    .value_counts()
)

# %%

# %%

# %%
# from itertools import product

# _dict_helper = list(
#     filter(
#         lambda _combo: _combo[0] != _combo[1],
#         product(
#             list(range(3)),
#             list(range(3)),
#             (
#                 potential_trades
#                 .assign(**{
#                     'price_id': lambda df: (
#                         df
#                         ['Id']
# %% '-' + (
#                             df
#                             ['Type']
#                             .map({
#                                 'Normal Quantity': 'prices.trendPrice',
#                                 'Holo Quantity': 'prices.trendPrice',
#                                 'Reverse Holo Quantity': 'prices.reverseHoloTrend',
#                             })
#                         )
#                     )
#                 })
#                 ['price_id']
#                 .drop_duplicates()
#                 .tolist()
#             ),
#         )
#     )
# )

# %%

# %%

# %%


# %%
trades = (
    suggested_trades
    .query('trade')
    .groupby('from')
    ['name']
    .apply(set)
    .to_dict()
)

# %%
[len(_x) for _x in trades.values()]

# %%

# %%

# %%

# %%
(
    suggested_trades
    .query('trade')
    .assign(**{
        'Id': lambda df: (
            df
            ['Id']
            .str
            .replace('sv35', 'sv3pt5')
            .str
            .replace('sv45', 'sv4pt5')
            # + '_' + (
            #     df
            #     ['from']
            # )
        )
    })
    .merge(
        (
            price_data
            .assign(**{
                'Id': lambda df: (
                    df
                    ['Id']
                    + '-' + (
                        df
                        ['price_type']
                    )
                )
            })
        ),
        on='Id',
        how='left'
    )
    .groupby('from')
    ['trend_price']
    .sum()
)


# %% [markdown]
# ## Experimental Cyclic Trader

# %%
def get_cyclic_trades_og(
    traders: list,
    cards: list,
) -> list:
    return list(
        map(
            flatten,
            product(
                unique_circular_permutations(
                    traders,
                    r=len(traders)
                ),
                unique_circular_permutations(
                    cards,
                    r=len(traders)
                )
            )
        )
    )

# %% [markdown]
# from itertools import combinations, permutations
# def flatten(t):
#     return tuple(chain.from_iterable(
#         map(flatten, t) 
#         if isinstance(t, tuple)
#         else (t,)
#     ))
#
# def unique_circular_permutations(items, r = None):
#     """ Generate unique circular permutations by ensuring the smallest element is always first. """
#     seen = set()
#     for perm in permutations(items, r or len(items)):
#         # Find the index of the minimum item (lexicographically smallest)
#         min_index = perm.index(min(perm))
#         # Rotate perm to start at the minimum item
#         normalized_perm = perm[min_index:] + perm[:min_index]
#         # Use a tuple (frozen representation) to keep track in a set
#         normalized_tuple = tuple(normalized_perm)
#         # Check if this normalized version has been seen
#         if normalized_tuple not in seen:
#             seen.add(normalized_tuple)
#             yield perm
#             
# def get_cyclic_trades(traders, cards):
#
#     logger.debug('Getting all trader combinations.')
#     trader_combinations = (
#         combinations(traders, n) 
#         for n in range(
#             2,
#             len(traders) + 1
#         )
#     )
#     logger.debug('Getting all card combinations.')
#     card_combinations = (
#         combinations(cards, n) 
#         for n in range(
#             2, 
#             len(cards) + 1
#         )
#     )
#     
#     logger.debug('Getting all possible trades.')
#     return list(map(
#         flatten,
#         chain.from_iterable(
#             (
#                 product(
#                     unique_circular_permutations(trader_set),
#                     unique_circular_permutations(card_set)
#                 )
#                 for trader_set, card_set in filter(
#                     # Number of traders equals number of cards traded
#                     lambda x: len(x[0]) == len(x[1]), 
#                     product(
#                         chain.from_iterable(trader_combinations), 
#                         chain.from_iterable(card_combinations)
#                     )
#                 )
#             )
#         )
#     ))
#
# traders = 'A B C'.split(' ')
# cards = list(map(str, range(5)))
# possible_cyclic_trades = get_cyclic_trades(
#     traders=traders,
#     cards=cards
# )
# possible_cyclic_trades
