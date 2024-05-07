from typing import Dict, Sequence
from functools import reduce
import polars as pl
from pokebroker import logger

def add_column(
    df: pl.DataFrame,
    column_name: str,
    value: str
) -> pl.DataFrame:
    
    logger.debug('Adding column "%s" with value: %s', column_name, value)
    return df.with_columns(
        pl.lit(value).alias(column_name)
    )


def get_price_id(
    df: pl.DataFrame,
    using_columns: Sequence[str] = [
        'id',
        'quantity_type'
    ]
) -> pl.DataFrame:
    
    logger.debug('Generating "price_id" using columns: %s', using_columns)
    return (
        df
        .with_columns(
            (
                pl.concat_str(
                    using_columns,
                    separator='_'
                )
                .alias('price_id')
            )
        )
    )


def replace_substrings(
    df: pl.DataFrame,
    column_name: str,
    substring_mapping: Dict[str, str]
) -> pl.DataFrame:
    
    logger.debug(
        'Replacing substrings in column "%s" with mapping: %s',
        column_name,
        substring_mapping
    )
    return df.with_columns(
        reduce(
            lambda _col, _old: (
                _col
                .str
                .replace(
                    _old, 
                    substring_mapping[_old]
                )
            ),
            substring_mapping,
            pl.col(column_name)
        )
        .alias(column_name)
    )


def select_rename(
    df: pl.DataFrame,
    column_mapping: Dict[str, str]
) -> pl.DataFrame:
    
    logger.debug('Selecting and renaming columns with mapping: %s', column_mapping)
    return (
        df
        .select([
            pl.col(_original).alias(_new)
            for _original, _new in column_mapping.items()
        ])
    )

