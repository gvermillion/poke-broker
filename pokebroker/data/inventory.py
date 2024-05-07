from typing import Any, Sequence
from functools import reduce
import polars as pl
from pokebroker import utils, logger, config
from pokebroker.data.base import DataHandlerBase

class TraderInventory(DataHandlerBase):
    NAME: str = 'Trader Inventory'

    def __init__(
        self,
        path: str,
        trader_id: str 
    ) -> None:
        self.trader_id: str = trader_id
        super().__init__(path=path)
        logger.debug('Trader ID: %s', self.trader_id)
    
    def _get_data(self) -> pl.DataFrame:
        return utils.IO.read_csv(
            path=self.path,
            separator=';',
            schema=config.SCHEMA['trader_inventory']
        )
    
    def _align_schema(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        return (
            df
            .pipe(
                utils.etl.select_rename,
                column_mapping=(
                    config.COLUMN_MAPPINGS
                    ['trader_inventory']
                )
            )
            .pipe(
                utils.etl.add_column,
                column_name='trader',
                value=self.trader_id
            )
            .pipe(
                utils.etl.replace_substrings,
                column_name='id',
                substring_mapping=config.SET_ALIGNMENT
            )
            .pipe(
                self._melt_by_variant
            )
            .pipe(
                utils.etl.get_price_id
            )
            .pipe(self._sparsify_quantity)
        )  

    @staticmethod
    def _melt_by_variant(
        df: pl.DataFrame
    ) -> pl.DataFrame:
        
        variants: Sequence[str] = [
            'normal_quantity',
            'holo_quantity',
            'reverse_holo_quantity',
        ]
        logger.debug('Melting by variant: %s', variants)
        return (
            df
            .melt(
                id_vars=[
                    'trader',
                    'id',
                    'name',
                    'set'
                ],
                value_vars=variants,
                variable_name='quantity_type',
                value_name='quantity',
            )
        )
    
    @staticmethod
    def _sparsify_quantity(
        df: pl.DataFrame
    ) -> pl.DataFrame:
        
        logger.debug(
            'Dropping %d rows with vanishing quantity.', 
            (
                df
                .filter(
                    pl.col('quantity').le(0)
                )
                .height
            )
        )
        return (
            df
            .filter(
                pl.col('quantity').gt(0)
            )
        )