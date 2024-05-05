import polars as pl
from pokebroker import utils, logger
from pokebroker.data.base import DataHandlerBase

class PriceData(DataHandlerBase):
    NAME: str = 'Price Data'
    
    def _get_data(self) -> pl.DataFrame:
        return utils.IO.read_dataset(
            path=self.path,
        )
    
    def _align_schema(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        return (
            df
            .drop(
                'price_id'
            )
            .with_columns(
                (
                    pl.col('price_type')
                    .str.replace_all('normal_market', 'normal_quantity')
                    .str.replace_all('holofoil_market', 'holo_quanitity')
                    .str.replace_all('reverseHolofoil_market', 'reverse_holo_quantity')
                    .alias('price_type')
                ),
            )
            .with_columns(
                (
                    pl.concat_str(
                    [
                            pl.col('id'),
                            pl.col('price_type')
                        ],
                        separator='_'
                    )
                    .alias('price_id')
                )
            )
        )  
