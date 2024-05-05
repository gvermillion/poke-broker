from pyarrow import parquet as pq
import polars as pl

from pokebroker.utils import logger

class IO:

    @staticmethod
    def read_dataset(
        # TODO: pathlike type
        path: str,
        **kwargs
    ):
        
        logger.debug(f'Reading dataset: {path}')
        return pl.from_arrow(
            pq.read_table(path),
            **kwargs
        )
    
    def read_csv(
         # TODO: pathlike type
        path: str,
        **kwargs
    ):
        
        logger.debug(f'Reading csv: {path}')
        return pl.read_csv(
            path,
            **kwargs
        )