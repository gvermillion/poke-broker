from abc import ABC, abstractmethod
import polars as pl
from pokebroker import logger, utils

class DataHandlerBase(ABC):
    @property
    @abstractmethod
    def NAME(self) -> str:
        pass

    def __init__(
        self,
        path: str
    ):
        self.path: str = path

        logger.info(
            'Intialized data handler %s using path: %s',
            self.NAME.upper(),
            self.path
        )

    @abstractmethod
    def _get_data(self, **kwargs) -> pl.DataFrame:
        pass

    def _align_schema(
        self,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        logger.debug('No schema alignment required.')
        return df
    
    def load(
        self,
        **kwargs
    ):
        return (
            self._get_data(
                **kwargs
            )
            .pipe(self._align_schema)
        )