import os
from dotenv import load_dotenv
from typing import (
    Dict,
    Sequence,
)
import polars as pl
from polars.datatypes.classes import DataTypeClass as PolarsDtype
from pokemontcgsdk import RestClient

load_dotenv()
assert os.getenv('POKEMON_TCG_API_KEY') is not None, (
    'API key not found. Please ensure you have set the POKEMON_TCG_API_KEY environment variable.'
)
RestClient.configure(
    api_key=os.getenv('POKEMON_TCG_API_KEY')
)

SCHEMA: Dict[str, Dict[str, PolarsDtype]] = {
    # TODO: Use Enum for data asset
    'trader_inventory': {
        'Category': pl.String,
        'Locale': pl.String,
        'Series': pl.String,
        'Set': pl.String,
        'Id': pl.String,
        'Name': pl.String,
        'Total Quantity': pl.Int64,
        'Normal Quantity': pl.Int64,
        'Holo Quantity': pl.Int64,
        'Reverse Holo Quantity': pl.Int64,
        'Shadowless Quantity': pl.Int64,
        'First Edition Quantity': pl.Int64,
    },
}

COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {
    # TODO: Use Enum for data asset
    'trader_inventory': {
        'Id': 'id',
        'Name': 'name',
        'Set': 'set',
        # 'Number': 'number',
        # 'Rarity': 'rarity',
        'Normal Quantity': 'normal_quantity',
        'Reverse Holo Quantity': 'reverse_holo_quantity',
        'Holo Quantity': 'holo_quantity',
    }
}

SET_ALIGNMENT: Dict[str, str] = {
    'sv35': 'sv3pt5',
    'sv45': 'sv4pt5',
    'swsh125': 'swsh12pt5',
    'swsh125gg': 'swsh12pt5gg',
}