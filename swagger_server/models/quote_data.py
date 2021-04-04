# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.quotes import Quotes  # noqa: F401,E501
from swagger_server import util


class QuoteData(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, symbol: str=None, predictions: List[Quotes]=None):  # noqa: E501
        """QuoteData - a model defined in Swagger

        :param symbol: The symbol of this QuoteData.  # noqa: E501
        :type symbol: str
        :param predictions: The predictions of this QuoteData.  # noqa: E501
        :type predictions: List[Quotes]
        """
        self.swagger_types = {
            'symbol': str,
            'predictions': List[Quotes]
        }

        self.attribute_map = {
            'symbol': 'symbol',
            'predictions': 'predictions'
        }
        self._symbol = symbol
        self._predictions = predictions

    @classmethod
    def from_dict(cls, dikt) -> 'QuoteData':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The QuoteData of this QuoteData.  # noqa: E501
        :rtype: QuoteData
        """
        return util.deserialize_model(dikt, cls)

    @property
    def symbol(self) -> str:
        """Gets the symbol of this QuoteData.


        :return: The symbol of this QuoteData.
        :rtype: str
        """
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: str):
        """Sets the symbol of this QuoteData.


        :param symbol: The symbol of this QuoteData.
        :type symbol: str
        """
        if symbol is None:
            raise ValueError("Invalid value for `symbol`, must not be `None`")  # noqa: E501

        self._symbol = symbol

    @property
    def predictions(self) -> List[Quotes]:
        """Gets the predictions of this QuoteData.


        :return: The predictions of this QuoteData.
        :rtype: List[Quotes]
        """
        return self._predictions

    @predictions.setter
    def predictions(self, predictions: List[Quotes]):
        """Sets the predictions of this QuoteData.


        :param predictions: The predictions of this QuoteData.
        :type predictions: List[Quotes]
        """
        if predictions is None:
            raise ValueError("Invalid value for `predictions`, must not be `None`")  # noqa: E501

        self._predictions = predictions
