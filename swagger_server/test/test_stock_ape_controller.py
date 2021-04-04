# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.history_data import HistoryData  # noqa: E501
from swagger_server.models.predict_data import PredictData  # noqa: E501
from swagger_server.models.quote_data import QuoteData  # noqa: E501
from swagger_server.test import BaseTestCase


class TestStockApeController(BaseTestCase):
    """StockApeController integration test stubs"""

    def test_predict_stock(self):
        """Test case for predict_stock

        Returns
        """
        query_string = [('ticker', 'ticker_example')]
        response = self.client.open(
            '/edems_swag/stock_api/1.0.0/prediction',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_stock_history(self):
        """Test case for stock_history

        Returns
        """
        query_string = [('ticker', 'ticker_example')]
        response = self.client.open(
            '/edems_swag/stock_api/1.0.0/history',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_stock_quote(self):
        """Test case for stock_quote

        Returns
        """
        query_string = [('ticker', 'ticker_example')]
        response = self.client.open(
            '/edems_swag/stock_api/1.0.0/qouote',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
