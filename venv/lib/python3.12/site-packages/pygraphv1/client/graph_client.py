#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/12/11
# @Author  : chujianfei
# @File    : graph_client.py
"""
from baidubce.http import http_methods, http_content_types
from bceinternalsdk.client.bce_internal_client import BceInternalClient

from pygraphv1.client.graph_api_operator import CreateOperatorRequest


class GraphClient(BceInternalClient):
    """
    JobClient is the client for JobService API.
    """

    @staticmethod
    def _build_base_operator_uri() -> str:
        """
        build the base uri for operator.
        """
        return "/v1/operators"

    def create_operator(self, request: CreateOperatorRequest):
        """
        create_operator
        """
        return self._send_request(
            http_method=http_methods.POST,
            headers={b"Content-Type": http_content_types.JSON},
            path=bytes(
                self._build_base_operator_uri(),
                encoding="utf-8",
            ),
            body=request.json(by_alias=True).encode("utf-8"),
        )