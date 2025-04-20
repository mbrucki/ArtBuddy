#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/12/13
# @Author  : yanxiaodong
# @File    : registry.py.py
"""
from typing import Dict, Type, Optional
from collections import defaultdict

import bcelogger

from pygraphv1.client.graph_client import GraphClient
from pygraphv1.client.graph_api_operator import CreateOperatorRequest


class Registry(object):
    """
    A registry to map strings to classes.
    """
    def __init__(self, scope):
        self._scope = scope
        self._module_dict: Dict[str, Dict[str, Type]] = defaultdict(dict)

    def get(self, name: str, version: Optional[str] = "latest"):
        """
        Get the registry record.
        """
        assert name in self._module_dict, f"Module named '{name}' was not registered in the '{self._scope}' scope"

        modules = self._module_dict[name]
        if version == "latest":
            sort_modules = sorted(modules.items(), key=lambda x: int(x[0]))
            return sort_modules[-1][1]
        else:
            return modules[version]

    def _register_module(self, module_name: str, version: str, module: Type):
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        self._module_dict[module_name].update({version: module})

    def register_module(self, name: str, version: str, module: Optional[Type] = None):
        """
        Register a module.
        """
        if module is not None:
            self._register_module(module_name=name, version=version, module=module)
            return module

        def _register(module):
            self._register_module(module_name=name, version=version, module=module)
            return module

        return _register

    def __call__(self, endpoint: str, ak: str, sk: str):
        client = GraphClient(endpoint=endpoint, ak=ak, sk=sk)
        for name, modules in self._module_dict.items():
            for version, module in modules.items():
                bcelogger.info(f"Create operator {name} with version {version}")
                request = CreateOperatorRequest(**module().dict())
                client.create_operator(request=request)