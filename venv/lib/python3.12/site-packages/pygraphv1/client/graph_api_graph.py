#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/7/19
# @Author  : yanxiaodong
# @File    : types.py
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from .graph_api_variable import Variable
from .graph_api_operator import Operator


class EdgeTarget(BaseModel):
    """
    EdgeTarget
    """
    operator: str = None
    property: str = None
    input: str = None
    output: str = None
    state: str = None


class Edge(BaseModel):
    """
    Edge
    """
    from_: EdgeTarget = Field(None, alias="from")
    to: EdgeTarget = None


class GraphContent(BaseModel):
    """
    Graph Content
    """
    name: Optional[str] = None
    local_name: Optional[str] = Field(None, alias="localName")
    environment: Optional[str] = None
    properties: Optional[List[Variable]] = None
    inputs: Optional[List[Variable]] = None
    outputs: Optional[List[Variable]] = None
    nodes: Optional[List[Operator]] = None
    edges: Optional[List[Edge]] = None
    visuals: Optional[str] = None


    def get_nodes(self, kind: str) -> Optional[List[Operator]]:
        """
        根据输入的kind名称获取所有的节点

        :param kind: kind 的名称。
        :return: 匹配的 List[Operator]，如果未找到则返回 None。
        """
        if self.nodes is None:
            return None

        # 初始化一个空列表，用于存储所有匹配的节点
        matched_nodes = []

        # 遍历所有节点，检查它们的 'kind' 属性
        for node in self.nodes:
            if node.kind == kind:
                matched_nodes.append(node)

        # 如果没有找到任何匹配的节点，则返回 None
        if not matched_nodes:
            return None

        return matched_nodes

