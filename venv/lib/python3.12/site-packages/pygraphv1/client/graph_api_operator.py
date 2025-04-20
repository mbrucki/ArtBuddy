#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/7/25
# @Author  : yanxiaodong
# @File    : graph_api_operator.py
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

from .graph_api_variable import Variable


class CategoryVisual(BaseModel):
    """
    CategoryVisual
    """
    name: str = None
    display_name: str = Field(None, alias="displayName")


class Operator(BaseModel):
    """
    Operator
    """
    name: Optional[str] = None
    local_name: str = Field(None, alias="localName")
    display_name: str = Field(None, alias="displayName")
    kind: str = None
    kind_display_name: str = Field(None, alias="kindDisplayName")
    parent_kind: str = Field(None, alias="parentKind")
    description: str = None
    category: List[str] = None
    category_visual: Optional[CategoryVisual] = Field(None, alias="categoryVisual")
    inputs: Optional[List[Variable]] = None
    outputs: Optional[List[Variable]] = None
    properties: Optional[List[Variable]] = None
    states: Optional[List[Variable]] = None
    visuals: Optional[str] = None
    runtime: str = None
    version: str = None

    model_config = ConfigDict(populate_by_name=True)

    def get_input(self, name: str) -> Optional[Variable]:
        """
        根据输入的名称获取对应的 input。

        :param name: Variable 的名称。
        :return: 匹配的 Variable，如果未找到则返回 None。
        """
        if not self.inputs:
            return None
        return next((var for var in self.inputs if var.name == name), None)

    def get_output(self, name: str) -> Optional[Variable]:
        """
        根据输入的名称获取对应的 output。

        :param name: Variable 的名称。
        :return: 匹配的 Variable，如果未找到则返回 None。
        """
        if not self.outputs:
            return None
        return next((var for var in self.outputs if var.name == name), None)

    def get_property(self, name: str) -> Optional[Variable]:
        """
        根据输入的名称获取对应的 property。

        :param name: Variable 的名称。
        :return: 匹配的 Variable，如果未找到则返回 None。
        """
        if not self.properties:
            return None
        return next((var for var in self.properties if var.name == name), None)

    def get_state(self, name: str) -> Optional[Variable]:
        """
        根据输入的名称获取对应的 state。

        :param name: Variable 的名称。
        :return: 匹配的 Variable，如果未找到则返回 None。
        """
        if not self.states:
            return None
        return next((var for var in self.states if var.name == name), None)


class CreateOperatorRequest(Operator):
    """
    CreateOperatorRequest
    """
    file_name: str = Field(None, alias="fileName", description="算子文件名", example="MongoDatasource.yaml")
    need_upload: bool = Field(None, description="是否需要上传算子文件", alias="needUpload", example=True)
