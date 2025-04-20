#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/7/25
# @Author  : yanxiaodong
# @File    : graph_api_variable.py
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class RangeConstraint(BaseModel):
    """
    RangeConstraint
    """
    min: int
    max: int


class FloatRangeConstraint(BaseModel):
    """
    FloatRangeConstraint
    """
    min: float
    max: float


class PatternConstraint(BaseModel):
    """
    PatternConstraint
    """
    regex: str
    description: str


class ChoiceItem(BaseModel):
    """
    ChoiceItem
    """
    value: str
    text: str


class ChoicesConstraint(BaseModel):
    """
    ChoicesConstraint
    """
    choice_type: str = Field(None, alias="choiceType")
    choices: List[ChoiceItem]


class Constraint(BaseModel):
    """
    Constraint
    """
    range: Optional[RangeConstraint] = None
    float_range: Optional[FloatRangeConstraint] = Field(None, alias="floatRange")
    pattern: Optional[PatternConstraint] = None
    choice: Optional[ChoicesConstraint] = None


class Variable(BaseModel):
    """
    Variable
    """
    name: Optional[str] = None
    display_name: Optional[str] = Field(None, alias="displayName")
    type: Optional[str] = None
    schema_: Optional[List['Variable']] = Field(None, alias="schema")
    default_value: Optional[str] = Field(None, alias="defaultValue")
    value: Optional[str] = None
    description: Optional[str] = None
    optional: Optional[bool] = None
    readonly: Optional[bool] = None
    constraint: Optional[Constraint] = None
    visuals: Optional[str] = None