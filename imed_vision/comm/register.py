# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:register
    author: 12718
    time: 2022/5/14 14:45
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division

class Register:
    def __init__(self, name):
        self._name = name
        self._obj_dict = dict()


    def __len__(self):
        return len(self._obj_dict)

    def __repr__(self):
        text = self.__class__.__name__ + f"{self._name}_item_{self._obj_dict}"
        return text
    def get_names(self):
        return self._obj_dict.keys()

    @property
    def obj_dict(self):
        return self._obj_dict

    @property
    def name(self):
        return self._name

    def _register(self, fuc_module_name, obj):
        assert (fuc_module_name not in self._obj_dict.keys()), \
            f"object {fuc_module_name} is already registered in {self._name}"
        self._obj_dict[fuc_module_name.lower()] = obj

    def register(self, obj=None):
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._register(name, func_or_class)
                return func_or_class
            return deco
        #used as a full function call
        self._register(obj.__name__, obj)

    def get(self, name):
        return self._obj_dict[name]

    __str__ = __repr__