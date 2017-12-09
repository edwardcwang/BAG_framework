# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################


"""This module defines parsing utility methods.
"""

import ast


class ExprVarScanner(ast.NodeVisitor):
    """
    This node visitor collects all variable names found in the
    AST, and excludes names of functions.  Variables having
    dotted names are not supported.
    """
    def __init__(self):
        self.varnames = set()

    # noinspection PyPep8Naming
    def visit_Name(self, node):
        self.varnames.add(node.id)

    # noinspection PyPep8Naming
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

    # noinspection PyPep8Naming
    def visit_Attribute(self, node):
        # ignore attributes
        pass


def get_variables(expr):
    """Parses the given Python expression and return a list of all variables.

    Parameters
    ----------
    expr : str
        An expression string that we want to parse for variable names.

    Returns
    -------
    var_list : list[str]
        Names of variables from the given expression.
    """
    root = ast.parse(expr, mode='exec')
    scanner = ExprVarScanner()
    scanner.visit(root)
    return list(scanner.varnames)
