import types
import numpy as np
from ..package_T1 import gauss_newton_fast
from ..package_T1 import dog_leg
from ..package_T1 import adam
from ..package_T1 import lbfgs


def get_func_method(argument):
    switch_dict = {
        'gauss-newton': gauss_newton_fast,
        'dog-leg': dog_leg,
        'adam': adam,
        'l-bfgs': lbfgs,
    }

    result = switch_dict.get(argument)
    if result is None:
        raise ValueError("Unknown method")

    return result


def get_func_research(f_label):
    compiled_function = compile(f_label, "<string>", "exec")
    exec(compiled_function)

    # Создаем объект функции из скомпилированного кода
    return types.FunctionType(compiled_function.co_consts[0], globals())
