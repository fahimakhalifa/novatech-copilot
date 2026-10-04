"""Safe arithmetic for the copilot ("2 NovaPhones with 10% off?")."""
import ast
import math
import operator

from app.tools.context import InvalidInput

MAX_EXPR_LEN = 200
MAX_EXPONENT = 100

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"round": round, "min": min, "max": max, "abs": abs}


def _eval(node: ast.AST) -> float:
    match node:
        # `type(...) in (int, float)` rather than isinstance: bool is a subclass of int.
        case ast.Constant(value=v) if type(v) in (int, float):
            return float(v)   # floats overflow to an error instead of growing without bound
        case ast.BinOp(left=l, op=op, right=r) if type(op) in _BINARY:
            left, right = _eval(l), _eval(r)
            if isinstance(op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise InvalidInput("Exponent too large.")
            return _BINARY[type(op)](left, right)
        case ast.UnaryOp(op=op, operand=x) if type(op) in _UNARY:
            return _UNARY[type(op)](_eval(x))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            values = [_eval(a) for a in args]
            if name == "round" and len(values) == 2:
                values[1] = int(values[1])   # round(x, 2.0) is a TypeError
            return float(_FUNCTIONS[name](*values))
        case _:
            raise InvalidInput(f"Unsupported element in expression: {ast.dump(node)[:40]}")


def calculate(expression: str) -> float:
    expression = (expression or "").strip()
    if not expression:
        raise InvalidInput("Empty expression.")
    if len(expression) > MAX_EXPR_LEN:
        raise InvalidInput(f"Expressions are limited to {MAX_EXPR_LEN} characters.")
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except SyntaxError:
        raise InvalidInput(f"'{expression}' is not a valid arithmetic expression.")
    except ZeroDivisionError:
        raise InvalidInput("Division by zero.")
    except (OverflowError, TypeError, ValueError):
        raise InvalidInput("The expression could not be computed.")
    # (-8) ** 0.5 is a complex number in Python, not an error.
    if isinstance(result, complex) or not math.isfinite(result):
        raise InvalidInput("The result is not a finite real number.")
    # Hide float noise (0.1 + 0.2 = 0.30000000000000004) without rounding real cents away.
    return round(result, 10)
