import re
from typing import Optional, Union, Tuple
from src.query_ast import ASTNode, FilterExpression, LogicalExpression, NotExpression

class MetadataParser:
    def __init__(self, schema_def: Optional[dict] = None):
        """
        schema_def specifies the expected types, e.g. {"year": int, "category": str, "rating": float}
        """
        self.schema_def = schema_def or {}

    def validate_type(self, field: str, value: str) -> Union[str, int, float]:
        expected_type = self.schema_def.get(field)
        if expected_type is int:
            return int(value)
        elif expected_type is float:
            return float(value)
        elif expected_type is bool:
            return value.lower() in ("true", "1", "yes")
        
        # Try automatic coercion if no schema exists
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("'\"")

    def parse(self, filter_str: str) -> Optional[ASTNode]:
        """
        Parses filter strings supporting operators: AND, OR, NOT, ==, !=, >=, <=, >, <
        Example: (dataset == 'quora' AND has_title == 1)
        """
        if not filter_str or not filter_str.strip():
            return None
            
        filter_str = filter_str.strip()

        # Remove a single set of outer parentheses before applying precedence rules.
        filter_str = self._strip_outer_parens(filter_str)
            
        # Parse logical operators by precedence: OR -> AND -> NOT -> atomic expression.
        or_split = self._split_by_operator_outside_parens(filter_str, "OR")
        if or_split:
            left_ast = self.parse(or_split[0])
            right_ast = self.parse(or_split[1])
            return LogicalExpression("OR", left_ast, right_ast)

        and_split = self._split_by_operator_outside_parens(filter_str, "AND")
        if and_split:
            left_ast = self.parse(and_split[0])
            right_ast = self.parse(and_split[1])
            return LogicalExpression("AND", left_ast, right_ast)

        not_match = re.match(r"^NOT\s+(.+)$", filter_str, flags=re.IGNORECASE)
        if not_match:
            operand = self.parse(not_match.group(1))
            return NotExpression(operand) if operand else None
            
        # Parse basic expressions: field op val
        match = re.match(r"^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$", filter_str)
        if not match:
            raise ValueError(f"Invalid filter syntax: {filter_str}")
            
        field, op, val = match.groups()
        val = val.strip().strip("'\"")
        typed_val = self.validate_type(field, val)
        return FilterExpression(field, op, typed_val)

    def _strip_outer_parens(self, s: str) -> str:
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            if inner and self._check_parens_balance(inner):
                return inner
        return s

    def _check_parens_balance(self, s: str) -> bool:
        balance = 0
        for char in s:
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
            if balance < 0:
                return False
        return balance == 0

    def _split_by_operator_outside_parens(self, s: str, operator: str) -> Optional[Tuple[str, str]]:
        balance = 0
        op_len = len(operator)
        for i in range(len(s)):
            char = s[i]
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
            elif balance == 0:
                if s[i:i+op_len].upper() == operator:
                    # check boundary spaces around OR / AND to avoid substring match
                    left_boundary = (i == 0 or s[i-1].isspace() or s[i-1] == ")")
                    right_boundary = (i+op_len == len(s) or s[i+op_len].isspace() or s[i+op_len] == "(")
                    if left_boundary and right_boundary:
                        return s[:i].strip(), s[i+op_len:].strip()
        return None
