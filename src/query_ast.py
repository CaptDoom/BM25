class ASTNode:
    pass

class FilterExpression(ASTNode):
    def __init__(self, field: str, operator: str, value):
        self.field = field
        self.operator = operator
        self.value = value

    def __repr__(self):
        return f"FilterExpression({self.field} {self.operator} {repr(self.value)})"

class LogicalExpression(ASTNode):
    def __init__(self, operator: str, left: ASTNode, right: ASTNode):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"LogicalExpression({self.operator}, {self.left}, {self.right})"

class NotExpression(ASTNode):
    def __init__(self, operand: ASTNode):
        self.operand = operand

    def __repr__(self):
        return f"NotExpression({self.operand})"
