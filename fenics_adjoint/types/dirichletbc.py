import backend
import ufl

from .constant import Constant
from .function import Function
from .expression import Expression

from pyadjoint.tape import get_working_tape
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.block import Block

# TODO: Might need/want some way of creating a new DirichletBCBlock if DirichletBC is assigned
#       new boundary values/function.

class DirichletBC(OverloadedType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)

        # Pop kwarg to pass the kwargs check in backend.DirichletBC.__init__.
        self.annotate_tape = kwargs.pop("annotate_tape", True)
        
        backend.DirichletBC.__init__(self, *args, **kwargs)

        if self.annotate_tape:
            tape = get_working_tape()

            # Since DirichletBC behaves differently based on number of
            # args and arg types, we pass all args to block
            block = DirichletBCBlock(self, *args)
            
            tape.add_block(block)
            block.add_output(self.block_output)


class DirichletBCBlock(Block):
    def __init__(self, bc, *args):
        Block.__init__(self)
        self.bc = bc
        self.function_space = args[0]

        if len(args) >= 2 and isinstance(args[1], OverloadedType):
            self.add_dependency(args[1].block_output)
        else:
            # TODO: Implement the other cases.
            #       Probably just a BC without dependencies?
            #       In which case we might not even need this Block?
            pass

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        for block_output in self.get_dependencies():
            c = block_output.output
            if isinstance(c, Constant):
                # Constants have float adj values.
                block_output.add_adj_output(adj_input.sum())
            elif isinstance(c, Function):
                # TODO: This gets a little complicated.
                #       The function may belong to a different space,
                #       and with `Function.set_allow_extrapolation(True)`
                #       you can even use the Function outside its domain.
                # For now we will just assume the FunctionSpace is the same for
                # the BC and the Function.
                block_output.add_adj_output(adj_input)

    def recompute(self):
        # There is nothing to be recomputed.
        pass


