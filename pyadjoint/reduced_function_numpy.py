from .tape import no_annotations
from .enlisting import Enlist
from .reduced_function import ReducedFunction
from .control import Control
import numpy
from .overloaded_type import create_overloaded_object as coo


class ReducedFunctionNumPy(ReducedFunction):
    """This class implements the reduced function for given function and
    controls based on numpy data structures.

    This "NumPy version" of the ReducedFunction is created from
    an existing ReducedFunction object:
    rf_np = ReducedFunctionNumPy(rf = rf)
    """

    def __init__(self, reduced_function):
        if not isinstance(reduced_function, ReducedFunction):
            raise TypeError("reduced_function should be a ReducedFunction")

        self.rf = reduced_function

    def __getattr__(self, item):
        return getattr(self.rf, item)

    def __call__(self, m_array):
        """An implementation of the reduced function evaluation
        that accepts the control values as an array of scalars

        """
        output = self.rf(self.get_rf_input(m_array))
        return self.get_outputs_array(output)

    def get_rf_input(self, m_array):
        m_copies = [control.copy_data() for control in self.controls]
        return self.set_local(m_copies, m_array)

    def get_rf_adj_input(self, adj_array):
        adj_controls = [Control(output.output) for output in self.outputs]
        adj_copies = [control.copy_data() for control in adj_controls]
        return self.set_adj_local(adj_controls, adj_copies, adj_array)

    def set_adj_local(self, adj_controls, adj, adj_array):
        """Use the given numpy array to set the values of the given list of control
        values."""
        offset = 0
        for i, control in enumerate(adj_controls):
            adj[i], offset = control.assign_numpy(adj[i], adj_array, offset)
        return adj

    def set_local(self, m, m_array):
        """Use the given numpy array to set the values of the given list of control
        values."""
        offset = 0
        for i, control in enumerate(self.controls):
            m[i], offset = control.assign_numpy(m[i], m_array, offset)
        return m

    def get_global(self, m):
        """Converts the given list of control values m to a single numpy array."""
        m_global = []
        for i, v in enumerate(Enlist(m)):
            if isinstance(v, Control):
                m_global += v.fetch_numpy(v.control)
            elif hasattr(v, "_ad_to_list"):
                m_global += v._ad_to_list(v)
            else:
                m_global += self.controls[i].control._ad_to_list(v)
        return coo(numpy.array(m_global, dtype="d"))

    @no_annotations
    def jac_action(self, m_array):
        """An implementation of the reduced function jac_action evaluation
        that accepts the controls as an array of scalars.
        """

        dJdm = self.rf.jac_action(self.get_rf_input(m_array))
        return self.get_outputs_array(dJdm)
    
    @no_annotations
    def adj_jac_action(self, adj_inputs):
        """An implementation of the reduced functional adjoint evaluation
        that returns the derivatives as an array of scalars.
        """

        dJdm = self.rf.adj_jac_action(self.get_rf_adj_input(adj_inputs))
        dJdm = Enlist(dJdm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(dJdm[i])

        return coo(numpy.array(m_global, dtype="d"))

    
    @no_annotations
    def hessian(self, m_array, m_dot_array):
        """An implementation of the reduced function hessian action evaluation
        that accepts the controls as an array of scalars. If m_array is None,
        the Hessian action at the latest forward run is returned."""
        # Calling derivative is needed, see i.e. examples/stokes-shape-opt
        self.jac_action(m_array)
        Hm = self.rf.hessian(m_array, self.get_rf_input(m_dot_array))
        Hm = Enlist(Hm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(Hm[i])

        self.rf.tape.reset_variables()

        return coo(numpy.array(m_global, dtype="d"))

    @staticmethod
    def get_identities(controls, pointwise=True):
        # Each identity is a tensor with twice the rank of the corresponding control.
        identities = []
        controls = Enlist(controls)
        if not isinstance(pointwise, (list, tuple)):
            pointwise = [pointwise] * len(controls)
        for i, control in enumerate(controls):
            if hasattr(control, "value_shape"):
                shape = control.value_shape()
                if len(shape) == 0:
                    shape = (1,)
            elif hasattr(control, "shape"):
                shape = control.shape
            else:
                shape = (control._ad_dim(),)
            if pointwise[i] and len(shape) > 0:
                n = shape[0]
                shape = tuple(shape[1:])

                identity = numpy.zeros((n,) + shape * 2)
                indices = numpy.indices(shape)
                diagonal = [idx for idx in indices] * 2
                identity[(...,) + tuple(diagonal)] = 1
            else:
                identity = numpy.zeros(shape * 2)
                indices = numpy.indices(shape)
                diagonal = [idx for idx in indices] * 2
                identity[tuple(diagonal)] = 1

            c_identities = [None] * len(controls)
            c_identities[i] = identity
            identities.append(c_identities)
        return controls.delist(identities)

    def obj_to_array(self, obj):
        return self.get_global(obj)

    def get_controls(self, controls=None):
        if controls is None:
            controls = self.controls
        m = [p.data() for p in controls]
        return self.obj_to_array(m)

    def set_controls(self, array):
        m = [p.data() for p in self.controls]
        return self.set_local(m, array)

    def get_outputs_array(self, values):
        m_global = []
        values = Enlist(values)
        for i, output in enumerate(self.outputs):
            if values[i] is not None:
                m_global += output.output._ad_to_list(values[i])
            else:
                m_global += [0] * output.output._ad_dim()
        return coo(numpy.array(m_global, dtype="d"))
