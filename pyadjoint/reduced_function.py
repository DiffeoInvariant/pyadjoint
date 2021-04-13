from .enlisting import Enlist
from .tape import stop_annotating, get_working_tape
from .drivers import (
    compute_gradient,
    compute_jacobian_action,
    compute_jacobian_matrix,
    compute_hessian_action,
)
from typing import Union, Optional
from .reduced_functional import marked_controls

class ReducedFunction(object):
    """Class representing a function that may or may not not be scalar-valued
    (as opposed to :class:`ReducedFunctional`, which always represents a 
    scalar-valued functional). A reduced function maps some number of control 
    values to the given function, and can compute derivatives of the function
    with respect to any of the controls.

    Args:
        outputs (Union[OverloadedType, list[OverloadedType]): An instance of an
            :class:`OverloadedType`, or a list of multiple such instances 
            representing the output of the function.
        controls (Union[Control, list[Control]]): A :class:`Control` or a list
            of controls representing the arguments to the function you may
            want to differentiate with respect to, or whose values you may
            want to change.

    """
    def __init__(
        self,
        outputs,
        controls,
        tape=None,
        eval_cb_pre=None,
        eval_cb_post=None,
        jac_action_cb_pre=None,
        jac_action_cb_post=None,
        adj_jac_action_cb_pre=None,
        adj_jac_action_cb_post=None,
        hess_action_cb_pre=None,
        hess_action_cb_post=None,
    ):
        outputs = Enlist(outputs)
        outputs = outputs.delist([output.block_variable for output in outputs])
        self.outputs = Enlist(outputs)
        self.outputs = Enlist(outputs)
        self.controls = Enlist(controls)
        self.tape = get_working_tape() if tape is None else tape

        nothing = lambda *args: None
        self.eval_cb_pre = nothing if eval_cb_pre is None else eval_cb_pre
        self.eval_cb_post = nothing if eval_cb_post is None else eval_cb_post
        self.jac_action_cb_pre = (
            nothing if jac_action_cb_pre is None else jac_action_cb_pre
        )
        self.jac_action_cb_post = (
            nothing if jac_action_cb_post is None else jac_action_cb_post
        )
        self.adj_jac_action_cb_pre = (
            nothing if adj_jac_action_cb_pre is None else adj_jac_action_cb_pre
        )
        self.adj_jac_action_cb_post = (
            nothing if adj_jac_action_cb_post is None else adj_jac_action_cb_post
        )
        self.hess_action_cb_pre = (
            nothing if hess_action_cb_pre is None else hess_action_cb_pre
        )
        self.hess_action_cb_post = (
            nothing if hess_action_cb_post is None else hess_action_cb_post
        )

    def jac_action(self, inputs, options=None):
        """Computes the action of the Jacobian of this :class:`ReducedFunction`
        on the input tangents.

        Args:
            inputs (Union[OverloadedType, list[OverloadedType]]): The input
                tangents. Must be the same shape as the controls for this
                :class:`ReducedFunction` (with `None` used to indicate no 
                derivative should be taken with respect to that particular 
                input)
            options (dict): An optional dictionary of options to pass to
                :func:`compute_jacobian_action`

        Returns:
            Union[OverloadedType, list[OverloadedType]]: The action of the 
                Jacobian applied to the input tangents.

        """
        inputs = Enlist(inputs)
        if len(inputs) != len(self.controls):
            raise TypeError(
                "The length of inputs must match the length of function controls."
            )

        values = [c.data() for c in self.controls]
        self.jac_action_cb_pre(
            self.controls.delist(values), self.controls.delist(inputs)
        )

        derivatives = compute_jacobian_action(
            self.outputs, self.controls, inputs, options=options, tape=self.tape
        )
        # Call callback
        self.jac_action_cb_post(
            self.outputs.delist([bv.saved_output for bv in self.outputs]),
            self.outputs.delist(derivatives),
            self.controls.delist(values),
        )

        return self.outputs.delist(derivatives)

    def adj_jac_action(self, inputs, options=None):
        """Computes the action of the adjoint of this :class:`ReducedFunction`
        on the given adjoint inputs (cotangents).

        Args:
            inputs (Union[OverloadedType, list[OverloadedType]]): The input
                cotangents. Must be the same shape as the outputs of this 
                :class:`ReducedFunction` (with `None` used to indicate no 
                derivative should be taken with respect to that particular 
                output).
            options (dict): An optional dictionary of options to pass to
                :func:`compute_gradient`
        Returns:
            Union[OverloadedType, list[OverloadedType]]: The action of the 
                adjoint applied to the adjoint inputs (i.e. the input cotangents).

        """
        inputs = Enlist(inputs)
        if len(inputs) != len(self.outputs):
            raise TypeError(
                "The length of inputs must match the length of function outputs."
            )

        values = [c.data() for c in self.controls]
        self.adj_jac_action_cb_pre(self.controls.delist(values))

        derivatives = compute_gradient(
            self.outputs,
            self.controls,
            options=options,
            tape=self.tape,
            adj_value=inputs,
        )

        # Call callback
        self.adj_jac_action_cb_post(
            self.outputs.delist([bv.saved_output for bv in self.outputs]),
            self.controls.delist(derivatives),
            self.controls.delist(values),
        )

        return self.controls.delist(derivatives)

    def jac_matrix(self, m_jac=None):
        """Computes the Jacobian matrix of this :class:`ReducedFunction`
        and multiplies the input Jacobian against it to conform to the chain
        rule.

        Args:
            m_jac (Optional[Union[OverloadedType, list[OverloadedType]]]): The
            input Jacobian to the function. Leave this `None` to use an 
            efficient proxy for the identity matrix.

        Returns: 
            Union[OverloadedType, list[OverloadedType]]: The accumulated Jacobian
                matrix.
        """
        if m_jac is not None:
            m_jac = Enlist(m_jac)
            if len(m_jac) != len(self.controls):
                raise TypeError(
                    "The length of m_jac must match the length of function controls."
                )

            for i, jac in enumerate(m_jac):
                m_jac[i] = Enlist(jac)
                if len(m_jac[i]) != len(self.controls):
                    raise TypeError(
                        "The length of each identity must match the length of function controls."
                    )

        outputs = [bv.output for bv in self.outputs]

        jacobian = compute_jacobian_matrix(
            outputs, self.controls, m_jac, tape=self.tape
        )
        for i, jac in enumerate(jacobian):
            if jac is not None:
                jacobian[i] = self.controls.delist(jac)
        jacobian = self.outputs.delist(jacobian)
        return jacobian

    def hess_action(self, m_dot, adj_input, options=None):
        """Computes the action of the Hessian of this :class:`ReducedFunction`
        on the input tangents (on the right) and cotantents (on the left).

        Args:
            m_dot (Union[OverloadedType, list[OverloadedType]]): The input
                tangents. Must be the same shape as the controls for this
                :class:`ReducedFunction` (with `None` used to indicate no 
                derivative should be taken with respect to that particular 
                input)
            adj_input (Union[OverloadedType, list[OverloadedType]]): The input
                cotangents. Must be the same shape as the outputs of this 
                :class:`ReducedFunction` (with `None` used to indicate no 
                derivative should be taken with respect to that particular 
                output).
            options (dict): An optional dictionary of options to pass to
                :func:`compute_jacobian_action`

        Returns:
            Union[OverloadedType, list[OverloadedType]]: The action of the 
                Hessian applied to the input tangents on the right and the input
                cotangents on the left.

        """
        m_dot = Enlist(m_dot)
        if len(m_dot) != len(self.controls):
            raise TypeError(
                "The length of m_dot must match the length of function controls."
            )

        adj_input = Enlist(adj_input)
        if len(adj_input) != len(self.outputs):
            raise TypeError(
                "The length of adj_input must match the length of function outputs."
            )

        values = [c.data() for c in self.controls]
        self.hess_action_cb_pre(self.controls.delist(values))

        derivatives = compute_gradient(
            self.outputs,
            self.controls,
            options=options,
            tape=self.tape,
            adj_value=adj_input,
        )

        # TODO: there should be a better way of generating hessian_input.
        zero = [0 * v for v in adj_input]
        hessian = compute_hessian_action(
            self.outputs,
            self.controls,
            m_dot,
            options=options,
            tape=self.tape,
            hessian_value=zero,
        )

        # Call callback
        self.hess_action_cb_post(
            self.outputs.delist([bv.saved_output for bv in self.outputs]),
            self.controls.delist(hessian),
            self.controls.delist(values),
        )

        return self.controls.delist(hessian)

    def __call__(self, inputs):
        """Re-evaluates the :class:`ReducedFunction` with the supplied controls.

        Args:
            inputs (Union[OverloadedType, list[OverloadedType]]): The input
                controls. Must be the same shape as the controls for this
                :class:`ReducedFunction`

        Returns:
            Union[OverloadedType, list[OverloadedType]]: The value of this
                :class:`ReducedFunction` evaluated at the new inputs.

        """
        inputs = Enlist(inputs)
        if len(inputs) != len(self.controls):
            raise TypeError("The length of inputs must match the length of controls.")

        # Call callback.
        self.eval_cb_pre(self.controls.delist(inputs))

        for i, value in enumerate(inputs):
            self.controls[i].update(value)

        # self.tape.reset_blocks()
        with self.marked_controls():
            with stop_annotating():
                self.tape.recompute()

        outputs = [output.checkpoint for output in self.outputs]
        outputs = self.outputs.delist(outputs)

        # Call callback
        self.eval_cb_post(outputs, self.controls.delist(inputs))

        return outputs

    def marked_controls(self):
        return marked_controls(self)

