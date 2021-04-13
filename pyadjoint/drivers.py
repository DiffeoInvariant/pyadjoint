from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating


def compute_gradient(J, m, options=None, tape=None, adj_value=1.0):
    """
    Compute the gradient of J with respect to the initialisation value of m,
    that is the value of m at its creation.

    Args:
        J (OverloadedType, list[OverloadedType]):  The objective functional.
        m (Union[list[Control], Control]): The (list of) controls.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape (Tape): The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The derivative with respect to the control. Should be an instance of the same type as
            the control.
    """
    options = options or {}
    tape = tape or get_working_tape()
    tape.reset_variables()
    J.adj_value = adj_value
    m = Enlist(m)

    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_adj(markings=True)

    grads = [i.get_derivative(options=options) for i in m]
    return m.delist(grads)


def compute_jacobian_action(J, m, m_dot, options=None, tape=None):
    """
    Compute the action of the Jacobian of J on m_dot with respect to the
    initialisation value of m, that is the value of m at its creation.

    Args:
        J (Union[List[OverloadedType], OverloadedType]):  The outputs of the function.
        m (Union[List[OverloadedType], OverloadedType]): The (list of) controls.
        m_dot (Union[List[OverloadedType], OverloadedType]): The input tangents.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape (Tape): The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The action on m_dot of the Jacobian of J with respect to the control. Should be an instance of the same type as the output of J.
    """
    options = {} if options is None else options
    tape = get_working_tape() if tape is None else tape
    tape.reset_tlm_values()
    m_dot = Enlist(m_dot)
    J = Enlist(J)
    m = Enlist(m)

    for i in range(len(m_dot)):
        m[i].tlm_value = m_dot[i]

    with stop_annotating():
        tape.evaluate_tlm(inputs=m, outputs=J)

        Jmdots = []
        for Ji in J:
            if isinstance(Ji.tlm_value, numpy.ndarray):
                output = Ji.output._ad_copy()
                output, offset = output._ad_assign_numpy(
                    output, Ji.tlm_value.flatten(), offset=0
                )
            else:
                output = Ji.tlm_value
            Jmdots.append(output)
    return J.delist(Jmdots)


def compute_jacobian_matrix(J, m, m_jac=None, tape=None,)
    """
    Compute dJdm matrix.

    Args:
        J (Union[list[OverloadedType], OverloadedType]): The outputs of the function.
        m (Union[list[Control], Control]): The (list of) controls.
        m_jac: An input Jacobian to multiply with. By default, this will be an identity Jacobian.
            If m is a list, this should be a list of lists with len(m_jac) == len(m) and
            len(m_jac[i]) == len(m) for each i-th entry in m_jac.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The Jacobian with respect to the control. Should be an instance of the same type as the control.
    """
    tape = get_working_tape() if tape is None else tape

    m = Enlist(m)
    if m_jac is None:
        m_jac = make_jacobian_identities(len(m))
    else:
        m_jac = Enlist(m_jac)

    tape.reset_tlm_matrix_values()

    J = Enlist(J)

    for i, input_jac in enumerate(m_jac):
        m[i].tlm_matrix = Enlist(input_jac)

    with stop_annotating():
        tape.evaluate_tlm_matrix(m, J)

    r = [v.block_variable.tlm_matrix for v in J]
    return J.delist(r)


def compute_hessian(J, m, m_dot, options=None, tape=None):
    """
    Compute the Hessian of J in a direction m_dot at the current value of m

    Args:
        J (AdjFloat):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        m_dot (list or instance of the control type): The direction in which to compute the Hessian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The second derivative with respect to the control in direction m_dot. Should be an instance of
            the same type as the control.
    """
    tape = tape or get_working_tape()
    options = options or {}

    tape.reset_tlm_values()
    tape.reset_hessian_values()

    m = Enlist(m)
    m_dot = Enlist(m_dot)
    for i, value in enumerate(m_dot):
        m[i].tlm_value = m_dot[i]

    with stop_annotating():
        tape.evaluate_tlm()

    J.block_variable.hessian_value = 0.0
    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_hessian(markings=True)

    r = [v.get_hessian(options=options) for v in m]
    return m.delist(r)


def compute_hessian_action(J, m, m_dot, options=None, tape=None, hessian_value=1.0,
) -> Union[List[OverloadedType], OverloadedType]:
    """
    Compute the Hessian of J in a direction m_dot at the current value of m

    Args:
        J (Union[list[OverloadedType], OverloadedType]):  The objective function.
        m (Union[list[Control], Control]): The (list of) controls.
        m_dot (Union[list[OverloadedType], OverloadedType]): The direction in which
            to compute the Hessian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The second derivative with respect to the control in direction m_dot. Should be an instance of
            the same type as the control.
    """
    tape = get_working_tape() if tape is None else tape
    options = {} if options is None else options

    tape.reset_tlm_values()
    tape.reset_hessian_values()

    m = Enlist(m)
    m_dot = Enlist(m_dot)
    for i, value in enumerate(m_dot):
        m[i].tlm_value = m_dot[i]

    with stop_annotating():
        tape.evaluate_tlm()

    hessian_value = Enlist(hessian_value)
    J = Enlist(J)
    for i in range(len(hessian_value)):
        J[i].hessian_value = hessian_value[i]

    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_hessian(markings=True)

    r = [v.get_hessian(options=options) for v in m]
    return m.delist(r)


def solve_adjoint(J, tape=None, adj_value=1.0):
    """
    Solve the adjoint problem for a functional J.

    This traverses the entire tape backwards, unlike `compute_gradient` which only works out those
    parts of the adjoint necessary to compute the sensitivity with respect to the specified control.
    As a result sensitivities with respect to all intermediate states are accumulated in the
    `adj_value` attribute of the associated block-variables. The adjoint solution of each solution
    step is stored in the `adj_sol` attribute of the corresponding solve block.

    Args:
        J (AdjFloat):  The objective functional.
        tape: The tape to use. Default is the current tape.
    """
    tape = tape or get_working_tape()
    tape.reset_variables()
    J.adj_value = adj_value

    with stop_annotating():
        tape.evaluate_adj(markings=False)
