# Python implementation of the mass-balance solver for supramolecular copolymerization 
# of two monomers types A and B from Matlab code by H. ten Eikelder
# Reference: 
#   Equilibrium Model for Supramolecular Copolymerizations
#   Huub M. M. ten Eikelder, Beatrice Adelizzi, Anja R. A. Palmans, and Albert J. Markvoort
#   The Journal of Physical Chemistry B 2019 123 (30), 6627-6642
#   DOI: 10.1021/acs.jpcb.9b04373


import numpy as np


def myerror(msg: str):
    """Raise a ValueError with the provided message."""
    raise ValueError(msg)


def ComputePL(a, b, p, Ktable, sigmatable):
    """Compute polymer concentrations and average chain lengths for all polymer types.
    
    Solves the equilibrium equations for polymer chain distributions given monomer
    concentrations. Computes the equivalent A and B concentrations in each polymer
    type and the average (concentration-weighted) chain length.
    
    Args:
        a (float): Equilibrium concentration of monomer A
        b (float): Equilibrium concentration of monomer B
        p (int): Number of polymer types
        Ktable (np.ndarray): Array of shape (p,) containing 2x2 equilibrium constant matrices
        sigmatable (np.ndarray): Array of shape (p,) containing initial monomer concentrations
        
    Returns:
        Ptable (np.ndarray): Shape (2, p) - Equivalent A and B concentrations in each polymer
        Ltable (np.ndarray): Shape (1, p) - Average chain length for each polymer type
        
    Raises:
        ValueError: If any polymer type has eigenvalue > 1 (outside stability region)
    """
    Ptable = np.zeros((2, p), float)
    Ltable = np.zeros((1, p), float)
    for i in range(p):
        sigma_i = sigmatable[i]
        K_i = Ktable[i]

        Mc = K_i * np.array([[a, a], [b, b]])
        M_A = K_i * np.array([[a, a], [0, 0]])
        M_B = K_i * np.array([[0, 0], [b, b]])

        # State-transition matrix for (PA, PB, LA, LB, SA, SB)
        M = np.block(
            [
                [Mc, np.zeros((2, 4), float)],
                [M_A, Mc, np.zeros((2, 2), float)],
                [M_B, np.zeros((2, 2), float), Mc],
            ]
        )

        u1 = np.concatenate(
            [
                sigma_i * np.array([a, b]).reshape(-1, 1),
                sigma_i * np.array([a, 0]).reshape(-1, 1),
                sigma_i * np.array([0, b]).reshape(-1, 1),
            ]
        )

        lambda_max = np.max(np.abs(np.linalg.eig(Mc)[0]))
        if lambda_max > 1:
            myerror(
                "error in ComputePL, polymer type %d outside allowed region, lambda-1=%8.3e\n"
                % (i, lambda_max - 1)
            )

        U = np.linalg.lstsq(np.eye(6) - M, np.dot(M, u1), rcond=None)[0].squeeze()
        Atot = U[2] + U[3]
        Btot = U[4] + U[5]
        Ptable[:, i] = np.array([Atot, Btot])

        polconctot = U[0] + U[1]
        Ltable[0, i] = (Atot + Btot) / polconctot

    return Ptable, Ltable


def Computefa(a, b, p, a_tot, Ktable, sigmatable):
    """Mass balance residual for monomer A: fa = a + sum(PA_i) - a_tot
       for fixed equilibrium concentration of monomer B. 

    Computes the residual of the mass balance equation for component A, which
    equals zero when the equilibrium concentration of A is correct.
    
    Args:
        a (float): Current equilibrium concentration of monomer A
        b (float): Fixed equilibrium concentration of monomer B
        p (int): Number of polymer types
        a_tot (float): Total concentration of component A (constraint)
        Ktable (np.ndarray): Array of shape (p,) containing 2x2 equilibrium constant matrices
        sigmatable (np.ndarray): Array of shape (p,) containing initial monomer concentrations
        
    Returns:
        fa (float): Mass balance residual for component A
        
    Raises:
        ValueError: If any polymer type has eigenvalue > 1 (outside stability region)
    """
    fa = a - a_tot
    for i in range(p):
        sigma_i = sigmatable[i]
        K_i = Ktable[i]
        Mc = K_i * np.array([[a, a], [b, b]])
        M_A = K_i * np.array([[a, a], [0, 0]])
        M = np.block([[Mc, np.zeros((2, 2), float)], [M_A, Mc]])

        u1 = np.concatenate(
            [sigma_i * np.array([a, b]).reshape(-1, 1), sigma_i * np.array([a, 0]).reshape(-1, 1)]
        )

        lambda_max = np.max(np.abs(np.linalg.eig(Mc)[0]))
        if lambda_max > 1:
            myerror(
                "error in Computefa, polymer type %d outside allowed region, lambda-1=%8.3e\n"
                % (i, lambda_max - 1)
            )

        U = np.linalg.lstsq(np.eye(4) - M, np.dot(M, u1), rcond=None)[0].squeeze()
        fa = fa + U[2] + U[3]
    return fa


def Computefb(a, b, p, b_tot, Ktable, sigmatable):
    """Mass balance residual for monomer B: fb = b + sum(PB_i) - b_tot.
       for fixed equilibrium concentration of monomer A. 
    
    Computes the residual of the mass balance equation for component B, which
    equals zero when the equilibrium concentration of B is correct.
    
    Args:
        a (float): Fixed equilibrium concentration of monomer A
        b (float): Current equilibrium concentration of monomer B
        p (int): Number of polymer types
        b_tot (float): Total concentration of component B (constraint)
        Ktable (np.ndarray): Array of shape (p,) containing 2x2 equilibrium constant matrices
        sigmatable (np.ndarray): Array of shape (p,) containing initial monomer concentrations
        
    Returns:
        fb (float): Mass balance residual for component B
        
    Raises:
        ValueError: If any polymer type has eigenvalue > 1 (outside stability region)
    """
    fb = b - b_tot
    for i in range(p):
        sigma_i = sigmatable[i]
        K_i = Ktable[i]
        Mc = K_i * np.array([[a, a], [b, b]])
        M_B = K_i * np.array([[0, 0], [b, b]])
        M = np.block([[Mc, np.zeros((2, 2), float)], [M_B, Mc]])

        u1 = np.concatenate(
            [sigma_i * np.array([a, b]).reshape(-1, 1), sigma_i * np.array([0, b]).reshape(-1, 1)]
        )

        lambda_max = np.max(np.abs(np.linalg.eig(Mc)[0]))
        if lambda_max > 1:
            myerror(
                "error in Computefb, polymer type %d outside allowed region, lambda-1=%8.3e\n"
                % (i, lambda_max - 1)
            )

        U = np.linalg.lstsq(np.eye(4) - M, np.dot(M, u1), rcond=None)[0].squeeze()
        fb = fb + U[2] + U[3]
    return fb


def Solvefa(b, a1_0, a2_0, rel_errorA, a_tot, b_tot, p, Ktable, sigmatable):
    """Solve mass balance for component A using bisection for fixed B.
    
    Given a fixed B concentration, finds the equilibrium A concentration using
    bisection to locate where fa(a) = 0. Also computes the B mass balance residual
    at the solution to check overall equilibrium.
    
    Args:
        b (float): Fixed equilibrium concentration of B for this bisection step
        a1_0 (float): Lower bound for bisection interval (initial guess)
        a2_0 (float): Upper bound for bisection interval (initial guess)
        rel_errorA (float): Relative error tolerance for bisection convergence
        a_tot (float): Total concentration of component A (mass balance constraint)
        b_tot (float): Total concentration of component B (mass balance constraint)
        p (int): Number of polymer types
        Ktable (np.ndarray): Array of shape (p,) containing 2x2 equilibrium constant matrices
        sigmatable (np.ndarray): Array of shape (p,) containing initial monomer concentrations
        
    Returns:
        resb12 (float): B mass balance residual at the solution
        a1 (float): Lower bound after bisection convergence
        a2 (float): Upper bound after bisection convergence
        aa12 (float): Equilibrium A concentration (solution via linear interpolation)
        
    Raises:
        ValueError: If initial bounds do not bracket a solution
    """
    if a_tot == 0:
        a1 = a2 = aa12 = 0
        resb12 = Computefb(aa12, b, p, b_tot, Ktable, sigmatable)
    else:
        aa1 = a1_0
        fa1 = Computefa(aa1, b, p, a_tot, Ktable, sigmatable)

        a_max = np.inf
        for pp in range(p):
            K_AB = Ktable[pp]
            N1 = K_AB[0, 0] - np.linalg.det(K_AB) * b
            if N1 != 0:
                a_max_pp = (1 - K_AB[1, 1] * b) / N1
                a_max = min(a_max, a_max_pp)

        if a_max <= a2_0:  # always a2_0 <= a_tot
            aa2 = a_max
            fa2 = np.inf
        else:
            aa2 = a2_0
            fa2 = Computefa(aa2, b, p, a_tot, Ktable, sigmatable)

        if (fa1 * fa2 > 0) or (aa1 > aa2):
            print(
                "aa1=%15.10e  aa2= %15.10e  fa1= %15.10e  fa2= %15.10e\n"
                % (aa1, aa2, fa1, fa2)
            )
            myerror(
                "b=%10.4e,c=%10.4e, error in Solve_fa2, wrong init values for aa1, aa2 in bin search"
                % (b)
            )

        while abs(fa2 - fa1) > rel_errorA:
            aa3 = (aa1 + aa2) / 2
            fa3 = Computefa(aa3, b, p, a_tot, Ktable, sigmatable)
            if fa1 * fa3 > 0:
                aa1 = aa3
                fa1 = fa3
            else:
                aa2 = aa3
                fa2 = fa3

        a1 = aa1
        a2 = aa2
        if fa2 < np.inf:
            s = -fa1 / (fa2 - fa1)
            aa12 = a1 + s * (a2 - a1)
        else:
            aa12 = (a1 + a2) / 2

        resb12 = Computefb(aa12, b, p, b_tot, Ktable, sigmatable)

    return resb12, a1, a2, aa12


def Solvefab(b1_0, b2_0, a_tot, b_tot, rel_errorA, rel_errorB, p, Ktable, sigmatable):
    """Solve full mass balance using nested bisection for both A and B.
    
    Finds equilibrium concentrations for both components A and B using a nested
    bisection approach: outer bisection on B, inner bisection on A. Converges when
    both mass balance residuals are below tolerance.
    
    Args:
        b1_0 (float): Lower bound for B bisection (typically 0)
        b2_0 (float): Upper bound for B bisection (typically b_tot or b_max)
        a_tot (float): Total concentration of component A (mass balance constraint)
        b_tot (float): Total concentration of component B (mass balance constraint)
        rel_errorA (float): Relative error tolerance for A bisection
        rel_errorB (float): Relative error tolerance for B bisection
        p (int): Number of polymer types
        Ktable (np.ndarray): Array of shape (p,) containing 2x2 equilibrium constant matrices
        sigmatable (np.ndarray): Array of shape (p,) containing initial monomer concentrations
        
    Returns:
        aa12 (float): Equilibrium concentration of component A
        b1 (float): Final lower B bound after convergence
        b2 (float): Final upper B bound after convergence
        bb12 (float): Equilibrium concentration of component B
        
    Raises:
        ValueError: If initial bounds do not bracket a solution
    """
    if b_tot == 0:
        b1 = b2 = bb12 = 0
        resb12, a1, a2, aa12 = Solvefa(0, 0, a_tot, rel_errorA, a_tot, b_tot, p, Ktable, sigmatable)
    else:
        bb1 = b1_0
        fb1, a11, a12, aa112 = Solvefa(bb1, 0, a_tot, rel_errorA, a_tot, b_tot, p, Ktable, sigmatable)

        b_max = np.inf
        for pp in range(p):
            K_AB = Ktable[pp]
            b_max = min(b_max, 1 / K_AB[1, 1])

        if b_max <= b2_0:  # always b2_0 <= b_tot
            bb2 = b_max
            fb2 = np.inf
            a21 = a22 = aa212 = 0
        else:
            bb2 = b2_0
            fb2, a21, a22, aa212 = Solvefa(bb2, 0, a_tot, rel_errorA, a_tot, b_tot, p, Ktable, sigmatable)

        if (fb1 * fb2 > 0) or (bb1 > bb2):
            print("b1=%15.10e  b2= %15.10e  fb1= %15.10e  fb2= %15.10e\n" % (bb1, bb2, fb1, fb2))
            myerror("error in Solve_fab, wrong init values for b1, b2 in bin search")

        while abs(fb2 - fb1) > rel_errorB:
            bb3 = (bb1 + bb2) / 2
            fb3, a31, a32, aa312 = Solvefa(bb3, a21, a12, rel_errorA, a_tot, b_tot, p, Ktable, sigmatable)
            if fb3 * fb1 > 0:
                bb1 = bb3
                fb1 = fb3
                a12 = a32
                aa112 = aa312
            else:
                bb2 = bb3
                fb2 = fb3
                a21 = a31
                aa212 = aa312

        b1 = bb1
        b2 = bb2
        if fb2 < np.inf:
            s = -fb1 / (fb2 - fb1)
            bb12 = b1 - s * (b2 - b1)
            aa12 = aa112 + s * (aa212 - aa112)
        else:
            bb12 = (b1 + b2) / 2
            aa12 = (aa112 + aa212) / 2

    return aa12, b1, b2, bb12


def solve_massbalance_2comp(totconc, sigmatable, Ktable):
    """Solve mass balance for two-component (A and B) copolymerization system.
    
    Core solver for binary copolymerization. Finds equilibrium monomer concentrations
    and polymer distributions that satisfy mass balance constraints.
    
    Args:
        totconc (np.ndarray): Shape (2, 1) - Total concentrations [a_tot, b_tot]
        sigmatable (np.ndarray): Shape (p, 1) - Initial monomer concentrations per polymer type
        Ktable (np.ndarray): Shape (p, 2, 2) - Equilibrium constant matrices for each type
        
    Returns:
        moneqconcs (np.ndarray): Shape (2, 1) - Equilibrium concentrations [a_eq, b_eq]
        res_ab (np.ndarray): Shape (2, 1) - Mass balance residuals [res_a, res_b]
        Pout (np.ndarray): Shape (2, p) - Equivalent A and B conc in each polymer
        Lout (np.ndarray): Shape (1, p) - Average chain length per polymer type
    """
    p = sigmatable.shape[0]  # number of copolymer types

    a_tot = totconc[0][0]
    b_tot = totconc[1][0]
    rel_errorA = 1e-6 * a_tot
    rel_errorB = 1e-6 * b_tot

    a_eq, _, _, b_eq = Solvefab(0, b_tot, a_tot, b_tot, rel_errorA, rel_errorB, p, Ktable, sigmatable)

    moneqconcs = np.array([a_eq, b_eq]).reshape(-1, 1)
    Pout, Lout = ComputePL(a_eq, b_eq, p, Ktable, sigmatable)
    res_a = sum(Pout[0, :]) + a_eq - a_tot
    res_b = sum(Pout[1, :]) + b_eq - b_tot
    res_ab = np.array([res_a, res_b]).reshape(-1, 1)

    return moneqconcs, res_ab, Pout, Lout


def solve_massbalance(totconc, sigmatable, Ktable, verbose=False):
    """Public interface to solve mass balance equations for copolymerization.
    
    Validates input shapes and dimensions, handles padding for <2 components,
    and dispatches to the two-component solver.
    
    Reference:
        Equilibrium Model for Supramolecular Copolymerizations
        ten Eikelder et al., J. Phys. Chem. B 2019, 123(30), 6627-6642
    
    Args:
        totconc (np.ndarray): Shape (ncomp, 1) - Total concentrations [a_tot, ...], ncomp <= 2
        sigmatable (np.ndarray): Shape (p, ncomp, 1) - Initial concentrations per polymer type
        Ktable (np.ndarray): Shape (p, ncomp, ncomp) - Equilibrium constant matrices
        verbose (bool): If True, print diagnostic information during solving
        
    Returns:
        moneqconcs (np.ndarray): Shape (ncomp, 1) - Equilibrium concentrations
        res_abc (np.ndarray): Shape (ncomp, 1) - Mass balance residuals
        Pout (np.ndarray): Shape (ncomp, p) - Equivalent concentrations per component/polymer
        Lout (np.ndarray): Shape (ncomp, p) - Average chain lengths per component/polymer
        
    Raises:
        ValueError: If input shapes are inconsistent or ncomp > 2
    """
    if len(totconc.shape) != 2:
        myerror("totconc should be 2-dimensional array: ncomp x 1")
    if len(Ktable.shape) != 3:
        myerror("Ktable should be 3-dimensional array: npol x ncomp x ncomp")
    if len(sigmatable.shape) != 3:
        myerror("sigmatable should be 3-dimensional array: npol x ncomp x 1")

    ncomp = totconc.shape[0]
    p = sigmatable.shape[0]
    if verbose:
        print("Number of components:", ncomp)
        print("Number of polymer types:", p)

    if sigmatable.shape[1] != ncomp:
        print("Number of conponents in totconc does not match sigmatable")
        myerror("sigmatable should be 3-dimensional array: npol x ncomp x 1")
    if sigmatable.shape[2] != 1:
        print("sigmatable should be 3-dimensional array: npol x 1 x ncomp")
        myerror("sigmatable should be 3-dimensional array: npol x ncomp x 1")

    if Ktable.shape[0] != p:
        print("Number of polymer types in sigmatable does not match Ktable")
        myerror("Ktable should be 3-dimensional array: npol x ncomp x ncomp")
    if Ktable.shape[1] != ncomp:
        print("Number of conponents in totconc does not match Ktable")
        myerror("Ktable should be 3-dimensional array: npol x ncomp x ncomp")
    if Ktable.shape[2] != Ktable.shape[1]:
        print("second and third dimension of Ktable should match")
        myerror("Ktable should be 3-dimensional array: npol x ncomp x ncomp")

    if ncomp == 1:
        c = totconc.flat[0]
        totconc = np.array([c, 0], float).reshape(-1, 1)
        t = sigmatable[:, 0, 0]
        sigmatable = np.zeros([p, 2, 1], float)
        sigmatable[:, 0, 0] = t
        t = Ktable[:, 0, 0]
        Ktable = np.zeros([p, 2, 2], float)
        Ktable[:, 0, 0] = t
    elif ncomp > 2:
        myerror("Bin search only implemented for at most 2 components")

    moneqconcs, res_abc, Pout, Lout = solve_massbalance_2comp(totconc, sigmatable, Ktable)

    return moneqconcs[:ncomp, :], res_abc[:ncomp, :], Pout[:ncomp, :], Lout[:ncomp, :]
    