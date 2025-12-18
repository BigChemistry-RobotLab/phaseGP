# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:18:51 2022

@author: amarkvoo
"""
import sys
import numpy as np

#from numba import jit


def myerror(s):
    print(s)
    sys.exit(-1)

#@jit(forceobj=True) #(nopython=True)
def ComputePL(a,b,c,p,Ktable,sigmatable):
    # Compute PA,PB,PC and L for all p (p=1,2,...) copolymer types 
    # a, b, c = input concentrations 
    # sigmatable = 3 x p matrix of [sigmaA;sigmaB;sigmaC] vectors for the p copolymer types
    # Ktable = 3 x 3p matrix of p  3x3 K matrices
   
    Ptable = np.zeros([3,p],float); # for equiv A, B and C conc in the p copolymer types
    Ltable = np.zeros([1,p],float); # for average conc weighted length of p copolymer types
    for i in range(p):
        sigmai = sigmatable[i]
        Ki = Ktable[i]
        Mc = Ki*np.array([[a,a,a],[b,b,b],[c,c,c]])
        M_A = Ki*np.array([[a,a,a],[0,0,0],[0,0,0]])
        M_B = Ki*np.array([[0,0,0],[b,b,b],[0,0,0]])
        M_C = Ki*np.array([[0,0,0],[0,0,0],[c,c,c]])
        M = np.concatenate(
            [np.concatenate([Mc,np.zeros([3,9],float)],1),
             np.concatenate([M_A,Mc,np.zeros([3,6],float)],1),
             np.concatenate([M_B,np.zeros([3,3],float),Mc,np.zeros([3,3],float)],1),
             np.concatenate([M_C,np.zeros([3,6],float),Mc],1)]
            ) # 12 x 12
        u1=np.concatenate(
            [sigmai*(np.array([a,b,c])).reshape(-1,1), 
             sigmai*(np.array([a,0,0])).reshape(-1,1),
             sigmai*(np.array([0,b,0])).reshape(-1,1),
             sigmai*(np.array([0,0,c])).reshape(-1,1)]
            )

        lambdamax=np.max(np.abs(np.linalg.eig(Mc)[0]))
        if lambdamax>1:
            myerror('error in ComputePL, polymer type %d outside allowed region, lambda-1=%8.3e\n' % (i,lambdamax-1))

        U = np.linalg.lstsq(np.eye(12)-M, np.dot(M,u1),rcond=None)[0].squeeze()
        Atot = U[3]+U[4]+U[5]
        Btot = U[6]+U[7]+U[8]
        Ctot = U[9]+U[10]+U[11]
        Ptable[:,i] = np.array([Atot,Btot,Ctot])
        polconctot = U[0]+U[1]+U[2]
        Ltable[0,i] = (Atot+Btot+Ctot)/polconctot

    return Ptable,Ltable

#@jit(forceobj=True) #(nopython=True)
def Computefa(a,b,c,p,a_tot,Ktable,sigmatable):
    # Compute a LHS: fa=a+ sum_i PA_i-a_tot   
    # for p (p=1,2,...) copolymer types 
    # a,b,c = input concentrations  
    # sigmatable = 3 x p matrix of [sigmaA;sigmaB;sigmaC] vectors for the p copolymer types
    # Ktable= 3x 3p matrix of p  3x3 K matrices
  
    fa = a - a_tot
    #print('---',fa,a,a_tot)
    for i in range(p):
        sigmai = sigmatable[i]
        Ki = Ktable[i]
        Mc = Ki*np.array([[a,a,a],[b,b,b],[c,c,c]])
        M_A = Ki*np.array([[a,a,a],[0,0,0],[0,0,0]])
        M = np.concatenate(
            [np.concatenate([Mc,np.zeros([3,3],float)],1),
             np.concatenate([M_A,Mc],1)]
            )
        #print('qqq')
        #print(sigmai)
        #print(np.array([a,b,c]).reshape(-1,1))
        u1=np.concatenate(
            [sigmai*(np.array([a,b,c])).reshape(-1,1), 
             sigmai*(np.array([a,0,0])).reshape(-1,1)]
            )
        #print('helahola',a, b, c)
        #print(Mc)
        #print(np.linalg.eig(Mc))
        #print(np.abs(np.linalg.eig(Mc)[0]))
        eigenvals = np.linalg.eig(Mc)[0]
        #print(eigenvals)
        lambdamax = np.max(np.abs(eigenvals))
        if lambdamax>1:
            myerror('error in Computefa, polymer type %d outside allowed region, lambda-1=%8.3e\n' % (i,lambdamax-1))

        #print(np.eye(6)-M)
        #print(M)
        #print(u1)
        #print(np.dot(M,u1))
        U = np.linalg.lstsq(np.eye(6)-M, np.dot(M,u1),rcond=None)[0].squeeze()
        #print('U',U)
        fa = fa + U[3]+U[4]+U[5]
        #print('fa',fa)
    return fa


#@jit(forceobj=True) #(nopython=True)
def Computefb(a,b,c,p,b_tot,Ktable,sigmatable):
    # Compute b LHS: fb=b+ sum_i PB_i -b_tot 
    # for p (p=1,2,...) copolymer types 
    # a,b,c = input concentrations  
    # sigmatable = 3 x p matrix of [sigmaA;sigmaB;sigmac] vectors for the p copolymer types
    # Ktable = 3 x 3p matrix of p  3x3 K matrices

    fb = b-b_tot
    for i in range(p):
        sigmai = sigmatable[i]
        Ki = Ktable[i]
        Mc = Ki*np.array([[a,a,a],[b,b,b],[c,c,c]])
        M_B = Ki*np.array([[0,0,0],[b,b,b],[0,0,0]])
        M = np.concatenate(
            [np.concatenate([Mc,np.zeros([3,3],float)],1),
             np.concatenate([M_B,Mc],1)]
            )
        u1=np.concatenate(
            [sigmai*(np.array([a,b,c])).reshape(-1,1), 
             sigmai*(np.array([0,b,0])).reshape(-1,1)]
            )

        #print('bbb')
        #print(Mc)
        eigenvals = np.linalg.eig(Mc)[0]
        #print(eigenvals)
        lambdamax = np.max(np.abs(eigenvals))
        if lambdamax>1:
            myerror('error in Computefb, polymer type %d outside allowed region, lambda-1=%8.3e\n' % (i,lambdamax-1))

        U = np.linalg.lstsq(np.eye(6)-M, np.dot(M,u1),rcond=None)[0].squeeze()
        fb = fb + U[3]+U[4]+U[5]
    return fb


#@jit(forceobj=True) #(nopython=True)
def Computefc(a,b,c,p,c_tot,Ktable,sigmatable):
    # Compute c LHS: fc = c + sum_i PC_i - c_tot 
    # for p (p=1,2,...) copolymer types 
    # a,b,c= input concentrations  
    # sigmatable = 3 x p matrix of [sigmaA;sigmaB;sigmac] vectors for the p copolymer types
    # Ktable = 3 x 3p matrix of p  3x3 K matrices

    fc = c-c_tot
    for i in range(p):
        sigmai = sigmatable[i]
        Ki = Ktable[i]
        Mc = Ki*np.array([[a,a,a],[b,b,b],[c,c,c]])
        M_C = Ki*np.array([[0,0,0],[0,0,0],[c,c,c]])
        M = np.concatenate(
            [np.concatenate([Mc,np.zeros([3,3],float)],1),
             np.concatenate([M_C,Mc],1)]
            )
        u1=np.concatenate(
            [sigmai*(np.array([a,b,c])).reshape(-1,1), 
             sigmai*(np.array([0,0,c])).reshape(-1,1)]
            )

        lambdamax=np.max(np.abs(np.linalg.eig(Mc)[0]))
        if lambdamax>1:
            myerror('error in Computefb, polymer type %d outside allowed region, lambda-1=%8.3e\n' % (i,lambdamax-1))

        U = np.linalg.lstsq(np.eye(6)-M, np.dot(M,u1),rcond=None)[0].squeeze()
        fc = fc + U[3]+U[4]+U[5]
    return fc

#@jit(forceobj=True) #(nopython=True)
def Solvefa(b,c,a1_0,a2_0,rel_errorA,a_tot,b_tot,c_max,p,Ktable,sigmatable):
    # find solution a for fixed b and c
    # precond: a1_0 <= solution a of a mass bal eq <= a2_0, 
    # resa_a1=fa(a1,b)=PA(a1,b)+a1-a_tot<=0,  resa_a2=fa(a2,b)>0,
    # aa12 by lin interpol between a1 and a2, such that fa(aa12,b)=0
    # resb12=fb(aa2,b,c) = (P(aa12,b,c)+b-btot = b residu
    # resc12=fc(aa2,b,c) = (P(aa12,b,c)+c-ctot = c residu
    
    #if b >= b_max  
    #    error('b=%10.4e > min(1./KPBBrow), b too large, in Solvefa',b);
    #end  
    if c >= c_max:
        myerror('c=%10.4e > min(1./KPCCrow), c too large, in Solvefa' % c)  
    if a_tot==0: # solution must be:  a1=0, a2=0
        a1=0; a2=0; aa12=0;
        resb12 = Computefb(aa12,b,c,p,b_tot,Ktable,sigmatable)\
        #resc12=Computefc(aa12,b,c,p,c_tot,Ktable,sigmatable)
    else:  # a_tot>0, bisection method:  
        aa1=a1_0  
        fa1=Computefa(aa1,b,c,p,a_tot,Ktable,sigmatable)

        a_max=np.inf
        for pp in range(p):
            K_ABC=Ktable[pp]
            K_AB=K_ABC[0:2,0:2]
            K_BC=K_ABC[1:3,1:3]
            K_AC=K_ABC[[0,2],:][:,[0,2]]
            N1 = K_ABC[0,0] - np.linalg.det(K_AB)*b - np.linalg.det(K_AC)*c + \
                np.linalg.det(K_ABC)*b*c
            if N1!=0:                         
                a_max_pp = (1 - K_ABC[1,1]*b - K_ABC[2,2]*c + np.linalg.det(K_BC)*b*c) / N1
                a_max = min(a_max,a_max_pp)

        if a_max<=a2_0:  # always a2_0<=a_tot 
            aa2=a_max; fa2=np.inf; 
        else:
            aa2=a2_0;
            fa2 = Computefa(aa2,b,c,p,a_tot,Ktable,sigmatable)

        #print('  ',b,fa1,aa1)
        #print('  ',b,fa2,aa2)
        # bisection method:
        if ((fa1*fa2)>0) or (aa1>aa2):
            print('aa1=%15.10e  aa2= %15.10e  fa1= %15.10e  fa2= %15.10e\n' % (aa1,aa2,fa1,fa2))
            myerror('b=%10.4e,c=%10.4e, error in Solve_fa2, wrong init values for aa1, aa2 in bin search' % (b,c))

        while abs(fa2-fa1)>rel_errorA: 
            aa3 = (aa1+aa2)/2
            fa3 = Computefa(aa3,b,c,p,a_tot,Ktable,sigmatable);
            if (fa1*fa3)>0:
                aa1=aa3; fa1=fa3; 
            else:
                aa2=aa3; fa2=fa3; 

        a1 = aa1
        a2 = aa2
        # fa1<0,   fa2>=0
        if fa2<np.inf:
            # use best lin appr for point aa12 where fa(aa12,b)=0 is:
            s = -fa1/(fa2-fa1)
            aa12 = a1+s*(a2-a1) 
        else:
            aa12 = (a1+a2)/2
   
        resb12 = Computefb(aa12,b,c,p,b_tot,Ktable,sigmatable)
        #resc12=Computefc(aa12,b,c,p,c_tot,Ktable,sigmatable);
        
    return resb12,a1,a2,aa12


#@jit(forceobj=True) #(nopython=True)
def Solvefab(c,b1_0,b2_0,a_tot,b_tot,c_tot,c_max,rel_errorA,rel_errorB,p,Ktable,sigmatable):
    # find solution b (and internally a) for fixed c
    
    if c >= c_max:
        myerror('c=%10.4e > min(1./KPCCrow), c too large, in Solvefab' % c)
    if b_tot == 0:
        b1=0; b2=0; bb12=0; 
        [resb12,a1,a2,aa12] = Solvefa(0,c,0,a_tot,rel_errorA,a_tot,b_tot,c_max,p,Ktable,sigmatable)
    else:
        bb1=b1_0;
        [fb1,a11,a12,aa112] = Solvefa(bb1,c,0,a_tot,rel_errorA,a_tot,b_tot,c_max,p,Ktable,sigmatable)

        b_max=np.inf;
        for pp in range(p):
            K_ABC=Ktable[pp]
            K_BC=K_ABC[1:3,1:3]
            N1=K_ABC[1,1]-np.linalg.det(K_BC)*c
            if N1 != 0:
                b_max_pp=(1-K_ABC[2,2]*c)/N1
                b_max=min(b_max,b_max_pp)

        if b_max <= b2_0:  # always b2_0<=b_tot 
            bb2=b_max; fb2=np.inf; 
            a21=0; a22=0; aa212=0;
        else:
            bb2=b2_0;
            [fb2,a21,a22,aa212] = Solvefa(bb2,c,0,a_tot,rel_errorA,a_tot,b_tot,c_max,p,Ktable,sigmatable)

        # bisection method:
        if (fb1*fb2>0) or (bb1>bb2):
            print('b1=%15.10e  b2= %15.10e  fb1= %15.10e  fb2= %15.10e\n' % (b1,b2,fb1,fb2))
            myerror('c=%10.4e, error in Solve_fab, wrong init values for b1, b2 in bin search' % c)

        #print('---',fb1,fb2)
        while abs(fb2-fb1)>rel_errorB: 
            bb3=(bb1+bb2)/2
            #print('  ',bb3)
            [fb3,a31,a32,aa312] = Solvefa(bb3,c,a21,a12,rel_errorA,a_tot,b_tot,c_max,p,Ktable,sigmatable)
            #print(a21,a12,a31,a32,aa312,fb1,fb2,fb3)
            if fb3*fb1>0:
                #bb1=bb3; fb1=fb3; a12=a32; aa112=aa312; # a11=a31;
                bb1=bb3; fb1=fb3; a12=a_tot; aa112=aa312; # a11=a31;
            else:
                #bb2=bb3; fb2=fb3; a21=a31; aa212=aa312; # a22=a32;
                bb2=bb3; fb2=fb3; a21=0.0; aa212=aa312; # a22=a32;
            #print(bb1,fb1,a12)
            #print(bb2,fb2,a21)
            #print()
                
        b1=bb1;
        b2=bb2;
        # fb1<0,   fb2>=0
        if fb2<np.inf:
            # use best lin appr for ...%%%point bb12 where fb(bb12,c)=0 is:
            s=-fb1/(fb2-fb1)
            bb12= b1-s*(b2-b1) 
            aa12=aa112+s*(aa212-aa112)
        else:
            bb12=(b1+b2)/2
            aa12=(aa112+aa212)/2
               
    resc12 = Computefc(aa12,bb12,c,p,c_tot,Ktable,sigmatable);        
    return resc12,aa12,b1,b2,bb12

#@jit(forceobj=True) #(nopython=True)
def solve_massbalance_3comp(totconc,sigmatable,Ktable):
    """ Solves mass balance equations, copolymerization with two monomers types A and B
    and p (p=1,2,...) different copolymer types. 
    inputs:
    totconc= total A, B and C concentrations , 3 x 1 vector  
    sigmatable= 3 x p matrix of [sigmaA;sigmaB;sigmaC] vectors for the p copolymer types
    Ktable= 3 x 3p matrix of p  3x3 K matrices
    outputs:
    moneqconcs: 3 x 1 vector of monomer equilibrium concentrations for A, B and C
    res_abc: 3 x 1 vector with the residus of the a, b and c mass-bal equation
    Pout: p x 3 vector, Pout(q,:) contains equivalent A, B and C concentrations
    in i th copolymer type
    Lout(i)= average (concentration weighted) length of i th copolymers type
    Matlab version by H ten Eikelder, June 2019.   
    Extended from 2 to 3 components by A.J. Markvoort, september 2022.   
    Converted to Python by A.J. Markvoort, oktober 2022
    """

    p = sigmatable.shape[0] # number of copolymer types

    a_tot = totconc[0][0]
    b_tot = totconc[1][0]
    c_tot = totconc[2][0]
    rel_errorA = 1e-6*a_tot
    rel_errorB = 1e-6*b_tot
    rel_errorC = 1e-6*c_tot
    rel_errorC2 = 1e-12*c_tot
  
    KPCCrow = Ktable[:,2,2]
    if np.max(KPCCrow) > 0:
        c_max = 1. / np.max(KPCCrow)
    else:
        c_max = np.inf
    
    if c_tot == 0:
        c_eq = 0
        [resc12,a_eq,b1,b2,b_eq] = Solvefab(0,0,b_tot,a_tot,b_tot,c_tot,c_max,rel_errorA,rel_errorB,p,Ktable,sigmatable);
        #print('ttt',resc12,a_eq,b1,b2,b_eq)
    else:
        c1 = 0;
        [fc1,aa112,b11,b12,bb112] = Solvefab(c1,0,b_tot,a_tot,b_tot,c_tot,c_max,rel_errorA,rel_errorB,p,Ktable,sigmatable);
        #print('sss',resc12,a_eq,b1,b2,b_eq)

        # find c2 and fc2:    
        if c_max <= c_tot:
            c2=c_max; fc2=np.inf
            a21=0; aa212=0; a22=0;
            b21=0; bb212=0; b22=0;
        else:
            # c_tot<c_max
            c2=c_tot;
            [fc2,aa212,b21,b22,bb212] = Solvefab(c2,0,b_tot,a_tot,b_tot,c_tot,c_max,rel_errorA,rel_errorB,p,Ktable,sigmatable)

        # bisection method
        if (fc1*fc2) > 0:
            print('c1=%15.10e  c2= %15.10e  fc1= %15.10e  fc2= %15.10e' % (c1,c2,fc1,fc2));
            myerror('error in SolveMassBal, wrong init values in bin search for c');

        while (abs(fc2-fc1)>rel_errorC) and ((c2-c1)>rel_errorC2): 
            c3 = (c1+c2)/2;
            #print('c3 = %.3e,   fc1 = %.3e,   fc2 = %.3e,  abs(fc2-fc1)=%.3e, rel_errorC=%.3e \n',c3,fc1,fc2,abs(fc2-fc1),rel_errorC);
            #[fc3,aa312,b31,b32,bb312] = Solvefab(c3,b21,b12,a_tot,b_tot,c_tot,c_max,rel_errorB,Ktable,sigmatable)
            [fc3,aa312,b31,b32,bb312] = Solvefab(c3,0,b_tot,a_tot,b_tot,c_tot,c_max,rel_errorA,rel_errorB,p,Ktable,sigmatable)
            if (fc3*fc1) > 0:
                c1=c3; fc1=fc3; 
                aa112=aa312;
                b12=b32; bb112=bb312; # b11=b31;
            else:
                c2=c3; fc2=fc3; 
                aa212=aa312;
                b21=b31; bb212=bb312; # b22=b32;

        if fc2 < np.inf:
            #  lin inerpolation between (aa112,c1) and (aa212,c2), using values of fc 
            # Note: fc1<0, fc2>=0
            s=-fc1/(fc2-fc1);
            a_eq=aa112+s*(aa212-aa112);
            b_eq=bb112+s*(bb212-bb112);
            c_eq=c1+s*(c2-c1); 
        else: # fc2=Inf
            a_eq=(aa112+aa212)/2;
            b_eq=(bb112+bb212)/2;
            c_eq=(c1+c2)/2;
            
        #c_eq = c1;
        #b_eq = bb112;
        #a_eq = aa112;

    moneqconcs = np.array([a_eq,b_eq,c_eq]).reshape(-1,1) 
    [Pout,Lout] = ComputePL(a_eq,b_eq,c_eq,p,Ktable,sigmatable)
    res_a = sum(Pout[0,:])+a_eq-a_tot
    res_b = sum(Pout[1,:])+b_eq-b_tot
    res_c = sum(Pout[2,:])+c_eq-c_tot
    res_abc = np.array([res_a,res_b,res_c]).reshape(-1,1) 

    return moneqconcs,res_abc,Pout,Lout


def solve_massbalance(totconc,sigmatable,Ktable,verbose=False):

    if not(len(totconc.shape)==2):
        myerror('totconc should be 2-dimensional array: ncomp x 1')
    if not(len(Ktable.shape)==3):
        myerror('Ktable should be 3-dimensional array: npol x ncomp x ncomp')
    if not(len(sigmatable.shape)==3):
        myerror('sigmatable should be 3-dimensional array: npol x ncomp x 1')

    ncomp = totconc.shape[0] # nr of components
    p = sigmatable.shape[0] # number of copolymer types
    if verbose:
        print('Number of components:', ncomp)
        print('Number of polymer types:', p)

    if not(sigmatable.shape[1]==ncomp): 
        print('Number of conponents in totconc does not match sigmatable')
        myerror('sigmatable should be 3-dimensional array: npol x ncomp x 1')
    if not(sigmatable.shape[2]==1):
        print('sigmatable should be 3-dimensional array: npol x 1 x ncomp')
        myerror('sigmatable should be 3-dimensional array: npol x ncomp x 1')

    if not(Ktable.shape[0]==p): 
        print('Number of polymer types in sigmatable does not match Ktable')
        myerror('Ktable should be 3-dimensional array: npol x ncomp x ncomp')
    if not(Ktable.shape[1]==ncomp): 
        print('Number of conponents in totconc does not match Ktable')
        myerror('Ktable should be 3-dimensional array: npol x ncomp x ncomp')
    if not(Ktable.shape[2]==Ktable.shape[1]): 
        print('second and third dimension of Ktable should match')
        myerror('Ktable should be 3-dimensional array: npol x ncomp x ncomp')
        
        
    if (ncomp == 1):
        c = totconc[0]
        totconc = np.array([c,0,0],float).reshape(-1,1)
        t = sigmatable[:,0,0]
        sigmatable = np.zeros([p,3,1],float)
        sigmatable[:,0,0] = t
        t = Ktable[:,0,0]
        Ktable = np.zeros([p,3,3],float)
        Ktable[:,0,0] = t
        #print(totconc.shape)
        #print(sigmatable.shape)
        #print(Ktable.shape)
    elif (ncomp == 2):    
        totconc = np.concatenate([totconc,np.array([[0.]])])
        t = sigmatable[:,:2,0]
        sigmatable = np.zeros([p,3,1],float)
        sigmatable[:,:2,0] = t
        t = Ktable[:,:2,:2]
        Ktable = np.zeros([p,3,3],float)
        Ktable[:,:2,:2] = t
    elif (ncomp == 3):
        if verbose:
            print('Warning: the scipy minimize version is probably faster')
    elif (ncomp > 3):    
        myerror('Bin search only implemented for at most 3 components')
        
    moneqconcs,res_abc,Pout,Lout = solve_massbalance_3comp(totconc,sigmatable,Ktable)

    return moneqconcs[:ncomp,:],res_abc[:ncomp,:],Pout[:ncomp,:],Lout[:ncomp,:]
    