__precompile__
include("quadrature.jl")
include("SpatialBasis.jl")
include("TimeSolver.jl");
using SparseArrays

using ProgressMeter

using LinearAlgebra

struct DGSolver
    # spatial grid of cell interfaces
    x::Array{Float64,1};
    xQ

    # quadrature
    q::Quadrature;

    # Solver settings
    settings::Settings;

    # spatial basis functions
    basis::SpatialBasis;

    # time solver
    time::TimeSolver;

    # preallocate memory for performance
    outRhs::Array{Float64,2};
    outRhsT::Array{Float64,1};

    # parameters for Shu Limiter
    cp::Float64;
    c01::Float64;
    
    # squared L2 norms of Legendre coeffs
    gamma::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};
    # Roe matrix
    AbsA::Array{Float64,2};

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;
    sigmaA::Float64;
    c::Float64;
    sigmaSB::Float64;
    aRad::Float64;

    L1x::SparseMatrixCSC{Float64, Int64};
    L2x::SparseMatrixCSC{Float64, Int64};
    PQ::SparseMatrixCSC{Float64, Int64};
    QP::SparseMatrixCSC{Float64, Int64};

    # constructor
    function DGSolver(settings)
        x = settings.x;
        Nq = settings.Nq
        q = Quadrature(settings.Nq,settings.quadratureType);
        basis = SpatialBasis(x,q,q,settings);
        time = TimeSolver(settings);

        outRhs = zeros(settings.NCells*settings.N,settings.nPN);
        outRhsT = zeros(settings.NCells*settings.N);

        # setup coefficients for Shu Limiter
        cp = 0;
        for i = 2:settings.N
            ciInf = sqrt(2.0*(i-1.0)+1.0);
            cp = cp + ciInf*sqrt(2*(i-1)+1)*abs.(basis.PhiTildeQuad[:,i])'*q.w*settings.dx*0.5; 
        end
        c01 = abs.(basis.PhiTildeQuad[:,1])'*q.w*settings.dx*0.5;
        #println("cp = ",cp)
        #println("c01 = ",c01)

        # setup gamma vector
        gamma = zeros(settings.nPN);
        for i = 1:settings.nPN
            n = i-1;
            gamma[i] = 2/(2*n+1);
        end
        
        # setup DG system
        A = zeros(settings.nPN,settings.nPN)
        for i = 1:(settings.nPN-1)
            n = i-1;
            A[i,i+1] = (n+1)/(2*n+1);
        end

        for i = 2:settings.nPN
            n = i-1;
            A[i,i-1] = n/(2*n+1);
        end

        # compute Roe Matrix
        D = abs.(Diagonal(eigvals(A)));
        T = eigvecs(A);
        AbsA = T*D*inv(T);

        c = settings.c;                # speed of light in [cm/s]
        sigmaSB = settings.sigmaSB;                # Stefan Boltzmann constant in [erg/cm^2/s/K^4]
        aRad = settings.aRad;   

        # setupt stencil matrix
        nx = settings.NCells;
        m = settings.N; # polynomial degree
        L1x = spzeros(nx*m,nx*m);

        # setup index arrays and values for allocation of stencil matrices

        L2k = -basis.PhiTildeRight*0.5*basis.PhiRight'+basis.PhiTildeLeft*0.5*basis.PhiLeft';
        L2kP = -basis.PhiTildeRight*0.5*basis.PhiLeft';
        L2kM = basis.PhiTildeLeft*0.5*basis.PhiRight';
    
        L1k = -basis.PhiTildeRight*0.5*basis.PhiRight'-basis.PhiTildeLeft*0.5*basis.PhiLeft';
        L1kP = basis.PhiTildeRight*0.5*basis.PhiLeft';
        L1kM = basis.PhiTildeLeft*0.5*basis.PhiRight';

        LInt = basis.DPhiTildeQuad*diagm(q.w)*basis.PhiQuad*(settings.dx)*0.5;

        I_idx = zeros(3*(nx-2)*m^2); J_idx = zeros(3*(nx-2)*m^2); vals = zeros(3*(nx-2)*m^2);
        counter = -2;

        for k = 2:nx-1
            for i = 1:m
                index = vectorIndex(m,k,i);
                for j = 1:m
                    counter += 3;
                    indexJ = vectorIndex.(m,k,j);
                    indexJPlus = vectorIndex.(m,k+1,j);
                    indexJMinus = vectorIndex.(m,k-1,j);

                    I_idx[counter+1] = index;
                    J_idx[counter+1] = indexJ;
                    vals[counter+1] = L1k[i,j];
                    if k > 1
                        I_idx[counter] = index;
                        J_idx[counter] = indexJMinus;
                        vals[counter] = L1kM[i,j];
                    end
                    if k < nx
                        I_idx[counter+2] = index;
                        J_idx[counter+2] = indexJPlus;
                        vals[counter+2] = L1kP[i,j];
                    end
                end
            end
        end
        L1x = sparse(I_idx,J_idx,vals,nx*m,nx*m);

        I_idx = zeros(3*(nx-2)*m^2); J_idx = zeros(3*(nx-2)*m^2); vals = zeros(3*(nx-2)*m^2);
        counter = -2;

        for k = 2:nx-1
            for i = 1:m
                index = vectorIndex(m,k,i);
                for j = 1:m
                    counter += 3;
                    indexJ = vectorIndex(m,k,j);
                    indexJPlus = vectorIndex(m,k+1,j);
                    indexJMinus = vectorIndex(m,k-1,j);

                    I_idx[counter+1] = index;
                    J_idx[counter+1] = indexJ;
                    vals[counter+1] = L2k[i,j] + LInt[i,j];
                    if k > 1
                        I_idx[counter] = index;
                        J_idx[counter] = indexJMinus;
                        vals[counter] = L2kM[i,j];
                    end
                    if k < nx
                        I_idx[counter+2] = index;
                        J_idx[counter+2] = indexJPlus;
                        vals[counter+2] = L2kP[i,j];
                    end
                end
            end
        end
        L2x = sparse(I_idx,J_idx,vals,nx*m,nx*m);

        I_idx = zeros(nx*m*Nq); J_idx = zeros(nx*m*Nq); vals = zeros(nx*m*Nq);
        vals2 = zeros(nx*m*Nq);
        counter = 0;

        for k = 1:nx
            for i = 1:Nq
                index = vectorIndex(Nq,k,i);
                for j = 1:m
                    counter += 1;
                    indexJ = vectorIndex(m,k,j);
                    I_idx[counter] = index;
                    J_idx[counter] = indexJ;
                    vals[counter] = basis.PhiQuad[i,j];
                    vals2[counter] = basis.PhiTildeWQuad[i,j]*(settings.dx)*0.5;
                end
            end
        end
        QP = sparse(I_idx,J_idx,vals,nx*Nq,nx*m);
        PQ = sparse(J_idx,I_idx,vals2,nx*m,nx*Nq);

        xQ::Array{Float64,1} = [];
        for k = 1:nx # leave out right ghost cells   
            xQ=[xQ; ((x[k].*(1.0.-q.xi)+x[k+1].*(1.0.+q.xi))./2.0)[end:-1:1]];
        end

        new(x,xQ,q,settings,basis,time,outRhs,outRhsT,cp,c01,gamma,A,AbsA,settings.sigmaT,settings.sigmaS,settings.sigmaA,c,sigmaSB,aRad,L1x,L2x,PQ,QP);
    end
end

# source term depends only on x now
function Q(obj::DGSolver,t::Float64,x::Array{Float64,1})
    Nq = length(x);
    y = zeros(Nq,obj.settings.nPN)
    for k = 1:Nq
        if obj.settings.problem == "ManufacturedSolution"
            y[k,1] = 2*exp(t)*cos.(x[k]);
            y[k,2] = -2/3 * exp(t)*sin.(x[k]);
        elseif obj.settings.problem == "ManufacturedSolutionLinear"
            y[k,1] = 2.0*cos.(x[k]);
            y[k,2] = -2/3 * (t+1.0)*sin.(x[k]);
        elseif obj.settings.problem == "ManufacturedSolutionSteady"
            y[k,2] = -2/3 * sin.(x[k]);
        end
    end
    return y;
end

# system flux for PN
function f(obj::DGSolver,u::Array{Float64,2})
    B = zeros(obj.settings.Nq,obj.settings.nPN) 
    for l = 1:obj.settings.Nq
        B[l,:] = obj.A*u[l,:];
    end
    return B;
end

# Lax-Friedrichs flux for PN
function numFlux(obj::DGSolver,ul::Array{Float64,1},ur::Array{Float64,1})
    #return 0.5*(obj.A*(ul+ur).-(obj.settings.dx/obj.settings.dt).*(ur-ul)/obj.c);
    return 0.5*(obj.A*(ul+ur).-obj.AbsA*(ur-ul));
end

function minmod(a,b,c)
    if sign(a)==sign(b) && sign(b) == sign(c)
        return sign(a)*minimum([abs(a),abs(b),abs(c)]);
    else
        return 0.0;
    end
end

function modMinmod(obj::DGSolver,a::Float64,b::Float64,c::Float64)
    H = obj.settings.dx;
    h = obj.settings.dx;
    M2 = 5000.0; # M2 > |\partial_xx u| 50
    alpha = H/(4*h)*(obj.cp+2*(obj.c01+2));
    beta = 2*alpha+0.5*obj.c01+1;
    Mg = 0.5*M2*(alpha+0.25*obj.cp);
    M = H^2/(H^2+abs(b)+abs(c))*Mg*(1+M2*beta);
    if abs(a) <= M*H^2
        return a;
    else
        return minmod(a,b,c);
    end
end

function vanLeerSlope(obj::DGSolver,ukBarP,ukM0,uk0,ukP0)
    DeltaP = ukP0 - uk0;
    DeltaM = uk0 - ukM0;
    #return minmod(ukBarP,DeltaP,DeltaM); # van Leer
    return modMinmod(obj,ukBarP,DeltaP,DeltaM); # Shu
    #return ukBarP;
    #return minmod(ukBarP,-1,1); # make sure limiter is switched off
    #return 0.0;
end

function SetupIC(obj::DGSolver)
    u = zeros(obj.settings.NCells,obj.settings.N,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    T = zeros(obj.settings.NCells,obj.settings.N);
    TVal = 80.0 * 11604.0;
    for j = 1:obj.settings.NCells
        if j == 1
            TVal = 80.0 * 11604.0;
        else
            TVal = 0.02 * 11604.0;
        end
        for i = 1:obj.settings.N
            T[j,i] = Integral(obj.q, xVal->TVal.*PhiTilde(obj.basis,j,i-1,xVal),obj.x[j],obj.x[j+1]);
            u[j,i,1] = 0.5*2.0*Integral(obj.q, xVal->obj.aRad*obj.c/4/pi .* TVal^4 .*PhiTilde(obj.basis,j,i-1,xVal),obj.x[j],obj.x[j+1]); # 2.0 * ... might be wrong here -> corrected by multiplying 0.5
        end
    end
    return Mat2Vec(u),Mat2Vec(T);
end

function RhsNoMatrix(obj::DGSolver,u::Array{Float64,3},T::Array{Float64,2},t::Float64=0.0)   
    intTerm = zeros(obj.settings.N,obj.settings.nPN);
    for k = 2:(obj.settings.NCells-1) # leave out ghost cells
        # evaluate integral term of right-hand-side
        A = f(obj,EvalAtQuad(obj.basis,u[k,:,:]));
        for i = 1:obj.settings.nPN
            for l = 1:obj.settings.Nq
                A[l,i] = A[l,i]*obj.q.w[l]*(obj.settings.dx)*0.5;
            end
        end
        intTerm .= obj.basis.DPhiTildeQuad*A

        xiQuad=(obj.x[k].*(1.0.-obj.q.xi)+obj.x[k+1].*(1.0.+obj.q.xi))./2.0;
        sourceTerm = obj.basis.PhiTildeWQuad'*Q(obj,t,xiQuad)*(obj.settings.dx)*0.5;

        # evaluate interface k (left interface of cell k)
        ukMRight = EvalRight(obj.basis,u[k-1,:,:]);
        ukLeft = EvalLeft(obj.basis,u[k,:,:]);

        # evaluate interface k+1 (right interface of cell k)
        ukRight = EvalRight(obj.basis,u[k,:,:]);
        ukPLeft = EvalLeft(obj.basis,u[k+1,:,:]);

        # compute coupling with temperature
        TContribution = obj.basis.PhiTildeWQuad'*EvalAtQuad(obj.basis,T[k,:]).^4*(obj.settings.dx)*0.5;


        obj.outRhs[k,:,:] = -(obj.basis.PhiTildeRight*numFlux(obj,ukRight,ukPLeft)'-obj.basis.PhiTildeLeft*numFlux(obj,ukMRight,ukLeft)')+intTerm;
        obj.outRhs[k,:,:] = obj.outRhs[k,:,:] - obj.sigmaT*u[k,:,:]; # add out-scattering and absorption contribution
        obj.outRhs[k,:,1] = obj.outRhs[k,:,1] + obj.sigmaS*u[k,:,1]; # add in-scattering contribution
        obj.outRhs[k,:,1] = obj.outRhs[k,:,1] + obj.sigmaA*obj.aRad*obj.c/4/pi * TContribution; # add temperature contribution
        obj.outRhs[k,:,:] = obj.outRhs[k,:,:] + sourceTerm; # add source contribution
    end

    # Apply Source in left boundary cell
    xiQuad=(obj.x[1].*(1.0.-obj.q.xi)+obj.x[1+1].*(1.0.+obj.q.xi))./2.0;
    sourceTerm = obj.basis.PhiTildeWQuad'*Q(obj,t,xiQuad)*(obj.settings.dx)*0.5;
    TContribution = obj.basis.PhiTildeWQuad'*EvalAtQuad(obj.basis,T[1,:]).^4*(obj.settings.dx)*0.5;
    obj.outRhs[1,:,:] = obj.outRhs[1,:,:] - obj.sigmaT*u[1,:,:]; # add out-scattering and absorption contribution
    obj.outRhs[1,:,1] = obj.outRhs[1,:,1] + obj.sigmaS*u[1,:,1]; # add in-scattering contribution
    obj.outRhs[1,:,1] = obj.outRhs[1,:,1] + obj.sigmaA*obj.aRad*obj.c/4/pi * TContribution; # add temperature contribution
    obj.outRhs[1,:,:] = obj.outRhs[1,:,:] + sourceTerm; # add source contribution
    return obj.outRhs*obj.c;
end

function Rhs(obj::DGSolver,u::Array{Float64,2},T::Array{Float64,1},t::Float64=0.0)   
    obj.outRhs .= obj.L2x*u*obj.A' +  obj.L1x*u*obj.AbsA' - obj.sigmaT*u + obj.PQ*Q(obj,t,obj.xQ);
    obj.outRhs[:,1] .+= obj.sigmaS*u[:,1] + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4);
    return obj.outRhs*obj.c;
end

function RhsT(obj::DGSolver,u::Array{Float64,2},T::Array{Float64,1},t::Float64=0.0)   
    return obj.sigmaA*(4*pi*u[:,1]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;
end

function SolveNoMatrix(obj::DGSolver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,:,l] = LimitCoefficients(obj,u[:,:,l]);
    end
    #u = Mat2Vec(u)
    
    Nt = round(tEnd/dt);
    
    # time loop
    @showprogress 0.1 "Progress " for n = 1:Nt

        # Update time by dt
        #u,T = Update(obj.time,obj,u,T,t);
        u,T = Update(obj.time,obj,u,T);
        
        t = t+dt;
    end

    # return end time and solution
    return t, u, T;

end

function Solve(obj::DGSolver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end
    
    Nt = round(tEnd/dt);
    
    # time loop
    @showprogress 0.1 "Progress " for n = 1:Nt

        # Update time by dt
        #u,T = Update(obj.time,obj,u,T,t);


        u,T = Update(obj.time,obj,u,T);
        
        t = t+dt;
    end

    # return end time and solution
    return t, u, T;

end

function SolveNaiveUnconventional(obj::DGSolver)
    # Get rank
    r=obj.settings.r;
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    for n=1:Nt

        ################## K-step ##################
        K .= X*S;

        u .= K*W';
        y .= obj.L2x*u*obj.A' +  obj.L1x*u*obj.AbsA' - obj.sigmaT*u + obj.PQ*Q(obj,t,obj.xQ);
        y[:,1] .+= obj.sigmaS*u[:,1] + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4);

        K .= K .+ dt*obj.c*y*W;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        XNew,STmp = qr(K);
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';
        
        u .= X*L';
        y .= obj.L2x*u*obj.A' +  obj.L1x*u*obj.AbsA' - obj.sigmaT*u + obj.PQ*Q(obj,t,obj.xQ);
        y[:,1] .+= obj.sigmaS*u[:,1] + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4);

        L .= L .+ dt*obj.c*y'*X;    
                
        WNew,STmp = qr(L);
        WNew = WNew[:,1:r];

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        u .= X*S*W';
        y .= obj.L2x*u*obj.A' +  obj.L1x*u*obj.AbsA' - obj.sigmaT*u + obj.PQ*Q(obj,t,obj.xQ);
        y[:,1] .+= obj.sigmaS*u[:,1] + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4);

        S .= S .+ dt*obj.c*X'*y*W;

        # update temperature
        T .= T + dt*RhsT(obj,X*S*W',T,t);

        next!(prog) # update progress bar
        t += dt;
               
    end

    # return end time and solution
    return t, X*S*W', T;

end

function SolveUnconventional(obj::DGSolver)
    # Get rank
    r=obj.settings.r;
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    yK = zeros(obj.settings.NCells*obj.settings.N,r)
    yL = zeros(obj.settings.nPN,r)
    yS = zeros(r,r)
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    for n=1:Nt

        ################## K-step ##################
        K .= X*S;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        yK .= obj.L2x*K*WAW +  obj.L1x*K*WAbsAW - obj.sigmaT*K + obj.PQ*Q(obj,t,obj.xQ)*W;
        yK .+= obj.sigmaS*K*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        XNew,STmp = qr(K);
        XNew = Matrix(XNew)
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';
        XL2X = X'*obj.L2x'*X;
        XL1X = X'*obj.L1x'*X;
        
        yL .= obj.A*L*XL2X +  obj.AbsA*L*XL1X - obj.sigmaT*L + Q(obj,t,obj.xQ)'*obj.PQ'*X;
        yL[1,:] .+= obj.sigmaS*L[1,:] .+ obj.sigmaA*obj.aRad*obj.c/4/pi*vec(((obj.QP*T).^4)'obj.PQ'*X);

        L .= L .+ dt*obj.c*yL;    
                
        WNew,STmp = qr(L);
        WNew = Matrix(WNew)
        WNew = WNew[:,1:r];

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        XL2X = X'*obj.L2x*X;
        XL1X = X'*obj.L1x*X;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        u .= X*S*W';
        yS .= XL2X*S*WAW +  XL1X*S*WAbsAW - obj.sigmaT*S + X'*obj.PQ*Q(obj,t,obj.xQ)*W;
        yS .+= obj.sigmaS*S*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*X'*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        S .= S .+ dt*obj.c*yS;

        # update temperature
        T .= T + dt*obj.sigmaA*(4*pi*X*S*W[1,:]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;

        next!(prog) # update progress bar
        t += dt;
               
    end

    # return end time and solution
    return t, X*S*W', T;

end

function SolveUnconventionalAdaptive(obj::DGSolver)
    # Get rank
    r=2;
    rmin = 2;
    rMaxTotal = Int(floor(obj.settings.r/2));

    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    yK = zeros(obj.settings.NCells*obj.settings.N,r)
    yL = zeros(obj.settings.nPN,r)
    yS = zeros(r,r)
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    rankInTime = zeros(2,Nt);

    for n=1:Nt

        ################## K-step ##################
        K = X*S;
        KOld = X*S;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        yK = obj.L2x*K*WAW +  obj.L1x*K*WAbsAW - obj.sigmaT*K + obj.PQ*Q(obj,t,obj.xQ)*W;
        yK .+= obj.sigmaS*K*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
            KOld[:,l] = LimitCoefficients(obj,KOld[:,l]);
        end

        XNew,STmp = qr([K KOld]);
        XNew = Matrix(XNew)
        XNew = XNew[:,1:2*r];

        MUp = XNew' * X;

        ################## L-step ##################
        L = W*S';
        XL2X = X'*obj.L2x'*X;
        XL1X = X'*obj.L1x'*X;
        
        yL = obj.A*L*XL2X +  obj.AbsA*L*XL1X - obj.sigmaT*L + Q(obj,t,obj.xQ)'*obj.PQ'*X;
        yL[1,:] .+= obj.sigmaS*L[1,:] .+ obj.sigmaA*obj.aRad*obj.c/4/pi*vec(((obj.QP*T).^4)'obj.PQ'*X);

        L .= L .+ dt*obj.c*yL;    
                
        WNew,STmp = qr([L W]);
        WNew = Matrix(WNew)
        WNew = WNew[:,1:2*r];

        NUp = WNew' * W;
        W = WNew;
        X = XNew;

        ################## S-step ##################
        S = MUp*S*(NUp')

        XL2X = Matrix(X'*obj.L2x*X);
        XL1X = Matrix(X'*obj.L1x*X);
        WAW = Matrix(W'*obj.A'*W);
        WAbsAW = Matrix(W'*obj.AbsA'*W);

        yS = XL2X*S*WAW +  XL1X*S*WAbsAW - obj.sigmaT*S + X'*obj.PQ*Q(obj,t,obj.xQ)*W;
        yS .+= obj.sigmaS*S*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*X'*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        S .= S .+ dt*obj.c*yS;

        ################## truncate ##################

        # Compute singular values of S1 and decide how to truncate:
        U,D,V = svd(Matrix(S));
        U = Matrix(U); V = Matrix(V)
        rmaxTmp = -1;
        S .= zeros(size(S));

        tmp = 0.0;
        tol = obj.settings.epsAdapt*norm(D)^obj.settings.adaptIndex;
        
        rmax = Int(floor(size(D,1)/2));
        for j=1:2*rmax
            tmp = sqrt(sum(D[j:2*rmax]).^2);
            
            if tmp < tol
                rmaxTmp = j;
                break;
            end
        end
        rmax = rmaxTmp;

        # if 2*r was actually not enough move to highest possible rank
        if rmax == -1
            rmax = rMaxTotal;
        end

        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,rmin);
        
        for l = 1:rmax
            S[l,l] = D[l];
        end



        # update solution with new rank
        XNew = XNew*U;
        WNew = WNew*V;

        # update solution with new rank
        S = S[1:rmax,1:rmax];
        X = XNew[:,1:rmax];
        W = WNew[:,1:rmax];

        # update rank
        r = rmax;

        # update temperature
        T .= T + dt*obj.sigmaA*(4*pi*X*S*W[1,:]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;

        rankInTime[1,n] = t;
        rankInTime[2,n] = r;

        next!(prog) # update progress bar
        t += dt;

        #println(r)
               
    end

    # return end time and solution
    return t, X*S*W', T,rankInTime;

end

# Update function time dependent
function Update(obj::TimeSolver,DG::DGSolver,u::Array{Float64,3},T::Array{Float64,2},t::Float64)
    NCells = DG.settings.NCells;
    obj.uRK[1,:,:,:] = u;
    obj.TRK[1,:,:] = T;
    for s = 1:obj.N
        obj.uRK[s+1,:,:,:] = zeros(obj.Nx-1,obj.N,DG.settings.nPN);
        obj.TRK[s+1,:,:] = zeros(obj.Nx-1,obj.N);
        for j = 1:s
            # skip if beta and alpha are zero
            if abs(obj.beta[s,j]) < 1e-10 && abs(obj.alpha[s,j]) < 1e-10
                continue;
            end
            obj.uRK[s+1,:,:,:] = obj.uRK[s+1,:,:,:]+obj.alpha[s,j].*obj.uRK[j,:,:,:]+obj.dt*obj.beta[s,j].*Rhs(DG,obj.uRK[j,:,:,:],obj.TRK[j,:,:],t+obj.dt*obj.gamma[s]);
            obj.TRK[s+1,:,:] = obj.TRK[s+1,:,:]+obj.alpha[s,j].*obj.TRK[j,:,:]+obj.dt*obj.beta[s,j].*RhsT(DG,obj.uRK[j,:,:,:],obj.TRK[j,:,:],t+obj.dt*obj.gamma[s]);

            for l = 1:DG.settings.nPN
                obj.uRK[s+1,:,:,l] = LimitCoefficients(DG,obj.uRK[s+1,:,:,l]);
            end

            # boundary cells with periodic BCs
            if DG.settings.BCType == "sin"
                obj.uRK[s+1,1,:,:] = obj.uRK[s+1,NCells-1,:,:];
                obj.uRK[s+1,NCells,:,:] = obj.uRK[s+1,2,:,:];
            elseif DG.settings.BCType == "exact"
                for i = 1:DG.settings.N
                    obj.uRK[s+1,1,i,1] = Integral(DG.q, xVal->ExactSolution(t+obj.dt*obj.gamma[s],xVal,DG.settings.problem).*PhiTilde(DG.basis,1,i-1,xVal),DG.x[1],DG.x[2]);
                    obj.uRK[s+1,NCells,i,1] = Integral(DG.q, xVal->ExactSolution(t+obj.dt*obj.gamma[s],xVal,DG.settings.problem).*PhiTilde(DG.basis,NCells,i-1,xVal),DG.x[end-1],DG.x[end]);
                end
            end
        end
    end

    return obj.uRK[obj.N+1,:,:,:],obj.TRK[obj.N+1,:,:]; # do I need deepcopy here?
end

function SolveProjectorSplitting(obj::DGSolver)
    # Get rank
    r=obj.settings.r;
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    yK = zeros(obj.settings.NCells*obj.settings.N,r)
    yL = zeros(obj.settings.nPN,r)
    yS = zeros(r,r)
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    for n=1:Nt

        ################## K-step ##################
        K .= X*S;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        yK .= obj.L2x*K*WAW +  obj.L1x*K*WAbsAW - obj.sigmaT*K + obj.PQ*Q(obj,t,obj.xQ)*W;
        yK .+= obj.sigmaS*K*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];

        ################## S-step ##################
        XL2X = X'*obj.L2x*X;
        XL1X = X'*obj.L1x*X;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        u .= X*S*W';
        yS .= XL2X*S*WAW + XL1X*S*WAbsAW - obj.sigmaT*S + X'*obj.PQ*Q(obj,t,obj.xQ)*W;
        yS .+= obj.sigmaS*S*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*X'*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        S .= S .- dt*obj.c*yS;

        ################## L-step ##################
        L = W*S';
        XL2X = X'*obj.L2x'*X;
        XL1X = X'*obj.L1x'*X;
        
        yL .= obj.A*L*XL2X +  obj.AbsA*L*XL1X - obj.sigmaT*L + Q(obj,t,obj.xQ)'*obj.PQ'*X;
        yL[1,:] .+= obj.sigmaS*L[1,:] .+ obj.sigmaA*obj.aRad*obj.c/4/pi*vec(((obj.QP*T).^4)'obj.PQ'*X);

        L .= L .+ dt*obj.c*yL;    
                
        W,S = qr(L);
        W = Matrix(W)
        W = W[:,1:r];
        S .= S';

        # update temperature
        T .= T + dt*obj.sigmaA*(4*pi*X*S*W[1,:]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;

        next!(prog) # update progress bar
        t += dt;
               
    end

    # return end time and solution
    return t, X*S*W', T;

end

function SolveProjectorSplittingFixS(obj::DGSolver)
    # Get rank
    r=obj.settings.r;
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    yK = zeros(obj.settings.NCells*obj.settings.N,r)
    yL = zeros(obj.settings.nPN,r)
    yS = zeros(r,r)
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    for n=1:Nt

        ################## K-step ##################
        K .= X*S;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        yK .= obj.L2x*K*WAW +  obj.L1x*K*WAbsAW - obj.sigmaT*K + obj.PQ*Q(obj,t,obj.xQ)*W;
        yK .+= obj.sigmaS*K*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];

        ################## S-step ##################
        XL2X = X'*obj.L2x*X;
        XL1X = 0.0*X'*obj.L1x*X; 
        WAW = W'*obj.A'*W;
        WAbsAW = -W'*obj.AbsA'*W; # choose "correct sign" for stabilization

        u .= X*S*W';
        yS .= XL2X*S*WAW + XL1X*S*WAbsAW - obj.sigmaT*S + X'*obj.PQ*Q(obj,t,obj.xQ)*W;
        yS .+= obj.sigmaS*S*W[1,:]*W[1,:]' + obj.sigmaA*obj.aRad*obj.c/4/pi*X'*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        S .= S .- dt*obj.c*yS;

        ################## L-step ##################
        L = W*S';
        XL2X = X'*obj.L2x'*X;
        XL1X = 0.0*X'*obj.L1x'*X;
        
        yL .= obj.A*L*XL2X +  obj.AbsA*L*XL1X - obj.sigmaT*L + Q(obj,t,obj.xQ)'*obj.PQ'*X;
        yL[1,:] .+= obj.sigmaS*L[1,:] .+ obj.sigmaA*obj.aRad*obj.c/4/pi*vec(((obj.QP*T).^4)'obj.PQ'*X);

        L .= L .+ dt*obj.c*yL;    
                
        W,S = qr(L);
        W = Matrix(W)
        W = W[:,1:r];
        S .= S';

        # update temperature
        T .= T + dt*obj.sigmaA*(4*pi*X*S*W[1,:]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;

        next!(prog) # update progress bar
        t += dt;
               
    end

    # return end time and solution
    return t, X*S*W', T;

end

function SolveProjectorSplitting2(obj::DGSolver)
    # Get rank
    r=obj.settings.r;
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    yK = zeros(obj.settings.NCells*obj.settings.N,r)
    yL = zeros(obj.settings.nPN,r)
    yS = zeros(r,r)
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    for n=1:Nt

        ################## K-step ##################
        K .= X*S;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        yK .= obj.L2x*K*WAW +  obj.L1x*K*WAbsAW + obj.PQ*Q(obj,t,obj.xQ)*W;

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];

        # update temperature
        T .= T + dt*obj.sigmaA*(4*pi*X*S*W[1,:]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;

        ################## S-step ##################
        XL2X = X'*obj.L2x*X;
        XL1X = X'*obj.L1x*X;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        u .= X*S*W';
        yS .= XL2X*S*WAW + XL1X*S*WAbsAW + X'*obj.PQ*Q(obj,t,obj.xQ)*W;

        S .= S .- dt*obj.c*yS;

        ################## L-step ##################
        L = W*S';
        XL2X = X'*obj.L2x'*X;
        XL1X = X'*obj.L1x'*X;
        
        yL .= obj.A*L*XL2X +  obj.AbsA*L*XL1X - obj.sigmaT*L + Q(obj,t,obj.xQ)'*obj.PQ'*X;

        L .= L .+ dt*obj.c*yL;    

        ################## L-step scattering ##################

        yL .= - obj.sigmaT*L
        yL[1,:] .+= obj.sigmaS*L[1,:];

        L .= L .+ dt*obj.c*yL;    
                
        W,S = qr(L);
        W = Matrix(W)
        W = W[:,1:r];
        S .= S';

        K = X*S;
        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end
        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];

        ################## K-step Temperature ##################
        yK = obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];
        next!(prog) # update progress bar
        t += dt;
               
    end

    # return end time and solution
    return t, X*S*W', T;

end

function SolveProjectorSplitting2FixS(obj::DGSolver)
    # Get rank
    r=obj.settings.r;
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    for l = 1:obj.settings.nPN
        u[:,l] = LimitCoefficients(obj,u[:,l]);
    end

    # Low-rank approx of init data:
    X,S,W = svd(u);

    y = zeros(size(u));
    yK = zeros(obj.settings.NCells*obj.settings.N,r)
    yL = zeros(obj.settings.nPN,r)
    yS = zeros(r,r)
    
    # rank-r truncation:
    X = X[:,1:r];
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r];
    K = zeros(size(X));

    Nt = Int(round(tEnd/dt));

    prog = Progress(Nt,1)

    for n=1:Nt

        ################## K-step ##################
        K .= X*S;
        WAW = W'*obj.A'*W;
        WAbsAW = W'*obj.AbsA'*W;

        yK .= obj.L2x*K*WAW +  obj.L1x*K*WAbsAW + obj.PQ*Q(obj,t,obj.xQ)*W;

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];

        # update temperature
        T .= T + dt*obj.sigmaA*(4*pi*X*S*W[1,:]-obj.aRad*obj.c*obj.PQ*(obj.QP*T).^4)/obj.settings.cV;

        ################## S-step ##################
        XL2X = X'*obj.L2x*X;
        XL1X = X'*obj.L1x*X;
        WAW = W'*obj.A'*W;
        WAbsAW = -W'*obj.AbsA'*W;

        u .= X*S*W';
        yS .= XL2X*S*WAW + XL1X*S*WAbsAW + X'*obj.PQ*Q(obj,t,obj.xQ)*W;

        S .= S .- dt*obj.c*yS;

        ################## L-step ##################
        L = W*S';
        XL2X = X'*obj.L2x'*X;
        XL1X = X'*obj.L1x'*X;
        
        yL .= obj.A*L*XL2X +  obj.AbsA*L*XL1X - obj.sigmaT*L + Q(obj,t,obj.xQ)'*obj.PQ'*X;

        L .= L .+ dt*obj.c*yL;    

        ################## L-step scattering ##################

        yL .= - obj.sigmaT*L
        yL[1,:] .+= obj.sigmaS*L[1,:];

        L .= L .+ dt*obj.c*yL;    
                
        W,S = qr(L);
        W = Matrix(W)
        W = W[:,1:r];
        S .= S';

        K = X*S;
        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end
        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];

        ################## K-step Temperature ##################
        yK = obj.sigmaA*obj.aRad*obj.c/4/pi*obj.PQ*((obj.QP*T).^4)*W[1,:]';

        K .= K .+ dt*obj.c*yK;

        # limit coefficients
        for l = 1:r
            K[:,l] = LimitCoefficients(obj,K[:,l]);
        end

        X,S = qr(K);
        X = Matrix(X)
        X = X[:,1:r];
        next!(prog) # update progress bar
        t += dt;
               
    end

    # return end time and solution
    return t, X*S*W', T;

end

# Update function
function UpdateNoMatrix(obj::TimeSolver,DG::DGSolver,u::Array{Float64,3},T::Array{Float64,2})
    NCells = DG.settings.NCells;
    obj.uRK[1,:,:,:] = u;
    obj.TRK[1,:,:] = T;
    for s = 1:obj.N
        obj.R[s,:,:,:]=Rhs(DG,obj.uRK[s,:,:,:],obj.TRK[s,:,:]);
        obj.RT[s,:,:]=RhsT(DG,obj.uRK[s,:,:,:],obj.TRK[s,:,:]);
        obj.uRK[s+1,:,:,:] = zeros(obj.Nx-1,obj.N,DG.settings.nPN);
        obj.TRK[s+1,:,:] = zeros(obj.Nx-1,obj.N);
        for j = 1:s
            # skip if beta and alpha are zero
            if abs(obj.beta[s,j]) < 1e-10 && abs(obj.alpha[s,j]) < 1e-10
                continue;
            end
            obj.uRK[s+1,:,:,:] = obj.uRK[s+1,:,:,:]+obj.alpha[s,j].*obj.uRK[j,:,:,:]+obj.dt*obj.beta[s,j].*obj.R[j,:,:,:];
            obj.TRK[s+1,:,:] = obj.TRK[s+1,:,:]+obj.alpha[s,j].*obj.TRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.RT[j,:,:];
            
            for l = 1:DG.settings.nPN
                obj.uRK[s+1,:,:,l] = LimitCoefficients(DG,obj.uRK[s+1,:,:,l]);
            end

            # boundary cells with periodic BCs
            if DG.settings.BCType == "sin"
                obj.uRK[s+1,1,:,:] = obj.uRK[s+1,NCells-1,:,:];
                obj.uRK[s+1,NCells,:,:] = obj.uRK[s+1,2,:,:];
            elseif DG.settings.BCType == "exact"
                for i = 1:DG.settings.N
                    obj.uRK[s+1,1,i,1] = Integral(DG.q, xVal->ExactSolution(t+obj.dt*obj.gamma[s],xVal,DG.settings.problem).*PhiTilde(DG.basis,1,i-1,xVal),DG.x[1],DG.x[2]);
                    obj.uRK[s+1,NCells,i,1] = Integral(DG.q, xVal->ExactSolution(t+obj.dt*obj.gamma[s],xVal,DG.settings.problem).*PhiTilde(DG.basis,NCells,i-1,xVal),DG.x[end-1],DG.x[end]);
                end
            end
        end
    end

    return obj.uRK[obj.N+1,:,:,:],obj.TRK[obj.N+1,:,:]; # do I need deepcopy here?
end

# Update function
function Update(obj::TimeSolver,DG::DGSolver,u::Array{Float64,2},T::Array{Float64,1})
    NCells = DG.settings.NCells;
    obj.uRK[1,:,:] = u;
    obj.TRK[1,:] = T;
    for s = 1:obj.N
        obj.R[s,:,:]=Rhs(DG,obj.uRK[s,:,:],obj.TRK[s,:]);
        obj.RT[s,:]=RhsT(DG,obj.uRK[s,:,:],obj.TRK[s,:]);
        obj.uRK[s+1,:,:] = zeros((obj.Nx-1)*obj.N,DG.settings.nPN);
        obj.TRK[s+1,:] = zeros(obj.Nx-1,obj.N);
        for j = 1:s
            # skip if beta and alpha are zero
            if abs(obj.beta[s,j]) < 1e-10 && abs(obj.alpha[s,j]) < 1e-10
                continue;
            end
            obj.uRK[s+1,:,:] = obj.uRK[s+1,:,:]+obj.alpha[s,j].*obj.uRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.R[j,:,:];
            obj.TRK[s+1,:] = obj.TRK[s+1,:]+obj.alpha[s,j].*obj.TRK[j,:]+obj.dt*obj.beta[s,j].*obj.RT[j,:];
            
            for l = 1:DG.settings.nPN
                obj.uRK[s+1,:,l] = LimitCoefficients(DG,obj.uRK[s+1,:,l]);
            end

            # boundary cells with periodic BCs
            if DG.settings.BCType == "sin"
                obj.uRK[s+1,1,:,:] = obj.uRK[s+1,NCells-1,:,:];
                obj.uRK[s+1,NCells,:,:] = obj.uRK[s+1,2,:,:];
            elseif DG.settings.BCType == "exact"
                for i = 1:DG.settings.N
                    obj.uRK[s+1,1,i,1] = Integral(DG.q, xVal->ExactSolution(t+obj.dt*obj.gamma[s],xVal,DG.settings.problem).*PhiTilde(DG.basis,1,i-1,xVal),DG.x[1],DG.x[2]);
                    obj.uRK[s+1,NCells,i,1] = Integral(DG.q, xVal->ExactSolution(t+obj.dt*obj.gamma[s],xVal,DG.settings.problem).*PhiTilde(DG.basis,NCells,i-1,xVal),DG.x[end-1],DG.x[end]);
                end
            end
        end
    end

    return obj.uRK[obj.N+1,:,:],obj.TRK[obj.N+1,:]; # do I need deepcopy here?
end

# returns limited coefficients
function LimitCoefficients(obj::DGSolver,u::Array{Float64,1})
    P = obj.settings.N-1;
    u = Vec2Mat(u,obj.settings.NCells,obj.settings.N);
    y = u;

    if P == 0
        return y;
    end

    
    NCells = obj.settings.NCells;
    # polynomial coefficients
    alpha = zeros(P+1,P+1);
    beta = zeros(P+1,P+1);
    
    alpha[1,1] = 1;
    if P > 0
        alpha[2,1] = 0; alpha[2,2] = 1;
    end
    if P > 1
        alpha[3,1] = -0.5; alpha[3,2] = 0; alpha[3,3] = 1.5;
    end

    for i = 0:P
        for l = 0:P
            beta[i+1,l+1] = sqrt(2*i+1)*alpha[i+1,l+1]*2^l;
        end
    end

    if P == 2
        A = zeros(P,P);
        A[1,1] = 0.5*beta[2,2];
        A[1,2] = beta[3,3]/4+beta[3,1];
        A[2,1] = 0.5*beta[2,2];
        A[2,2] = -beta[3,3]/4-beta[3,1];
    end

    for k = 2:(NCells-1) # leave out ghost cells
        # compute limited interface values at cell k
        # evaluate interface k (left interface of cell k)
        uKTildePlus = vanLeerSlope(obj,-EvalBarLeft(obj.basis,u[k,:]),u[k-1,1],u[k,1],u[k+1,1]);
        # evaluate interface k+1 (right interface of cell k)
        uKTildeMinus = vanLeerSlope(obj,EvalBarRight(obj.basis,u[k,:]),u[k-1,1],u[k,1],u[k+1,1]);

        # compute limited coefficients according to DG script Section 6.4
        if P == 1
            y[k,1] = u[k,1];
            y[k,2] = uKTildePlus/sqrt(3);
        elseif P == 2
            y[k,1] = u[k,1];
            y[k,2:3] = A\[uKTildeMinus; uKTildePlus];
        end
    end
    return Mat2Vec(y);
end

function tenToMat(u::Array{Float64,3})
    Nx = size(u,1);
    Np = size(u,2);
    r = size(u,3);
    
    y = u[1,:,:];
    for i=2:Nx
        
        tmp = u[i,:,:];
        y = [y; tmp];
    end

    return y
end

function matToTen(v::Array{Float64,2}, s::Settings)
    Nx = s.NCells;
    Np = s.N; 
    r = size(v,2);

    y = zeros(Nx,Np,r);

    y[1, : ,:] = v[1:Np, 1:r];
    for i=2:Nx
        y[i,:,:] = v[ (i-1)*Np+1:i*Np, 1:r];
    end

    return y
end

function RK_SSP(obj::DGSolver, fun, y0, dt)
    N = obj.settings.N;

    if N==1
        y1 = y0 +dt*fun(y0);

    elseif N==2
        k1 = y0 +dt*fun(y0);
        y1 = 0.5*y0 + 0.5*(k1+dt*fun(k1));

    elseif N==3
        k1 =  y0 +dt*fun(y0);
        k2 =  (1/4)*(3*y0 + k1 +dt*fun(k1));    
        y1 =  1/3*(y0 +2*k2 +2*dt*fun(k2));
    end

    return y1
end

function Limiter(obj, u)
    u = matToTen(u, obj.settings)

    r = size(u);
    r = r[end];

    for l = 1:r
        u[:,:,l] = LimitCoefficients(obj,u[:,:,l]);
    end

    return tenToMat(u);
end

function vectorIndex(nx,i,j)
    return (i-1)*nx + j;
end

function vectorIndex(n,m,i,j,k)
    return (i-1)*n*m + vectorIndex(m,j,k)
end

function Mat2Vec(mat::Array{Float64,2})
    n = size(mat,1)
    m = size(mat,2)
    v = zeros(n*m);
    for i = 1:n
        for j = 1:m
            v[(i-1)*m + j] = mat[i,j]
        end
    end
    return v;
end

function Vec2Mat(v::Array{Float64,1},n,m)
    mat = zeros(n,m);
    for i = 1:n
        for j = 1:m
            mat[i,j] = v[vectorIndex(m,i,j)]
        end
    end
    return mat;
end

function Vec2Mat(v::Array{Float64,2},n,m)
    m1 = size(v,2)
    mat = zeros(n,m,m1);
    for i = 1:n
        for j = 1:m
            mat[i,j,:] = v[vectorIndex(m,i,j),:]
        end
    end
    return mat;
end

function Mat2Vec(mat::Array{Float64,3})
    n = size(mat,1)
    m = size(mat,2)
    m2 = size(mat,3)
    v = zeros(n*m,m2);
    for i = 1:n
        for j = 1:m
            v[(i-1)*m + j,:] = mat[i,j,:]
        end
    end
    return v;
end