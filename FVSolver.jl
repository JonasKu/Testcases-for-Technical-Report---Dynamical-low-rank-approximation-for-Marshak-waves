__precompile__
include("quadrature.jl")
include("SpatialBasis.jl")
include("TimeSolver.jl");

using ProgressMeter

using LinearAlgebra

struct FVSolver
    # spatial grid of cell interfaces
    x::Array{Float64,1};

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


    # constructor
    function FVSolver(settings)
        x = settings.x;
        q = Quadrature(settings.Nq,settings.quadratureType);
        basis = SpatialBasis(x,q,q,settings);
        time = TimeSolver(settings);

        outRhs = zeros(settings.NCells,settings.nPN);
        outRhsT = zeros(settings.NCells);

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

        new(x,q,settings,basis,time,outRhs,outRhsT,gamma,A,AbsA,settings.sigmaT,settings.sigmaS,settings.sigmaA,c,sigmaSB,aRad);
    end
end

# Lax-Friedrichs flux for PN
function numFlux(obj::FVSolver,ul::Array{Float64,1},ur::Array{Float64,1})
    #return 0.5*(obj.A*(ul+ur).-(obj.settings.dx/obj.settings.dt).*(ur-ul)/obj.c);
    return 0.5*(obj.A*(ul+ur).-obj.AbsA*(ur-ul));
end

function SetupIC(obj::FVSolver)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    T = zeros(obj.settings.NCells);
    TVal = 80.0 * 11604.0;
    for j = 1:obj.settings.NCells
        if j == 1
            TVal = 80.0 * 11604.0;
        else
            TVal = 0.02 * 11604.0;
        end
        T[j] = TVal;
        u[j,1] = obj.aRad*obj.c/4/pi.*TVal^4;
    end
    return u,T;
end

function Rhs(obj::FVSolver,u::Array{Float64,2},T::Array{Float64,1},t::Float64=0.0)   
    for k = 2:(obj.settings.NCells-1) # leave out ghost cells

        # evaluate interface k (left interface of cell k)
        ukMRight = u[k-1,:];
        ukLeft = u[k,:];

        # evaluate interface k+1 (right interface of cell k)
        ukRight = u[k,:];
        ukPLeft = u[k+1,:];

        # compute coupling with temperature
        TContribution = T[k]^4;

        obj.outRhs[k,:] = -(numFlux(obj,ukRight,ukPLeft).-numFlux(obj,ukMRight,ukLeft))/obj.settings.dx;
        obj.outRhs[k,:] = obj.outRhs[k,:] .- obj.sigmaT*u[k,:]; # add out-scattering and absorption contribution
        obj.outRhs[k,1] = obj.outRhs[k,1] .+ obj.sigmaS*u[k,1]; # add in-scattering contribution
        obj.outRhs[k,1] = obj.outRhs[k,1] .+ obj.sigmaA*obj.aRad*obj.c/4/pi * TContribution; # add temperature contribution
    end

    #TContribution = T[1]^4;
    #obj.outRhs[1,:] = obj.outRhs[1,:] - obj.sigmaT*u[1,:]; # add out-scattering and absorption contribution
    #obj.outRhs[1,1] = obj.outRhs[1,1] + obj.sigmaS*u[1,1]; # add in-scattering contribution
    #obj.outRhs[1,1] = obj.outRhs[1,1] + obj.sigmaA*obj.aRad*obj.c/4/pi * TContribution; # add temperature contribution
    return obj.outRhs*obj.c;
end

function RhsT(obj::FVSolver,u::Array{Float64,2},T::Array{Float64,1},t::Float64=0.0)   
    for k = 2:(obj.settings.NCells-1) # use Dirichlet values at boundary to make sure temperature does not change here??
        # compute coupling with temperature
        TContribution = T[k]^4;
        obj.outRhsT[k] = obj.sigmaA*(4*pi*u[k,1]-obj.aRad*obj.c*TContribution)/obj.settings.cV;
    end
    return obj.outRhsT;
end

function Solve(obj::FVSolver)
    t = 0.0;
    dt = obj.settings.dt;
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);
    
    Nt = round(tEnd/dt);
    
    # time loop
    @showprogress 0.1 "Progress " for n = 1:Nt

        # Update time by dt
        u,T = F(obj,u,T);
        
        t = t+dt;
    end

    # return end time and solution
    return t, u, T;

end

function F(obj::FVSolver,u::Array{Float64,2},T::Array{Float64,1},t::Float64=0.0)
    NCells = obj.settings.NCells;
    uNew = u.+obj.settings.dt*Rhs(obj,u,T,t);
    T .= T.+obj.settings.dt*RhsT(obj,u,T,t);
    return uNew,T;
end

function SolveNaiveBackward(obj::FVSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    Nx = obj.settings.NCells;
    N = obj.settings.nPN;
    r = 2; # DLR rank
    tEnd = obj.settings.tEnd;

    # Set up initial condition
    u,T = SetupIC(obj);

    uHatL = u[1,:];
    uHatR = u[end,:];

    TNew = T;

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r]; 

    A = zeros(r,r,r);
    B = zeros(r,r,N);
    Y = zeros(r,r,r);
    SFlux = zeros(r,r);
    numFlux = zeros(r);
    K = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    LFlux = zeros(N,r);
    
    Nt = Integer(round(tEnd/dt));

    prog = Progress(Nt,1)
    
    # time loop
    #@showprogress 0.1 "Progress " 
    #@gif 
    for n = 1:Nt

        # Update T
        TNew .= T + dt.*RhsT(obj,X*S*W',T);

        #println("min T = ",minimum(T))
        #if minimum(T) < 0
        #    break;
        #end

        #println("min u_0 = ",minimum((X*S*W')[:,1]))
        #if minimum((X*S*W')[:,1]) < 0
            #break;
        #end

        ###### K-step ######
        K .= X*S;

        # impose BCs
        K[1,:] = uHatL'*W;
        K[end,:] = uHatR'*W;

        yK = Rhs(obj,K*W',T);
        K .= K .+ dt*yK*W;

        X,S = qr(K); # optimize by choosing XFull, SFull
        X = X[:, 1:r]; 
        S = S[1:r, 1:r];

        ###### S-step ######

        yS = Rhs(obj,X*S*W',T);
        S .= S .- dt.*X'*yS*W;

        ###### L-step ######
        L = W*S';

        yL = Rhs(obj,X*L',T);
        L .= L .+ dt*(X'*yL)';
                
        W,S = qr(L);
        #W,S = qr(L);
        W = W[:, 1:r];
        S = S[1:r, 1:r];

        S .= S';

        # update T
        T .= TNew;
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X*S*W',T;
end