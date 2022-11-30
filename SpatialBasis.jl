__precompile__
import GSL

struct SpatialBasis
    # number of moments
    N::Int64;
    Nq::Int64;
    # spatial grid size
    dx::Float64;
    # spatial grid
    x::Array{Float64,1};
    # precomputed Legendre Polynomials at cell interfaces
    PhiLeft::Array{Float64,1};
    PhiRight::Array{Float64,1};
    PhiTildeLeft::Array{Float64,1};
    PhiTildeRight::Array{Float64,1};

    # precomputed Legendre Polynomials and derivatives at quadrature points
    PhiQuad::Array{Float64,2};
    PhiTildeGrid::Array{Float64,2};
    PhiTildeQuad::Array{Float64,2};
    PhiTildeWQuad::Array{Float64,2};
    DPhiTildeQuad::Array{Float64,2};

    function SpatialBasis(x,quadrature::Quadrature,qGrid::Quadrature,settings::Settings)
        N = settings.N;
        Nq = quadrature.Nq;
        dx = settings.dx;

        # precompute Legendre basis functions at interfaces and quad points
        PhiLeft = zeros(N);
        PhiRight = zeros(N);
        PhiTildeLeft = zeros(N);
        PhiTildeRight = zeros(N);
        PhiQuad = zeros(Nq,N);
        PhiTildeQuad = zeros(Nq,N);
        PhiTildeWQuad = zeros(Nq,N);
        PhiTildeGrid = zeros(Nq,N);
        DPhiTildeQuad = zeros(N,Nq);
        for i = 1:N
            PhiLeft[i] = sqrt(2.0*(i-1.0)+1.0)*Phi(i-1,-1.0);
            PhiRight[i] = sqrt(2.0*(i-1.0)+1.0)*Phi(i-1,+1.0);
            PhiTildeLeft[i] = (1.0/dx)*sqrt(dx)*sqrt((2.0*(i-1.0)+1.0)/dx)*Phi.(i-1,-1.0);
            PhiTildeRight[i] = (1.0/dx)*sqrt(dx)*sqrt((2.0*(i-1.0)+1.0)/dx)*Phi.(i-1,+1.0);
            for k = 1:Nq
                PhiQuad[k,i] = sqrt(2.0*(i-1.0)+1.0)*Phi.(i-1,quadrature.xi[k]);
                PhiTildeGrid[k,i] = (1.0/dx)*sqrt(dx)*sqrt((2.0*(i-1.0)+1.0)/dx)*Phi.(i-1,qGrid.xi[k]);
                PhiTildeQuad[k,i] = (1.0/dx)*sqrt(dx)*sqrt((2.0*(i-1.0)+1.0)/dx)*Phi.(i-1,quadrature.xi[k]);
                PhiTildeWQuad[k,i] = PhiTildeQuad[k,i]*quadrature.w[k];
                DPhiTildeQuad[i,k] = (1.0/dx)*sqrt(dx)*sqrt((2.0*(i-1.0)+1.0)/dx)*DPhi.(i-1,quadrature.xi[k])*2.0/dx;
            end
        end

        new(N,Nq,dx,x,PhiLeft,PhiRight,PhiTildeLeft,PhiTildeRight,PhiQuad,PhiTildeGrid,PhiTildeQuad,PhiTildeWQuad,DPhiTildeQuad);
    end

end

# Legendre Polynomials on [-1,1]
function Phi(n::Int64,xVal::Float64) ## can I use this with input vectors xVal?
    if xVal > 1.0
        xVal = 1.0;
    elseif xVal < -1.0
        xVal = -1.0;
    end
    return GSL.sf_legendre_Pl.(n,xVal);
end

# Derivative of Legendre Polynomials on [-1,1]
function DPhi(n::Int64,xVal::Float64)
    tupleY = GSL.sf_legendre_Pl_deriv_array(n,xVal);
    y = collect(tupleY);
    return y[2][n+1];
end

# Legendre Polynomials on spatial cell k
function PhiBar(obj::SpatialBasis,k::Int64,n::Int64,xVal::Float64)
    return sqrt((2.0*n+1.0)/obj.dx)*Phi.(n,2.0*(xVal-obj.x[k])/obj.dx-1.0);
end

# Derivative of Legendre Polynomials on spatial cell k
function DPhiBar(obj::SpatialBasis,k::Int64,n::Int64,xVal::Float64)
    return sqrt((2.0*n+1.0)/obj.dx)*DPhi.(n,2.0*(xVal-obj.x[k])/obj.dx-1.0)*2.0/obj.dx;
end

# Basis on spatial cell k
function Phi(obj::SpatialBasis,k::Int64,n::Int64,xVal::Float64)
    return sqrt(obj.dx)*PhiBar(obj,k,n,xVal);
end

# Test functions on spatial cell k
function PhiTilde(obj::SpatialBasis,k::Int64,n::Int64,xVal::Float64)
    return (1.0/obj.dx)*Phi(obj,k,n,xVal);
end

# Test functions on spatial cell k
function DPhiTilde(obj::SpatialBasis,k::Int64,n::Int64,xVal::Float64)
    return (1.0/obj.dx)*sqrt(obj.dx)*sqrt((2.0*n+1.0)/obj.dx)*DPhi.(n,2.0*(xVal-obj.x[k])/obj.dx-1.0)*2.0/obj.dx;
    #return (1.0/obj.dx)*DPhiBar.(obj,k,n,xVal);
end

# evaluate polynomial with moments u at spatial position x
function Eval(obj::SpatialBasis,k::Int64,u::Array{Float64,1},x::Float64)
    y = zeros(size(x));
    for i = 1:obj.N
        y = y+u[i]*Phi.(obj,k,i-1,x);
        #y = y+u[i]*sqrt(2.0*(i-1.0)+1.0)*Phi.(i-1,2.0*(x-obj.x[k])/obj.dx-1.0);
    end
    return y;
end

# evaluate polynomial with moments u at left cell interface of cell k
function EvalLeft(obj::SpatialBasis,u::Array{Float64,2})
    return (u')*obj.PhiLeft;
end

# evaluate polynomial with moments u at right cell interface of cell k
function EvalRight(obj::SpatialBasis,u::Array{Float64,2})
    return (u')*obj.PhiRight;
end

# evaluate polynomial with moments u at left cell interface of cell k leaving out 0 order moment
function EvalBarLeft(obj::SpatialBasis,u::Array{Float64,1})
    return u[2:end]'obj.PhiLeft[2:end];
end

# evaluate polynomial with moments u at right cell interface of cell k
function EvalBarRight(obj::SpatialBasis,u::Array{Float64,1})
    return u[2:end]'obj.PhiRight[2:end];
end

function EvalAtQuad(obj::SpatialBasis,u)
    return obj.PhiQuad*u;
end

# functions for evaluating entropic variables

# evaluate dual State at quadrature points for given dual variables v
function EvalVAtQuad(obj::SpatialBasis,v::Array{Float64,1})
    return obj.PhiTildeQuad*v;
end

# evaluate dual State at grid quadrature points for given dual variables v
function EvalVAtGrid(obj::SpatialBasis,v::Array{Float64,1})
    return obj.PhiTildeGrid*v;
end


# evaluate polynomial with moments u at left cell interface of cell k
function EvalVLeft(obj::SpatialBasis,u::Array{Float64,1})
    return u'obj.PhiTildeLeft;
end

# evaluate polynomial with moments u at right cell interface of cell k
function EvalVRight(obj::SpatialBasis,u::Array{Float64,1})
    return u'obj.PhiTildeRight;
end