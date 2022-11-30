__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # number spatial cells
    NCells::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    dx::Float64

    # time settings
    # end time
    tEnd::Float64;
    # time increment
    dt::Float64;
    # Runge Kutta stages of time integrator
    rkStages::Int64;

    # number of quadrature points
    Nq::Int64;
    # definition of quadrature type
    quadratureType::String;

    # maximal polynomial degree
    N::Int64;
    
    # degree PN
    nPN::Int64;

    # DLRA settings
    r::Int;
    epsAdapt::Float64;
    adaptIndex::Int;

    # solution values
    uL::Float64;
    uR::Float64;
    x0::Float64;
    x1::Float64;

    # spatial grid
    x

    # problem definitions
    problem::String;

    # advection speed if advection is defined as problem
    advectionSpeed::Float64;

    # indicator which volume integration should be chosen
    animation::Int;
    # IC definitions
    ICType::String;
    BCType::String;

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;
    sigmaA::Float64;
    cV::Float64;
    
    c::Float64;                
    sigmaSB::Float64;                
    aRad::Float64; 

    function Settings(Nx::Int=51,N::Int=2,r::Int=50)
        # spatial grid setting
        #Nx = 100; # number of spatial grid points
        NCells = Nx - 1;
        a = 0.0; # left boundary
        b = 0.002;#0.05;#0.08; # right boundary
        
        # time settings
        tEnd = 0.05 * 1e-10;# 0.05 * 1e-9;
        cfl = 0.8; # CFL condition

        # moment system parameters
        #N = 3; # number of spatial expansion coefficients per cell
        Nq = 5; # number of quadrature points
        quadratureType = "Gauss"; # Possibilities are Gauss and ClenshawCurtis
        ICType = "MarshakWave"; # Possibilities are MarshakWave

        # initial condition parameters
        uL = 12.0;
        uR = 1.0;
        x0 = 0.5;
        x1 = x0;#1.0;
        
        # number PN moments
        nPN = 100;

        animation = 0; # 1 records gif

        problem = "MarshakWave"; # Possibilities are ManufacturedSolution, ManufacturedSolutionLinear, ManufacturedSolutionSteady and LineSource
        advectionSpeed = 299792458.0 * 100.0;  # speed of light

        x = collect(range(a,stop = b,length = NCells));
        dx = x[2]-x[1];
        #x = [x[1]-dx;x;x[NCells-1]+dx]; # add ghost cells
        x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        println("NCells = ",NCells);
        println("size x = ",size(x));
        # a lies on x(3), b lies on x(N-1). Cells 1,2 and N-1,N are ghost cells

        dt = cfl*dx/((2*(N-1)+1)*advectionSpeed);

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 1.0 / 0.926 / 1e-6 / 100.0#1.0 / 92.6 / 1e-6 / 100.0;
        sigmaT = sigmaA + sigmaS;
        density = 0.01;#2.7;
        cV = density * 0.831 * 1e7;    # heat capacity: [kJ/(kg K)] = [1000 m^2 / s^2 / K] therefore density * 0.831 * 1e7

        c = 299792458.0 * 100.0;                # speed of light in [cm/s]
        sigmaSB = 5.6704 * 1e-5;                # Stefan Boltzmann constant in [erg/cm^2/s/K^4]
        aRad = 4.0 * sigmaSB / c;               # radiation constant [erg/(cm^3 K^4)]

        BCType = "dirichlet" # periodic,dirichlet,exact

        epsAdapt = 1e-3;
        adaptIndex = 1;

        # build class
        new(Nx,NCells,a,b,dx,tEnd,dt,N,Nq,quadratureType,N,nPN,r,epsAdapt,adaptIndex,uL,uR,x0,x1,x,problem,advectionSpeed,animation,ICType,BCType,sigmaT,sigmaS,sigmaA,cV,c,sigmaSB,aRad);
    end

end

function IC(obj::Settings,x)
    y = zeros(size(x));
    if obj.ICType == "shock"
        for j = 1:length(y);
            if x[j] < obj.x0
                y[j] = obj.uL;
            else
                y[j] = obj.uR;
            end
        end
    elseif obj.ICType == "sin"
        for j = 1:length(y)
            y[j] = sin(2*pi*x[j]);
        end
    elseif obj.ICType == "LS"
        x0 = 0.0
        s2 = 0.03^2
        floor = 1e-4
        for j = 1:length(y);
            #println(1.0/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0))/4.0/s2))
            y[j] = max(floor,1.0/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0))/4.0/s2))
        end
    elseif obj.ICType == "ManufacturedSolution"
        for j = 1:length(y);
            y[j] = cos(x[j]);
        end
    elseif obj.ICType == "MarshakWave"
        T = 80.0 * 11604.0;
        y[1] = obj.aRad*obj.c/4/pi * T^4;
        for j = 2:length(y);
            T = 0.02 * 11604.0;
            y[j] = obj.aRad*obj.c/4/pi * T^4;
        end
    end
    return y;
end

function IC0Deterministic(obj::Settings,x)
    y = zeros(size(x));
    for j = 1:length(y);
        if x[j] < obj.x0
            y[j] = obj.uL;
        else
            y[j] = obj.uR;
        end
    end
    return y;
end

function IC1Deterministic(obj::Settings,x)
    y = zeros(size(x));
    x0 = 0.0
    s2 = 0.03^2
    floor = 1e-4
    println("x = ",x)

    for j = 1:length(y);
        println(1.0/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0))/4.0/s2))
        y[j] = max(floor,1.0/(4.0*pi*s2) *exp(-((x[j]-x0)*(x[j]-x0))/4.0/s2))
    end

    return y;
end

function IC3Deterministic(obj::Settings,x)
    y = zeros(size(x));
    for j = 1:length(y);
        if x[j] < obj.x0
            y[j] = obj.uL;
        elseif x[j] < obj.x1
            y[j] = 0.5*(obj.uR+obj.uL);            
        else
            y[j] = obj.uR;
        end
    end
    return y;
end

function IC1DeterministicExact(obj::Settings,t::Float64,x)
    y = zeros(size(x));

    if obj.problem == "Burgers"
        if t >= (obj.x1-obj.x0)/(obj.uL-obj.uR);
            tS = (obj.x1-obj.x0)/(obj.uL-obj.uR);
            x0BeforeShock = obj.x0 + tS*obj.uL;
            x1BeforeShock = obj.x1 + tS*obj.uR;
            x0 = x0BeforeShock + (t-tS)*(obj.uL+obj.uR)*0.5;
            x1 = x0 - 1.0;
        else
            x0 = obj.x0 + t*obj.uL;
            x1 = obj.x1 + t*obj.uR;
        end
        for j = 1:length(y);
            if x[j] < x0
                y[j] = obj.uL;
            elseif x[j] < x1
                y[j] = obj.uL + (obj.uR - obj.uL)*(x[j]-x0)/(x1-x0);
            else
                y[j] = obj.uR;
            end
        end
    elseif obj.problem == "Advection"
        if obj.ICType == "shock"
            x0 = obj.x0 + t*obj.advectionSpeed;
            x1 = obj.x1 + t*obj.advectionSpeed;
            for j = 1:length(y)
                if x[j] < x0
                    y[j] = obj.uL;
                elseif x[j] < x1
                    y[j] = obj.uL + (obj.uR - obj.uL)*(x[j]-x0)/(x1-x0);
                else
                    y[j] = obj.uR;
                end
            end
        elseif obj.ICType == "sin"
            for j = 1:length(y)
                y[j] = sin(2*pi*(x[j]-t*obj.advectionSpeed));
            end
        end
    end
    return y;
end

function ExactSolution(t::Float64,x::Float64,problem::String)
    if problem == "ManufacturedSolution"
        return exp(t)*cos(x)*2;
    elseif problem == "ManufacturedSolutionLinear"
        return (t+1.0)*cos(x)*2;
    elseif problem == "ManufacturedSolutionSteady"
        return cos(x)*2;
    else
        return 0.0;
    end
end
