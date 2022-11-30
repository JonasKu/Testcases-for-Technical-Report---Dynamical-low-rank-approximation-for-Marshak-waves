__precompile__
struct TimeSolver
    alpha::Array{Float64,2};
    beta::Array{Float64,2};
    gamma::Array{Float64,1};
    N::Int64;
    Nx::Int64;
    dt::Float64;

    uRK::Array{Float64,3};
    TRK::Array{Float64,2};
    R::Array{Float64,3};
    RT::Array{Float64,2};

    # constructor
    function TimeSolver(settings::Settings)
        N = settings.N;
        Nx = settings.Nx;
        dt = settings.dt;

        # set up alpha and beta values for SSP RK method
        alpha, beta = SetUpRKTable(N);

        # setup gamma according to https://gkyl.readthedocs.io/en/latest/dev/ssp-rk.html
        gamma = zeros(N);
        if N == 1
            gamma[1] = 0.0;
        elseif N == 2
            gamma[1] = 0.0;
            gamma[2] = 1.0;
        elseif N == 3
            gamma[1] = 0.0;
            gamma[2] = 1.0;
            gamma[3] = 0.5;
        end

        # allocate memory for matrices used in Update function
        uRK = zeros(N+1,(Nx-1)*N,settings.nPN);
        TRK = zeros(N+1,(Nx-1)*N);
        R = zeros(N,(Nx-1)*N,settings.nPN);
        RT = zeros(N,(Nx-1)*N);

        new(alpha,beta,gamma,N,Nx,dt,uRK,TRK,R,RT)
    end
end

function SetUpRKTable(p::Int64)
    alpha = zeros(p,p);
    beta = zeros(p,p);
    if p == 1
        alpha[1,1] = 1.0;
        beta[1,1] = 1.0;
    elseif p == 2
        alpha[1,1] = 1.0;
        alpha[2,1] = 0.5;
        alpha[2,2] = 0.5;
        beta[1,1] = 1.0;
        beta[2,1] = 0.0;
        beta[2,2] = 0.5;
    elseif p == 3
        alpha[1,1] = 1.0;
        alpha[2,1] = 3.0/4.0;
        alpha[2,2] = 1.0/4.0;
        alpha[3,1] = 1.0/3.0;
        alpha[3,3] = 2.0/3.0;
        beta[1,1] = 1.0;
        beta[2,1] = 0.0;
        beta[2,2] = 1.0/4.0;   
        beta[3,3] = 2.0/3.0; 
    end
    return alpha, beta
end
