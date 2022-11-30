__precompile__
using PyPlot

struct Plotting
    settings::Settings;
    basis::SpatialBasis;
    q::Quadrature
    NqFine::Int

    function Plotting(settings::Settings)
        NqFine = 20;
        q = Quadrature(NqFine,settings.quadratureType);
        b = SpatialBasis(settings.x,q,q,settings);
        
        new(settings,b,q,NqFine);
    end
end

function L2Error(obj::Plotting, u::Array{Float64,3})
    error = 0.0;
    for k = 2:(obj.settings.NCells-1) # leave out ghost cells
        xGrid = (obj.settings.x[k]*(1.0.-obj.q.xi)+obj.settings.x[k+1]*(1.0.+obj.q.xi))/2.0;
        #println("boundaries x_k = ", obj.settings.x[k], "; x_k+1 = ",obj.settings.x[k+1]);
        #println(xGrid)
        error = error + ((ExactSolution.(obj.settings.tEnd,xGrid,obj.settings.problem).-EvalAtQuad(obj.basis,u[k,:,1])).^2)'*obj.q.w*(obj.settings.dx)*0.5;
    end
    return sqrt(error);
end

function GetInXCell(obj::Plotting, u::Array{Float64,1},j::Int64)
    xCoarse = (obj.settings.x[j]*(1.0.-obj.q.xi)+obj.settings.x[j+1]*(1.0.+obj.q.xi))/2.0;
    y = EvalAtQuad(obj.basis,u);
    return xCoarse,y;
end

function PlotInXCell(obj::Plotting, u::Array{Float64,1},j)
    xCoarse = (obj.settings.x[j]*(1.0-obj.q.xi)+obj.settings.x[j+1]*(1+obj.q.xi))/2.0;
    ax.plot(xCoarse,EvalAtQuad(obj.basis,u), "b--", linewidth=2, label=L"$u_{DG}$", alpha=0.6)
end

function PlotInX(obj::Plotting,u::Array{Float64,3},T::Array{Float64,2})
    Nq = obj.NqFine;
    Nx = obj.settings.Nx;
    uPlot = zeros(Nq*(Nx-1));
    xPlot = zeros(Nq*(Nx-1));
    TPlot = zeros(Nq*(Nx-1));
    for j = 1:(Nx-1)
        xPlot[((j-1)*Nq+1):j*Nq] , uPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
    end

    # compute exact solution in mid-point grid
    NFine = 1000;
    xGrid = x = range((obj.settings.a+0.5*obj.settings.dx),stop =(obj.settings.b-0.5*obj.settings.dx),length = obj.settings.Nx-1);
    xGridFine = range(obj.settings.a,stop = obj.settings.b,length = NFine-1)
    uExact = ExactSolution.(obj.settings.tEnd,xGrid,obj.settings.problem);
    uExactFine = ExactSolution.(obj.settings.tEnd,xGridFine,obj.settings.problem);

    if obj.settings.ICType == "shock"
        ylimMinus = -0.5;
        ylimPlus = 16.0
    elseif obj.settings.ICType == "sin"
        ylimMinus = -1.5;
        ylimPlus = 1.5;
    else
        ylimMinus = -0.5;#0.0*minimum(uPlot);
        ylimPlus = 1.2*maximum(uPlot);
    end
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((obj.settings.x[j], obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)
    ax.plot(xPlot,uPlot, "ro", linewidth=1, label=L"$\phi_{DG}$", alpha=0.6)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a-2*obj.settings.dx,obj.settings.b+2*obj.settings.dx])
    ax.set_xlabel("x", fontsize=20);
    ax.tick_params("both",labelsize=20);

    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((obj.settings.x[j], obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)
    ylimMinus = -0.5;#0.0*minimum(uPlot);
    ylimPlus = 1.2*maximum(TPlot);
    ax.plot(xPlot,TPlot, "go", linewidth=1, label=L"$T_{DG}$", alpha=0.6)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a-2*obj.settings.dx,obj.settings.b+2*obj.settings.dx])
    ax.set_xlabel("x", fontsize=20);
    ax.tick_params("both",labelsize=20);
end

function PlotInX(obj::Plotting,u::Array{Float64,3},T::Array{Float64,2},u1::Array{Float64,3},T1::Array{Float64,2},u2::Array{Float64,3},T2::Array{Float64,2},labelsU::Array{LaTeXString,1},labelsT::Array{LaTeXString,1})
    Nq = obj.NqFine;
    Nx = obj.settings.Nx;
    uPlot = zeros(Nq*(Nx-1));
    u1Plot = zeros(Nq*(Nx-1));
    u2Plot = zeros(Nq*(Nx-1));
    xPlot = zeros(Nq*(Nx-1));
    TPlot = zeros(Nq*(Nx-1));
    T1Plot = zeros(Nq*(Nx-1));
    T2Plot = zeros(Nq*(Nx-1));
    fsize = 15; # 35 in proposal
    fsizelegend = 20; # 35 in proposal
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
    j=1
    xPlot[((j-1)*Nq+1):j*Nq] , uPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u[j,:,1],j);
    xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , u1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u1[j,:,1],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , u2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u2[j,:,1],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],uPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, label=labelsU[1], alpha=0.6)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, label=labelsU[2], alpha=1.0)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, label=labelsU[3], alpha=1.0)
    for j = 2:(Nx-1)
        xPlot[((j-1)*Nq+1):j*Nq] , uPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , u1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u1[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , u2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u2[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);

        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],uPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, alpha=0.6)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, alpha=1.0)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, alpha=1.0)
    end

    # compute exact solution in mid-point grid
    NFine = 1000;
    xGrid = x = range((obj.settings.a+0.5*obj.settings.dx),stop =(obj.settings.b-0.5*obj.settings.dx),length = obj.settings.Nx-1);
    xGridFine = range(obj.settings.a,stop = obj.settings.b,length = NFine-1)
    uExact = ExactSolution.(obj.settings.tEnd,xGrid,obj.settings.problem);
    uExactFine = ExactSolution.(obj.settings.tEnd,xGridFine,obj.settings.problem);

    if obj.settings.ICType == "shock"
        ylimMinus = -0.5;
        ylimPlus = 16.0
    elseif obj.settings.ICType == "sin"
        ylimMinus = -1.5;
        ylimPlus = 1.5;
    else
        ylimMinus = -0.5;#0.0*minimum(uPlot);
        ylimPlus = 1.2*maximum(uPlot);
    end
    

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((1000 .* obj.settings.x[j], 1000 .* obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)

    ax.legend(loc="upper right", fontsize=fsizelegend)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim(1000 .* [obj.settings.a-2*obj.settings.dx,obj.settings.b+2*obj.settings.dx])
    ax.set_xlabel("x", fontsize=fsize);
    ax.tick_params("both",labelsize=fsize);
    ax.yaxis.offsetText.set_fontsize(fsize)
    tight_layout()

    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((1000 .* obj.settings.x[j], obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)
    ylimMinus = -0.5;#0.0*minimum(uPlot);
    ylimPlus = 1.2*maximum(TPlot);

    j=1
    xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],TPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, label=labelsT[1], alpha=0.6)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, label=labelsT[2], alpha=1.0)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, label=labelsT[3], alpha=1.0)
    for j = 2:(Nx-1)
        xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);

        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],TPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, alpha=0.6)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, alpha=1.0)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, alpha=1.0)
    end

    ax.legend(loc="upper right", fontsize=fsizelegend)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim(1000 .* [obj.settings.a-2*obj.settings.dx,obj.settings.b+2*obj.settings.dx])
    ax.set_xlabel("x", fontsize=fsize);
    ax.tick_params("both",labelsize=fsize);
    ax.yaxis.offsetText.set_fontsize(fsize)
    tight_layout()
end

"""
    PlotInXZoom(obj::Plotting,u::Array{Float64,3},T::Array{Float64,2},u1::Array{Float64,3},T1::Array{Float64,2},u2::Array{Float64,3},T2::Array{Float64,2},labelsU::Array{LaTeXString,1},labelsT::Array{LaTeXString,1})

TBW
"""
function PlotInXZoom(obj::Plotting,u::Array{Float64,3},T::Array{Float64,2},u1::Array{Float64,3},T1::Array{Float64,2},u2::Array{Float64,3},T2::Array{Float64,2},labelsU::Array{LaTeXString,1},labelsT::Array{LaTeXString,1})
    Nq = obj.NqFine;
    Nx = obj.settings.Nx;
    uPlot = zeros(Nq*(Nx-1));
    u1Plot = zeros(Nq*(Nx-1));
    u2Plot = zeros(Nq*(Nx-1));
    xPlot = zeros(Nq*(Nx-1));
    TPlot = zeros(Nq*(Nx-1));
    T1Plot = zeros(Nq*(Nx-1));
    T2Plot = zeros(Nq*(Nx-1));
    fig, ax = subplots(figsize=(12, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
    j=1
    xPlot[((j-1)*Nq+1):j*Nq] , uPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u[j,:,1],j);
    xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , u1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u1[j,:,1],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , u2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u2[j,:,1],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],uPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, label=labelsU[1], alpha=0.6)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, label=labelsU[2], alpha=1.0)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, label=labelsU[3], alpha=1.0)
    for j = 2:(Nx-1)
        xPlot[((j-1)*Nq+1):j*Nq] , uPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , u1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u1[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , u2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u2[j,:,1],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);

        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],uPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, alpha=0.6)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, alpha=1.0)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],u2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, alpha=1.0)
    end

    # compute exact solution in mid-point grid
    NFine = 1000;
    xGrid = x = range((obj.settings.a+0.5*obj.settings.dx),stop =(obj.settings.b-0.5*obj.settings.dx),length = obj.settings.Nx-1);
    xGridFine = range(obj.settings.a,stop = obj.settings.b,length = NFine-1)
    uExact = ExactSolution.(obj.settings.tEnd,xGrid,obj.settings.problem);
    uExactFine = ExactSolution.(obj.settings.tEnd,xGridFine,obj.settings.problem);

    if obj.settings.ICType == "shock"
        ylimMinus = -0.5;
        ylimPlus = 16.0
    elseif obj.settings.ICType == "sin"
        ylimMinus = -1.5;
        ylimPlus = 1.5;
    else
        ylimMinus = -0.5;#0.0*minimum(uPlot);
        ylimPlus = 1.2*maximum(uPlot);
    end
    

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((1000 .* obj.settings.x[j], 1000 .* obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)

    #ax.legend(loc="upper right", fontsize=30)
    ax.set_ylim([ylimMinus,2.3e18])
    ax.set_xlim([1000 .* 0.000714,1000 .* 0.001082])
    ax.set_xlabel("x", fontsize=40);
    ax.tick_params("both",labelsize=40);
    ax.yaxis.offsetText.set_fontsize(35)
    tight_layout()
    

    fig, ax = subplots(figsize=(12, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((1000 .* obj.settings.x[j], 1000 .* obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)
    ylimMinus = -0.5;#0.0*minimum(uPlot);
    ylimPlus = 53;

    j=1
    xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
    xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],TPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, label=labelsT[1], alpha=0.6)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, label=labelsT[2], alpha=1.0)
    ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, label=labelsT[3], alpha=1.0)
    for j = 2:(Nx-1)
        xPlot[((j-1)*Nq+1):j*Nq] , TPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T1Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T1[j,:],j);
        xPlot[((j-1)*Nq+1):j*Nq] , T2Plot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,T2[j,:],j);

        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],TPlot[((j-1)*Nq+1):j*Nq], "k-", linewidth=2, alpha=0.6)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T1Plot[((j-1)*Nq+1):j*Nq], "r:", linewidth=2, alpha=1.0)
        ax.plot(1000 .* xPlot[((j-1)*Nq+1):j*Nq],T2Plot[((j-1)*Nq+1):j*Nq], "b--", linewidth=2, alpha=1.0)
    end

    #ax.legend(loc="upper right", fontsize=30)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim([1000 .* 0.000714,1000 .* 0.001082])
    ax.set_xlabel("x", fontsize=40);
    ax.tick_params("both",labelsize=40);
    ax.yaxis.offsetText.set_fontsize(35)
    tight_layout()

    tight_layout()
end

function PlotInX(obj::Plotting,u::Array{Float64,2},T::Array{Float64,1})
    Nq = obj.NqFine;
    Nx = obj.settings.NCells;

    xPlot = obj.settings.x[1:Nx];
    uPlot = u[:,1];
    TPlot = T;

    if obj.settings.ICType == "shock"
        ylimMinus = -0.5;
        ylimPlus = 16.0
    elseif obj.settings.ICType == "sin"
        ylimMinus = -1.5;
        ylimPlus = 1.5;
    else
        ylimMinus = -0.5;#0.0*minimum(uPlot);
        ylimPlus = 1.2*maximum(uPlot);
    end
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((obj.settings.x[j], obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)
    println(size(xPlot))
    println(size(uPlot))
    ax.plot(xPlot,uPlot, "ro", linewidth=1, label=L"$\phi_{DG}$", alpha=0.6)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a-2*obj.settings.dx,obj.settings.b+2*obj.settings.dx])
    ax.set_xlabel("x", fontsize=20);
    ax.tick_params("both",labelsize=20);

    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((obj.settings.x[j], obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    #ax.plot(xGridFine,uExactFine, "k-", linewidth=2, label=L"$u_{exact}$", alpha=0.6)
    #ax.plot(xGrid,uExact, "ko", linewidth=2)
    ylimMinus = -0.5;#0.0*minimum(uPlot);
    ylimPlus = 1.2*maximum(TPlot);
    ax.plot(xPlot,TPlot, "go", linewidth=1, label=L"$T_{DG}$", alpha=0.6)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a-2*obj.settings.dx,obj.settings.b+2*obj.settings.dx])
    ax.set_xlabel("x", fontsize=20);
    ax.tick_params("both",labelsize=20);
    tight_layout()
end

function PlotInX(obj::Plotting,u::Array{Float64,2})
    Nq = obj.NqFine;
    Nx = obj.settings.Nx;
    uPlot = zeros(Nq*(Nx-1));
    xPlot = zeros(Nq*(Nx-1));
    for j = 1:(Nx-1)
        xPlot[((j-1)*Nq+1):j*Nq] , uPlot[((j-1)*Nq+1):j*Nq] = GetInXCell(obj,u[j,:],j);
    end

    # compute exact solution in mid-point grid
    NFine = 1000;
    xGrid = x = range((obj.settings.a+0.5*obj.settings.dx),stop =(obj.settings.b-0.5*obj.settings.dx),length = obj.settings.Nx-1);
    xGridFine = range(obj.settings.a,stop = obj.settings.b,length = NFine-1);

    if obj.settings.ICType == "shock"
        ylimMinus = -0.5;
        ylimPlus = 16.0
    elseif obj.settings.ICType == "sin"
        ylimMinus = -1.5;
        ylimPlus = 1.5;
    else
        ylimMinus = -0.5;#0.0*minimum(uPlot);
        ylimPlus = 1.2*maximum(uPlot);
    end
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    # draw cell lines
    for j = 1:obj.settings.Nx
        ax.plot((obj.settings.x[j], obj.settings.x[j]), (ylimMinus, ylimPlus), "k:");
    end

    ax.plot(xPlot,uPlot, "ro", linewidth=1, label=L"$u_{DG}$", alpha=0.6)
    ax.legend(loc="upper right", fontsize=20)
    ax.set_ylim([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a-0.05,obj.settings.b+0.05])
    ax.set_xlabel("x", fontsize=20);
    ax.tick_params("both",labelsize=20);
end

