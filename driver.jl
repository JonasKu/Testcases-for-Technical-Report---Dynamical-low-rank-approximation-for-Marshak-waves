include("settings.jl")
include("DGSolver.jl")
include("plotting.jl")

#pointer_from_objref(x);

close("all")


############################

# number of cells
Nx = [10; 25; 50; 75; 100; 200; 300];

errors0 = zeros(length(Nx));
errors1 = zeros(length(Nx));
errors2 = zeros(length(Nx));

for i = 1:length(Nx)
    s = Settings(Nx[i],1);
    solver = DGSolver(s);
    @time tEnd, u = Solve(solver);
    s.tEnd = tEnd;
    plotSolution = Plotting(s);
    errors0[i] = L2Error(plotSolution,u);
end

for i = 1:length(Nx)
    s = Settings(Nx[i],2);
    solver = DGSolver(s);
    @time tEnd, u = Solve(solver);
    s.tEnd = tEnd;
    plotSolution = Plotting(s);
    errors1[i] = L2Error(plotSolution,u);
end

for i = 1:length(Nx)
    s = Settings(Nx[i],3);
    solver = DGSolver(s);
    @time tEnd, u = Solve(solver);
    s.tEnd = tEnd;
    plotSolution = Plotting(s);
    errors2[i] = L2Error(plotSolution,u);
end

fig, ax = subplots(figsize=(15, 8), dpi=100)
ax.loglog(Nx,errors0, "ko", linewidth=2, label=L"DG$_1$", alpha=1.0)
ax.loglog(Nx,errors1, "r+", linewidth=2, label=L"DG$_2$", alpha=1.0)
ax.loglog(Nx,errors2, "b*", linewidth=2, label=L"DG$_3$", alpha=1.0)
ax.loglog(Nx,errors0[1]*(Nx).^(-1)*Nx[1]^1, "k:", linewidth=2, alpha=0.6)
ax.loglog(Nx,errors1[1]*(Nx).^(-2)*Nx[1]^2, "r:", linewidth=2, alpha=0.6)
ax.loglog(Nx,errors2[1]*(Nx).^(-3)*Nx[1]^3, "b:", linewidth=2, alpha=0.6)

ax.legend(loc="lower left", fontsize=20)
#ax.set_ylim([ylimMinus,ylimPlus])
#ax.set_xlim([obj.settings.a,obj.settings.b])
ax.set_xlabel(L"N_x", fontsize=20);
ax.tick_params("both",labelsize=20);
