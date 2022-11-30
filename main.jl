include("settings.jl")
include("DGSolver.jl")
include("FVSolver.jl")
include("plotting.jl")

close("all")

s = Settings();

############################
solver = DGSolver(s);

s.r = 50;
@time tEnd, u_ad, T_ad, rankInTime = SolveUnconventionalAdaptive(solver);
s.r = 5;
@time tEnd, u_lr, T_lr = SolveUnconventional(solver);
@time tEnd, u, T = Solve(solver);
solver = DGSolver(s);
@time tEnd, u_ps, T_ps = SolveProjectorSplitting(solver); # not working for higher ranks!


s.tEnd = tEnd;

plotSolution = Plotting(s);

PlotInX(plotSolution,Vec2Mat(u,s.NCells,s.N),Vec2Mat(T,s.NCells,s.N)./11604.0,Vec2Mat(u_lr,s.NCells,s.N),Vec2Mat(T_lr,s.NCells,s.N)./11604.0,Vec2Mat(u_ps,s.NCells,s.N),Vec2Mat(T_ps,s.NCells,s.N)./11604.0,[L"\phi_{DG}",L"\phi_{BUG}",L"\phi_{PSI}"],[L"T_{DG}",L"T_{BUG}",L"T_{PSI}"]);

#PlotInXZoom(plotSolution,Vec2Mat(u,s.NCells,s.N),Vec2Mat(T,s.NCells,s.N)./11604.0,Vec2Mat(u_lr,s.NCells,s.N),Vec2Mat(T_lr,s.NCells,s.N)./11604.0,Vec2Mat(u_ps,s.NCells,s.N),Vec2Mat(T_ps,s.NCells,s.N)./11604.0,[L"\phi_{DG}",L"\phi_{BUG}",L"\phi_{PSI}"],[L"T_{DG}",L"T_{BUG}",L"T_{PSI}"]);


PlotInX(plotSolution,Vec2Mat(u,s.NCells,s.N),Vec2Mat(T,s.NCells,s.N)./11604.0,Vec2Mat(u_ad,s.NCells,s.N),Vec2Mat(T_ad,s.NCells,s.N)./11604.0,Vec2Mat(u_ps,s.NCells,s.N),Vec2Mat(T_ps,s.NCells,s.N)./11604.0,[L"\phi_{DG}",L"$\phi_{BUG}$, adaptive",L"\phi_{PSI}"],[L"T_{DG}",L"$T_{BUG}$, adaptive",L"T_{PSI}"]);

#PlotInXZoom(plotSolution,Vec2Mat(u,s.NCells,s.N),Vec2Mat(T,s.NCells,s.N)./11604.0,Vec2Mat(u_ad,s.NCells,s.N),Vec2Mat(T_ad,s.NCells,s.N)./11604.0,Vec2Mat(u_ps,s.NCells,s.N),Vec2Mat(T_ps,s.NCells,s.N)./11604.0,[L"\phi_{DG}",L"\phi_{BUG}",L"\phi_{PSI}"],[L"T_{DG}",L"T_{BUG}",L"T_{PSI}"]);



fig = figure("rank evolution",figsize=(10, 10), dpi=100)
ax = gca()

ax.plot(rankInTime[1,:],rankInTime[2,:], "k-", linewidth=2, alpha=1.0)

ax.set_xlim([0.0,s.tEnd])
#ax.set_ylim([0.0,440])
ax.set_xlabel("t", fontsize=20);
ax.set_ylabel("rank", fontsize=20);
ax.tick_params("both",labelsize=20) 
#ax.legend(loc="upper left", fontsize=20)
tight_layout()
fig.canvas.draw() # Update the figure

println("main finished")
