import numpy as np
import PHD as phd
import pandas as pd
import gospapy
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from scipy.stats import poisson
from confidence_ellipse import confidence_ellipse

def AnimateStep(measurement,true_measurements, ax, targets_to_plot, t):
    ax.cla()
    ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='blue')
    ax.coastlines(resolution=res, color='black')
    ax.stock_img()
    for bird_place in bird_places:
        confidence_ellipse([bird_place[0], bird_place[1]], airportCov, ax=ax, edgecolor="red", label="Airport")
    
    ax.set_title('t = ' + str(t), fontstyle='italic')
    for x,y in measurement:
        ax.plot(x, y, '.', color='gray')
    for x,y in true_measurements:
        ax.plot(x, y, 'X', color='red')

    for filter in targets_to_plot: 
        ax.plot(filter.m[0], filter.m[1], "+", color="black", label="PHD")
        ax.annotate('_____' + str(filter.label) +  '_____', xy=(filter.m[0], filter.m[1]),transform=crs.Geodetic(), textcoords='data')
        confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax, edgecolor="blue")

if __name__ == "__main__":
    dt = 1
    lam = 0.000001
    Pd = 0.98
    Ps = 0.95
    res = '110m'
    offset = 0.0001

    PLOT = False
    PLOT_TRAJECTORIES  = True
    PLOT_TRUE_TRAJECTORIES = False
    CLUTTER = False
    RUNS = 11

    projPC = crs.PlateCarree()

    # One - Charlie
    # BIRD = 'Charlie'
    # lonW = -10
    # lonE = 40
    # latS = 50
    # latN = 71

    # One - Ava
    # BIRD = 'Ava'
    # lonW = -20
    # lonE = 20
    # latS = 10
    # latN = 60

    # One - Baby
    # BIRD = 'Baby'
    # lonW = -20
    # lonE = 20
    # latS = 10
    # latN = 60

    # One - Penny
    # BIRD = 'Penny'
    # lonW = -10
    # lonE = 40
    # latS = 50
    # latN = 71

    # One - Skittles
    # BIRD = 'Skittles'
    # lonW = -10
    # lonE = 60
    # latS = 50
    # latN = 71

    # One - Greta
    # BIRD = 'Greta'
    # lonW = -20
    # lonE = 20
    # latS = 10
    # latN = 60

    # One - Greta + Skittles + Penny + Baby + Ava + Charlie
    # BIRD = 'SixIndividuals'
    # lonW = -20
    # lonE = 60
    # latS = 10
    # latN = 71

    # ALL
    BIRD = 'day_aggr'
    lonW = -20
    lonE = 54
    latS = 6
    latN = 71 
    
    df = pd.read_csv(BIRD + '.csv')
    gb = df.groupby('time_step')

    for run in range(RUNS):
        m_list = [gb.get_group(x) for x in gb.groups]
        n_dat = len(m_list)

        measurements = []
        true_measurements = []
        x_s = []
        y_s = []

        clutterCount = poisson.rvs(6250000 *  lam, size=n_dat) 
        for t in range(n_dat):
            data = m_list[t]  
            measurements_at_time = []
            true_measurements_at_time =[ ]
            for index, row in data.iterrows():     
                x = row['location-long']
                y = row['location-lat']
                measurements_at_time.append((x, y))  
                true_measurements_at_time.append((x, y))  
                
                if CLUTTER:
                    X0min = x/1.01
                    X0max = 1.01*x

                    X1min = y/1.01
                    X1max = 1.01*y

                    x_s = np.random.uniform(low=X0min * (1 - np.sign(X0min) * offset), high=X0max * (1 + np.sign(X0max) * offset), size=clutterCount[t])
                    y_s = np.random.uniform(low=X1min * (1 - np.sign(X1min) * offset), high=X1max * (1 + np.sign(X1max) * offset), size=clutterCount[t])
                    for x_c, y_c in zip(x_s, y_s):
                        measurements_at_time.append((x_c, y_c))

            measurements.append(measurements_at_time)
            true_measurements.append(true_measurements_at_time)

        F = np.array([  [1,       0,       dt,        0],
                        [0,       1,       0,         dt],
                        [0,       0,       1,         0],
                        [0,       0,       0,         1]])   

        Q = np.array([  [dt,    0,       dt/2,      0],
                        [0,       dt,    0,         dt/2],
                        [dt/2,    0,       dt,        0],
                        [0,       dt/2,    0,         dt]])
        
        Q = Q.dot(0.2 ** 2)

        H = np.array([  [1,       0,       0,         0],
                        [0,       1,       0,         0]])

        R = np.array([  [0.4**2,    0],
                        [0,       0.4**2]]
                        )

        d = 0.001
        airportCov = np.array([[d, 0, 0, 0],
                            [0, d, 0, 0],
                            [0, 0, d, 0],
                            [0, 0, 0, d]])
        
        # bird_places = [[1.906831622123718, 50.16376495361328, 0, 0],[2.7221153179804483, 50.9518076578776, 0, 0],[3.538073682785034, 51.24851913452149, 0, 0],[4.43301714791192, 51.14955181545682, 0, 0],[5.028501510620117, 50.76594543457031, 0, 0]] # ,[29.96021270751953, 67.13046264648438, 0, 0]
        bird_places = [[3.538073682785034, 51.24851913452149, 0, 0]]
        sources = []
        targets_to_plot = []
        gospas = []

        if PLOT:
            fig, ax = plt.subplots(figsize=(20, 16)) 
            ax = plt.axes(projection=projPC)

        for coordinates in bird_places:
            sources.append(np.array(coordinates))

        model = phd.PHD(sources=sources, F=F, Q=Q, H=H, R=R, lam=lam, Ps=0.98, Pd=0.95, U=9.2, T=1.e-5, T_plot=0.5, gamma=5, w_th=0.5, trajectory_prune=50)

        for t in range(n_dat):
                    
            model.predict(cov=airportCov)
            model.gate(measurements=measurements[t])  
            model.update()
            model.mergeLabels()
            targets_to_plot = [x for x in model.targets if x.w > model.T_plot]
            gospa_targets = [x.m[0:2] for x in targets_to_plot ]
            gospas.append(gospapy.calculate_gospa(targets=gospa_targets, tracks=true_measurements[t], c=0.5, alpha=2, p=2))
            if PLOT:
                AnimateStep(measurement=measurements[t], true_measurements=true_measurements[t], ax=ax, targets_to_plot=targets_to_plot, t=t)
                plt.pause(0.1)
        model.fixTrajectories()

        if PLOT_TRAJECTORIES:
            fig_2, ax_2 = plt.subplots(figsize=(16,12))
            ax_2 = plt.axes(projection=projPC)
            ax_2.set_extent([lonW, lonE, latS, latN], crs=projPC)
            ax_2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='blue')
            ax_2.coastlines(resolution=res, color='black')
            ax_2.stock_img()

        trajectory_counter = 0
        for trajectory, key in zip(model.trajectories.values(), model.trajectories.keys()): 
            xs = [t.m[0] for t in trajectory]
            ys = [t.m[1] for t in trajectory] 
            np.savetxt('trajectories/trajectory_' + BIRD + '_' + str(trajectory_counter) +  '_run_' + str(run) + '.txt', (xs, ys))
            trajectory_counter += 1
            if PLOT_TRAJECTORIES:
                ax_2.plot(xs,ys,linestyle='--', marker='o', label=str(key))

        if PLOT_TRUE_TRAJECTORIES:
            xs = [t[0] for t in true_measurements]
            ys = [t[1] for t in true_measurements] 
            ax_2.plot(xs,ys ,linestyle='--', marker='o', label='True trajectory', color='red')

        if PLOT_TRAJECTORIES:
            plt.legend(loc="lower right", prop={'size': 20}, markerscale=4, ncol=3)
        
        PHD_loc_error = []
        for g in gospas:
            PHD_loc_error.append(g[2])

        np.savetxt('gospas/gospa_' + BIRD + '_run_' + str(run) + '.txt', PHD_loc_error)
    
        if PLOT_TRAJECTORIES:
            plt.plot()
            while True:
                plt.pause(1)
        

