from Project_simulation_12 import simulation
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

num_steps = 100
num_stat = 664 # max 664
num_trucks = 3
lambda_bike = 0.0187 # see Excel sheet for numbers
norm_factor = 0.148
load_rate = 0.005
truck_cap = 10
time_of_day = 'afternoon' # pick between 'night','morning','noon','afternoon','evening'
plot_animation = True
save_animation = False

times, bikes_avail, bikes_in_stations, broken_bikes, delta_distance, clusters = simulation(num_stat,num_trucks,lambda_bike,norm_factor,load_rate,truck_cap,time_of_day,num_steps,plot_animation,save_animation)

# running statistics
distance = [np.cumsum(i) for i in delta_distance]
avg_distance = [[distance[i][j] / (j+1) for j in range(num_steps+1)] for i in range(num_trucks)]
bikes_in_clusters = [[bikes_in_stations[i] for i in range(len(bikes_avail)) if clusters[i] == j] for j in range(num_trucks)] # used for perc_good_bikes
bikes_per_cluster = [sum(i) for i in bikes_in_clusters] # used for perc_good_bikes
print(bikes_per_cluster)
perc_good_bikes = [[100*(1 - (broken_bikes[i][j] / bikes_per_cluster[i])) for j in range(num_steps+1)] for i in range(num_trucks)]

#final statistics
total_distance = [sum(i) for i in delta_distance]
average_speed = [total_distance[i] / times[-1] for i in range(num_trucks)]
perc_idle_time = [sum([times[i+1] - times[i] for i in range(num_steps) if delta_distance[j][i] == 0]) / times[-1] for j in range(num_trucks)]
print(perc_idle_time)
print(total_distance)
print(average_speed) # set norm_factor such that average speed is 16km/h at perc_idle_time = 0 (http://www.wnyc.org/story/traffic-speeds-slow-nyc-wants-curb-car-service-growth/)

# plotting results
rcParams['grid.alpha'] = 15
rcParams['grid.color'] = 'k'
rcParams['legend.edgecolor'] = 'k'
rcParams['legend.facecolor'] = 'w'
#print(rcParams.keys())
cluster_col = ['blue','green','gray','orange','cyan']
f, axarr = plt.subplots(2, 2)
for i in range(num_trucks):
	axarr[0, 0].plot(times,distance[i],color=cluster_col[i],label = 'Truck ' + str(i+1))
	axarr[0, 1].plot(times,avg_distance[i],color=cluster_col[i],label = 'Truck ' + str(i+1))
	axarr[1, 0].plot(times,broken_bikes[i],color=cluster_col[i],label = 'Truck ' + str(i+1))
	axarr[1, 1].plot(times,perc_good_bikes[i],color=cluster_col[i],label = 'Truck ' + str(i+1))
# setting titles for plots
axarr[0, 0].set_title('Total Distance Traveled [km]')
axarr[0, 1].set_title('Average Distance per time step [km]')
axarr[1, 0].set_title('Number of Broken Bikes in System')
axarr[1, 1].set_title('% of Functioning Bikes to Total Number of Bikes')
# setting y axis limits
axarr[0, 0].set_ylim([0, 12000])
axarr[0, 1].set_ylim([0, 2.5])
axarr[1, 0].set_ylim([0, 1000])
axarr[1, 1].set_ylim([50, 101])
# setting axis labels
axarr[0, 0].set_xlabel('time [hours]')
axarr[0, 1].set_xlabel('time [hours]')
axarr[1, 0].set_xlabel('time [hours]')
axarr[1, 1].set_xlabel('time [hours]')
axarr[0, 0].set_ylabel('Distance [km]')
axarr[0, 1].set_ylabel('Distance [km]')
axarr[1, 0].set_ylabel('# Broken Bikes')
axarr[1, 1].set_ylabel('% Functioning Bikes')
# setting background color
axarr[0,0].set_axis_bgcolor('none')
axarr[0,1].set_axis_bgcolor('none')
axarr[1,0].set_axis_bgcolor('none')
axarr[1,1].set_axis_bgcolor('none')
# setting legends
axarr[0,0].legend(loc='upper left')
axarr[0,1].legend(loc='upper right')
axarr[1,0].legend(loc='upper left')
axarr[1,1].legend(loc='lower right')

plt.show()

