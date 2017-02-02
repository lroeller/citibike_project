from Project_simulation_12 import simulation
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import rcParams
import numpy as np

# constants
num_steps = 10000
num_stat = 664
num_trucks = 5
norm_factor = 0.148
load_rate = 0.005
plot_animation = False
save_animation = False
last_num_hours = 200 # only look at this many of the last hours

# pick one of these to iterate over
lambda_bike = 0.0187 # see Excel sheet for numbers #[0.001 + i*0.001 for i in range(3)]
truck_cap = [5 + 5*i for i in range(3)]
fixed_cost_truck = 60 # dollars per hour per truck
distance_cost_truck = 5 # dollars per kilometer
time_of_day = 'afternoon' # pick between 'night','morning','noon','afternoon','evening'

# plotting style parameters
rcParams['grid.alpha'] = 15
rcParams['grid.color'] = 'k'
rcParams['legend.edgecolor'] = 'k'
rcParams['legend.facecolor'] = 'w'

# initializing plot
f, axarr = plt.subplots(2, 2)
cluster_cols = ['blue','red','black','green','gray'] # colors to distinguish different runs
counter = 0
for n in truck_cap: # change to whichever one you are iterating over
	# list to store statistics
	total_distance_list = []
	perc_idle_time_list = []
	perc_good_bikes_list = []
	cost_list = []
	for m in range(num_trucks):
		# change name in simulation call to n for the variable that you want to iterate over
		times, bikes_avail, bikes_in_stations, broken_bikes, delta_distance, clusters = simulation(num_stat,m+1,lambda_bike,norm_factor,load_rate,n,time_of_day,num_steps,plot_animation,save_animation)

		# finding step number that starts last hour of simulation run
		index = next((i for i, v in enumerate(times) if v >= (times[-1] - last_num_hours)), -1)
		print(index)
		# statistics (last hours of run)
		delta_distance_concat = [delta_distance[i][index:] for i in range(m+1)]
		total_distance = [sum(i)/(last_num_hours*(m+1)) for i in delta_distance_concat]
		perc_idle_time = [sum([times[i+1] - times[i] for i in range(index,num_steps) if delta_distance[j][i] == 0])*100 / last_num_hours for j in range(m+1)]
		total_broken_bikes = sum([broken_bikes[i][-1] for i in range(m+1)])
		perc_good_bikes = 1 - (total_broken_bikes / sum(bikes_in_stations))

		# cost per hour single run
		cost = (fixed_cost_truck + sum(total_distance)*distance_cost_truck)*(m+1)

		# storing statistics in list
		total_distance_list.append(sum(total_distance))
		perc_idle_time_list.append(np.mean(perc_idle_time))
		perc_good_bikes_list.append(perc_good_bikes)
		cost_list.append(cost)

	# plotting results
	axarr[0, 0].plot([i+1 for i in range(num_trucks)],total_distance_list,'-o',color = cluster_cols[counter],label = 'truck capacity = ' + str(n))
	axarr[0, 1].plot([i+1 for i in range(num_trucks)],perc_idle_time_list,'-o',color = cluster_cols[counter],label = 'truck capacity = ' + str(n))
	axarr[1, 0].plot([i+1 for i in range(num_trucks)],perc_good_bikes_list,'-o',color = cluster_cols[counter],label = 'truck capacity = ' + str(n))
	axarr[1, 1].plot([i+1 for i in range(num_trucks)],cost_list,'-o',color = cluster_cols[counter],label = 'truck capacity = ' + str(n))
	counter += 1

# setting x-axis ticker intervals
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
axarr[0,0].xaxis.set_major_locator(loc)
axarr[0,1].xaxis.set_major_locator(loc)
axarr[1,0].xaxis.set_major_locator(loc)
axarr[1,1].xaxis.set_major_locator(loc)

# setting axis labels
axarr[0, 0].set_xlabel('# Trucks')
axarr[0, 1].set_xlabel('# Trucks')
axarr[1, 0].set_xlabel('# Trucks')
axarr[1, 1].set_xlabel('# Trucks')
axarr[0, 0].set_ylabel('Distance [km]')
axarr[0, 1].set_ylabel('Idle Time [%]')
axarr[1, 0].set_ylabel('% Functioning Bikes')
axarr[1, 1].set_ylabel('Total Cost [$ / hour]')

# setting titles for plots
axarr[0, 0].set_title('Total Distance Traveled per Truck [km]')
axarr[0, 1].set_title('Idle Time [%]')
axarr[1, 0].set_title('Percentage of Functioning Bikes to Total Number of Bikes')
axarr[1, 1].set_title('Total Cost [$ / hour]')
# axarr[1, 1].set_ylim([0.9, 1])

# setting legends
axarr[0,0].legend(loc='upper right')
axarr[0,1].legend(loc='upper left')
axarr[1,0].legend(loc='lower right')
axarr[1,1].legend(loc='lower right')

# turning on grid
axarr[0,0].grid()
axarr[0,1].grid()
axarr[1,0].grid()
axarr[1,1].grid()

plt.show()

