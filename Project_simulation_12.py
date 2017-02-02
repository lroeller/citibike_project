# Transportation Project Simulation
def simulation(num_stat,num_trucks,lambda_bike,norm_factor,load_rate,truck_cap,time_of_day,
	num_steps,plot_animation,save_animation):
	# import libraries
	import matplotlib.pyplot as plt
	from matplotlib import style
	import random as rd
	import math
	from live_feed import station_status, station_info
	import google_map_api_cornerpoints
	from io import BytesIO
	from PIL import Image
	from matplotlib import rcParams
	from sklearn.cluster import KMeans
	import numpy as np
	np.seterr(all='ignore') # ignore division by 0 error (we set equal to Inf in that case)
	import csv
	import pandas as pd

	# events are: 	t_a[i]: bike breaks down on station i
	#				t_d[i]: truck i arrives at a station

	# creating styled window for graphs
	rcParams['figure.figsize'] = (10, 10) #Size of figure
	rcParams['grid.alpha'] = 0
	style.use('fivethirtyeight')

	# repair station locations
	rep_station_x_pos = [-73.9952, -74.0124]
	rep_station_y_pos = [40.7512 , 40.65]
	num_rep_stat = len(rep_station_x_pos)
	rep_locations = [[rep_station_x_pos[i],rep_station_y_pos[i]] for i in range(num_rep_stat)]

	# bike stations
	station_lat = station_info('lat')
	station_lon = station_info('lon')
	station_x_pos = station_lon[0:num_stat]
	station_y_pos = station_lat[0:num_stat]
	markersize = np.array([20 for i in range(num_stat)])

	# finding clusters
	data_cluster = list(zip(station_x_pos,station_y_pos))
	kmeans = KMeans(n_clusters=num_trucks, random_state=0).fit(data_cluster)
	cluster_labels = kmeans.labels_.tolist()
	cluster_centers = kmeans.cluster_centers_

	# finding closest repair station for each cluster
	def closest_node(node, nodes):
		nodes = np.asarray(nodes)
		dist_2 = np.sum((nodes - node)**2, axis=1)
		return np.argmin(dist_2)

	closest_rep_station = [closest_node(cluster_centers[i],rep_locations) for i in range(num_trucks)]

	# adding repair station location to end of bike station positions
	# (needed so that we can send the trucks to the repair stations simply by sending them to
	# the last station_x_pos and station_y_pos)
	station_x_pos = station_x_pos + rep_station_x_pos
	station_y_pos = station_y_pos + rep_station_y_pos

	# initial truck positions (set to first bike station location of each cluster)
	truck_x_pos = [station_x_pos[cluster_labels.index(i)] for i in range(num_trucks)]
	truck_y_pos = [station_y_pos[cluster_labels.index(i)] for i in range(num_trucks)]

	# mapping between latitude/longitude and pixels for plotting
	# needed because the google map in the background does not use latitude/ longitude information
	# but the station and truck locations do
	centerLat = 40.73
	centerLon = -73.9851
	zoom = 12
	mapWidth = 640
	mapHeight = 640
	centerPoint = google_map_api_cornerpoints.G_LatLng(centerLat, centerLon)
	corners = google_map_api_cornerpoints.getCorners(centerPoint, zoom, mapWidth, mapHeight)
	# the mapping is (x_plot, y_plot) = (longitude * m_x + n_x,latitude * m_y + n_y)
	m_x = 1279.5 / (corners['E']-corners['W'])
	n_x = -corners['W']*m_x
	m_y = 1279.5 / (corners['S']-corners['N'])
	n_y = -corners['N']*m_y

	# map the bike station positions to the plotting reference frame
	station_plot_x = np.array([i*m_x + n_x for i in station_x_pos])
	station_plot_y = np.array([i*m_y + n_y for i in station_y_pos])

	# map the repair station positions to the plotting reference frame
	rep_station_plot_x = [i*m_x + n_x for i in rep_station_x_pos]
	rep_station_plot_y = [i*m_y + n_y for i in rep_station_y_pos]

	# image of NYC (background)
	image = Image.open("file.png")

	# colors to use for plotting clusters (up to 5 clusters supported)
	cluster_col = ['blue','green','gray','orange','cyan']

	# stores list of station-ids. This is the order in which all stations will be referenced
	station_order = list(map(int, station_info('station_id')))

	# create a pandas data frame of zeros with colum names "columns" and index station_id
	columns = ['night','morning','noon','afternoon','evening']
	data = np.empty((len(station_order),5,))
	data[:] = np.zeros(1)
	station_df  = pd.DataFrame(data,index = station_order,columns = columns)
	station_df.index.name = 'station_id'

	# read in the bike ride distribution data gathered in R
	station_df2 = pd.read_csv('frequency.csv',index_col='station_id')

	# superimpose the data onto the pandas dataframe. This is necessary because stations are
	# omitted from the bike ride distribution if there were no bikes going to them. This 
	# step sets the values for those stations to zero.
	station_df.update(station_df2)

	# the bike ride distribution is "bikes_avail". Multiply by the single bike breakdown rate
	# to get the overall bike breakdown rate at each station (lambda stations)
	bikes_avail = list(np.array(station_df[time_of_day])*14/(60*30))[0:num_stat]
	lambda_stations = np.array([i*lambda_bike for i in bikes_avail])

	# number of bikes at each station. Used to find total number of bikes in the streets
	bikes_in_stations = station_status('num_bikes_available')[0:num_stat]

	# initial truck rates to reach destinations
	mu = [norm_factor / (10 + load_rate) for i in range(num_trucks)]

	# lists that will be returned by the function for analysis
	time_list = [0] # list of times at which the events occur
	# list of lists where each list records the number of broken bikes in the system at each event time
	num_broken_bikes = [[0] for i in range(num_trucks)]
	# distance traveled by each truck between the last event time and current event time
	delta_dist = [[0] for i in range(num_trucks)]

	# font used for text on the plot
	font_dict = {'family':'serif','color':'black','size':15}

	# definitions for loop
	R = 6371 # earth radius(used to find distance traveled by the trucks)
	t = 0 # initializing current time
	# initializing current number of broken bikes in each cluster (list of length(num_trucks))
	n = [0 for i in range(num_trucks)]
	# order is a list of list where each list stores station id's of the broken bikes.
	order = [[] for i in range(num_trucks)]	# Each time there is a new broken bike it's
											# station-id is appended to order
	last = [cluster_labels.index(i) for i in range(num_trucks)] # stores last station-id each truck was at
	current = [cluster_labels.index(i) for i in range(num_trucks)] # stores station-id each truck is going to
	next_dest = [float('NaN') for i in range(num_trucks)] 	# stores next station-id that each truck will
															# go to after "current"
	current_num_bikes = [0 for i in range(num_trucks)] 	# stores the current number of bikes that need
														# to be picked up at "current"
	bikes_on_truck = [0 for i in range(num_trucks)] # stores the number of bikes on each truck
	# stores the last time each truck arrived at a bike station location 
	t_last_d = [0 for i in range(num_trucks)]
	t_perc = [0 for i in range(num_trucks)] # stores how much of the current trip each
											# truck has completed (between 0 and 1)

	# initializing next arrivals
	t_d = [float('Inf') for i in range(num_trucks)] # Next arrival time for each truck. Infinity because
													# the trucks won't move until there is a broken bike
	t_a = list(-math.log(rd.uniform(0,1)) / lambda_stations) # time of next broken bike at each station

	# main loop
	for i in range(num_steps):
		if plot_animation: # only plot if set to True since plotting takes a LOT of time
			plt.clf() # erase current plot
			plt.imshow(image) # insert background of Manhattan
			ax = plt.gca()
			ax.set_axis_bgcolor('none')
			# map the updated truck positions to the plotting reference frame
			truck_plot_x = [i*m_x + n_x for i in truck_x_pos]
			truck_plot_y = [i*m_y + n_y for i in truck_y_pos]
			for clusters in range(num_trucks): # for each of the clusters
				# only plot the current cluster stations (we are looping through each)
				which_points = [i == clusters for i in cluster_labels]
				plt.scatter(station_plot_x[0:-num_rep_stat][which_points], 
							station_plot_y[0:-num_rep_stat][which_points], 
							color=cluster_col[clusters],
							s = markersize[which_points])
			plt.scatter(truck_plot_x, truck_plot_y, color='red',s=100) # plot updated truck positions
			plt.scatter(rep_station_plot_x,rep_station_plot_y,color = 'black', s = 40) # plot repair station positions
			plt.text(0, -20,"broken bikes = " + str(n), fontdict=font_dict) # plot number of broken bikes for each cluster
			#plt.text(1400, 50,"mu = " + str(['{0:.3g}'.format(mu[i]) for i in range(num_trucks)]), fontdict=font_dict)
			#plt.text(1400, 100,"t = " + str('{0:.3g}'.format(t)), fontdict=font_dict)
			plt.text(500, -20,"bikes on truck = " + str(bikes_on_truck), fontdict=font_dict) # plot number of bikes on each truck

			plt.pause(0.00001) # pause a short time so the plot is shown on screen
			# If set to True, each animation step will be saved as a png file.
			# This is used to create gifs of the animation. It slows down the simulation by a lot!
			if save_animation:
				plt.savefig(str(i) + '.png', dpi=100) # save as individual png files (to create a gif)


		# find which t_d or t_a is the minimum
		val, counter = min((val, idx) for (idx, val) in enumerate(t_d+t_a))

		# if the next event is a truck arrival
		if counter < len(t_d):
			t_perc[counter] = 0 # the truck has completed 0% of its trip to the next destination
								# (after the one at which it has just arrived)
			t_last_d[counter] = t_d[counter] 	# the last arrival time is the current arrival time
												# (since it has just arrived)
			t = t_d[counter] # the current time is now the arrival time of the truck
			time_list.append(t) # append current event time to list of event times (used for analysis of run)
			n[counter] -= current_num_bikes[counter] # the truck picks up all bikes from the current location
													 # so there are less broken bikes on the streets now
			[num_broken_bikes[i].append(n[i]) for i in range(num_trucks)] # record current number of broken bikes
																		  # on streets (used for analysis of run)
			if current_num_bikes[counter] == 0: # if we have arrived at a repair station location
				bikes_on_truck[counter] = 0 # unload all the bikes currently on the truck
			else: # if we are at a bike station location
				bikes_on_truck[counter] += current_num_bikes[counter]	# load all currently broken bikes from the
																		# station to the truck
				# update markersize for station on plot
				markersize[current[counter]] = markersize[current[counter]] / (2**current_num_bikes[counter])
			
			# set updated position of truck
			x_dist = math.radians(station_x_pos[current[counter]] - truck_x_pos[counter]) \
			* math.cos(math.radians(station_y_pos[current[counter]]+truck_y_pos[counter]) / 2)
			y_dist = math.radians(station_y_pos[current[counter]] - truck_y_pos[counter])
			delta_dist[counter].append(math.sqrt(x_dist**2 + y_dist**2) * R)
			truck_x_pos[counter] = station_x_pos[current[counter]]
			truck_y_pos[counter] = station_y_pos[current[counter]]

			# set position of other trucks
			for z in range(num_trucks):
				if z != counter:
					t_perc[z] = (t - t_last_d[z]) / (t_d[z] - t_last_d[z])
					x_dist = math.radians(station_x_pos[last[z]] + t_perc[z]*(station_x_pos[current[z]] \
						- station_x_pos[last[z]]) - truck_x_pos[z]) * math.cos(math.radians(station_y_pos[last[z]] \
						+ t_perc[z]*(station_y_pos[current[z]] - station_y_pos[last[z]])+truck_y_pos[z]) / 2)
					y_dist = math.radians(station_y_pos[last[z]] + t_perc[z]*(station_y_pos[current[z]] \
						- station_y_pos[last[z]]) - truck_y_pos[z])
					delta_dist[z].append(math.sqrt(x_dist**2 + y_dist**2) * R)
					truck_x_pos[z] = station_x_pos[last[z]] + t_perc[z]*(station_x_pos[current[z]] \
						- station_x_pos[last[z]])
					truck_y_pos[z] = station_y_pos[last[z]] + t_perc[z]*(station_y_pos[current[z]] \
						- station_y_pos[last[z]])

			# if there are no trucks left to pick up and if the truck has not reached its full capacity yet
			if n[counter] == 0 and bikes_on_truck[counter] < truck_cap:
				last[counter] = current[counter] # the last arrival station-id is set to the current one
				#current = float('NaN')
				t_d[counter] = float('Inf') # the next truck arrival time is infinity since it has nowhere to go for now
				current_num_bikes[counter] = 0 # there are no bikes at the next location (since there is no next location)
				order[counter] = [value for value in order[counter] if value != current[counter]] # delete the bikes that
																								  # have been picked up
																								  # from "order"
			else: # if there are other bikes to pick up or if the truck capacity has been reached
				if bikes_on_truck[counter] >= truck_cap: # if the truck capacity has been reached
					next_dest[counter] = num_stat + closest_rep_station[counter] # set the next_destination to repair station
					current_num_bikes[counter] = 0 # there are no bikes to pick up at the repair station location
				else: # if there are other bikes to pick up (at other stations)
					next_dest[counter] = max(set(order[counter]), key=order[counter].count) # set the next destination to the
																							# one with the most number of 
																							# broken bikes
					# set to number of broken bikes at next location
					current_num_bikes[counter] = len([i for i, j in enumerate(order[counter]) if j == next_dest[counter]])
					# delete the bikes that have been picked up from "order"
					order[counter] = [value for value in order[counter] if value != next_dest[counter]]
				
				# update travel time rate to account for distance between current location and next location
				mu[counter] = norm_factor / (load_rate + math.sqrt((station_x_pos[next_dest[counter]] \
					- truck_x_pos[counter])**2 + (station_y_pos[next_dest[counter]] - truck_y_pos[counter])**2))
				t_d[counter] = t - math.log(rd.uniform(0,1)) / mu[counter] # simulate next arrival time of this truck

				# update "last" and "current" since the truck has arrived at a station and is now going to the next station
				last[counter] = current[counter]
				current[counter] = next_dest[counter]
				next_dest[counter] = float('NaN') # we don't know yet what the next destination will be after the current one

		else: # if the next event is a bike breakdown
			counter = counter - len(t_d) # counter is the index of which station it is
			t = t_a[counter] # update the time to reflect that an arrival has happened
			time_list.append(t) # append current time to list of event times (for later analysis)
			cluster = cluster_labels[counter] # figure out in which cluster the arrival happened
			# update how far trucks have traveled to their destinations (since time has progressed)
			for z in range(num_trucks):
				t_perc[z] = (t - t_last_d[z]) / (t_d[z] - t_last_d[z])

			n[cluster] += 1 # add 1 to the number of broken bikes on the streets
			[num_broken_bikes[i].append(n[i]) for i in range(num_trucks)] # append the current number of broken
																		  # bikes on the streets (for later analysis)
			t_a[counter] = t - math.log(rd.uniform(0,1)) /  lambda_stations[counter] # simulate next broken bike
																					 # time at that station
			
			if n[cluster] == 1 and bikes_on_truck[cluster] < truck_cap: # if the broken bike is the first one in the
																		# cluster and if the truck capacity has
																		# not been reached
				last[cluster] = current[cluster] # update last station-id to current one
				current[cluster] = counter # update the current station to the one in which the broken bike happened

				# updated travel time rate
				mu[cluster] = norm_factor / (load_rate + math.sqrt((station_x_pos[counter] - truck_x_pos[cluster])**2 \
					+ (station_y_pos[counter] - truck_y_pos[cluster])**2))
				t_d[cluster] = t - math.log(rd.uniform(0,1)) / mu[cluster] # simulate travel time of truck
				current_num_bikes[cluster] = 1 # set number of broken bikes at the station that we are going to to 1

				# update position of all other trucks
				for y in range(num_trucks):
					if y != cluster:
						x_dist = math.radians(station_x_pos[last[y]] + t_perc[y]*(station_x_pos[current[y]] \
							- station_x_pos[last[y]]) - truck_x_pos[y]) * math.cos(math.radians(station_y_pos[last[y]] \
							+ t_perc[y]*(station_y_pos[current[y]] - station_y_pos[last[y]])+truck_y_pos[y]) / 2)
						y_dist = math.radians(station_y_pos[last[y]] + t_perc[y]*(station_y_pos[current[y]] \
							- station_y_pos[last[y]]) - truck_y_pos[y])
						delta_dist[y].append(math.sqrt(x_dist**2 + y_dist**2) * R)
						truck_x_pos[y] = station_x_pos[last[y]] + t_perc[y]*(station_x_pos[current[y]] - station_x_pos[last[y]])
						truck_y_pos[y] = station_y_pos[last[y]] + t_perc[y]*(station_y_pos[current[y]] - station_y_pos[last[y]])
				delta_dist[cluster].append(0)

			else: # if the broken bike is not the first one in the cluster or if the truck capacity has been reached
				# update all truck positions
				for y in range(num_trucks):
					x_dist = math.radians(station_x_pos[last[y]] + t_perc[y]*(station_x_pos[current[y]] - station_x_pos[last[y]]) \
						- truck_x_pos[y]) * math.cos(math.radians(station_y_pos[last[y]] + t_perc[y]*(station_y_pos[current[y]] \
							- station_y_pos[last[y]])+truck_y_pos[y]) / 2)
					y_dist = math.radians(station_y_pos[last[y]] + t_perc[y]*(station_y_pos[current[y]] \
						- station_y_pos[last[y]]) - truck_y_pos[y])
					delta_dist[y].append(math.sqrt(x_dist**2 + y_dist**2) * R)
					truck_x_pos[y] = station_x_pos[last[y]] + t_perc[y]*(station_x_pos[current[y]] - station_x_pos[last[y]])
					truck_y_pos[y] = station_y_pos[last[y]] + t_perc[y]*(station_y_pos[current[y]] - station_y_pos[last[y]])
				if current[cluster] == counter: # if the bike breakdown happened at the station that we are currently driving to
					current_num_bikes[cluster] += 1 # add 1 to the number of broken bikes at the station we are going to
				else: # if the bike breakdown did not happen at the station that we are currently driving to
					order[cluster].append(counter) # add the station-id of the broken bike to "order"

			markersize[counter] = markersize[counter] * 2 # update markersize of the station at which the breakdown happened

	return time_list, bikes_avail, bikes_in_stations, num_broken_bikes, delta_dist, cluster_labels