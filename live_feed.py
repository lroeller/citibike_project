# station status feed
def station_status(key):
	# keys:'eightd_has_available_keys', 'is_installed', 'is_renting',
    #  'is_returning', 'last_reported', 'num_bikes_available',
    #  'num_bikes_disabled', 'num_docks_available', 'num_docks_disabled',
    #  'station_id'
	from urllib.request import urlopen
	import json
	import pandas as pd

	data_stations = urlopen("https://gbfs.citibikenyc.com/gbfs/en/station_status.json").read().decode("utf-8")
	data_stations = json.loads(data_stations)
	data_stations = data_stations["data"]
	data_stations = data_stations["stations"]
	data_stations = pd.DataFrame(data_stations)

	return list(data_stations[key])
	
# system alerts
def system_alerts(key):
	from urllib.request import urlopen
	import json
	import pandas as pd

	data_alerts = urlopen("https://gbfs.citibikenyc.com/gbfs/en/system_alerts.json").read().decode("utf-8")
	data_alerts = json.loads(data_alerts)
	data_alerts = data_alerts["data"]
	data_alerts = data_alerts["alerts"]
	data_alerts = pd.DataFrame(data_alerts)
	return data_alerts.keys()
	#return list(data_alerts[key])

# station information
def station_info(key):
	# 'capacity', 'eightd_has_key_dispenser', 'lat', 'lon', 'name',
    #  'region_id', 'rental_methods', 'short_name', 'station_id'
	from urllib.request import urlopen
	import json
	import pandas as pd

	data_station_info = urlopen("https://gbfs.citibikenyc.com/gbfs/en/station_information.json").read().decode("utf-8")
	data_station_info = json.loads(data_station_info)
	data_station_info = data_station_info["data"]
	data_station_info = data_station_info["stations"]
	data_station_info = pd.DataFrame(data_station_info)
	return list(data_station_info[key])