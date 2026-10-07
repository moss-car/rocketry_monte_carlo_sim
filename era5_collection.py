from dotenv import load_dotenv
import xarray as xr
from aiohttp import client_exceptions
import datetime, random, os
import numpy as np
import rocketpy as rc
from simulate_spaceshot import *

LAT=-12.378
LON=136.821
MONTH=7
ITER=2

PATH = os.path.dirname(__file__) + f"/era5_{LAT}_{LON}.nc"

#dictionary for rocketpy
ROCKETPY_TO_ERA5 = {
    "time": "time",
    "latitude": "latitude",
    "longitude": "longitude",
    "level": "pressureLevel",
    "temperature": "t",
    "surface_geopotential_height": None,
    "geopotential_height": None,
    "geopotential": "z",
    "u_wind": "u",
    "v_wind": "v",
}

def fetch_dataset():
    """fetch data closest to LAT and LON from ERA5 6 hourly 1940-2025 dataset, saving
    relevant entries as 'era5_LAT_LON_reanalysis.nc'"""
    load_dotenv()
    cdsapi_key = os.getenv("CDSAPI_KEY")

    if not cdsapi_key:
        raise RuntimeError("CDSAPI_KEY not found")

    url = (   
        "https://arco.datastores.ecmwf.int/"
        "cadl-arco-geo-048/arco/"
        "reanalysis_era5_pressure_levels/pl/"
        "geoChunked.zarr/"
    )

    try:
        ds = xr.open_zarr(
            url,
            consolidated=True,
            storage_options={
                "headers": {"Authorization":f"Bearer {cdsapi_key}"}
            }
        )
    except client_exceptions.ClientResponseError as e:
        print(f"http request failed: {e.message} ({e.status}). Make sure you have"
            "the correct credentials")
        exit(1)

    # get data around LAT/LON so we can bilinearly interpolate exact location
    # don't drop lat/lon so rocketpy can grab it as part of reanalysis
    dp = ds.sel(
        latitude=slice(LAT -0.25, LAT + 0.25), 
        longitude=slice(LON - 0.25, LON + 0.25), 
        time=slice("2000-01-01", "2025-12-31"),
        drop=False,
    )

    dp = dp[["u", "v", "t", "z"]]
    dp.to_netcdf(PATH)

def gen_rand_time(month):
    #dire generation method but oh well
    start = datetime.datetime(year=2000, month=1, day=1)
    end = datetime.datetime(year=2026, month=1, day=1) - datetime.timedelta(seconds=1)
    total_secs = int((end - start).total_seconds())
    rand_time=start + datetime.timedelta(seconds=random.randint(0, total_secs))
    while (rand_time.month != month):
        rand_time=start + datetime.timedelta(seconds=random.randint(0, total_secs))
    return rand_time

def summarise(flight : rc.Flight, time : datetime.datetime):
    t_end = float(flight.t_final)
    out = {
        "launch_month" : time.month,
        "apogee_km": float(flight.apogee - flight.env.elevation) / 1000.0,
        "landing_distance_km": flight.drift(t_end) / 1000.0,
        "impact_speed_ms": abs(flight.vz(t_end)),
        "flight_time_s": t_end, 
        "parachute_events": [(round(float(t), 1), p.name) for t, p in flight.parachute_events],
    }
    return out
    

if not os.path.exists(PATH) or os.stat(PATH).st_size == 0:
    print(f"No valid file found at {PATH}, fetching...")
    fetch_dataset()

print(f"Atmospheric data downloaded")

env = rc.Environment(latitude=LAT, longitude=LON)

run = xr.DataArray()

#TODO: create an array and select 50, 95 and 99% results!
# Altitude and velocity of the deployment of a drogue, main (is a secondary parachute required?)
# Parachute dimensions
# Contribution of launch corridor from descent (from Coriolis drift) - 0m/s sim
for i in range(ITER):
    time=gen_rand_time(MONTH)

    #TODO: is ERA5 time UTC or local (I suspect UTC, but should check). Currently I assume UTC
    env.set_date(time)
    env.set_atmospheric_model(type="reanalysis", file=PATH, dictionary=ROCKETPY_TO_ERA5)

    rocket, _ = build_rocket()
    flight = rc.Flight(rocket=rocket, environment=env, rail_length=TOWER_LENGTH,
                        inclination=90.0, heading=0.0,max_time=3000.0)

    _ = xr.DataArray(summarise(flight, time))
    print(fs)