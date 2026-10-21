from dotenv import load_dotenv
import xarray as xr
from aiohttp import client_exceptions
import datetime, random, os, sys
import numpy as np
import rocketpy as rc
from simulate_spaceshot import *

LAT=-12.378
LON=136.821
MONTH=7
ITER=10

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

def fetch_dataset(lat, lon, path):
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
        latitude=slice(lat -0.25, lat + 0.25), 
        longitude=slice(lon - 0.25, lon + 0.25), 
        time=slice("2000-01-01", "2025-12-31"),
        drop=False,
    )
    dp = dp[["u", "v", "t", "z"]]
    
    dp.to_netcdf(path)

def gen_rand_time(month):
    #dire generation method but oh well
    start = datetime.datetime(year=2000, month=1, day=1)
    end = datetime.datetime(year=2026, month=1, day=1) - datetime.timedelta(seconds=1)
    total_secs = int((end - start).total_seconds())
    rand_time=start + datetime.timedelta(seconds=random.randint(0, total_secs))
    if month is not None:
        while (rand_time.month != month):
            rand_time = start + datetime.timedelta(seconds=random.randint(0, total_secs))
    return rand_time

def summarise(flight : rc.Flight, time : datetime):
    t_end = float(flight.t_final)
    
    p_events = [(t, flight.altitude(t), abs(flight.vz(t))) for t, _ in flight.parachute_events]
    ds = xr.Dataset(
        {
            "apogee" : ("time", [flight.apogee_time]),
            "apogee_disp" : ("time", [flight.drift(flight.apogee_time)]),
            "landing_distance" : ("time", [flight.drift(t_end)]),
            "impact_speed" : ("time", [abs(flight.vz(t_end))]),
            # "flight_time" : ("time", [t_end]),
            "drogue_time" : ("time", [p_events[0][0]]),
            "drogue_alt" : ("time", [p_events[0][1]]),
            "drogue_acc" : ("time", [p_events[0][2]]),
            "main_time" : ("time", [p_events[1][0]]),
            "main_alt" : ("time", [p_events[1][1]]),
            "main_acc" : ("time", [p_events[1][2]]),
        },
        coords={
            "time" : [time]
        }
    )
    return ds
    
def main(lat: float = LAT, lon: float = LON, month: int | None = None, iter: int = ITER):
    path = os.path.dirname(__file__) + f"/era5_{lat}_{lon}.nc"

    if not os.path.exists(path) or os.stat(path).st_size == 0:
        print(f"No valid file found at {path}, fetching...")
        fetch_dataset(lat, lon, path)

    print(f"Atmospheric data downloaded")

    env = rc.Environment(latitude=lat, longitude=lon)

    for i in range(iter):
        time=gen_rand_time(month)

        env.set_date(time)
        env.set_atmospheric_model(type="reanalysis", file=path, dictionary=ROCKETPY_TO_ERA5)

        rocket, _ = build_rocket()
        flight = rc.Flight(rocket=rocket, environment=env, rail_length=TOWER_LENGTH,
                            inclination=90.0, heading=0.0,max_time=3000.0)
        trial = summarise(flight, time)
        if i == 0:
            ds = trial
        else:
            ds = xr.concat([ds, trial], dim="time")
    
    apog_mean = ds["apogee"].mean(dim="time")
    lvel_mean = ds["impact_speed"].mean(dim="time")
    lvel_quant = ds["impact_speed"].quantile([0.5, 0.9, 0.95], dim="time")
    lvel_qstr = ", ".join([f"{x * 100}th_percentile={y}" for x,y in zip(lvel_quant["quantile"].values, lvel_quant.values)])
    ld_mean = ds["landing_distance"].mean(dim="time")
    ld_quant = ds["landing_distance"].quantile([0.5, 0.9, 0.95], dim="time")
    ld_qstr = ", ".join([f"{x * 100}th_percentile={y}" for x,y in zip(ld_quant["quantile"].values, ld_quant.values)])

# TODO: Print TF
# Altitude and velocity of the deployment of a drogue, main DONE!
# Parachute dimensions DONE!
# Landing velocity DONE!
# Contribution of launch corridor from descent (total) DONE!
# Contribution of launch corridor from descent (from Coriolis drift)
    print(f"\n---------\n RESULTS \n---------")
    print(f"main_drag_coeff={MAIN_CD_S}, drogue_drag_coeff={DROGUE_CD_S}")
    print(f"landing_velocity_z(m/s): mean={lvel_mean.values}, {lvel_qstr}")
    print(f"landing_distance(m): mean={ld_mean.values}, {ld_qstr}")
    print(f"apogee(s)={apog_mean.values}")
    print(f"parachute_events:")
    print(f"  drogue:")
    print(f"    time={ds["drogue_time"].mean(dim="time").values}, alt={ds["drogue_alt"].mean(dim="time").values}, vel_z={ds["drogue_acc"].mean(dim="time").values}")
    print(f"  main")
    print(f"    time={ds["main_time"].mean(dim="time").values}, alt={ds["main_alt"].mean(dim="time").values}, vel_z={ds["drogue_acc"].mean(dim="time").values}")

#nasty argument parser
if __name__ == "__main__":
    args = sys.argv
    if len(sys.argv) == 5:
        n = int(sys.argv[2])
        lat = float(sys.argv[3])
        lon = float(sys.argv[4])
        month = int(sys.argv[1])
        print(f"Starting sim for month={month}, iter={n}, lat={lat}, lon={lon}")
        main(lat, lon, month, n)
    elif len(sys.argv) == 3:
        n = int(sys.argv[2])
        month = int(sys.argv[1])
        print(f"Starting sim for month={month}, iter={n}")
        main(month=month, iter=n)
    elif len(sys.argv) == 2:
        month = int(sys.argv[1])
        print(f"Starting sim for month={month}")
        main(month=month)
    elif len(sys.argv) > 1:
        print("usage: python3 era5_collection.py [<MONTH> [<ITER [<LAT> <LON>]]]")
    else:
        main()
    