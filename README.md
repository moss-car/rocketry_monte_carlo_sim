TO FETCH DATA YOURSELF YOU NEED AN API KEY FOR ECMWF. Follow the following steps:
1. create an EMCWF account by going to https://cds.climate.copernicus.eu/how-to-api and start following instructions for your OS (MacOS/linux only, sorry Windows users)
2. copy the part after key to your clipboard (you only need to get to this part)
3. create a file called `.env` in this directory
4. type `CDSAPI_KEY=<your_key>` into `.env`
It should work now

You can use as a command line tool with the following inputs
```bash
usage: python3 era5_collection.py [<MONTH> [<ITER [<LAT> <LON>]]]
```

Otherwise I can send data to you if you want to test yourself :)