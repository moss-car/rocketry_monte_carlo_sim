import cdsapi

LAT=-12.378
LON=136.821

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": ["2024"],
    "month": [
        "01",
    ],
    "day": [
        "01",
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "pressure_level": [
        "1", "2", "3",
        "5", "7", "10",
        "20", "30", "50",
        "70", "100", "125",
        "150", "175", "200",
        "225", "250", "300",
        "350", "400", "450",
        "500", "550", "600",
        "650", "700", "750",
        "775", "800", "825",
        "850", "875", "900",
        "925", "950", "975",
        "1000"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [90, 179.5, 89.5, 180]
}

target = "download.nc"

client = cdsapi.Client(debug=True)

#this dataset is hella slow, and we can only get about 1 day out of it at a time. I'm not sure if its worthwhile
#if it takes us 2m to retrieve one day of data.
try:
    client.retrieve(dataset, request, target)
except Exception as e:
    print(e)