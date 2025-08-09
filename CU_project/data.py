import xarray as xr

# Replace this path with the actual .nc file
ds = xr.open_dataset("C:/Users/Mohamed.Benzarti/Downloads/AIFS_test_2t_tp_2023062500.nc")

print(ds)
