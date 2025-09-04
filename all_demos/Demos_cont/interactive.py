import xarray as xr
import panel as pn
import hvplot.xarray
import numpy as np
import geoviews as gv
import geopandas as gpd
import plotly.graph_objs as go

gv.extension('bokeh')
pn.extension('plotly')

# Load dataset
ds = xr.open_dataset("C:/Users/Mohamed.Benzarti/Downloads/AIFS_test_2t_tp_2023062500.nc")

# Adjust longitudes
if ds.lon.max() > 180:
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

# Load world shapefile
world = gpd.read_file("C:/Users/Mohamed.Benzarti/Desktop/shapes/ne_110m_admin_0_countries.shp")

# Variables and forecast steps
variables = ['2t', 'tp']
steps = ds.sizes['step']

# Domain definitions
domains = {
    'Global':   {"lon_min": float(ds.lon.min()), "lon_max": float(ds.lon.max()),
                 "lat_min": float(ds.lat.min()), "lat_max": float(ds.lat.max()), "country": None},

    'Kenya':    {"lon_min": 33.5, "lon_max": 42.2, "lat_min": -5.0, "lat_max": 5.5, "country": "Kenya"},
    'Nigeria':  {"lon_min": 3.5,  "lon_max": 14.5, "lat_min": 4.0,  "lat_max": 14.5, "country": "Nigeria"},
    'Chile':    {"lon_min": -75,  "lon_max": -66,  "lat_min": -56,  "lat_max": -17, "country": "Chile"},
}

# UI widgets
var_selector = pn.widgets.Select(name='Variable', options=variables, value='2t')
step_slider = pn.widgets.IntSlider(name='Forecast Step', start=0, end=steps - 2, value=0)
region_selector = pn.widgets.Select(name='Zoom Region', options=list(domains.keys()), value='Global')


@pn.depends(var_selector, step_slider, region_selector)
def make_output(variable, step_index, region):
    bounds = domains[region]
    zoom = dict(
        lon=slice(bounds["lon_min"], bounds["lon_max"]),
        lat=slice(bounds["lat_max"], bounds["lat_min"])  # lat is descending
    )

    # Map: use selected step
    da = ds[variable].isel(time=0, step=step_index)
    if variable == '2t':
        da = da - 273.15
        da.attrs['units'] = '°C'
    elif variable == 'tp':
        da.attrs['units'] = 'mm'

    da_interp = da.interp(
        lat=np.linspace(float(da.lat.max()), float(da.lat.min()), da.sizes['lat'] * 2),
        lon=np.linspace(float(da.lon.min()), float(da.lon.max()), da.sizes['lon'] * 2)
    )

    valid_time = np.datetime64(ds.time.values[0]) + np.timedelta64(int(ds.step.values[step_index]), 'h')

    img = da_interp.hvplot.image(
        x='lon', y='lat',
        geo=True, coastline=True, cmap='viridis',
        clim=(da.min().item(), da.max().item()),
        frame_height=600, frame_width=1190,
        title=f"{variable.upper()} at step {int(ds.step.values[step_index])}h (valid: {valid_time})"
    )

    if bounds["country"]:
        country_geom = world[world["NAME"] == bounds["country"]]
        country_border = gv.Polygons(country_geom).opts(
            line_color='black', line_width=3, fill_alpha=0
        )
        img = img * country_border

    img = img.opts(
        xlim=(bounds["lon_min"], bounds["lon_max"]),
        ylim=(bounds["lat_min"], bounds["lat_max"])
    )

    # Line chart: simulate 6 hourly values via interpolation
    da_full = ds[variable].isel(time=0).sel(**zoom)
    if variable == '2t':
        da_full = da_full - 273.15

    y_vals = []
    if step_index + 1 < steps:
        v0 = da_full.isel(step=step_index).mean().item()
        v1 = da_full.isel(step=step_index + 1).mean().item()
        for h in range(7):  # hours 0 to 6
            frac = h / 6
            interp_val = v0 + frac * (v1 - v0)
            y_vals.append(interp_val)
    else:
        val = da_full.isel(step=step_index).mean().item()
        y_vals = [val] * 7

    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=list(range(7)),
        y=y_vals,
        mode='lines+markers',
        line=dict(color='firebrick'),
        name=variable.upper()
    ))

    chart.update_layout(
        title=f"{variable.upper()} estimated values for next 6 hours",
        xaxis_title="Hour",
        yaxis_title=da.attrs['units'],
        height=400,
        width=1190
    )

    return pn.Column(img, pn.pane.Plotly(chart))


layout = pn.Column(
    "# 🌍 Atmospheric Interactive Map",
    pn.Row(var_selector, step_slider, region_selector),
    make_output
)

layout.servable()