# Standard library imports
import os
import sys
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error

import ipywidgets as widgets
from ipywidgets import (
    Dropdown,
    IntSlider,
    Checkbox,
    HBox,
    VBox,
    Button,
    Output,
    Layout,
    interact,
    interactive_output,
)
from IPython.display import display, clear_output

# Local application imports
import logger
# import torch  # Uncomment if needed

# Suppress warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (12, 6)


# data loading, derivations
def load_wrf_dataset(wrf_path, group):
    import h5py
    import xarray as xr
    import numpy as np

    with h5py.File(wrf_path, "r") as f:
        group = f[group]
        data_vars = {}

        time = group["time"][:]
        lat = group["latitude"][:]
        lon = group["longitude"][:]

        for var_name in group.keys():
            if var_name in ["time", "latitude", "longitude"]:
                continue  # skip coord vars

            data = group[var_name][:]
            if data.ndim == 3:
                dims = ("time", "lat", "lon")
                data_vars[var_name] = (dims, data)

        # Build coords dictionary
        coords = {"time": time, "lat": lat, "lon": lon}

        ds_wrf = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds_wrf


def print_ds_info(ds, name):
    print(f"Dataset: {name}")
    print(ds)
    print("\nVariables:\n", list(ds.data_vars))
    print("\nCoordinates:\n", ds.coords)
    print("-" * 100)


CC = 0
K1 = 0.4
K2 = 0.4
K3 = 0.1
K4 = 0.1


def fog_index(T2, Q2, PSFC, U10, V10):

    T0 = 273.16
    # WS = torch.sqrt(U10**2+V10**2)
    WS = np.sqrt((U10**2) + (V10**2))
    # RH = (0.263*PSFC*Q2)/(torch.exp((17.67*(T2-T0))/(T2-29.65)))
    RH = relative_humidity(T2, Q2, PSFC)
    # TD = 243.04 * (torch.log(RH/100)+((17.625*T2)/(243.04+T2)))/(17.625-torch.log(RH/100)-((17.625*T2)/(243.04+T2)))
    TD = (
        243.04
        * (np.log(RH / 100) + ((17.625 * T2) / (243.04 + T2)))
        / (17.625 - np.log(RH / 100) - ((17.625 * T2) / (243.04 + T2)))
    )
    TDD = T2 - TD
    FI = K1 * (100 - TDD) + K2 * RH + K3 * (5 - WS) + K4 * (1 - CC)

    return FI


def relative_humidity(t2, q2, psfc):
    tc = t2 - 273.15
    es = 6.112 * np.exp((17.67 * tc) / (tc + 243.5))
    r = q2 / (1 - q2)
    p_hpa = psfc / 100
    e = (r * p_hpa) / (0.622 + r)
    rh = (e / es) * 100
    rh = rh.clip(0, 100)

    return rh


def heat_index(t2, rh):
    def hi_func(tf, rh):
        hi = (
            -42.379
            + 2.04901523 * tf
            + 10.14333127 * rh
            - 0.22475541 * tf * rh
            - 0.00683783 * tf**2
            - 0.05481717 * rh**2
            + 0.00122874 * tf**2 * rh
            + 0.00085282 * tf * rh**2
            - 0.00000199 * tf**2 * rh**2
        )
        return hi

    tf = (t2 - 273.15) * 9 / 5 + 32

    hi = hi_func(tf, rh)

    return hi


def compute_derived_variables(
    ds,
    t2="T2",
    q2="Q2",
    psfc="PSFC",
):

    rh = relative_humidity(ds["T2"], ds["Q2"], ds["PSFC"])
    hi = heat_index(ds["T2"], rh)
    fi = fog_index(ds["T2"], ds["Q2"], ds["PSFC"], ds["U10"], ds["V10"])


    ds["RH"] = rh
    ds["HI"] = hi

    ds["HI"] = (5 / 9) * (ds["HI"] - 32) + 273.15
    ds["FI"] = fi

    return ds


# main interactive vis
def interactive_dataset_viewer(datasets, main_title, cmap="coolwarm"):
    max_datasets = len(datasets)
    clear_output(wait=True)

    def find_sample_or_ensemble_dim(da):
        for d in da.dims:
            if "ensemble" in d.lower() or "sample" in d.lower():
                return d
        return None

    def get_colormap_from_corrdiff(var):
        # Can customize this function if needed
        return "coolwarm"

    def get_data_slice(
        ds, var, lead_idx=None, sample_idx=None, mean=False, vertical_idx=None
    ):
        da = ds[var]
        for dim_name in ["lead_time", "time", "valid_time", "Time"]:
            if dim_name in da.dims and lead_idx is not None:
                da = da.isel({dim_name: lead_idx})
                break
        vertical_dim = None
        for vdim in ["bottom_top", "level", "altitude"]:
            if vdim in da.dims:
                vertical_dim = vdim
                break
        if vertical_dim and vertical_idx is not None:
            da = da.isel({vertical_dim: vertical_idx})
        sdim = find_sample_or_ensemble_dim(da)
        if sdim:
            if mean:
                da = da.mean(dim=sdim)
            elif sample_idx is not None:
                da = da.isel({sdim: sample_idx})
        spatial_dims = [
            d
            for d in da.dims
            if "lat" in d
            or "lon" in d
            or "y" in d
            or "x" in d
            or "south_north" in d
            or "west_east" in d
        ]
        extra_dims = [d for d in da.dims if d not in spatial_dims]
        if extra_dims:
            da = da.isel({d: 0 for d in extra_dims})
        return da

    def plot_raster(da, ax, title="", cmap="coolwarm"):
        spatial_dims = [
            d
            for d in da.dims
            if "lat" in d
            or "lon" in d
            or "y" in d
            or "x" in d
            or "south_north" in d
            or "west_east" in d
        ]
        if len(spatial_dims) == 2:
            y_dim, x_dim = spatial_dims
            im = ax.imshow(da.values, origin="lower", cmap=cmap, aspect="equal")
            ax.set_title(title)
            ax.set_xlabel(x_dim)
            ax.set_ylabel(y_dim)
            return im
        else:
            ax.text(0.5, 0.5, f"Cannot plot: not 2D (dims: {da.dims})", ha="center")
            ax.set_title(title)
            return None

    def gather_inputs(**kwargs):
        datasets_vars = []
        for i in range(max_datasets):
            ds = kwargs[f"ds_{i}"]
            var = kwargs[f"var_{i}"]
            lead = kwargs[f"lead_{i}"]
            sample = kwargs[f"sample_{i}"]
            mean = kwargs[f"mean_{i}"]
            vertical = kwargs.get(f"vertical_{i}", None)
            datasets_vars.append((ds, var, lead, sample, mean, vertical))
            # datasets_vars.append((ds, var, lead, sample, mean))

        interactive_plot(datasets, *datasets_vars)

    def interactive_plot(datasets, *datasets_vars):
        clear_output(wait=True)
        datasets_vars = [
            x for x in datasets_vars if x[0] is not None and x[1] is not None
        ]
        n_axes = len(datasets_vars)
        if n_axes == 0:
            print("No datasets selected")
            return
        corrdiff_var = None
        for ds_name, var, *_ in datasets_vars:
            if ds_name == "CorrDiff 2KM" and var is not None:
                corrdiff_var = var
                break

        # Respect user-supplied cmap unless special override
        cmap_to_use = get_colormap_from_corrdiff(corrdiff_var) if corrdiff_var else cmap

        fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
        if n_axes == 1:
            axes = [axes]
        last_im = None
        for ax, (ds_name, var, lead_idx, sample_idx, mean, vertical_idx) in zip(
            axes, datasets_vars
        ):
            da = get_data_slice(
                datasets[ds_name], var, lead_idx, sample_idx, mean, vertical_idx
            )
            last_im = plot_raster(da, ax, title=f"{ds_name} – {var}", cmap=cmap_to_use)
        if last_im is not None:
            cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
            fig.colorbar(
                last_im, cax=cbar_ax, label=corrdiff_var if corrdiff_var else ""
            )
        fig.suptitle(main_title)
        plt.tight_layout()
        plt.show()

    ds_w_list, var_w_list, lead_w_list = [], [], []
    sample_w_list, mean_w_list, vertical_w_list = [], [], []
    row_boxes = []

    def make_update_fn(idx):
        def update_widgets(*args):
            ds_value = ds_w_list[idx].value
            if ds_value:
                ds = datasets[ds_value]
                var_w_list[idx].options = list(ds.data_vars)
                lead_dim = None
                for dim_name in ["lead_time", "time", "valid_time", "Time"]:
                    if dim_name in ds.dims:
                        lead_dim = dim_name
                        break
                if lead_dim:
                    lead_w_list[idx].min = 0
                    lead_w_list[idx].max = ds.dims[lead_dim] - 1
                    lead_w_list[idx].disabled = False
                else:
                    lead_w_list[idx].min = lead_w_list[idx].max = 0
                    lead_w_list[idx].disabled = True
                vertical_dim = None
                for vdim in ["bottom_top", "level", "altitude"]:
                    if vdim in ds.dims:
                        vertical_dim = vdim
                        break
                if vertical_dim:
                    vertical_w_list[idx].min = 0
                    vertical_w_list[idx].max = ds.dims[vertical_dim] - 1
                    vertical_w_list[idx].disabled = False
                else:
                    vertical_w_list[idx].min = vertical_w_list[idx].max = 0
                    vertical_w_list[idx].disabled = True
                sdim = find_sample_or_ensemble_dim(ds)
                if sdim:
                    sample_w_list[idx].min = 0
                    sample_w_list[idx].max = ds.dims[sdim] - 1
                    sample_w_list[idx].disabled = False
                    mean_w_list[idx].disabled = False
                else:
                    sample_w_list[idx].min = sample_w_list[idx].max = 0
                    sample_w_list[idx].disabled = True
                    mean_w_list[idx].disabled = True
            else:
                var_w_list[idx].options = []
                lead_w_list[idx].disabled = True
                sample_w_list[idx].disabled = True
                mean_w_list[idx].disabled = True
                vertical_w_list[idx].disabled = True

        return update_widgets

    for i in range(max_datasets):
        ds_w = Dropdown(
            options=[None] + list(datasets.keys()),
            description=f"Dataset {i + 1}",
            layout=Layout(width="200px"),
        )
        var_w = Dropdown(description=f"Variable {i + 1}", layout=Layout(width="200px"))
        lead_w = IntSlider(
            description=f"Lead {i + 1}", min=0, max=0, layout=Layout(width="300px")
        )
        sample_w = IntSlider(
            description=f"Ensemble {i + 1}",
            min=0,
            max=0,
            disabled=True,
            layout=Layout(width="250px"),
        )
        mean_w = Checkbox(
            value=False,
            description=f"Ensembles Mean {i + 1}",
            disabled=True,
            layout=Layout(width="250px"),
        )
        vertical_w = IntSlider(
            description=f"Vertical {i + 1}",
            min=0,
            max=0,
            disabled=True,
            layout=Layout(width="300px"),
        )
        ds_w_list.append(ds_w)
        var_w_list.append(var_w)
        lead_w_list.append(lead_w)
        sample_w_list.append(sample_w)
        mean_w_list.append(mean_w)
        vertical_w_list.append(vertical_w)
        row_boxes.append(HBox([ds_w, var_w, lead_w, sample_w, mean_w, vertical_w]))

    for i in range(max_datasets):
        ds_w_list[i].observe(make_update_fn(i), names="value")
        make_update_fn(i)()

    widget_dict = {}
    for i in range(max_datasets):
        widget_dict[f"ds_{i}"] = ds_w_list[i]
        widget_dict[f"var_{i}"] = var_w_list[i]
        widget_dict[f"lead_{i}"] = lead_w_list[i]
        widget_dict[f"sample_{i}"] = sample_w_list[i]
        widget_dict[f"mean_{i}"] = mean_w_list[i]
        widget_dict[f"vertical_{i}"] = vertical_w_list[i]

    ui = VBox(row_boxes)
    out = widgets.interactive_output(gather_inputs, widget_dict)
    display(ui, out)



# loss functions
def loss_functions(datasets):
    def find_time_dim(da):
        """Return the first matching time-like dimension name or None."""
        for name in ("lead_time", "time", "valid_time"):
            if name in da.dims:
                return name
        return None

    def slice_time_sample(da, lead_idx, sample_idx=None, ensemble_mean=False):
        """
        Robustly slice a DataArray by whichever time-like dim exists and by sample if present.
        - If desired index > length, it will be clipped to last available index (safe).
        """
        da = da.copy()
        time_dim = find_time_dim(da)
        if time_dim is not None:
            max_idx = da.sizes[time_dim] - 1
            idx = int(min(max_idx, int(lead_idx)))
            da = da.isel({time_dim: idx})
        else:
            # no time-like dim -> do nothing (assume already single-time slice)
            pass

        if "sample" in da.dims:
            if ensemble_mean:
                da = da.mean(dim="sample")
            elif sample_idx is not None:
                max_s = da.sizes["sample"] - 1
                sidx = int(min(max_s, int(sample_idx)))
                da = da.isel(sample=sidx)
        return da

    def spatial_shape(da):
        """Return the two spatial dims and their sizes (assume last two dims are spatial if ambiguous)."""
        # exclude common non-spatial dims
        non_spatial = set(["time", "lead_time", "valid_time", "sample"])
        s_dims = [d for d in da.dims if d not in non_spatial]
        if len(s_dims) >= 2:
            y_dim, x_dim = s_dims[-2], s_dims[-1]
            return y_dim, x_dim, (da.sizes[y_dim], da.sizes[x_dim])
        # fallback: last two dims
        y_dim, x_dim = da.dims[-2], da.dims[-1]
        return y_dim, x_dim, (da.sizes[y_dim], da.sizes[x_dim])

    def has_latlon_coords(da):
        """Return True if da has coords named lat and lon (1D or 2D)."""
        return "lat" in da.coords and "lon" in da.coords

    def make_grid_ds_from_da(da):
        """
        Build a minimal xr.Dataset suitable for xESMF.Regridder from a DataArray.
        It ensures coords 'lat' and 'lon' exist (1D or 2D). If missing, it will
        fabricate index-based coords (NOT geographic) — used only as a last resort.
        """
        da = da.copy()
        # If lat/lon exist, attempt to use them (fine if 1D or 2D)
        if "lat" in da.coords and "lon" in da.coords:
            ds = da.to_dataset(name="var")
            # xESMF accepts ds with 'lat' and 'lon' coords
            return ds

        # try to find candidate coords (e.g., corrdiff_y/corrdiff_x -> treat as index coords)
        y_dim, x_dim, _ = spatial_shape(da)
        # create integer coords
        y_coords = np.arange(da.sizes[y_dim])
        x_coords = np.arange(da.sizes[x_dim])
        # rename dims to 'y','x' for consistency
        da_renamed = da.rename({y_dim: "y", x_dim: "x"})
        # broadcast to 2D lat/lon arrays (index-based)
        lat2, lon2 = np.meshgrid(y_coords, x_coords, indexing="ij")
        ds = da_renamed.to_dataset(name="var")
        ds = ds.assign_coords(lat=(("y", "x"), lat2), lon=(("y", "x"), lon2))
        return ds

    def regrid_to_target(src_da, tgt_da, method="bilinear"):
        """
        Regrid src_da onto the grid of tgt_da and return an xarray.DataArray
        aligned to tgt_da dims/coords.
        Strategies:
        1) If shapes already match, align dims and return src_da aligned without regridding.
        2) If both have 'lat' and 'lon' coords, use xESMF regridder.
        3) Otherwise, fallback to scipy.ndimage.zoom to resize src to tgt shape (fast).
        """
        # slice away any non-spatial dims (we assume src_da and tgt_da already sliced in time/sample)
        src_y, src_x, src_shape = spatial_shape(src_da)
        tgt_y, tgt_x, tgt_shape = spatial_shape(tgt_da)

        # If shapes are identical: quick align by reordering dims if needed
        if src_shape == tgt_shape:
            # rename to same dim names as target if possible
            # try to align by position: map src last two dims -> tgt last two dims
            src_dims = list(src_da.dims)
            tgt_dims = list(tgt_da.dims)
            # create mapping from src spatial dims to tgt spatial dims
            s_y, s_x = src_dims[-2], src_dims[-1]
            t_y, t_x = tgt_dims[-2], tgt_dims[-1]
            # if names differ, rename src dims to tgt dims
            if (s_y != t_y) or (s_x != t_x):
                try:
                    src_da = src_da.rename({s_y: t_y, s_x: t_x})
                except Exception:
                    pass
            # reindex to target coords if coords exist
            try:
                src_aligned = src_da.transpose(*tgt_da.dims)
            except Exception:
                src_aligned = src_da
            return src_aligned

        # If both have lat/lon coords -> xESMF
        if has_latlon_coords(src_da) and has_latlon_coords(tgt_da):
            src_grid = make_grid_ds_from_da(src_da)
            tgt_grid = make_grid_ds_from_da(tgt_da)
            # build regridder (cached)
            key = ("xesmf", src_grid["lat"].shape, tgt_grid["lat"].shape, method)
            if key not in regridder_cache:
                regridder_cache[key] = xe.Regridder(
                    src_grid, tgt_grid, method, reuse_weights=False
                )
            regridder = regridder_cache[key]
            # xESMF expects DataArray with dims (y,x) matching src grid dims; rename as necessary:
            # ensure src_da dims match the src_grid dims ('y','x' or dims in src)
            try:
                res = regridder(src_da)
            except Exception as e:
                warnings.warn(
                    f"xESMF regrid failed: {e}. Falling back to zoom resizing."
                )
                res = None
            if res is not None:
                # ensure res dims align with tgt dims
                # if necessary, rename dims to match tgt
                try:
                    res = res.transpose(*tgt_da.dims)
                except Exception:
                    pass
                return res

        # fallback: numeric resizing via scipy.ndimage.zoom
        sy, sx = src_shape
        ty, tx = tgt_shape
        if sy <= 0 or sx <= 0 or ty <= 0 or tx <= 0:
            raise ValueError("Invalid spatial sizes for regridding.")
        zoom_y = ty / float(sy)
        zoom_x = tx / float(sx)
        arr = src_da.values
        # force 2D if arr has more dims (shouldn't)
        if arr.ndim > 2:
            # keep only the last two dims
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
        resized = zoom(arr, (zoom_y, zoom_x), order=1)  # bilinear-like
        # build DataArray with target dims and coords
        tgt_dims = list(tgt_da.dims)
        # assume last two dims are spatial
        ty_dim, tx_dim = tgt_dims[-2], tgt_dims[-1]
        coords = {d: tgt_da.coords[d] for d in tgt_dims if d in tgt_da.coords}
        # create DataArray; if target has coords for spatial dims, use them
        da_resampled = xr.DataArray(resized, dims=(ty_dim, tx_dim))
        # attach coords if available
        if ty_dim in tgt_da.coords:
            da_resampled = da_resampled.assign_coords({ty_dim: tgt_da.coords[ty_dim]})
        if tx_dim in tgt_da.coords:
            da_resampled = da_resampled.assign_coords({tx_dim: tgt_da.coords[tx_dim]})
        return da_resampled

    def compute_metrics(forecast, truth):
        """
        Calculate forecast verification metrics.

        METRICS EXPLANATION:
        -------------------
        1. RMSE (Root Mean Square Error):
        - Measures the average magnitude of forecast errors
        - Formula: sqrt(mean((forecast - truth)²))
        - Units: same as original data
        - Lower is better (0 = perfect)

        2. Bias (Mean Error):
        - Measures systematic over/under-prediction
        - Formula: mean(forecast - truth)
        - Positive = forecast too high, Negative = forecast too low
        - Units: same as original data
        - 0 = no systematic bias (ideal)

        3. Correlation (Pearson r):
        - Measures linear relationship between forecast and truth
        - Range: -1 to +1
        - +1 = perfect positive correlation
        - 0 = no linear relationship
        - -1 = perfect negative correlation
        - Higher absolute values are better

        Returns: (rmse, bias, correlation)
        """
        f = forecast.values.flatten()
        t = truth.values.flatten()
        mask = np.isfinite(f) & np.isfinite(t)
        f, t = f[mask], t[mask]
        if f.size == 0:
            return np.nan, np.nan, np.nan
        rmse = np.sqrt(np.mean((f - t) ** 2))
        bias = np.mean(f - t)
        corr = np.corrcoef(f, t)[0, 1] if f.size > 1 else np.nan
        return float(rmse), float(bias), float(corr)

    def show_metrics_table_and_plots(
        forecast_low_ds_name,
        forecast_high_ds_name,
        var_low,
        var_high,
        truth_var,
        lead_idx,
        sample_idx,
        ensemble_mean,
        truth_ds_name,
    ):
        """
        Enhanced metrics function with truth variable selection and visualization.
        """
        clear_output(wait=True)

        print("FORECAST VERIFICATION METRICS")
        print("=" * 50)
        print(f"Truth Dataset: {truth_ds_name} (Variable: {truth_var})")
        print(f"Dataset 1: {forecast_low_ds_name} (Variable: {var_low})")
        print(f"Dataset 2: {forecast_high_ds_name} (Variable: {var_high})")
        print(
            f"Lead Time: {lead_idx}, Sample: {sample_idx}, Ensemble Mean: {ensemble_mean}"
        )
        print()

        # get datasets
        ds_low = datasets[forecast_low_ds_name]
        ds_high = datasets[forecast_high_ds_name]
        truth_ds = datasets[truth_ds_name]

        # slice forecasts to requested lead/sample
        f_low_raw = ds_low[var_low]
        f_high_raw = ds_high[var_high]

        f_low = slice_time_sample(f_low_raw, lead_idx, sample_idx, ensemble_mean)
        f_high = slice_time_sample(f_high_raw, lead_idx, sample_idx, ensemble_mean)

        # slice truth using selected truth variable
        if truth_var not in truth_ds.data_vars:
            print(
                f"Error: Variable '{truth_var}' not found in truth dataset '{truth_ds_name}'"
            )
            print(f"Available variables: {list(truth_ds.data_vars)}")
            return

        truth_da = truth_ds[truth_var]
        time_dim_truth = find_time_dim(truth_da)
        if time_dim_truth is not None:
            # clip lead_idx to truth length if necessary
            idx = int(min(truth_da.sizes[time_dim_truth] - 1, int(lead_idx)))
            truth = truth_da.isel({time_dim_truth: idx})
        else:
            # truth has no time-like dim — assume 2D truth already
            truth = truth_da

        try:
            # regrid forecasts to truth grid
            low_on_truth = regrid_to_target(f_low, truth)
            high_on_truth = regrid_to_target(f_high, truth)

            # compute metrics
            rmse_low, bias_low, corr_low = compute_metrics(low_on_truth, truth)
            rmse_high, bias_high, corr_high = compute_metrics(high_on_truth, truth)

            # Create results dataframe
            metrics_data = {
                "Dataset": [forecast_low_ds_name, forecast_high_ds_name],
                "Variable": [var_low, var_high],
                "RMSE": [rmse_low, rmse_high],
                "Bias": [bias_low, bias_high],
                "Correlation": [corr_low, corr_high],
            }
            df = pd.DataFrame(metrics_data)

            # Display table
            print("METRICS TABLE:")
            print("-" * 30)
            display(df.round(4))
            print()

            # Create bar plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            datasets_short = [
                name.split()[0] for name in df["Dataset"]
            ]  # Shorten names for plots

            # RMSE plot
            bars1 = axes[0].bar(
                datasets_short, df["RMSE"], color=["skyblue", "lightcoral"], alpha=0.8
            )
            axes[0].set_title("RMSE")
            axes[0].set_ylabel("RMSE")
            axes[0].grid(axis="y", alpha=0.3)
            for i, v in enumerate(df["RMSE"]):
                axes[0].text(
                    i,
                    v + max(df["RMSE"]) * 0.01,
                    f"{v:.3f}",
                    ha="center",
                )

            # Bias plot
            bars2 = axes[1].bar(
                datasets_short, df["Bias"], color=["lightgreen", "orange"], alpha=0.8
            )
            axes[1].set_title("Bias")
            axes[1].set_ylabel("Bias")
            axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
            axes[1].grid(axis="y", alpha=0.3)
            for i, v in enumerate(df["Bias"]):
                axes[1].text(
                    i,
                    (
                        v + (max(df["Bias"]) - min(df["Bias"])) * 0.05
                        if v >= 0
                        else v - (max(df["Bias"]) - min(df["Bias"])) * 0.05
                    ),
                    f"{v:.3f}",
                    ha="center",
                )

            # Correlation plot
            bars3 = axes[2].bar(
                datasets_short,
                df["Correlation"],
                color=["gold", "mediumpurple"],
                alpha=0.8,
            )
            axes[2].set_title("Correlation")
            axes[2].set_ylabel("Correlation")
            axes[2].set_ylim(0, 1)
            axes[2].grid(axis="y", alpha=0.3)
            for i, v in enumerate(df["Correlation"]):
                if not np.isnan(v):
                    axes[2].text(
                        i, v + 0.02, f"{v:.3f}", ha="center"
                    )

            plt.tight_layout()
            plt.show()

            # Performance summary
            print("PERFORMANCE SUMMARY:")
            print("-" * 25)

            # Find best performing dataset for each metric
            best_rmse = (
                df.loc[df["RMSE"].idxmin(), "Dataset"]
                if not df["RMSE"].isna().all()
                else "N/A"
            )
            best_bias = (
                df.loc[df["Bias"].abs().idxmin(), "Dataset"]
                if not df["Bias"].isna().all()
                else "N/A"
            )
            best_corr = (
                df.loc[df["Correlation"].idxmax(), "Dataset"]
                if not df["Correlation"].isna().all()
                else "N/A"
            )

            print(f"Best RMSE: {best_rmse}")
            print(f"Best Bias (closest to 0): {best_bias}")
            print(f"Best Correlation: {best_corr}")

        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback

            traceback.print_exc()

    # Interactive UI with enhanced controls
    ds_names = list(datasets.keys())

    # Default picks - more robust selection
    default_high = next(
        (name for name in ds_names if "CorrDiff" in name),
        ds_names[0] if ds_names else "",
    )
    default_low = next(
        (name for name in ds_names if "ERA5" in name),
        ds_names[1] if len(ds_names) > 1 else ds_names[0],
    )
    default_truth = next((name for name in ds_names if "WRF" in name), ds_names[0])

    # Widgets
    forecast_low_w = Dropdown(
        options=ds_names, value=default_low, description="Dataset 1"
    )
    forecast_high_w = Dropdown(
        options=ds_names, value=default_high, description="Dataset 2"
    )
    var_low_w = Dropdown(description="Variable 1")
    var_high_w = Dropdown(description="Variable 2")
    truth_w = Dropdown(
        options=ds_names, value=default_truth, description="Truth Dataset"
    )
    truth_var_w = Dropdown(description="Truth Variable")  # NEW: Truth variable selector
    lead_w = IntSlider(min=0, max=24, step=1, description="Lead idx")
    sample_w = IntSlider(min=0, max=0, step=1, description="Ensemble")
    mean_w = Checkbox(value=False, description="Ensemble mean")

    def update_vars_and_ranges(*args):
        """Update variable options and slider ranges based on selected datasets."""
        # Update forecast variable lists
        if forecast_low_w.value in datasets:
            dsL = datasets[forecast_low_w.value]
            var_low_w.options = list(dsL.data_vars)

        if forecast_high_w.value in datasets:
            dsH = datasets[forecast_high_w.value]
            var_high_w.options = list(dsH.data_vars)

        # Update truth variable list
        if truth_w.value in datasets:
            truth_ds = datasets[truth_w.value]
            truth_var_w.options = list(truth_ds.data_vars)

        # Update sample slider based on high-res dataset
        if forecast_high_w.value in datasets:
            dsH = datasets[forecast_high_w.value]
            sample_max = max(dsH.dims.get("sample", 1) - 1, 0)
            sample_w.max = sample_max

        # Update lead slider to match max lead among selected datasets
        lead_maxes = []
        for ds_name in [forecast_low_w.value, forecast_high_w.value, truth_w.value]:
            if ds_name in datasets:
                ds = datasets[ds_name]
                lead_max = (
                    ds.dims.get(
                        "lead_time", ds.dims.get("time", ds.dims.get("valid_time", 1))
                    )
                    - 1
                )
                lead_maxes.append(lead_max)

        if lead_maxes:
            lead_w.max = max(0, int(max(lead_maxes)))

    # Attach observers
    forecast_low_w.observe(update_vars_and_ranges, names="value")
    forecast_high_w.observe(update_vars_and_ranges, names="value")
    truth_w.observe(update_vars_and_ranges, names="value")

    # Initialize
    update_vars_and_ranges()

    # Interactive interface
    @interact(
        forecast_low=forecast_low_w,
        forecast_high=forecast_high_w,
        var_low=var_low_w,
        var_high=var_high_w,
        truth_ds=truth_w,
        truth_var=truth_var_w,  # NEW: Truth variable selection
        lead_idx=lead_w,
        sample_idx=sample_w,
        ensemble_mean=mean_w,
    )
    def run_enhanced_metrics(
        forecast_low,
        forecast_high,
        var_low,
        var_high,
        truth_ds,
        truth_var,
        lead_idx,
        sample_idx,
        ensemble_mean,
    ):
        if all([forecast_low, forecast_high, var_low, var_high, truth_ds, truth_var]):
            show_metrics_table_and_plots(
                forecast_low,
                forecast_high,
                var_low,
                var_high,
                truth_var,
                lead_idx,
                sample_idx,
                ensemble_mean,
                truth_ds,
            )
        else:
            print("Please select all required options before running metrics.")


# difference maps
def difference_map(datasets):

    def choose_cmap(varname):
        if varname is None:
            return "RdBu_r"
        vn = varname.upper()
        if vn == "T2":
            return "coolwarm"
        if vn in ("RAIN", "Q2", "RH"):
            return "viridis"
        return "RdBu_r"

    def get_spatial_dims(da):
        """
        Identify spatial dimensions (y, x) in the DataArray.
        Handles lat/lon variants, y/x endings, and fallbacks.
        """
        dims = list(da.dims)
        y_dim = next((d for d in dims if "lat" in d or d.lower().endswith("y")), None)
        x_dim = next((d for d in dims if "lon" in d or d.lower().endswith("x")), None)

        if y_dim and x_dim:
            return y_dim, x_dim
        # fallback: assume last two are spatial
        return dims[-2], dims[-1]

    def get_spatial_dims(da):
        """
        Identify spatial dimensions (y, x) in the DataArray.
        Handles lat/lon variants, WRF-style (south_north, west_east),
        and CorrDiff (corrdiff_y, corrdiff_x) patterns.
        """
        dims = list(da.dims)

        spatial_y_keywords = ["lat", "y", "south_north", "corrdiff_y"]
        spatial_x_keywords = ["lon", "x", "west_east", "corrdiff_x"]

        def match_dim(dim, keywords):
            dim_lower = dim.lower()
            return any(k in dim_lower or dim_lower.endswith(k) for k in keywords)

        y_dim = next((d for d in dims if match_dim(d, spatial_y_keywords)), None)
        x_dim = next((d for d in dims if match_dim(d, spatial_x_keywords)), None)

        # If not found, fallback to last two dims
        if y_dim is None or x_dim is None:
            if len(dims) >= 2:
                return dims[-2], dims[-1]
            else:
                raise ValueError(
                    f"Cannot determine spatial dimensions from dims: {dims}"
                )

        return y_dim, x_dim

    def get_data_slice(
        da, var_name, time_idx=0, lead_idx=0, sample_idx=None, ensemble_mean=False
    ):
        """
        Returns a 2D slice (y, x) from a DataArray across CorrDiff, ERA5, WRF, etc.
        """

        # Handle Dataset input
        if isinstance(da, xr.Dataset):
            if var_name not in da:
                raise ValueError(
                    f"Variable {var_name} not found in dataset {list(da.data_vars)}"
                )
            da = da[var_name]

        # Handle time dim if present
        if "time" in da.dims:
            da = da.isel(time=time_idx)

        # Handle lead_time / valid_time if present
        if "lead_time" in da.dims:
            da = da.isel(lead_time=lead_idx)
        elif "valid_time" in da.dims:
            da = da.isel(valid_time=lead_idx)

        # Handle ensemble or sample dim
        sdim = next((d for d in ("sample", "ensemble") if d in da.dims), None)
        if sdim:
            if ensemble_mean:
                da = da.mean(dim=sdim)
            else:
                if sample_idx is None:
                    sample_idx = 0
                da = da.isel({sdim: sample_idx})

        # Identify spatial dims
        ydim, xdim = get_spatial_dims(da)

        # Remove any leftover non-spatial dims
        extra_dims = [d for d in da.dims if d not in (ydim, xdim)]
        if extra_dims:
            da = da.isel({d: 0 for d in extra_dims})

        # Ensure numeric
        if not np.issubdtype(da.dtype, np.number):
            da = da.astype(float)

        if da.ndim != 2:
            raise ValueError(f"Slice is not 2D: {da.dims}")

        return da

    # align or truncate incase of shape mismatch
    def align_arrays(gt_da, comp_da):
        """
        Align two DataArrays by spatial dims. If sizes differ by exactly 1 in either axis, truncate the larger.
        After truncation, rename comp_da dims to match gt_da dims for safe arithmetic.
        Returns (gt_aligned, comp_aligned, None) or (None, None, error_message).
        """
        try:
            y_gt, x_gt = get_spatial_dims(gt_da)
            y_c, x_c = get_spatial_dims(comp_da)
        except Exception as e:
            return None, None, f"Cannot get spatial dims: {e}"

        ny_gt, nx_gt = gt_da.sizes[y_gt], gt_da.sizes[x_gt]
        ny_c, nx_c = comp_da.sizes[y_c], comp_da.sizes[x_c]

        # exact match
        if ny_gt == ny_c and nx_gt == nx_c:
            # if coordinate names differ, rename comp to GT's names
            if (y_c != y_gt) or (x_c != x_gt):
                comp_da = comp_da.rename({y_c: y_gt, x_c: x_gt})
            return gt_da, comp_da, None

        # allow off-by-1 in each dimension -> truncate larger
        if abs(ny_gt - ny_c) <= 1 and abs(nx_gt - nx_c) <= 1:
            ny = min(ny_gt, ny_c)
            nx = min(nx_gt, nx_c)
            gt_trim = gt_da.isel({y_gt: slice(0, ny), x_gt: slice(0, nx)})
            comp_trim = comp_da.isel({y_c: slice(0, ny), x_c: slice(0, nx)})
            comp_trim = comp_trim.rename({y_c: y_gt, x_c: x_gt})
            return gt_trim, comp_trim, None

        # too large mismatch
        return (
            None,
            None,
            f"Grid mismatch too large: GT=({ny_gt},{nx_gt}) vs COMP=({ny_c},{nx_c})",
        )

    # main plotting
    def plot_three_way(
        gt_key,
        comp1_key,
        comp2_key,
        var_gt,
        var_c1,
        var_c2,
        lead_idx,
        sample_idx=None,
        ensemble_mean=False,
    ):
        clear_output(wait=True)
        print("Loading difference maps...")

        # basic selection validation
        if not (gt_key and comp1_key and comp2_key and var_gt and var_c1 and var_c2):
            print("Please select ground-truth, two comparisons and variables.")
            return

        # build slices
        # da_gt = get_data_slice(datasets[gt_key], var_gt, lead_idx, sample_idx, ensemble_mean)

        # da_c1 = get_data_slice(
        #     datasets[comp1_key], var_c1,
        #     lead_idx=lead_idx, sample_idx=sample_idx, ensemble_mean=ensemble_mean
        # )
        # da_c2 = get_data_slice(
        #     datasets[comp2_key], var_c2,
        #     lead_idx=lead_idx, sample_idx=sample_idx, ensemble_mean=ensemble_mean
        # )

        da_gt = get_data_slice(
            datasets[gt_key],
            var_gt,
            time_idx=0,
            lead_idx=lead_idx,
            sample_idx=sample_idx,
            ensemble_mean=ensemble_mean,
        )
        da_c1 = get_data_slice(
            datasets[comp1_key],
            var_c1,
            time_idx=0,
            lead_idx=lead_idx,
            sample_idx=sample_idx,
            ensemble_mean=ensemble_mean,
        )
        da_c2 = get_data_slice(
            datasets[comp2_key],
            var_c2,
            time_idx=0,
            lead_idx=lead_idx,
            sample_idx=sample_idx,
            ensemble_mean=ensemble_mean,
        )

        if da_gt is None or da_c1 is None or da_c2 is None:
            print("One or more slices are invalid — check messages above.")
            return

        print(
            f"Shapes before alignment: GT={da_gt.shape}, C1={da_c1.shape}, C2={da_c2.shape}"
        )

        # align/truncate
        da_gt_a1, da_c1_a, msg1 = align_arrays(da_gt, da_c1)
        da_gt_a2, da_c2_a, msg2 = align_arrays(da_gt, da_c2)

        if msg1:
            print("Left comparison alignment:", msg1)
        if msg2:
            print("Right comparison alignment:", msg2)
        if da_gt_a1 is None or da_gt_a2 is None:
            print("Cannot align grids for differences; aborting.")
            return

        print(
            f"Shapes after alignment: GT={da_gt_a1.shape}, C1={da_c1_a.shape}, C2={da_c2_a.shape}"
        )

        # compute diffs (GT minus others)
        try:
            diff1 = da_gt_a1 - da_c1_a
            diff2 = da_gt_a2 - da_c2_a
        except Exception as e:
            print("Error computing difference:", e)
            return

        cmap = choose_cmap(var_gt)

        # plot
        plt.close("all")
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        for ax, diff, comp_key in zip(axes, [diff1, diff2], [comp1_key, comp2_key]):

            try:
                # use xarray plotting (keeps coords) — colorbar per panel suppressed; we'll add a single cb at the end
                im = diff.plot(
                    ax=ax,
                    cmap=cmap,
                    robust=True,
                    add_colorbar=False,
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines()
                ax.set_title(f"{gt_key} - {comp_key}\nLead {lead_idx}")
                ax.set_aspect("equal", adjustable="box")
                cbar = fig.colorbar(im, orientation="vertical", ax=ax)
                cbar.set_label(f"{var_gt} (difference)")
            except Exception as e:
                ax.text(0.5, 0.5, f"Plot error: {e}", ha="center")
                print("Plot error for", comp_key, e)

        plt.tight_layout()
        plt.show()

    # widgets
    gt_ds_w = Dropdown(
        options=[None] + list(datasets.keys()), description="Ground Truth"
    )
    comp1_ds_w = Dropdown(
        options=[None] + list(datasets.keys()), description="Compare 1"
    )
    comp2_ds_w = Dropdown(
        options=[None] + list(datasets.keys()), description="Compare 2"
    )
    var_gt_w = Dropdown(options=[None], description="GT Var")
    var_c1_w = Dropdown(options=[None], description="Var 1")
    var_c2_w = Dropdown(options=[None], description="Var 2")
    lead_w = IntSlider(min=0, max=24, step=1, description="Lead Time")
    sample_w = IntSlider(min=0, max=0, step=1, description="Ensemble")
    mean_w = Checkbox(value=False, description="Use Ensemble Mean")

    # update var lists and lead/sample ranges
    sample_w = IntSlider(min=0, max=0, step=1, description="Ensemble")

    def update_vars(_=None):
        # GT
        if gt_ds_w.value:
            ds = datasets[gt_ds_w.value]
            var_gt_w.options = list(ds.data_vars)
            lead_dim = next(
                (d for d in ("lead_time", "time", "valid_time") if d in ds.dims), None
            )
            lead_w.max = ds.dims[lead_dim] - 1 if lead_dim else 0
            sdim = next((d for d in ("sample", "ensemble") if d in ds.dims), None)
            if sdim:
                sample_w.max = ds.dims[sdim] - 1
                sample_w.disabled = False
            else:
                sample_w.max = 0
                sample_w.disabled = True
        else:
            var_gt_w.options = [None]

        # comp1
        if comp1_ds_w.value:
            var_c1_w.options = list(datasets[comp1_ds_w.value].data_vars)
        else:
            var_c1_w.options = [None]

        # comp2
        if comp2_ds_w.value:
            var_c2_w.options = list(datasets[comp2_ds_w.value].data_vars)
        else:
            var_c2_w.options = [None]

    gt_ds_w.observe(update_vars, names="value")
    comp1_ds_w.observe(update_vars, names="value")
    comp2_ds_w.observe(update_vars, names="value")
    update_vars()

    widget_dict = {
        "gt_key": gt_ds_w,
        "comp1_key": comp1_ds_w,
        "comp2_key": comp2_ds_w,
        "var_gt": var_gt_w,
        "var_c1": var_c1_w,
        "var_c2": var_c2_w,
        "lead_idx": lead_w,
        "sample_idx": sample_w,
        "ensemble_mean": mean_w,
    }

    out = interactive_output(plot_three_way, widget_dict)

    ui = VBox(
        [
            HBox([gt_ds_w, comp1_ds_w, comp2_ds_w]),
            HBox([var_gt_w, var_c1_w, var_c2_w]),
            HBox([lead_w, sample_w, mean_w]),
        ]
    )
    display(ui, out)
