# processHRRR.py
import os
import shutil
import datetime
import glob
import json
import math
import sys
import re
import time
import threading
import contextlib
import functools
import concurrent.futures
import urllib.request
import numpy as np
import xarray as xr
import cv2
import boto3
import contourpy
import gzip
import struct

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ.setdefault("TQDM_DISABLE", "1")
sys.stdout.reconfigure(line_buffering=True)

# HRRR standard forecast hours: Hourly from F00 to F18
FORECAST_STEPS = list(range(0, 19))

MAX_TEXTURE_SIZE = 4096
MAX_CONCURRENT_WORKERS = int(os.environ.get("MAX_WORKERS", "12"))
GZIP_LEVEL = 6
GZIP_THREADS = 4
MAX_CONCURRENT_PARAMS = 2

CONFIG_FILE_PATH = os.path.join("config", "parametersHRRR.json")
DEFAULT_HF_REPO = "PhillyWeatherGuy/baroclinic-hrrr-data"

CONTOUR_COORD_SCALE = 1000
CONTOUR_FORMAT = "CTV2"
INT16_MAX = 32767

CLIMATOLOGY_CACHE = {}
GRID_REMAP_CACHE = {}

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
TIMINGS = {}
_T_LOCK = threading.Lock()


@contextlib.contextmanager
def tm(key):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _T_LOCK:
            t = TIMINGS.setdefault(key, [0.0, 0])
            t[0] += dt
            t[1] += 1


def timed(key):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with tm(key):
                return fn(*a, **k)
        return wrapper
    return deco


def _gzip_to_file(path, raw_bytes):
    with open(path, "wb") as f:
        f.write(gzip.compress(raw_bytes, compresslevel=GZIP_LEVEL))


def _print_timing(pid, phases, total):
    ph = " | ".join(f"{k} {v:.1f}s" for k, v in phases.items())
    work = " | ".join(f"{k} {v[0]:.1f}s/{v[1]}x" for k, v in sorted(TIMINGS.items()))
    print(f"⏱️ [{pid}] total {total:.1f}s || phases: {ph} || thread-summed: {work}", flush=True)


def load_parameter_config(param_key="2t"):
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            if param_key in params:
                return params[param_key]
    raise FileNotFoundError(f"Parameter '{param_key}' not found in {CONFIG_FILE_PATH}")


# ---------------------------------------------------------------------------
# Contours: vectorized extraction + CTV2 binary packing
# ---------------------------------------------------------------------------

def _pack_level(cont_gen, level):
    pts_list, off_list = cont_gen.lines(level)
    pts, off = pts_list[0], off_list[0]
    if pts is None or off is None or len(off) < 2:
        return None

    off = off.astype(np.intp)
    lens = np.diff(off)
    starts = off[:-1]

    x = np.minimum(pts[:, 0], 180.0)
    seam = (np.maximum.reduceat(x, starts) < -179.99) | (np.minimum.reduceat(x, starts) > 179.99)

    q = np.empty((len(pts), 2), dtype=np.int32)
    q[:, 0] = np.rint(x * CONTOUR_COORD_SCALE)
    q[:, 1] = np.rint(pts[:, 1] * CONTOUR_COORD_SCALE)

    keep_line = ~seam
    if not keep_line.all():
        q = q[np.repeat(keep_line, lens)]
        lens = lens[keep_line]
    if len(lens) == 0:
        return None

    n = len(q)
    is_start = np.zeros(n, dtype=bool)
    is_start[np.cumsum(lens) - lens] = True

    if n > 1:
        is_start[1:] |= (np.abs(np.diff(q, axis=0)) > INT16_MAX).any(axis=1)

    seg_starts = np.flatnonzero(is_start)
    seg_lens = np.diff(np.append(seg_starts, n))

    keep_seg = seg_lens >= 2
    if not keep_seg.all():
        q = q[np.repeat(keep_seg, seg_lens)]
        seg_lens = seg_lens[keep_seg]
        seg_starts = np.cumsum(seg_lens) - seg_lens
        if len(seg_lens) == 0:
            return None
        is_start = np.zeros(len(q), dtype=bool)
        is_start[seg_starts] = True

    deltas = np.diff(q, axis=0)
    inner = ~is_start[1:]
    dx = deltas[inner, 0].astype("<i2")
    dy = deltas[inner, 1].astype("<i2")

    return b"".join((
        struct.pack("<I", len(seg_lens)),
        seg_lens.astype("<u4").tobytes(),
        q[seg_starts, 0].astype("<i4").tobytes(),
        q[seg_starts, 1].astype("<i4").tobytes(),
        dx.tobytes(),
        dy.tobytes(),
    ))


def _level_style(level, contours_config):
    name = str(level)
    unit = None
    for c_def in contours_config:
        if c_def.get("dynamic") and c_def.get("unit") in {"dam", "mb", "hPa"}:
            unit = c_def.get("unit")
            name = str(int(level)) if float(level).is_integer() else str(level)
            break

    match = next(
        (c for c in contours_config if c.get("dynamic") or c.get("target") == level),
        {},
    )
    return {
        "name": name,
        "unit": unit or match.get("unit", ""),
        "color": match.get("color", "#000000"),
        "width": match.get("width", 1.6),
        "opacity": match.get("opacity", 0.9),
    }


@timed("contour_math")
def extract_contour_levels(raw_arr_k, contours_config=None, contour_settings=None):
    if not contours_config:
        return []

    try:
        contour_settings = contour_settings or {}
        coarsen_stride = max(1, int(contour_settings.get("coarsen_stride", 1)))
        blur_kernel = tuple(contour_settings.get("blur_kernel", (5, 5)))
        blur_sigma = float(contour_settings.get("blur_sigma", 1.2))

        working_arr = raw_arr_k.astype(np.float32)
        if coarsen_stride > 1:
            working_arr = cv2.resize(
                working_arr,
                (max(1, working_arr.shape[1] // coarsen_stride), max(1, working_arr.shape[0] // coarsen_stride)),
                interpolation=cv2.INTER_AREA,
            )

        frame_h, frame_w = working_arr.shape
        smoothed = cv2.GaussianBlur(working_arr, blur_kernel, blur_sigma)
        smoothed_flipped = np.flipud(smoothed)
        smoothed_cyclic = np.hstack([smoothed_flipped, smoothed_flipped[:, :1]])

        lon_step = 360.0 / frame_w
        lons = np.linspace(-180.0, 180.0 + lon_step, frame_w + 1)
        lats = np.linspace(-90.0, 90.0, frame_h)
        cont_gen = contourpy.contour_generator(
            x=lons, y=lats, z=smoothed_cyclic,
            line_type=contourpy.LineType.ChunkCombinedOffset,
        )

        interval = 6.0
        explicit_levels = []

        for c_def in contours_config:
            if c_def.get("dynamic"):
                interval = float(c_def.get("interval", 6.0))
                valid_mask = np.isfinite(working_arr) & (working_arr > 0)
                if not np.any(valid_mask):
                    explicit_levels = []
                    break

                valid_values = working_arr[valid_mask]
                data_min = float(np.nanmin(valid_values))
                data_max = float(np.nanmax(valid_values))
                if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                    start = math.ceil(data_min / interval) * interval
                    end = math.floor(data_max / interval) * interval
                    explicit_levels = [float(v) for v in np.arange(start, end + (interval * 0.5), interval)]
                break
            if "target" in c_def:
                explicit_levels.append(float(c_def["target"]))

        if not explicit_levels:
            return []

        levels = []
        for target_val in sorted(set(explicit_levels)):
            payload = _pack_level(cont_gen, target_val)
            if payload is not None:
                levels.append({"props": _level_style(target_val, contours_config), "payload": payload})

        return levels
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return []


@timed("pack")
def pack_step(levels):
    return {
        "blob": b"".join(l["payload"] for l in levels),
        "props": [l["props"] for l in levels],
    }


def assemble_contour_file(steps):
    parts = [CONTOUR_FORMAT.encode("ascii"), struct.pack("<I", len(steps))]
    for step in sorted(steps):
        entry = steps[step]
        parts.append(struct.pack("<III", int(step), len(entry["props"]), len(entry["blob"])))
        parts.append(entry["blob"])
    return b"".join(parts)


def normalize_array(raw_arr, param_config):
    scaling = param_config.get("scaling", {})
    mode = scaling.get("mode", "linear")

    if mode == "piecewise" and "val_points" in scaling and "byte_points" in scaling:
        multiplier = scaling.get("unit_multiplier", 1.0)
        v = np.nan_to_num(raw_arr, nan=0.0) * multiplier
        val_pts = scaling["val_points"]
        byte_pts = scaling["byte_points"]
        return np.interp(v, val_pts, byte_pts).astype(np.uint8)
    else:
        min_v = scaling.get("min_val", param_config.get("min_val", 0.0))
        max_v = scaling.get("max_val", param_config.get("max_val", 255.0))
        arr = np.nan_to_num(raw_arr, copy=False, nan=min_v)
        np.clip(arr, min_v, max_v, out=arr)
        arr -= min_v
        arr /= (max_v - min_v)
        arr *= 255.0
        return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Fast CONUS Equirectangular Reprojection
# ---------------------------------------------------------------------------
def reproject_hrrr_to_equirectangular(raw_arr, lats_2d, lons_2d, target_shape=(721, 1440), default_val=0.0):
    global GRID_REMAP_CACHE
    target_h, target_w = target_shape
    cache_key = f"{raw_arr.shape}_{target_shape}"

    # 🌟 ONLY NECESSARY CHANGE: Flip array vertically to fix upside-down GRIB orientation
    if lats_2d is not None and lats_2d.ndim == 2 and lats_2d[0, 0] < lats_2d[-1, 0]:
        raw_arr = np.flipud(raw_arr)

    if cache_key not in GRID_REMAP_CACHE:
        target_lats = np.linspace(90.0, -90.0, target_h)
        target_lons = np.linspace(-180.0, 180.0, target_w)
        
        lat_min, lat_max = np.min(lats_2d), np.max(lats_2d)
        lon_min, lon_max = np.min(lons_2d), np.max(lons_2d)
        
        y_indices = np.where((target_lats >= lat_min) & (target_lats <= lat_max))[0]
        x_indices = np.where((target_lons >= lon_min) & (target_lons <= lon_max))[0]

        GRID_REMAP_CACHE[cache_key] = {
            "y_indices": y_indices,
            "x_indices": x_indices,
            "sub_shape": (len(y_indices), len(x_indices))
        }

    info = GRID_REMAP_CACHE[cache_key]
    out_grid = np.full(target_shape, default_val, dtype=np.float32)
    
    if len(info["y_indices"]) > 0 and len(info["x_indices"]) > 0:
        conus_resized = cv2.resize(raw_arr.astype(np.float32), (len(info["x_indices"]), len(info["y_indices"])), interpolation=cv2.INTER_LINEAR)
        y_start, y_end = info["y_indices"][0], info["y_indices"][-1] + 1
        x_start, x_end = info["x_indices"][0], info["x_indices"][-1] + 1
        out_grid[y_start:y_end, x_start:x_end] = conus_resized

    return out_grid


# ---------------------------------------------------------------------------
# High-Speed AWS Open Data HTTP Range Downloader
# ---------------------------------------------------------------------------
def download_hrrr_idx_slice(target_date, chosen_run, step, search_pattern, file_type, dest_path):
    base_url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{target_date}/conus"
    prefix = f"hrrr.t{chosen_run}z.wrf{file_type}f{step:02d}.grib2"
    grib_url = f"{base_url}/{prefix}"
    idx_url = f"{grib_url}.idx"

    req = urllib.request.Request(idx_url, headers={"User-Agent": "BaroClinic-HRRR-Engine"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        idx_lines = resp.read().decode("utf-8").strip().split("\n")

    start_byte = None
    end_byte = None

    for i, line in enumerate(idx_lines):
        if search_pattern in line:
            parts = line.split(":")
            start_byte = int(parts[1])
            if i + 1 < len(idx_lines):
                end_byte = int(idx_lines[i + 1].split(":")[1]) - 1
            break

    if start_byte is None:
        raise ValueError(f"Pattern '{search_pattern}' not found in {prefix}.idx")

    range_header = f"bytes={start_byte}-" if end_byte is None else f"bytes={start_byte}-{end_byte}"
    range_req = urllib.request.Request(grib_url, headers={
        "Range": range_header,
        "User-Agent": "BaroClinic-HRRR-Engine"
    })

    with urllib.request.urlopen(range_req, timeout=30) as r_resp:
        with open(dest_path, "wb") as f_out:
            f_out.write(r_resp.read())

    return dest_path


@timed("decode_all")
def process_grib_to_array(grib_path, param_config, want_raster=True, target_date=None):
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={'errors': 'ignore'})

    lats = ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
    lons = ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values
    if np.nanmax(lons) > 180:
        lons = np.where(lons > 180, lons - 360, lons)

    grib_var = param_config["grib_param"]
    if isinstance(grib_var, list) and len(grib_var) == 2:
        raw_arr_k = np.hypot(np.squeeze(ds[grib_var[0]].values), np.squeeze(ds[grib_var[1]].values))
    else:
        target_var = grib_var if (isinstance(grib_var, str) and grib_var in ds) else list(ds.data_vars)[0]
        raw_arr_k = np.squeeze(ds[target_var].values)

    ds.close()

    raw_arr_reproj = reproject_hrrr_to_equirectangular(raw_arr_k, lats, lons)

    if param_config.get("category_mode"):
        return np.nan_to_num(raw_arr_reproj, nan=0.0).clip(0, 255).astype(np.uint8), []

    if param_config.get("unit") == "dam" or str(param_config.get("grib_param", "")).lower() in {"z", "gh", "hgt"}:
        raw_arr_reproj = raw_arr_reproj / 10.0
    elif param_config.get("unit") in {"mb", "hPa"} or str(param_config.get("grib_param", "")).lower() in {"msl", "mslma", "prmsl"}:
        raw_arr_reproj = raw_arr_reproj / 100.0

    contour_levels = extract_contour_levels(
        raw_arr_reproj,
        param_config.get("contours", []),
        param_config.get("contours_settings")
    )

    arr_8bit = normalize_array(raw_arr_reproj, param_config) if want_raster else None
    return arr_8bit, contour_levels


def fetch_and_process_step(target_date, chosen_run, step, param_config, model_name):
    pid = param_config["id"]
    file_type = param_config.get("file_type", "sfc")
    grib_file = f"{model_name}_{pid}_{step:02d}.grib2"

    try:
        with tm("download"):
            # 🌟 Handle dual-component wind vector parameters (U & V)
            if "idx_search_u" in param_config and "idx_search_v" in param_config:
                u_file = f"{model_name}_{pid}_u_{step:02d}.grib2"
                v_file = f"{model_name}_{pid}_v_{step:02d}.grib2"
                download_hrrr_idx_slice(target_date, chosen_run, step, param_config["idx_search_u"], file_type, u_file)
                download_hrrr_idx_slice(target_date, chosen_run, step, param_config["idx_search_v"], file_type, v_file)

                # Concatenate U and V into a single GRIB file for cfgrib
                with open(grib_file, "wb") as f_out:
                    with open(u_file, "rb") as f_u: f_out.write(f_u.read())
                    with open(v_file, "rb") as f_v: f_out.write(f_v.read())
                
                try: os.remove(u_file); os.remove(v_file)
                except Exception: pass
            else:
                search_pat = param_config.get("idx_search", f":{param_config['grib_param']}:")
                download_hrrr_idx_slice(target_date, chosen_run, step, search_pat, file_type, grib_file)

        if os.path.exists(grib_file):
            contour_sources = param_config.get("contour_sources")
            if not contour_sources and param_config.get("contour_source"):
                contour_sources = [param_config["contour_source"]]

            has_contours = bool(contour_sources)
            raster_config = {**param_config, "contours": []} if has_contours else param_config
            frame_arr, contour_levels = process_grib_to_array(grib_file, raster_config, target_date=target_date)
            contour_levels = list(contour_levels)
            ptype_arr = None

            try: os.remove(grib_file)
            except Exception: pass

            # 🌟 Categorical Simulated Radar P-Type (Rain / Snow / Sleet / Freezing Rain)
            ptype_sources = param_config.get("ptype_sources")
            if ptype_sources:
                ptype_masks = {}
                for p_name, p_search in ptype_sources.items():
                    pt_file = f"{model_name}_{pid}_{p_name}_{step:02d}.grib2"
                    try:
                        download_hrrr_idx_slice(target_date, chosen_run, step, p_search, "sfc", pt_file)
                        if os.path.exists(pt_file):
                            mask, _ = process_grib_to_array(pt_file, {"grib_param": "", "category_mode": True})
                            ptype_masks[p_name] = mask > 0.5
                    except Exception: pass
                    finally:
                        if os.path.exists(pt_file):
                            try: os.remove(pt_file)
                            except Exception: pass

                if ptype_masks:
                    shape = frame_arr.shape
                    ptype_arr = np.zeros(shape, dtype=np.uint8)
                    if "rain" in ptype_masks: ptype_arr[ptype_masks["rain"]] = 1
                    if "frzr" in ptype_masks: ptype_arr[ptype_masks["frzr"]] = 3
                    if "snow" in ptype_masks: ptype_arr[ptype_masks["snow"]] = 5
                    if "sleet" in ptype_masks: ptype_arr[ptype_masks["sleet"]] = 7

            packed = pack_step(contour_levels)
            print(f"  ⚡ [{pid}] Processed HRRR F{step:02d}")
            return step, frame_arr, packed, ptype_arr

    except Exception as e:
        print(f"  ❌ [{pid}] Error processing F{step:02d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass

    return step, None, None, None


@timed("stack")
def build_volume_chunks(frame_arrays, steps_written, model_name, param_config, target_date, chosen_run, ptype_arrays=None):
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    frames_per_volume = 10

    chunks = []
    patterns = param_config["filename_patterns"]
    pattern_key = "volume" if "volume" in patterns else "spritesheet"

    for chunk_idx, i in enumerate(range(0, len(frame_arrays), frames_per_volume)):
        chunk_frames = frame_arrays[i:i + frames_per_volume]
        chunk_steps = steps_written[i:i + frames_per_volume]
        chunk_ptype = ptype_arrays[i:i + frames_per_volume] if ptype_arrays else None

        volume_arr = np.stack(chunk_frames, axis=0).astype(np.uint8)

        volume_filename = patterns[pattern_key].format(
            model=model_name,
            param=param_config["id"],
            date=target_date,
            run=chosen_run,
            chunk_idx=chunk_idx
        )

        chunk_manifest = {
            "file": volume_filename,
            "forecast_steps": chunk_steps,
            "frame_count": len(chunk_frames)
        }
        chunk_entry = {
            "array": volume_arr,
            "manifest_data": chunk_manifest
        }
        if chunk_ptype and all(frame is not None for frame in chunk_ptype):
            ptype_filename = volume_filename.replace("_volume_", "_ptype_volume_")
            chunk_entry["ptype_array"] = np.stack(chunk_ptype, axis=0).astype(np.uint8)
            chunk_manifest["ptype_file"] = ptype_filename
        chunks.append(chunk_entry)

    return chunks, frame_w, frame_h


def upload_to_huggingface(folder_path, repo_id=DEFAULT_HF_REPO):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set in environment. Skipping Hugging Face upload.")
        return

    print(f"\n🤗 Uploading {folder_path} assets to Hugging Face ({repo_id})...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload HRRR grids: {folder_path}"
        )
        print(f"  ✅ Uploaded to Hugging Face: {folder_path}")
    except Exception as e:
        print(f"  ❌ Failed to upload to Hugging Face: {e}")


def prune_old_huggingface_runs(repo_id=DEFAULT_HF_REPO, days_to_keep=2):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return

    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    cutoff_int = int(cutoff_date.strftime("%Y%m%d"))

    print(f"\n🧹 Checking for HRRR runs older than {cutoff_date.strftime('%Y-%m-%d')} (>{days_to_keep} days)...")

    try:
        from huggingface_hub import HfApi, CommitOperationDelete
        api = HfApi(token=hf_token)
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        files_to_delete = []

        for fname in all_files:
            match = re.search(r'_(\d{8})_', fname)
            if match:
                file_date_int = int(match.group(1))
                if file_date_int < cutoff_int:
                    files_to_delete.append(fname)

        if files_to_delete:
            print(f"  🗑️ Deleting {len(files_to_delete)} expired HRRR files...")
            batch_size = 200
            for i in range(0, len(files_to_delete), batch_size):
                batch = files_to_delete[i:i + batch_size]
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Prune HRRR runs older than {days_to_keep} days ({i + len(batch)}/{len(files_to_delete)})"
                )
            print("  ✅ Pruning complete!")
        else:
            print("  ✨ No expired HRRR files found.")

    except Exception as e:
        print(f"  ⚠️ Could not prune old Hugging Face files: {e}")


def check_hrrr_run_ready(target_date, run_hour):
    """
    Checks if F18 of the requested HRRR run is available on AWS S3 before running.
    """
    test_url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{target_date}/conus/hrrr.t{run_hour}z.wrfsfcf18.grib2.idx"
    try:
        req = urllib.request.Request(test_url, method="HEAD", headers={"User-Agent": "BaroClinic-HRRR-Engine"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_master_pipeline(selected_param_key="2t", custom_run=None, custom_date=None):
    MODEL_NAME = "hrrr"
    param_config = load_parameter_config(selected_param_key)
    patterns = param_config["filename_patterns"]

    output_dist_dir = f"run_{MODEL_NAME}_{param_config['id']}"

    TIMINGS.clear()
    t_total = time.perf_counter()
    phases = {}

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if custom_run and custom_date:
        CHOSEN_RUN = str(custom_run).zfill(2)
        target_date = str(custom_date)
    else:
        # 🌟 Check latest hours until finding the run that NOAA has 100% completed (has F18)
        found = False
        for hours_back in range(1, 6):
            t_cand = now_utc - datetime.timedelta(hours=hours_back)
            cand_run = t_cand.strftime("%H")
            cand_date = t_cand.strftime("%Y%m%d")
            if check_hrrr_run_ready(cand_date, cand_run):
                CHOSEN_RUN = cand_run
                target_date = cand_date
                found = True
                break
        
        if not found:
            # Fallback to 2 hours ago
            t_cand = now_utc - datetime.timedelta(hours=2)
            CHOSEN_RUN = t_cand.strftime("%H")
            target_date = t_cand.strftime("%Y%m%d")

    init_time_iso = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}T{CHOSEN_RUN}:00:00Z"

    print(f"🌍 [{param_config['id']}] Model: {MODEL_NAME.upper()} | Param: {param_config['name']} | Run: {CHOSEN_RUN}z on {target_date}")

    for f in glob.glob(f"{MODEL_NAME}_{param_config['id']}_*.grib2"):
        try: os.remove(f)
        except Exception: pass

    os.makedirs(output_dist_dir, exist_ok=True)

    t0 = time.perf_counter()
    results = {}
    contour_steps = {}
    ptype_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_step = {
            executor.submit(fetch_and_process_step, target_date, CHOSEN_RUN, step, param_config, MODEL_NAME): step
            for step in FORECAST_STEPS
        }
        for future in concurrent.futures.as_completed(future_to_step):
            step, arr, packed, ptype_arr = future.result()
            if arr is not None:
                results[step] = arr
                if packed and packed["props"]:
                    contour_steps[step] = packed
                if ptype_arr is not None:
                    ptype_results[step] = ptype_arr

    phases["fetch_phase"] = time.perf_counter() - t0
    print(f"[{param_config['id']}] steps ok: {len(results)}/{len(FORECAST_STEPS)}", flush=True)

    sorted_steps = sorted(results.keys())
    frame_arrays = [results[s] for s in sorted_steps]
    ptype_arrays = [ptype_results.get(s) for s in sorted_steps]
    steps_written = sorted_steps

    if not frame_arrays:
        print(f"❌ [{param_config['id']}] No frames processed. Exiting pipeline.")
        return

    t0 = time.perf_counter()
    if contour_steps and "run_contours" in patterns:
        contour_binary = assemble_contour_file(contour_steps)
        compressed_contours = gzip.compress(contour_binary, compresslevel=6)

        if len(compressed_contours) >= 100:
            master_contours = {
                "model": MODEL_NAME,
                "parameter": param_config["id"],
                "run": f"{CHOSEN_RUN}z",
                "date": target_date,
                "steps": {
                    str(step): {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "MultiLineString", "coordinates": []},
                                "properties": props,
                            }
                            for props in contour_steps[step]["props"]
                        ],
                    }
                    for step in sorted(contour_steps)
                },
            }

            run_contour_filename = patterns["run_contours"].format(
                model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
            )
            run_binary_filename = run_contour_filename.replace(".json", ".bin.gz")
            master_contours["binary"] = {
                "format": CONTOUR_FORMAT,
                "scale": CONTOUR_COORD_SCALE,
                "file": run_binary_filename,
            }

            with open(os.path.join(output_dist_dir, run_contour_filename), 'w') as f:
                json.dump(master_contours, f, separators=(",", ":"))
            with open(os.path.join(output_dist_dir, run_binary_filename), "wb") as f:
                f.write(compressed_contours)

            if "latest_contours" in patterns:
                latest_contour_filename = patterns["latest_contours"].format(
                    model=MODEL_NAME, param=param_config["id"]
                )
                with open(os.path.join(output_dist_dir, latest_contour_filename), 'w') as f:
                    json.dump(master_contours, f, separators=(",", ":"))
    phases["contour_file"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks, frame_w, frame_h = build_volume_chunks(
        frame_arrays,
        steps_written,
        model_name=MODEL_NAME,
        param_config=param_config,
        target_date=target_date,
        chosen_run=CHOSEN_RUN,
        ptype_arrays=ptype_arrays if ptype_results else None
    )

    manifest_chunks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=GZIP_THREADS) as gz_pool:
        gz_futures = []
        for chunk in chunks:
            filename = chunk["manifest_data"]["file"]
            filepath = os.path.join(output_dist_dir, filename)

            if filename.endswith(".bin"):
                gz_futures.append(gz_pool.submit(_gzip_to_file, filepath, chunk["array"].tobytes()))
                ptype_filename = chunk["manifest_data"].get("ptype_file")
                if ptype_filename and "ptype_array" in chunk:
                    gz_futures.append(gz_pool.submit(
                        _gzip_to_file,
                        os.path.join(output_dist_dir, ptype_filename),
                        chunk["ptype_array"].tobytes(),
                    ))
            elif filename.endswith(".webp"):
                cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_WEBP_QUALITY), 101])
            else:
                cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_PNG_COMPRESSION), 6])

            manifest_chunks.append(chunk["manifest_data"])

        for fut in gz_futures:
            fut.result()
    phases["volumes+gzip"] = time.perf_counter() - t0

    manifest = {
        "model": MODEL_NAME,
        "parameter": param_config["id"],
        "name": param_config.get("name", param_config["id"]),
        "unit": param_config.get("unit", ""),
        "scaling": param_config.get("scaling", {}),
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
        "init_time": init_time_iso,
        "type": "volume_chunked",
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "temp_min_k": param_config.get("min_val", param_config.get("scaling", {}).get("min_val", 0.0)),
        "temp_max_k": param_config.get("max_val", param_config.get("scaling", {}).get("max_val", 255.0)),
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    run_manifest_filename = patterns["run_manifest"].format(
        model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
    )

    manifest_files_to_write = ["manifest.json", run_manifest_filename]

    if "latest_manifest" in patterns:
        latest_manifest_filename = patterns["latest_manifest"].format(
            model=MODEL_NAME, param=param_config["id"]
        )
        manifest_files_to_write.append(latest_manifest_filename)

    for m_fname in manifest_files_to_write:
        m_path = os.path.join(output_dist_dir, m_fname)
        with open(m_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    print(f"\n🎉 [{param_config['id']}] HRRR assets ready in {output_dist_dir}/")

    t0 = time.perf_counter()
    upload_to_huggingface(output_dist_dir, repo_id=DEFAULT_HF_REPO)
    phases["hf_upload"] = time.perf_counter() - t0

    try:
        shutil.rmtree(output_dist_dir)
        print(f"  ✅ [{param_config['id']}] Cleanup complete.")
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Failed to delete temp directory: {e}")

    _print_timing(param_config["id"], phases, time.perf_counter() - t_total)


if __name__ == "__main__":
    target_params = []
    custom_run = None
    custom_date = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--run" and i + 1 < len(args):
            custom_run = args[i + 1]
            i += 2
        elif args[i] == "--date" and i + 1 < len(args):
            custom_date = args[i + 1]
            i += 2
        else:
            target_params.append(args[i])
            i += 1

    if not target_params:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_params = list(data.get("parameters", {}).keys())
        else:
            target_params = ["2t"]

    print(f"🚀 Launching HRRR Pipeline for Parameters: {target_params}")

    batch_size = MAX_CONCURRENT_PARAMS
    t_run = time.perf_counter()

    for i in range(0, len(target_params), batch_size):
        batch = target_params[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = math.ceil(len(target_params) / batch_size)
        t_batch = time.perf_counter()

        print(f"\n📦 [Batch {batch_num}/{total_batches}] Running {len(batch)} HRRR parameter(s): {batch}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(run_master_pipeline, param, custom_run, custom_date): param for param in batch}
            for future in concurrent.futures.as_completed(futures):
                param = futures[future]
                try:
                    future.result()
                    print(f"✅ Finished HRRR parameter: {param}")
                except Exception as e:
                    print(f"❌ Error processing HRRR parameter '{param}': {e}")

        print(f"⏱️ Batch {batch_num}/{total_batches} took {time.perf_counter() - t_batch:.1f}s", flush=True)

    print("\n🎉 ALL HRRR PARAMETERS COMPLETED!")
    t_p = time.perf_counter()
    prune_old_huggingface_runs(repo_id=DEFAULT_HF_REPO, days_to_keep=2)
    print(f"⏱️ prune {time.perf_counter() - t_p:.1f}s | pipeline total {time.perf_counter() - t_run:.1f}s", flush=True)
