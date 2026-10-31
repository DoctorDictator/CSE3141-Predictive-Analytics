from pathlib import Path
import io
import zipfile
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
DATA.mkdir(parents=True, exist_ok=True)


def load_data():
    SOURCE = "https://archive.ics.uci.edu/static/public/492/metro%2Binterstate%2Btraffic%2Bvolume.zip"
    archive = DATA / "metro_interstate_traffic_volume.zip"
    if not archive.exists():
        try:
            response = requests.get(SOURCE, timeout=(15, 120))
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as check:
                assert any(name.endswith(".csv.gz") for name in check.namelist())
            archive.write_bytes(response.content)
        except Exception as error:
            raise RuntimeError(
                "UCI download failed. Download the ZIP from "
                + SOURCE
                + " into data/raw/metro_interstate_traffic_volume.zip and rerun. "
                + str(error)
            ) from error
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(".csv.gz"))
        raw = pd.read_csv(
            io.BytesIO(bundle.read(member)), compression="gzip", keep_default_na=False
        )
    return raw
