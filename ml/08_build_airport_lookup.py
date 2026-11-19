import pandas as pd

from config import ROOT

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OUT = ROOT / "ml" / "airport_lookup.csv"

df = pd.read_csv(URL, low_memory=False)
df = df[df["iata_code"].notna()
        & df["type"].isin(["large_airport", "medium_airport", "small_airport"])]
out = (df[["iata_code", "name", "iso_country", "latitude_deg", "longitude_deg"]]
       .drop_duplicates("iata_code"))
out.to_csv(OUT, index=False)
print(len(out), "airports saved to", OUT)