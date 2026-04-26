"""
Generates AirQuality_Dashboard.twbx:
  1. Writes cleaned data to a Tableau Hyper extract
  2. Builds a .twb XML workbook referencing the extract
  3. Zips both into .twbx
"""
import os, sys, shutil, zipfile, textwrap
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))
from analysis import load_cleaned, compute_h1_correlations, compute_h4_pca, REF_COLS

OUT_DIR   = os.path.dirname(__file__)
HYPER_REL = "AirQuality.hyper"
HYPER_ABS = os.path.join(OUT_DIR, HYPER_REL)
TWB_PATH  = os.path.join(OUT_DIR, "AirQuality_Dashboard.twb")
TWBX_PATH = os.path.join(OUT_DIR, "AirQuality_Dashboard.twbx")


# ── 1. Build enriched CSV / DataFrame ────────────────────────────────────────
def build_enriched_df():
    df = load_cleaned().copy()
    df["Hour"]    = df.index.hour
    df["Month"]   = df.index.month
    df["MonthName"] = df.index.strftime("%b")
    df["YearMonth"] = df.index.strftime("%Y-%m")
    df["Date"]    = df.index.date

    # Pearson r tier for each sensor pair from full dataset
    h1 = compute_h1_correlations(df)
    for name, res in h1.items():
        r = res["r"]
        df[f"r_{name}"] = round(r, 3)
        df[f"r_tier_{name}"] = ("High" if abs(r) > 0.80 else ("Medium" if abs(r) > 0.70 else "Low"))

    # PCA scores
    h4 = compute_h4_pca(df)
    scores = h4["scores"]
    idx = df[REF_COLS].dropna().index
    for i in range(2):
        df.loc[idx, f"PC{i+1}"] = scores[:, i]

    return df


# ── 2. Write Hyper extract ────────────────────────────────────────────────────
def write_hyper(df):
    from tableauhyperapi import (
        HyperProcess, Connection, Telemetry,
        TableDefinition, SqlType, TableName,
        Inserter, CreateMode
    )

    col_map = {
        "CO(GT)":       SqlType.double(),
        "PT08.S1(CO)":  SqlType.double(),
        "C6H6(GT)":     SqlType.double(),
        "PT08.S2(NMHC)":SqlType.double(),
        "NOx(GT)":      SqlType.double(),
        "PT08.S3(NOx)": SqlType.double(),
        "NO2(GT)":      SqlType.double(),
        "PT08.S4(NO2)": SqlType.double(),
        "PT08.S5(O3)":  SqlType.double(),
        "T":            SqlType.double(),
        "RH":           SqlType.double(),
        "AH":           SqlType.double(),
        "Hour":         SqlType.int(),
        "Month":        SqlType.int(),
        "MonthName":    SqlType.text(),
        "YearMonth":    SqlType.text(),
        "Date":         SqlType.text(),
        "r_CO":         SqlType.double(),
        "r_C6H6":       SqlType.double(),
        "r_NOx":        SqlType.double(),
        "r_NO2":        SqlType.double(),
        "r_tier_CO":    SqlType.text(),
        "r_tier_C6H6":  SqlType.text(),
        "r_tier_NOx":   SqlType.text(),
        "r_tier_NO2":   SqlType.text(),
        "PC1":          SqlType.double(),
        "PC2":          SqlType.double(),
        "Datetime":     SqlType.text(),
    }

    df2 = df.reset_index()
    df2["Datetime"] = df2["Datetime"].astype(str)
    df2["Date"]     = df2["Date"].astype(str)

    columns = [TableDefinition.Column(c, t) for c, t in col_map.items() if c in df2.columns]
    table_def = TableDefinition(TableName("Extract", "AirQuality"), columns)

    if os.path.exists(HYPER_ABS):
        os.remove(HYPER_ABS)

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(hyper.endpoint, HYPER_ABS, CreateMode.CREATE_AND_REPLACE) as conn:
            conn.catalog.create_schema_if_not_exists("Extract")
            conn.catalog.create_table_if_not_exists(table_def)

            col_names = [c.name.unescaped for c in columns]
            with Inserter(conn, table_def) as inserter:
                for _, row in df2[col_names].iterrows():
                    inserter.add_row([None if (isinstance(v, float) and np.isnan(v)) else v
                                      for v in row.values])
                inserter.execute()
    print(f"Hyper extract written: {HYPER_ABS}")


# ── 3. Build .twb XML ─────────────────────────────────────────────────────────
def build_twb():
    xml = textwrap.dedent(f"""\
    <?xml version='1.0' encoding='utf-8' ?>
    <workbook source-build='2023.1.0' source-platform='mac' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
      <datasources>
        <datasource caption='AirQuality' name='AirQuality' inline='true'>
          <connection class='hyper' dbname='{HYPER_REL}' schema='Extract' tablename='AirQuality'>
          </connection>
          <column datatype='real' name='[CO(GT)]' role='measure' type='quantitative'/>
          <column datatype='real' name='[PT08.S1(CO)]' role='measure' type='quantitative'/>
          <column datatype='real' name='[C6H6(GT)]' role='measure' type='quantitative'/>
          <column datatype='real' name='[NOx(GT)]' role='measure' type='quantitative'/>
          <column datatype='real' name='[NO2(GT)]' role='measure' type='quantitative'/>
          <column datatype='real' name='[T]' role='measure' type='quantitative'/>
          <column datatype='real' name='[RH]' role='measure' type='quantitative'/>
          <column datatype='integer' name='[Hour]' role='dimension' type='ordinal'/>
          <column datatype='integer' name='[Month]' role='dimension' type='ordinal'/>
          <column datatype='string' name='[MonthName]' role='dimension' type='nominal'/>
          <column datatype='string' name='[YearMonth]' role='dimension' type='nominal'/>
          <column datatype='string' name='[Datetime]' role='dimension' type='nominal'/>
          <column datatype='real' name='[r_CO]' role='measure' type='quantitative'/>
          <column datatype='real' name='[r_C6H6]' role='measure' type='quantitative'/>
          <column datatype='real' name='[r_NOx]' role='measure' type='quantitative'/>
          <column datatype='real' name='[r_NO2]' role='measure' type='quantitative'/>
          <column datatype='string' name='[r_tier_CO]' role='dimension' type='nominal'/>
          <column datatype='real' name='[PC1]' role='measure' type='quantitative'/>
          <column datatype='real' name='[PC2]' role='measure' type='quantitative'/>
        </datasource>
      </datasources>
      <worksheets>
        <worksheet name='Time Series'>
          <table>
            <view>
              <datasources>
                <datasource caption='AirQuality' name='AirQuality'/>
              </datasources>
              <datasource-dependencies datasource='AirQuality'>
                <column-instance column='[Datetime]' derivation='None' name='[none:Datetime:nk]' pivot='key' type='nominal'/>
                <column-instance column='[CO(GT)]' derivation='None' name='[sum:CO(GT):qk]' pivot='key' type='quantitative' role='measure'/>
                <column-instance column='[PT08.S1(CO)]' derivation='None' name='[sum:PT08.S1(CO):qk]' pivot='key' type='quantitative' role='measure'/>
              </datasource-dependencies>
              <shelf-sorts/>
              <aggregation value='true'/>
            </view>
            <style/>
          </table>
        </worksheet>
        <worksheet name='Sensor Accuracy'>
          <table>
            <view>
              <datasources>
                <datasource caption='AirQuality' name='AirQuality'/>
              </datasources>
              <datasource-dependencies datasource='AirQuality'>
                <column-instance column='[PT08.S1(CO)]' derivation='None' name='[none:PT08.S1(CO):qk]' pivot='key' type='quantitative'/>
                <column-instance column='[CO(GT)]' derivation='None' name='[none:CO(GT):qk]' pivot='key' type='quantitative'/>
                <column-instance column='[r_tier_CO]' derivation='None' name='[none:r_tier_CO:nk]' pivot='key' type='nominal'/>
              </datasource-dependencies>
              <aggregation value='true'/>
            </view>
            <style/>
          </table>
        </worksheet>
        <worksheet name='Temporal Heatmap'>
          <table>
            <view>
              <datasources>
                <datasource caption='AirQuality' name='AirQuality'/>
              </datasources>
              <datasource-dependencies datasource='AirQuality'>
                <column-instance column='[Hour]' derivation='None' name='[none:Hour:ok]' pivot='key' type='ordinal'/>
                <column-instance column='[MonthName]' derivation='None' name='[none:MonthName:nk]' pivot='key' type='nominal'/>
                <column-instance column='[CO(GT)]' derivation='None' name='[avg:CO(GT):qk]' pivot='key' type='quantitative' role='measure'/>
              </datasource-dependencies>
              <aggregation value='true'/>
            </view>
            <style/>
          </table>
        </worksheet>
        <worksheet name='PCA Structure'>
          <table>
            <view>
              <datasources>
                <datasource caption='AirQuality' name='AirQuality'/>
              </datasources>
              <datasource-dependencies datasource='AirQuality'>
                <column-instance column='[PC1]' derivation='None' name='[none:PC1:qk]' pivot='key' type='quantitative'/>
                <column-instance column='[PC2]' derivation='None' name='[none:PC2:qk]' pivot='key' type='quantitative'/>
                <column-instance column='[MonthName]' derivation='None' name='[none:MonthName:nk]' pivot='key' type='nominal'/>
              </datasource-dependencies>
              <aggregation value='true'/>
            </view>
            <style/>
          </table>
        </worksheet>
      </worksheets>
      <dashboards>
        <dashboard name='Air Quality Dashboard'>
          <style/>
          <zones>
            <zone h='50000' id='1' type='layout-flow' w='100000' x='0' y='0'>
              <zone h='50000' id='2' type='layout-flow' layoutFlow='horizontal' w='100000' x='0' y='0'>
                <zone h='50000' id='3' name='Time Series' type='worksheet' w='50000' x='0' y='0'/>
                <zone h='50000' id='4' name='Sensor Accuracy' type='worksheet' w='50000' x='50000' y='0'/>
              </zone>
              <zone h='50000' id='5' type='layout-flow' layoutFlow='horizontal' w='100000' x='0' y='50000'>
                <zone h='50000' id='6' name='Temporal Heatmap' type='worksheet' w='50000' x='0' y='50000'/>
                <zone h='50000' id='7' name='PCA Structure' type='worksheet' w='50000' x='50000' y='50000'/>
              </zone>
            </zone>
          </zones>
        </dashboard>
      </dashboards>
    </workbook>
    """)
    with open(TWB_PATH, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"TWB written: {TWB_PATH}")


# ── 4. Package as .twbx ───────────────────────────────────────────────────────
def package_twbx():
    if os.path.exists(TWBX_PATH):
        os.remove(TWBX_PATH)
    with zipfile.ZipFile(TWBX_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(TWB_PATH,  os.path.basename(TWB_PATH))
        zf.write(HYPER_ABS, HYPER_REL)
    print(f"TWBX packaged: {TWBX_PATH}")


if __name__ == "__main__":
    print("Building enriched dataset...")
    df_enriched = build_enriched_df()
    print("Writing Hyper extract...")
    write_hyper(df_enriched)
    print("Building TWB workbook XML...")
    build_twb()
    print("Packaging TWBX...")
    package_twbx()
    print("\nDone! Open tableau/AirQuality_Dashboard.twbx in Tableau Desktop.")
