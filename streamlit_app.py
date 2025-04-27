import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import time
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
from datetime import datetime, timedelta
import requests
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import geopandas as gpd
import shapely.geometry
import json
import pydeck as pdk
import matplotlib.cm as cm
import io
import base64
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import altair as alt

st.set_page_config(page_title='Indian Railways Visualization', layout='wide')

@st.cache_data
def load_schedule():
    return pd.read_csv('train_detail.csv')

@st.cache_data
def load_merged():
    return pd.read_csv('Merged_Output.csv')

schedule_df = load_schedule()
merged_df = load_merged()

main_sections = [
    "IndianRailViz",
    "Overview",
    "Delay Hotspots",
    "Route Analysis",
    "Stations Summary",
    "Advanced Viz",
    "Comparisions",
    "Rating"
]

main_tabs = st.tabs(main_sections)

with main_tabs[0]:
    
    

    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h2 style="color:yellow;">Welcome to the Indian Railways Visualization Portal</h2>
            <p style="font-size:17px; line-height:1.6;">
                This dashboard helps you understand Indian Railways better by showing easy-to-read charts and maps about train delays, station performance, cleanliness, earnings, and more. It’s useful for decision-makers, planners, and anyone interested in how the railway system works
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("## Why Indian Railways Matter")

    st.markdown(
       """
        - **India’s Main Travel System**: Every day, over 23 million people and 3 million tons of goods travel by train.  
        - **Huge Railway Network**: More than 13,000 passenger trains and 9,000 goods trains run daily, covering over 7,000 stations.  
        - **Boosts the Economy**: Helps in business, tourism, job creation, and connecting cities and villages.  
        - **Saves Energy**: Trains use less fuel and cause less pollution compared to cars or airplanes.  
        - **Brings People Together**: Connects different parts of the country and helps unite people from all backgrounds.
        """
    )

    st.markdown("## Key Features of This Platform")

    st.markdown(
        """
        - View train delays, top-performing trains, and route analysis.
        - Discover busy stations and their footfall, revenue, and cleanliness.
        - Explore maps that show where delays happen most often.
        - Compare different types of trains and how well they run across regions.
        """
    )

    st.markdown("---")

    st.success("Use the navigation tabs above to explore each section in detail.")

#tab1
with main_tabs[1]:
    sub_pages = ["Trains", "Stations"]
    sub_tabs = st.tabs(sub_pages)

    with sub_tabs[0]:
        categories = [
            "Rajdhani", "Duranto", "Vande Bharat", "Shatabdi", "Jan-Shatabdi",
            "HumSafar", "Garib Rath", "Subidha", "Superfast",
            "Mail/Exp", "Passenger"
        ]
        no_of_trains = [53, 51, 141, 63, 69, 94, 65, 406, 1387, 2304, 4505]
        no_of_tables = [270, 260, 710, 320, 350, 475, 330, 2035, 6929, 11506, 22529]

        df = pd.DataFrame({
            "Train Category": categories,
            "No. of Trains": no_of_trains,
            "No. of Tables": no_of_tables
        })

        st.subheader("Train and Table Donut Charts")
        st.title("Delay Data Summary")

        col1, col2, col3 = st.columns([4, 2, 4])

        with col1:
            st.subheader("No. of Trains")
            fig1 = go.Figure(data=[go.Pie(
                labels=categories,
                values=no_of_trains,
                hole=0.5,
                textinfo='label+value',
                textposition='inside',
                marker=dict(line=dict(color='#000000', width=0)),
                showlegend=False
            )])
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(f"**Total Trains:** {sum(no_of_trains)}")

        with col2:
            st.subheader("Train Category Data")
            st.table(df)

        with col3:
            st.subheader("No. of Tables")
            fig2 = go.Figure(data=[go.Pie(
                labels=categories,
                values=no_of_tables,
                hole=0.5,
                textinfo='label+value',
                textposition='inside',
                marker=dict(line=dict(color='#000000', width=0)),
                showlegend=False
            )])
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(f"**Total Tables:** {sum(no_of_tables)}")

    with sub_tabs[1]:
        st.subheader("3D India State Blocks by Station Count")

        @st.cache_data
        def load_geojson():
            with open("in.json") as f:
                return json.load(f)

        @st.cache_data
        def load_station_data():
            return pd.read_csv("corrected_file.csv")

        geojson = load_geojson()
        df = load_station_data()

        station_counts = df['State'].str.strip().str.title().value_counts().reset_index()
        station_counts.columns = ['State', 'Station_Count']

        state_corrections = {
            'Andaman And Nicobar Islands': 'Andaman & Nicobar Island',
            'Jammu And Kashmir': 'Jammu and Kashmir',
            'Odisha': 'Orissa'
        }
        station_counts['State'] = station_counts['State'].replace(state_corrections)

        geo_states = [feature['properties']['name'] for feature in geojson['features']]
        geo_states.sort()

        st.sidebar.header("Style and Color")
        color_map_name = st.sidebar.selectbox("Color Scheme", ["Set3", "Paired", "Dark2", "Accent", "tab10", "tab20"])
        cmap = plt.get_cmap(color_map_name)

        state_colors = {}
        for i, state in enumerate(geo_states):
            color = cmap(i % cmap.N)
            rgb = [int(255 * color[0]), int(255 * color[1]), int(255 * color[2])]
            state_colors[state] = rgb

        station_dict = dict(zip(station_counts['State'], station_counts['Station_Count']))
        for feature in geojson['features']:
            name = feature['properties']['name']
            feature['properties']['Station_Count'] = station_dict.get(name, 0)
            feature['properties']['fill_color'] = state_colors.get(name, [200, 200, 200])

        st.sidebar.header("Camera Controls")
        zoom = st.sidebar.slider("Zoom", 1, 10, 3, step=1)
        pitch = st.sidebar.slider("Tilt (Pitch)", 0, 85, 20, step=1)
        bearing = st.sidebar.slider("Rotation (Bearing)", 0, 360, 0)

        elevation_scale = st.sidebar.slider("Elevation Multiplier", 0, 3000, 350, step=50)

        view_state = pdk.ViewState(
            latitude=22,
            longitude=80,
            zoom=zoom,
            pitch=pitch,
            bearing=bearing
        )

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=True,
            get_elevation=f"properties.Station_Count * {elevation_scale}",
            get_fill_color="properties.fill_color",
            opacity=0.92
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}\nStations: {Station_Count}"},
            map_style="mapbox://styles/mapbox/light-v10"
        )

        st.pydeck_chart(deck)

        @st.cache_data
        def load_data():
            df = pd.read_csv('corrected_file.csv')
            return df

        df = load_data()

        st.title("Indian Railway Stations Visualization")

        st.sidebar.header("Filters")

        zone_options = sorted(df['Zone'].dropna().unique())
        selected_zones = st.sidebar.multiselect("Select Zones", zone_options, default=zone_options)

        state_options = sorted(df['State'].dropna().unique())
        selected_states = st.sidebar.multiselect("Select States", state_options, default=state_options)

        division_options = sorted(df['Division'].dropna().unique())
        selected_divisions = st.sidebar.multiselect("Select Divisions", division_options, default=division_options)

        filtered_df = df[
            (df['Zone'].isin(selected_zones)) &
            (df['State'].isin(selected_states)) &
            (df['Division'].isin(selected_divisions))
        ]

        st.sidebar.header("Filtered Results")
        st.dataframe(filtered_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Stations", len(df))
        col2.metric("Filtered Stations", len(filtered_df))
        col3.metric("Unique Zones", df['Zone'].nunique())

        stations_per_zone = filtered_df['Zone'].value_counts().reset_index()
        stations_per_zone.columns = ['Zone', 'Count']

        fig_zone = px.bar(
            stations_per_zone,
            x='Zone',
            y='Count',
            color='Zone',
            title="Stations per Zone",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig_zone, use_container_width=True)

        stations_per_state = filtered_df['State'].value_counts().reset_index()
        stations_per_state.columns = ['State', 'Count']

        fig_state = px.bar(
            stations_per_state,
            x='State',
            y='Count',
            color='State',
            title="Stations per State",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_state, use_container_width=True)

        st.sidebar.header("State-wise Zone Distribution of Stations")

        state_zone_distribution = filtered_df.groupby(['State', 'Zone']).size().reset_index(name='Station_Count')
        state_zone_distribution = state_zone_distribution.sort_values(by='State')

        fig_zone_dist = px.bar(
            state_zone_distribution,
            x="State",
            y="Station_Count",
            color="Zone",
            title="Distribution of Zones Across Each State",
            labels={"Station_Count": "No. of Stations"},
            height=600
        )

        fig_zone_dist.update_layout(barmode='stack', xaxis_tickangle=-45)
        st.plotly_chart(fig_zone_dist, use_container_width=True)
    
#tab2
with main_tabs[2]:
    sub_pages = ["Heatmap Hotspots", "Geospatial Hotspots"]
    sub_tabs = st.tabs(sub_pages)

    with sub_tabs[0]:
        st.title("Indian Railways — Stationwise Delay Map with Highlight and Theme Toggle")

        train_categories = [
            "Duronto", "Garib-Rath", "Humsafar", "Jan_Shatabdi",
            "Mail", "Passenger", "Rajdhani", "Shatabdi",
            "Superfast", "Suvidha", "Vande_Bharat"
        ]

        time_options = ["One Week", "One Month", "3 Months", "6 Months", "One Year"]

        train_mapping = {
            "Duronto": "D",
            "Garib-Rath": "GR",
            "Humsafar": "H",
            "Jan_Shatabdi": "J",
            "Mail": "E",
            "Passenger": "P",
            "Rajdhani": "R",
            "Shatabdi": "SH",
            "Superfast": "SU",
            "Suvidha": "SUV",
            "Vande_Bharat": "V"
        }

        time_mapping = {
            "One Week": "1W",
            "One Month": "1M",
            "3 Months": "3M",
            "6 Months": "6M",
            "One Year": "1Y"
        }

        min_size = 5
        max_size = 30

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_train_categories = st.multiselect(
                "Select Train Categories",
                options=train_categories,
                default=[]
            )

        with col2:
            selected_timeframe = st.selectbox(
                "Select Timeframe",
                options=time_options
            )

        filenames = []

        if selected_train_categories and selected_timeframe:
            for train_cat in selected_train_categories:
                mapped_train = train_mapping.get(train_cat)
                mapped_time = time_mapping.get(selected_timeframe)
                if mapped_train and mapped_time:
                    filenames.append(f"Master Delay/Stationwise_{mapped_train}{mapped_time}.csv")

        if filenames:
            st.subheader("Loaded Stationwise Database Files")
            for name in filenames:
                st.write(f"File: {name}")
        else:
            st.warning("Please select at least one Train Category and a Timeframe")

        temp_filename = "TempStationwiseMerged.csv"
        dataframes = []

        for file in filenames:
            if os.path.exists(file):
                df = pd.read_csv(file)
                df["Average_Delay"] = pd.to_numeric(df["Average_Delay"], errors="coerce")
                df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
                df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
                dataframes.append(df)
            else:
                st.warning(f"File not found: {file}")

        if dataframes:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            combined_df = pd.concat(dataframes)
            temp_df = combined_df.groupby(["Station_Name", "Station_Code", "Lat", "Long"], as_index=False).agg({
                "Average_Delay": "mean"
            })

            temp_df.to_csv(temp_filename, index=False)

            df_temp = temp_df.copy()
            df_temp["Average_Delay"] = pd.to_numeric(df_temp["Average_Delay"], errors="coerce")
            df_temp["Lat"] = pd.to_numeric(df_temp["Lat"], errors="coerce")
            df_temp["Long"] = pd.to_numeric(df_temp["Long"], errors="coerce")
            df_temp = df_temp.dropna(subset=["Lat", "Long", "Average_Delay"])

            delay_min = df_temp["Average_Delay"].min()
            delay_max = df_temp["Average_Delay"].max()

            def interpolate_size(delay):
                if delay_max == delay_min:
                    return (min_size + max_size) / 2
                return min_size + (delay - delay_min) * (max_size - min_size) / (delay_max - delay_min)

            df_temp["Size"] = df_temp["Average_Delay"].apply(interpolate_size)

            enable_highlight = st.toggle("Enable Highlighting")
            if enable_highlight:
                slider_value = st.slider(
                    "Select Delay Value to Highlight",
                    min_value=int(delay_min),
                    max_value=int(delay_max),
                    value=int(delay_min)
                )
                highlight_df = df_temp[
                    (df_temp["Average_Delay"] >= slider_value) &
                    (df_temp["Average_Delay"] <= slider_value + 10)
                ]
            else:
                highlight_df = pd.DataFrame()

            dark_mode = st.toggle("Switch to Dark Mode", value=False)
            map_style = "carto-darkmatter" if dark_mode else "carto-positron"

            fig = go.Figure()

            fig.add_trace(go.Scattermapbox(
                lat=df_temp["Lat"],
                lon=df_temp["Long"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=df_temp["Size"],
                    color=df_temp["Average_Delay"],
                    colorscale="Plasma_r",
                    cmin=delay_min,
                    cmax=delay_max,
                    showscale=True,
                    colorbar=dict(title="Avg Delay (min)")
                ),
                text=df_temp.apply(
                    lambda row: f"{row['Station_Name']}<br>Delay: {row['Average_Delay']:.2f} min<br>Lat: {row['Lat']:.4f}<br>Lon: {row['Long']:.4f}",
                    axis=1
                ),
                hoverinfo="text",
                name="All Stations"
            ))

            if not highlight_df.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=highlight_df["Lat"],
                    lon=highlight_df["Long"],
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=[30 for _ in highlight_df["Average_Delay"]],
                        color="cyan",
                        opacity=0.7
                    ),
                    text=highlight_df.apply(
                        lambda row: f"{row['Station_Name']}<br>Delay: {row['Average_Delay']:.2f} min<br>Lat: {row['Lat']:.4f}<br>Lon: {row['Long']:.4f}",
                        axis=1
                    ),
                    hoverinfo="text",
                    name="Highlighted Stations"
                ))

            fig.update_layout(
                mapbox=dict(
                    style=map_style,
                    zoom=4,
                    center={"lat": 22.9734, "lon": 78.6569}
                ),
                height=900,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                title="Stationwise Average Delay Map"
            )

            st.plotly_chart(fig, config={"scrollZoom": True})

        else:
            st.error("No data files were successfully loaded. Please select valid options.")

    with sub_tabs[1]:
        mapbox_token = "1e98e7bfc37485a1c384d8561a1eb85c83ab5dea"

        @st.cache_data
        def load_station_features():
            file_path = "Master_3M_Delay.csv"
            df = pd.read_csv(file_path)
            station_features = df.groupby('Station_Code').agg(avg_delay=('Delay', 'mean')).reset_index()
            return station_features

        @st.cache_data
        def load_station_coordinates():
            coords = []
            with open("stations.json", 'r') as f:
                data = json.load(f)
                for feature in data['features']:
                    prop = feature['properties']
                    geom = feature['geometry']
                    if geom is not None:
                        lon, lat = geom['coordinates']
                        coords.append({
                            'Station_Code': prop['code'],
                            'Latitude': lat,
                            'Longitude': lon
                        })
            coords_df = pd.DataFrame(coords)
            return coords_df

        def inverse_distance_weighting(x, y, z, xi, yi, power=2):
            tree = cKDTree(np.c_[x, y])
            dist, idx = tree.query(np.c_[xi, yi], k=6)
            weights = 1 / dist**power
            weights /= weights.sum(axis=1)[:, None]
            zi = np.sum(z[idx] * weights, axis=1)
            return zi

        @st.cache_data
        def load_india_shape():
            shapefile_path = "indianmaptrainpath//ne_110m_admin_0_countries.shp"
            world = gpd.read_file(shapefile_path)
            india = world[world['ADMIN'] == 'India']
            return india

        def mask_inside_india(x, y, india_polygon):
            points = gpd.GeoSeries([shapely.geometry.Point(xi, yi) for xi, yi in zip(x, y)])
            mask = points.within(india_polygon.unary_union)
            return mask

        st.title("Railway Station Delay Heatmap Masked to India")

        station_features = load_station_features()
        station_coords = load_station_coordinates()
        india = load_india_shape()

        merged = pd.merge(station_features, station_coords, on='Station_Code')

        lat_min, lat_max = merged['Latitude'].min() - 1, merged['Latitude'].max() + 1
        lon_min, lon_max = merged['Longitude'].min() - 1, merged['Longitude'].max() + 1

        grid_lon = np.linspace(lon_min, lon_max, 300)
        grid_lat = np.linspace(lat_min, lat_max, 300)
        grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

        grid_z = inverse_distance_weighting(
            merged['Longitude'].values,
            merged['Latitude'].values,
            merged['avg_delay'].values,
            grid_lon.ravel(),
            grid_lat.ravel()
        )

        mask = mask_inside_india(grid_lon.ravel(), grid_lat.ravel(), india)
        grid_z[~mask] = np.nan
        grid_z = grid_z.reshape(grid_lon.shape)

        fig = go.Figure()

        fig.add_trace(go.Contour(
            z=grid_z,
            x=np.linspace(lon_min, lon_max, grid_z.shape[1]),
            y=np.linspace(lat_min, lat_max, grid_z.shape[0]),
            colorscale='RdYlGn_r',
            contours_coloring='heatmap',
            line_width=2,
            showscale=True,
            colorbar_title="Avg Delay (min)",
            zmin=0,
            zmax=250,
            hovertemplate="Avg Delay: %{z:.2f} min<extra></extra>"
        ))

        fig.update_layout(
            title="Geospatial Railway Delay Heatmap with Isocurves (India only)",
            width=1000,
            height=750,
            xaxis_title="Longitude",
            yaxis_title="Latitude"
        )

        st.plotly_chart(fig, use_container_width=True)

with main_tabs[3]:
    sub_pages = ["Train Routes Visualization", "Train comparisions", "Route analysis"]
    sub_tabs = st.tabs(sub_pages)

    with sub_tabs[0]:
       
        @st.cache_data
        def load_stations(path="stations.json"):
            with open(path) as f:
                geo = json.load(f)
            coords, picklist = {}, []
            for feat in geo["features"]:
                geom = feat.get("geometry")
                if geom and geom["type"] == "Point":
                    lon, lat = geom["coordinates"]
                    c = feat["properties"].get("code", "")
                    n = feat["properties"].get("name", "")
                    coords[c] = (lat, lon)
                    picklist.append(f"{c} - {n}")
            return coords, sorted(picklist)

        @st.cache_data
        def load_delay_data(path="Master_3M_Delay.csv"):
            return pd.read_csv(path)

        @st.cache_data
        def get_trains_between(df, src, dst):
            out = []
            for tno, sub in df.groupby("Train_No"):
                route = sub.drop_duplicates("Station_Code")["Station_Code"].tolist()
                if src in route and dst in route:
                    i, j = route.index(src), route.index(dst)
                    if i < j:
                        name = sub.iloc[0].get("Train_Name", f"Train {tno}")
                        out.append({"Train_No": tno, "Train_Name": name, "route": route[i:j+1]})
            return out

        @st.cache_data
        def get_full_network_trains(df):
            out = []
            for tno, sub in df.groupby("Train_No"):
                route = sub.drop_duplicates("Station_Code")["Station_Code"].tolist()
                if len(route) > 1:
                    name = sub.iloc[0].get("Train_Name", f"Train {tno}")
                    out.append({"Train_No": tno, "Train_Name": name, "route": route})
            return out

        def build_path_segments(trains, coords):
            seen_seg = set()
            segments = []
            seen_dot = set()
            station_dots = []
            for tr in trains:
                route = tr["route"]
                for a, b in zip(route, route[1:]):
                    if a in coords and b in coords:
                        key = (a, b)
                        if key not in seen_seg:
                            seen_seg.add(key)
                            lat1, lon1 = coords[a]
                            lat2, lon2 = coords[b]
                            segments.append({"u": a, "v": b, "from": [lon1, lat1], "to": [lon2, lat2]})
                for s in route:
                    if s in coords and s not in seen_dot:
                        seen_dot.add(s)
                        lat, lon = coords[s]
                        station_dots.append({"Station_Code": s, "lat": lat, "lon": lon})
            return segments, station_dots

        @st.cache_data
        def station_avg_delay(df, station_codes):
            sub = df[df["Station_Code"].isin(station_codes)]
            return (sub.groupby("Station_Code")["Delay"].mean() / 60).to_dict()

        def color_segments(segments, station_delays, cmap_name="plasma_r"):
            delays = []
            for seg in segments:
                d1 = station_delays.get(seg["u"], np.nan)
                d2 = station_delays.get(seg["v"], np.nan)
                seg["delay"] = (d1 + d2) / 2
                delays.append(seg["delay"])

            arr = np.array(delays, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                for seg in segments:
                    seg["color"] = [200, 200, 200, 150]
                    seg["width"] = 2
                return segments

            low, high = np.percentile(arr, 5), np.percentile(arr, 95)
            cmap = cm.get_cmap(cmap_name)

            for seg in segments:
                d = seg["delay"]
                if np.isnan(d):
                    d = low
                d = min(max(d, low), high)
                norm = (d - low) / (high - low) if high > low else 0
                r, g, b, a = cmap(norm)
                seg["color"] = [int(r * 255), int(g * 255), int(b * 255), int(a * 200)]
                seg["width"] = 2 + 3 * norm
            return segments

        def draw_map(segments, station_dots, delay_thresh=0.0):
            filtered_segments = [s for s in segments if s.get("delay", 0) >= delay_thresh]
            if not filtered_segments:
                st.warning("No paths satisfy the selected delay threshold.")
                return

            arc_layer = pdk.Layer(
                "ArcLayer",
                data=filtered_segments,
                get_source_position="from",
                get_target_position="to",
                get_source_color="color",
                get_target_color="color",
                get_width="width",
                pickable=True,
                auto_highlight=True
            )

            dot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=station_dots,
                get_position='[lon,lat]',
                get_color='[0,0,0,0]',
                get_radius=500
            )

            india_center = pdk.ViewState(
                latitude=22.0,
                longitude=80.0,
                zoom=4.5,
                pitch=0,
                bearing=0
            )

            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=india_center,
                layers=[arc_layer, dot_layer],
                tooltip={"text": "{u} → {v}\nAvg Delay: {delay:.1f} hr"},
                height=700
            )

            st.pydeck_chart(deck, use_container_width=True)

        def show_colorbar_small(cmap_name="plasma_r"):
            fig, ax = plt.subplots(figsize=(4, 0.4))
            norm = plt.Normalize(vmin=0, vmax=1)
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name),
                            cax=ax, orientation='horizontal')
            cb.set_label("Delay Intensity")
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, transparent=True)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            css = f"""
            <div style='position: fixed; bottom: 20px; right: 20px; background: rgba(255,255,255,0.8); padding: 5px; border-radius: 5px;'>
                <img src="data:image/png;base64,{b64}" width="200">
            </div>
            """
            st.markdown(css, unsafe_allow_html=True)

        def main():
            st.title("Delay-Aware Railway Network Visualization")

            coords, station_list = load_stations()
            df = load_delay_data()

            mode = st.sidebar.radio("View Mode", ["Full Network", "Path A→B"], index=1)

            if mode == "Full Network":
                trains = get_full_network_trains(df)
                segments, station_dots = build_path_segments(trains, coords)
                codes = [d["Station_Code"] for d in station_dots]
                st_delays = station_avg_delay(df, codes)
                segments = color_segments(segments, st_delays)

                delay_thresh = st.sidebar.slider("Minimum delay (hr) to show", 0.0, 10.0, 0.0, step=0.1)
                draw_map(segments, station_dots, delay_thresh)
                show_colorbar_small()

            else:
                default_src = "NJP - NEW JALPAIGURI"
                default_dst = "CAPE - KANYAKUMARI"

                src = st.sidebar.selectbox("Source Station", station_list,
                                           index=station_list.index(default_src) if default_src in station_list else 0
                                           ).split(" - ")[0]

                dst = st.sidebar.selectbox("Destination Station", station_list,
                                           index=station_list.index(default_dst) if default_dst in station_list else 0
                                           ).split(" - ")[0]

                trains = get_trains_between(df, src, dst)
                if not trains:
                    st.warning(f"No trains found from {src} to {dst}.")
                    return

                temp_segments, temp_dots = build_path_segments(trains, coords)
                temp_codes = [d["Station_Code"] for d in temp_dots]
                station_delays = station_avg_delay(df, temp_codes)

                for tr in trains:
                    delays = [station_delays.get(s, 0) for s in tr["route"]]
                    tr["avg_delay"] = sum(delays) / len(delays) if delays else 0

                trains = sorted(trains, key=lambda x: x["avg_delay"])

                rank_df = pd.DataFrame([{
                    "Rank": i + 1,
                    "Train_No": tr["Train_No"],
                    "Train_Name": tr["Train_Name"],
                    "Avg_Delay (hr)": round(tr["avg_delay"], 2)
                } for i, tr in enumerate(trains)])

                st.sidebar.markdown("Trains Ranked by Avg Delay")
                st.sidebar.dataframe(rank_df, use_container_width=True)

                k = st.sidebar.number_input("Top k least-delay trains to plot", min_value=1, max_value=len(trains), value=1)
                trains = trains[:k]

                segments, station_dots = build_path_segments(trains, coords)
                codes = [d["Station_Code"] for d in station_dots]
                st_delays = station_avg_delay(df, codes)
                segments = color_segments(segments, st_delays)

                delay_thresh = st.sidebar.slider("Minimum delay (hr) to show", 0.0, 10.0, 0.0, step=0.1)
                draw_map(segments, station_dots, delay_thresh)
                show_colorbar_small()

        if __name__ == "__main__":
            main()

    with sub_tabs[1]:
        BASE_FOLDER = "RailViz_Data_Scraping-main"
        SCHEDULE_JSON_PATH = "schedules.json"
        MASTER_DELAY_FOLDER = os.path.join(BASE_FOLDER, "Master_Delay")

        TIME_PERIOD_FILES = {
            "Weekly": "Master_Weekly_Delay.csv",
            "Monthly": "Master_Monthly_Delay.csv",
            "3M": "Master_3M_Delay.csv",
            "6M": "Master_6M_Delay.csv",
            "Yearly": "Master_Yearly_Delay.csv"
        }

        @st.cache_data
        def load_schedule():
            with open(SCHEDULE_JSON_PATH, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["train_number"] = df["train_number"].astype(str).str.strip()
            df["station_name"] = df["station_name"].astype(str).str.strip().str.upper()
            df["departure"] = pd.to_datetime(df["departure"], format="%H:%M:%S", errors='coerce')
            df["arrival"] = pd.to_datetime(df["arrival"], format="%H:%M:%S", errors='coerce')
            return df

        @st.cache_data
        def load_delay(period):
            file_name = TIME_PERIOD_FILES[period]
            df = pd.read_csv(os.path.join(MASTER_DELAY_FOLDER, file_name))
            df.columns = df.columns.str.strip()
            df["Train_No"] = df["Train_No"].astype(str).str.strip()
            df["Station_Name"] = df["Station_Name"].astype(str).str.strip().str.upper()
            return df

        st.title("Train Schedule Delay Gantt Chart")

        period = st.sidebar.selectbox("Select Time Period", list(TIME_PERIOD_FILES.keys()))
        delay_df = load_delay(period)
        schedule_df = load_schedule()

        all_stations = sorted(schedule_df["station_name"].dropna().unique())
        src_station = st.sidebar.selectbox("Select Source Station", all_stations, index=all_stations.index("KANPUR CENTRAL") if "KANPUR CENTRAL" in all_stations else 0)
        dst_station = st.sidebar.selectbox("Select Destination Station", all_stations, index=all_stations.index("NEW DELHI") if "NEW DELHI" in all_stations else 1)

        if src_station == dst_station:
            st.warning("Source and destination must be different.")
            st.stop()

        train_segments = []

        for train_no, group in schedule_df.groupby("train_number"):
            group_sorted = group.reset_index(drop=True)
            if src_station in group_sorted["station_name"].values and dst_station in group_sorted["station_name"].values:
                src_idx = group_sorted[group_sorted["station_name"] == src_station].index[0]
                dst_idx = group_sorted[group_sorted["station_name"] == dst_station].index[0]

                if src_idx < dst_idx:
                    src_row = group_sorted.loc[src_idx]
                    dst_row = group_sorted.loc[dst_idx]

                    start = pd.to_datetime("2023-01-01 " + str(src_row["departure"].time()), errors='coerce')
                    end = pd.to_datetime("2023-01-01 " + str(dst_row["arrival"].time() if pd.notnull(dst_row["arrival"]) else dst_row["departure"].time()), errors='coerce')

                    if pd.notna(start) and pd.notna(end) and end <= start:
                        end += pd.Timedelta(days=1)

                    delay_src = delay_df[(delay_df["Train_No"] == train_no) & (delay_df["Station_Name"] == src_station)]["Delay"].mean()
                    delay_dst = delay_df[(delay_df["Train_No"] == train_no) & (delay_df["Station_Name"] == dst_station)]["Delay"].mean()

                    if pd.isna(delay_src) or pd.isna(delay_dst):
                        continue

                    train_segments.append({
                        "Train_No": train_no,
                        "Train_Name": src_row["train_name"],
                        "Start_Time": start,
                        "End_Time": end,
                        "Delay_Source": delay_src,
                        "Delay_Destination": delay_dst
                    })

        if not train_segments:
            st.warning("No valid trains between selected stations for this time period.")
            st.stop()

        chart_df = pd.DataFrame(train_segments)
        chart_df["Label"] = chart_df["Train_No"].astype(str) + " - " + chart_df["Train_Name"]
        chart_df["Color"] = "#FFFF00"
        chart_df = chart_df.sort_values(by="Start_Time")

        fig = px.timeline(
            chart_df,
            x_start="Start_Time",
            x_end="End_Time",
            y="Label",
            color="Color",
            color_discrete_map="identity",
            hover_data=["Train_No", "Train_Name", "Delay_Source", "Delay_Destination"]
        )

        label_to_y = {label: i for i, label in enumerate(chart_df["Label"])}

        for i, row in chart_df.iterrows():
            delay_start = row["Start_Time"] + pd.Timedelta(minutes=row["Delay_Source"])
            delay_end = row["End_Time"] + pd.Timedelta(minutes=row["Delay_Destination"])
            y_pos = label_to_y[row["Label"]]

            fig.add_shape(
                type="rect",
                x0=delay_start,
                x1=delay_end,
                y0=y_pos + 0.2,
                y1=y_pos - 0.4,
                xref="x",
                yref="y",
                fillcolor="red",
                opacity=0.3,
                layer="above",
                line_width=0
            )

        fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), categoryorder="array", categoryarray=chart_df["Label"])
        fig.update_traces(width=0.5)
        chart_height = max(300, len(chart_df) * 35)

        start_time = chart_df["Start_Time"].min().replace(minute=0, second=0)
        end_time = chart_df["End_Time"].max().replace(minute=0, second=0) + pd.Timedelta(hours=1)
        current = start_time
        while current <= end_time:
            fig.add_shape(
                type="line",
                x0=current, x1=current,
                y0=-0.5, y1=len(chart_df) - 0.5,
                xref="x", yref="y",
                line=dict(color="lightgray", width=1, dash="dot"),
                layer="below"
            )
            current += pd.Timedelta(hours=4)

        legend_x_start = 0.75
        legend_y_top = 1.05
        box_height = 0.02
        box_width = 0.025
        line_gap = 0.09

        fig.add_shape(type="rect", xref="paper", yref="paper", x0=legend_x_start, x1=legend_x_start + box_width,
                    y0=legend_y_top - box_height / 2, y1=legend_y_top + box_height / 2,
                    fillcolor="yellow", line=dict(width=1, color="white"), layer="above")

        fig.add_annotation(xref="paper", yref="paper", x=legend_x_start + box_width + 0.01, y=legend_y_top,
                        text="Scheduled Duration", showarrow=False, font=dict(size=12, color="white"))

        legend_y_bottom = legend_y_top - line_gap
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=legend_x_start, x1=legend_x_start + box_width,
                    y0=legend_y_bottom - box_height / 2, y1=legend_y_bottom + box_height / 2,
                    fillcolor="red", opacity=0.3, line=dict(width=1, color="white"), layer="above")

        fig.add_annotation(xref="paper", yref="paper", x=legend_x_start + box_width + 0.01, y=legend_y_bottom,
                        text="Average Delay", showarrow=False, font=dict(size=12, color="white"))

        fig.update_layout(
            height=chart_height,
            showlegend=False,
            margin=dict(l=20, r=20, t=70, b=20),
            title={
                "text": f"Train Schedules from {src_station.upper()} to {dst_station.upper()} ({period})",
                "x": 0.5,
                "xanchor": "center"
            },
            xaxis_title="",
            xaxis=dict(tickformat="%H:%M", dtick=14400000)
        )

        st.plotly_chart(fig, use_container_width=True)

with main_tabs[4]:
    sub_pages = ["Peak Hours", "Foot-fall and Revenue"]
    sub_tabs = st.tabs(sub_pages)

    with sub_tabs[0]:
        st.title("Indian Railways Rush Hour Analysis")

        @st.cache_data
        def load_data():
            try:
                df = pd.read_csv('isl_wise_train_detail_03082015_v1.csv')
                if len(df.columns) == 1:
                    df = pd.read_csv('isl_wise_train_detail_03082015_v1.csv', 
                        names=['Train No', 'Train Name', 'islno', 'Station Code', 'Station Name', 
                               'Arrival time', 'Departure time', 'Distance', 'Source Station Code', 
                               'source Station Name', 'Destination station Code', 'Destination Station Name'])
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace("'", "").str.strip()

                def parse_time(time_str):
                    try:
                        if pd.isna(time_str) or time_str == '':
                            return np.nan, np.nan
                        time_str = time_str.replace("'", "").strip()
                        parts = time_str.split(':')
                        hour = int(parts[0])
                        minute = int(parts[1]) if len(parts) > 1 else 0
                        return hour, minute
                    except:
                        return np.nan, np.nan

                df['ArrivalHour'], df['ArrivalMinute'] = zip(*df['Arrival time'].apply(parse_time))
                df['DepartureHour'], df['DepartureMinute'] = zip(*df['Departure time'].apply(parse_time))

                df['ArrivalMinutesSinceMidnight'] = df['ArrivalHour'] * 60 + df['ArrivalMinute']
                df['DepartureMinutesSinceMidnight'] = df['DepartureHour'] * 60 + df['DepartureMinute']

                df = df.dropna(subset=['ArrivalMinutesSinceMidnight', 'DepartureMinutesSinceMidnight'])
                return df
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return pd.DataFrame()

        with st.spinner("Loading and processing train data..."):
            df = load_data()

        if df.empty:
            st.error("Failed to load data. Please check your CSV file.")
            st.stop()

        station_list = sorted(df['Station Name'].unique())
        selected_station = st.selectbox("Select a station to analyze rush hours", station_list)

        bin_size = st.slider("Select Time Bin Size (minutes)", min_value=15, max_value=180, value=60, step=15)

        def classify_time_period(hour):
            if hour is None or pd.isna(hour):
                return "Unknown"
            hour = int(hour)
            if 5 <= hour < 10:
                return "Morning"
            elif 10 <= hour < 16:
                return "Midday"
            elif 16 <= hour < 20:
                return "Evening"
            else:
                return "Night"

        def calculate_rush_hour(station_name, bin_size_minutes=60):
            station_df = df[df['Station Name'] == station_name].copy()
            if station_df.empty:
                return pd.DataFrame(), "No data available for this station"

            events = []
            for _, row in station_df.iterrows():
                if not pd.isna(row['ArrivalMinutesSinceMidnight']):
                    events.append({'Time': row['ArrivalMinutesSinceMidnight'], 'Type': 'Arrival', 'Change': 1})
                if not pd.isna(row['DepartureMinutesSinceMidnight']):
                    events.append({'Time': row['DepartureMinutesSinceMidnight'], 'Type': 'Departure', 'Change': -1})

            events_df = pd.DataFrame(events)
            if events_df.empty:
                return pd.DataFrame(), "No arrival/departure data available for this station"

            events_df = events_df.sort_values(by=['Time', 'Type'], ascending=[True, False])
            train_count = 0
            for i, event in enumerate(events_df.itertuples()):
                train_count += event.Change
                events_df.loc[event.Index, 'TrainCount'] = train_count

            total_minutes_in_day = 24 * 60
            bins = list(range(0, total_minutes_in_day + bin_size_minutes, bin_size_minutes))

            bin_labels = []
            for start_min in bins[:-1]:
                end_min = start_min + bin_size_minutes
                start_hour, start_minute = divmod(start_min, 60)
                end_hour, end_minute = divmod(end_min, 60)
                if end_hour >= 24:
                    end_hour %= 24
                bin_labels.append(f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}")

            events_df['TimeBin'] = pd.cut(events_df['Time'], bins=bins, labels=bin_labels, right=False, include_lowest=True)
            bin_max_counts = events_df.groupby('TimeBin')['TrainCount'].max().reset_index()
            bin_max_counts['TrainCount'] = bin_max_counts['TrainCount'].abs()

            max_train_count = bin_max_counts['TrainCount'].max()
            non_zero_counts = bin_max_counts[bin_max_counts['TrainCount'] > 0]['TrainCount']
            if non_zero_counts.nunique() == 1 and non_zero_counts.max() <= 2:
                rush_hour_message = "No Rush Hour Detected - Small Station with Consistent Traffic"
            else:
                rush_hour_bins = bin_max_counts[bin_max_counts['TrainCount'] == max_train_count]['TimeBin'].tolist()
                rush_hour_message = f"Rush Hour(s): {', '.join(rush_hour_bins)} with {max_train_count} trains"

            all_bins = pd.DataFrame({'TimeBin': bin_labels})
            all_bins['SortKey'] = all_bins['TimeBin'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1].split('-')[0]))
            all_bins = all_bins.sort_values('SortKey')
            bin_max_counts = pd.merge(all_bins[['TimeBin']], bin_max_counts, on='TimeBin', how='left')
            bin_max_counts['TrainCount'] = bin_max_counts['TrainCount'].fillna(0)
            bin_max_counts['TimePeriod'] = bin_max_counts['TimeBin'].apply(lambda x: classify_time_period(int(x.split(':')[0])))

            return bin_max_counts, rush_hour_message

        rush_hour_data, rush_hour_message = calculate_rush_hour(selected_station, bin_size)

        if not rush_hour_data.empty:
            st.subheader(f"Rush Hour Profile for {selected_station} ({bin_size}-minute bins)")

            fig = px.bar(
                rush_hour_data,
                x="TimeBin",
                y="TrainCount",
                color="TimePeriod",
                color_discrete_map={
                    "Morning": "#1E3A8A",
                    "Midday": "#C2410C",
                    "Evening": "#9D174D",
                    "Night": "#4C1D95"
                },
                labels={"TimeBin": "Time of Day", "TrainCount": "Maximum Number of Trains"},
                height=500,
                category_orders={"TimeBin": rush_hour_data['TimeBin'].tolist()}
            )

            fig.update_layout(
                xaxis=dict(
                    title="Time of Day (24-hour format)",
                    tickangle=45,
                    showgrid=False,
                    type='category',
                    categoryorder='array',
                    categoryarray=rush_hour_data['TimeBin'].tolist()
                ),
                yaxis=dict(
                    title="Maximum Number of Trains at Station",
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                bargap=0.1,
                hovermode="closest",
                legend_title_text="Time Period"
            )

            fig.update_traces(hovertemplate="<b>%{x}</b><br>Max Trains: %{y}<extra></extra>")

            if "No Rush Hour" not in rush_hour_message:
                max_train_count = rush_hour_data['TrainCount'].max()
                rush_bins = rush_hour_data[rush_hour_data['TrainCount'] == max_train_count]['TimeBin'].tolist()
                for rush_bin in rush_bins:
                    fig.add_annotation(
                        x=rush_bin,
                        y=max_train_count,
                        text="RUSH HOUR",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(size=12, color="black", family="Arial"),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="rgba(0, 0, 0, 0.5)",
                        borderwidth=1,
                    )
            else:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="NO RUSH HOUR DETECTED",
                    showarrow=False,
                    font=dict(size=20, color="gray", family="Arial"),
                    opacity=0.7
                )

            st.plotly_chart(fig, use_container_width=True)

            if "No Rush Hour" in rush_hour_message:
                st.info(rush_hour_message)
        else:
            st.warning(f"No train data available for {selected_station} to calculate rush hours.")

    with sub_tabs[1]:
        st.subheader("Foot-fall and Revenue Summary")

        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #1E3A8A;
                text-align: center;
                margin-bottom: 1rem;
            }
            .section-header {
                font-size: 1.8rem;
                color: #1E3A8A;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f0f4f8;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: bold;
                color: #1E3A8A;
            }
            .metric-label {
                font-size: 1rem;
                color: #4B5563;
            }
            .filter-container {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h1 class='main-header'>Indian Railways Revenue Analysis</h1>", unsafe_allow_html=True)

        @st.cache_data
        def load_data():
            return pd.read_csv('top_100_Revenue_Vs_Football.csv')

        df = load_data()

        @st.cache_data
        def load_geojson():
            url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
            response = requests.get(url)
            geojson_data = response.json()
            for feature in geojson_data['features']:
                feature['id'] = feature['properties']['ST_NM']
            return geojson_data

        india_states_geojson = load_geojson()

        st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            all_states = sorted(df['State'].unique())
            selected_states = st.multiselect(
                "Select States",
                options=all_states,
                default=all_states,
                key="states_filter"
            )

        with col2:
            all_categories = sorted(df['Category'].unique())
            selected_categories = st.multiselect(
                "Select Categories",
                options=all_categories,
                default=all_categories,
                key="categories_filter"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if not selected_states and not selected_categories:
            filtered_df = df
        elif not selected_states:
            filtered_df = df[df['Category'].isin(selected_categories)]
        elif not selected_categories:
            filtered_df = df[df['State'].isin(selected_states)]
        else:
            filtered_df = df[(df['State'].isin(selected_states)) & (df['Category'].isin(selected_categories))]

        state_revenue = filtered_df.groupby('State')['Revenue'].sum().reset_index()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">₹{filtered_df['Revenue'].sum()/10000000:.2f} Cr</div>
                    <div class="metric-label">Total Revenue</div>
                </div>""", unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{filtered_df['Passengers Footfall'].sum()/1000000:.2f}M</div>
                    <div class="metric-label">Total Footfall</div>
                </div>""", unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{len(filtered_df['State'].unique())}</div>
                    <div class="metric-label">States Represented</div>
                </div>""", unsafe_allow_html=True
            )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("<h2 class='section-header'>Top States by Revenue</h2>", unsafe_allow_html=True)

            top_revenue_states = state_revenue.sort_values('Revenue', ascending=False).head(10)
            fig_top_revenue = px.bar(
                top_revenue_states,
                x='State',
                y='Revenue',
                color='Revenue',
                color_continuous_scale='Viridis',
                labels={'Revenue': 'Revenue (₹)'}
            )
            fig_top_revenue.update_layout(
                xaxis_title="State",
                yaxis_title="Revenue (₹)",
                height=500,
                xaxis={'categoryorder': 'total descending'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_top_revenue.update_traces(
                textfont_size=12,
                textangle=0,
                textposition="outside",
                cliponaxis=False,
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            st.plotly_chart(fig_top_revenue, use_container_width=True)

        with chart_col2:
            st.markdown("<h2 class='section-header'>Revenue vs Footfall Analysis</h2>", unsafe_allow_html=True)

            zoom_options = {
                "All Data": [None, None],
                "Low Range": [0, filtered_df['Passengers Footfall'].quantile(0.25)],
                "Mid Range": [filtered_df['Passengers Footfall'].quantile(0.25), filtered_df['Passengers Footfall'].quantile(0.75)],
                "High Range": [filtered_df['Passengers Footfall'].quantile(0.75), filtered_df['Passengers Footfall'].max()]
            }
            selected_zoom = st.selectbox("Select Zoom Range", options=list(zoom_options.keys()), index=0)

            jittered_df = filtered_df.copy()
            jitter_amount = filtered_df['Passengers Footfall'].max() * 0.01
            jittered_df['Passengers Footfall'] += np.random.uniform(-jitter_amount, jitter_amount, len(jittered_df))
            jittered_df['Station'] = jittered_df['Station'].fillna('N/A')

            fig_scatter = px.scatter(
                jittered_df,
                x="Passengers Footfall",
                y="Revenue",
                size="Revenue",
                color="State",
                hover_name="Station",
                hover_data=["State", "Revenue", "Passengers Footfall", "Code", "Zone", "Division", "Category"],
                size_max=20,
                opacity=0.8,
                template="plotly_white",
                height=500,
                log_y=True
            )

            fig_scatter.update_xaxes(
                showline=True,
                linewidth=2,
                linecolor='lightgray',
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                ticks="outside",
                tickfont=dict(size=10),
                rangeslider=dict(
                    visible=True,
                    thickness=0.05,
                    bgcolor="lightblue",
                    bordercolor="gray",
                    borderwidth=1
                ),
                range=zoom_options[selected_zoom]
            )
            fig_scatter.update_yaxes(
                showline=True,
                linewidth=2,
                linecolor='lightgray',
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                ticks="outside",
                tickfont=dict(size=10),
                tickformat=",d",
                exponentformat="none"
            )

            fig_scatter.update_layout(
                title_text="",
                plot_bgcolor='rgba(0,0,0,0)',
                legend_title="State",
                font=dict(family="Arial", size=12),
                title_font=dict(size=16),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    font_color="black",
                    bordercolor="#636363"
                ),
                hovermode="closest",
                margin=dict(l=40, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)'
            )

            top_stations = filtered_df.nlargest(3, 'Revenue')
            for i, row in top_stations.iterrows():
                fig_scatter.add_annotation(
                    x=row['Passengers Footfall'],
                    y=row['Revenue'],
                    text=row['Station'],
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=10, color="#1E3A8A")
                )

            st.plotly_chart(fig_scatter, use_container_width=True)

with main_tabs[5]:
    sub_pages = ["Station Delays", "Cleanliness Analysis", "Trains Analysis"]
    sub_tabs = st.tabs(sub_pages)

    with sub_tabs[0]:
        @st.cache_data
        def load_data(file_path):
            df = pd.read_csv(file_path)
            return df

        def engineer_features(df):
            df = df[['Station_Name', 'Station_Code', 'Delay']].dropna()
            grouped = df.groupby(['Station_Code', 'Station_Name']).agg(
                avg_delay=('Delay', 'mean'),
                max_delay=('Delay', 'max'),
                min_delay=('Delay', 'min'),
                std_delay=('Delay', 'std'),
                count_delays=('Delay', 'count')
            ).reset_index()
            grouped['std_delay'] = grouped['std_delay'].fillna(0)
            return grouped

        def reduce_dimensions(features_df):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df[['avg_delay', 'max_delay', 'min_delay', 'std_delay', 'count_delays']])
            reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap = reducer.fit_transform(X_scaled)
            return X_umap

        def perform_clustering(embedded_features, n_clusters=3):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embedded_features)
            return labels

        def plot_clusters(station_info, embedded_features, labels, sorted_clusters):
            plot_df = pd.DataFrame({
                'x': embedded_features[:, 0],
                'y': embedded_features[:, 1],
                'Station_Code': station_info['Station_Code'],
                'Station_Name': station_info['Station_Name'],
                'avg_delay': station_info['avg_delay'],
                'Cluster': labels
            })

            custom_colors = [
                "#90ee90", "#f0e68c", "#ffd700", "#ffa500", "#ff8c00",
                "#ff4500", "#dc143c", "#b22222", "#8b0000", "#660000"
            ]
            num_clusters = len(sorted_clusters)
            color_list = custom_colors[:num_clusters]

            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color=plot_df['Cluster'].astype(str),
                color_discrete_sequence=color_list,
                hover_data={
                    "Cluster": False,
                    "avg_delay": True,
                    "x": False,
                    "y": False,
                    "Station_Code": True,
                    "Station_Name": True,
                },
                title="Station Clustering Based on Delay Patterns",
                labels={"x": "UMAP Dimension 1", "y": "UMAP Dimension 2", "color": "Cluster ID"},
                width=1000,
                height=700,
            )

            fig.update_layout(
                title_font_size=24,
                xaxis_title_font_size=20,
                yaxis_title_font_size=20,
                legend_title_font_size=18,
                font=dict(size=16),
                legend=dict(itemsizing='constant')
            )

            st.plotly_chart(fig, use_container_width=True)

        def create_cluster_summary(station_features):
            summary = station_features.groupby('Cluster').agg(
                num_stations=('Station_Code', 'count'),
                avg_delay=('avg_delay', 'mean')
            ).reset_index()

            summary.rename(columns={'avg_delay': 'avg_delay (minutes)'}, inplace=True)
            summary = summary.sort_values(by='avg_delay (minutes)').reset_index(drop=True)

            def assign_label(avg_delay):
                if avg_delay <= 30:
                    return " Good (Low Delay)"
                elif avg_delay <= 60:
                    return " Medium Delay"
                else:
                    return " High Delay"

            summary['Cluster_Label'] = summary['avg_delay (minutes)'].apply(assign_label)
            return summary

        def main():
            st.title("Railway Station Clustering Based on Delay Patterns")

            file_options = {
                "Weekly": "RailViz_Data_Scraping-main/Master_Delay/Master_Weekly_Delay.csv",
                "Monthly": "RailViz_Data_Scraping-main/Master_Delay/Master_Monthly_Delay.csv",
                "3 Month": "RailViz_Data_Scraping-main/Master_Delay/Master_3M_Delay.csv",
                "6 Month": "RailViz_Data_Scraping-main/Master_Delay/Master_6M_Delay.csv",
                "Yearly": "RailViz_Data_Scraping-main/Master_Delay/Master_Yearly_Delay.csv",
            }

            period_choice = st.sidebar.selectbox("Select Time Period", list(file_options.keys()))
            file_path = file_options[period_choice]

            if os.path.exists(file_path):
                with st.spinner('Loading and processing data...'):
                    df = load_data(file_path)
                    station_features = engineer_features(df)
                    embedded_features = reduce_dimensions(station_features)

                    st.sidebar.header("Clustering Settings")
                    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

                    cluster_labels = perform_clustering(embedded_features, n_clusters=n_clusters)
                    station_features['Cluster'] = cluster_labels

                    cluster_summary = create_cluster_summary(station_features)
                    plot_clusters(station_features, embedded_features, cluster_labels, cluster_summary['Cluster'].tolist())

                    st.subheader("Cluster Summary (Sorted)")
                    st.dataframe(cluster_summary)
            else:
                st.error(f"CSV file not found at {file_path}")

            st.markdown("---")

        if __name__ == "__main__":
            main()

    with sub_tabs[1]:
        st.subheader("Tisnee")

        @st.cache_data
        def load_data():
            file_path = "final_unique_station_clean3.csv"
            df = pd.read_csv(file_path)
            df = df.dropna(subset=['longitude', 'latitude', 'Total_Score'])
            return df

        df = load_data()

        st.sidebar.header("Clustering Settings")
        num_clusters = st.sidebar.number_input("Select number of clusters (k)", min_value=2, max_value=20, value=5, step=1)

        features = df[['longitude', 'latitude', 'Total_Score']].copy()

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        features_scaled[:, 2] *= 5

        tsne = TSNE(
            n_components=3,
            perplexity=5,
            random_state=42,
            learning_rate='auto',
            init='random'
        )
        tsne_results = tsne.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tsne_results)

        hover_texts = [
            f"Station: {name}<br>Score: {score:.2f}<br>Cluster: {cluster}"
            for name, score, cluster in zip(df['Station_Name_lower'], df['Total_Score'], cluster_labels)
        ]

        fig = go.Figure(data=[go.Scatter3d(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            z=tsne_results[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=cluster_labels,
                colorscale='Rainbow',
                opacity=0.9,
                colorbar=dict(title='Cluster ID')
            ),
            text=hover_texts,
            hoverinfo='text'
        )])

        fig.update_layout(
            title="3D Railway Station Cleanliness Clustering",
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            height=850,
            width=1100,
            margin=dict(l=10, r=10, t=50, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with sub_tabs[2]:

        _slider_counter = 0
        def unique_slider(label, min_value, max_value, value, **kwargs):
            global _slider_counter
            if "key" not in kwargs:
                kwargs["key"] = f"slider_{_slider_counter}"
                _slider_counter += 1
            return st.sidebar.slider(label, min_value, max_value, value, **kwargs)

        @st.cache_data
        def load_all_data(main_folder_path):
            all_data = []
            for folder_name in os.listdir(main_folder_path):
                folder_path = os.path.join(main_folder_path, folder_name)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(folder_path, file))
                            df = df.dropna(subset=["Delay"])
                            df = df[df["Delay"] != 0]
                            if not df.empty:
                                df['Train_Type'] = folder_name.replace('List_of_All_', '').replace('_Trains', '')
                                df['Time_Period'] = file.replace('Category_', '').replace('_Delay.csv', '')
                                all_data.append(df)
            return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        def engineer_features(df):
            grouped = df.groupby(['Train_No', 'Train_Name']).agg(
                avg_delay=('Delay', 'mean'),
                max_delay=('Delay', 'max'),
                min_delay=('Delay', 'min'),
                std_delay=('Delay', 'std'),
                count_delays=('Delay', 'count')
            ).reset_index()
            grouped['std_delay'] = grouped['std_delay'].fillna(0)
            return grouped

        def reduce_dimensions_pca(features_df):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df[['avg_delay','max_delay','min_delay','std_delay','count_delays']])
            return PCA(n_components=3).fit_transform(X_scaled)

        def perform_clustering_and_sort(features_df, embedded_features, n_clusters=3):
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embedded_features)
            features_df['Cluster'] = labels
            means = features_df.groupby('Cluster')['avg_delay'].mean()
            order = means.sort_values().index.tolist()
            mapping = {old:new for new,old in enumerate(order)}
            features_df['Sorted_Cluster'] = features_df['Cluster'].map(mapping)
            return features_df['Sorted_Cluster'].values

        def plot_clusters_3d(train_info, embedded, sorted_labels):
            df = pd.DataFrame({
                'x': embedded[:,0], 'y': embedded[:,1], 'z': embedded[:,2],
                'Train_No': train_info['Train_No'],
                'Train_Name': train_info['Train_Name'],
                'Average_Delay': train_info['avg_delay'],
                'Cluster_ID': sorted_labels
            })
            fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                color=df['Cluster_ID'].astype(str),
                hover_data={'Train_No':True,'Train_Name':True,'Average_Delay':':.2f'},
                title="Train Clustering using PCA 3D",
                labels={"x":"PC1","y":"PC2","z":"PC3","color":"Cluster"}
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)

        def create_cluster_summary(features_df, labels):
            features_df['Sorted_Cluster'] = labels
            summary = features_df.groupby('Sorted_Cluster').agg(
                num_trains=('Train_No','count'),
                avg_delay=('avg_delay','mean')
            ).reset_index()
            def label_delay(x):
                return "Low Delay" if x<=30 else "Medium Delay" if x<=90 else "High Delay"
            summary['Cluster_Label'] = summary['avg_delay'].apply(label_delay)
            return summary

        def main():
            st.title("Train Delay Clustering (3D PCA)")

            folder = "RailViz_Data_Scraping-main"
            df = load_all_data(folder)
            if df.empty:
                st.error("No data found. Check the folder path.")
                return

            st.sidebar.header("Filter Options")
            types = sorted(t for t in df['Train_Type'].unique() if t!="Master_Delay")
            sel_type = st.sidebar.selectbox("Select Train Type", types, key="type_select")
            times = ["Weekly","Monthly","3M","6M","Yearly"]
            sel_time = st.sidebar.selectbox("Select Time Period", times, key="time_select")

            filtered = df[(df['Train_Type']==sel_type)&(df['Time_Period']==sel_time)]
            if filtered.empty:
                st.warning("No data for chosen filters.")
                return

            feats = engineer_features(filtered)
            embedded = reduce_dimensions_pca(feats)
            k = unique_slider("Number of Clusters", 2, 10, 3)
            sorted_labels = perform_clustering_and_sort(feats, embedded, n_clusters=k)

            plot_clusters_3d(feats, embedded, sorted_labels)

            st.subheader("Cluster Summary")
            summary = create_cluster_summary(feats, sorted_labels)
            st.dataframe(summary)

            st.markdown("---")

        if __name__ == "__main__":
            main()

with main_tabs[6]:
    sub_pages = ["Train Types Comparision", "Region Wise Comparision", "Best Trains", "Worst Trains"]
    sub_tabs = st.tabs(sub_pages)

    with sub_tabs[0]:
        st.subheader("Train Types vs Delay Comparison")

        train_type_files = {
            "Duronto": "Duronto_Monthly_Delay.csv",
            "Express": "Express_Monthly_Delay.csv",
            "Garib Rath": "Garib_Rath_Monthly_Delay.csv",
            "Humsafar": "Humsafar_Monthly_Delay.csv",
            "Jan Shatabdi": "Jan_Shatabdi_Delay.csv",
            "Local": "Local_Monthly_Delay.csv",
            "Rajdhani": "Rajdhani_Monthly_Delay.csv",
            "Shatabdi": "Shatabdi_Monthly_Delay.csv",
            "Super Fast": "Super_Fast_Monthly_Delay.csv"
        }

        delay_data = []

        for train_type, filename in train_type_files.items():
            try:
                df = pd.read_csv(filename)
                df.columns = df.columns.str.strip()
                df["Train_No"] = df["Train_No"].astype(str).str.strip()
                df["Delay"] = pd.to_numeric(df["Delay"], errors="coerce")
                df_grouped = df.groupby("Train_No", as_index=False)["Delay"].mean()
                df_grouped["Train_Type"] = train_type
                delay_data.append(df_grouped)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")

        if delay_data:
            combined_df = pd.concat(delay_data, ignore_index=True)

            sns.set_theme(style="darkgrid", rc={
                'axes.facecolor': '#1e1e1e',
                'figure.facecolor': '#1e1e1e',
                'axes.labelcolor': 'grey',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'text.color': 'white',
                'axes.edgecolor': 'gray'
            })

            fig, ax = plt.subplots(figsize=(12, 6))

            sns.stripplot(
                data=combined_df,
                x="Train_Type",
                y="Delay",
                palette="Set2",
                size=3.5,
                jitter=True,
                ax=ax
            )

            ax.set_title("Train Delay Distribution by Train Type (Past 3 Months)", fontsize=16)
            ax.set_ylabel("Average Delay (Minutes)", fontsize=12)
            ax.set_xlabel("Train Type", fontsize=12)

            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("No valid delay data found in uploaded files.")

    with sub_tabs[1]:
        st.subheader("Zone vs Delay Comparison")

        period_option = st.selectbox("Select the Time Period:", ("Weekly", "Monthly", "3 Month", "6 Month", "Yearly"))

        period_file_mapping = {
            "Weekly": "Master_Weekly_Delay.csv",
            "Monthly": "Master_Monthly_Delay.csv",
            "3 Month": "Master_3M_Delay.csv",
            "6 Month": "Master_6M_Delay.csv",
            "Yearly": "Master_Yearly_Delay.csv"
        }

        selected_file = period_file_mapping.get(period_option)

        try:
            df = pd.read_csv(selected_file)
            df.columns = df.columns.str.strip()
            df["Train_No"] = df["Train_No"].astype(str).str.strip()
            df["Delay"] = pd.to_numeric(df["Delay"], errors="coerce")
            df["Zone"] = df["Zone"].astype(str).str.strip()
            df = df.dropna(subset=["Delay", "Zone"])

            df_grouped = df.groupby(["Train_No", "Zone"], as_index=False)["Delay"].mean()

            zone_list = sorted(df_grouped["Zone"].unique())
            zone_mapping = {zone: idx * 5 for idx, zone in enumerate(zone_list)}
            df_grouped["Zone_idx"] = df_grouped["Zone"].map(zone_mapping)
            df_grouped["Zone_idx_jitter"] = df_grouped["Zone_idx"] + np.random.uniform(-1.5, 1.5, size=len(df_grouped))

            chart = alt.Chart(df_grouped).mark_circle(size=70, opacity=0.7).encode(
                x=alt.X(
                    'Zone_idx_jitter:Q',
                    title='Region (Zone)',
                    axis=alt.Axis(
                        grid=False,
                        values=[zone_mapping[z] for z in zone_list],
                        labelExpr='{"' + '","'.join([str(zone_mapping[z])+":"+z for z in zone_list]) + '"}[datum.value]',
                        labelAngle=-45,
                        labelOverlap=False
                    )
                ),
                y=alt.Y(
                    'Delay:Q',
                    title='Average Delay (Minutes)',
                    axis=alt.Axis(
                        grid=True,
                        gridColor='lightgray',
                        gridOpacity=0.5,
                        tickColor='gray',
                        tickWidth=0.5
                    )
                ),
                color=alt.Color('Zone:N', legend=None),
                tooltip=['Train_No:N', 'Zone:N', 'Delay:Q']
            ).properties(
                width=900,
                height=550,
                title=f'Train Delay by Region ({period_option})'
            ).configure_axis(
                grid=True,
                gridColor='lightgray',
                gridDash=[2, 2],
                gridOpacity=0.4,
                tickColor='gray',
                domain=False
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load the file: {e}")

        image_path = "india.jpg"
        st.image(image_path, caption="Train Delay by Region")

    with sub_tabs[2]:
        st.subheader("Best Trains")

        @st.cache_data
        def load_all_data(main_folder_path):
            all_data = []
            for folder_name in os.listdir(main_folder_path):
                folder_path = os.path.join(main_folder_path, folder_name)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(folder_path, file))
                            df = df.dropna(subset=["Delay"])
                            df = df[df["Delay"] != 0]
                            if not df.empty:
                                df['Train_Type'] = folder_name.replace('List_of_All_', '').replace('_Trains', '')
                                df['Time_Period'] = file.replace('Category_', '').replace('_Delay.csv', '')
                                all_data.append(df)
            return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        def plot_top_trains(df, train_type, top_n, time_period_display):
            fig = px.bar(
                df,
                x="Train_Label",
                y="Average_Delay",
                color="Average_Delay",
                color_continuous_scale="Blues",
                title=f"Top {top_n} {train_type} Trains with Least Delay ({time_period_display})",
                labels={"Average_Delay": "Average Delay (Minutes)", "Train_Label": "Train"}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis_tickfont=dict(size=10)
            )
            st.plotly_chart(fig, use_container_width=True, key=train_type)

        def main():
            st.title("Top Trains with Least Delay")

            main_folder_path = "RailViz_Data_Scraping-main"
            df = load_all_data(main_folder_path)

            if df.empty:
                st.error("No data found. Please check the folder path.")
                return

            time_mapping = {
                "Weekly": "Weekly",
                "Monthly": "Monthly",
                "3M": "3 Months",
                "6M": "6 Months",
                "Yearly": "Yearly"
            }
            time_period_display = st.selectbox("Select Time Period", list(time_mapping.values()))
            reverse_mapping = {v: k for k, v in time_mapping.items()}
            selected_period = reverse_mapping[time_period_display]

            top_n = st.selectbox("Select Number of Top Trains to Show", [10, 20, 50, 100])

            df = df[df['Time_Period'] == selected_period]

            if df.empty:
                st.warning(f"No data available for {time_period_display}.")
                return

            df['Train_Label'] = df['Train_No'].astype(str) + " - " + df['Train_Name']
            grouped = df.groupby(['Train_Label', 'Train_Type']).agg(Average_Delay=('Delay', 'mean')).reset_index()

            for train_type in grouped['Train_Type'].unique():
                sub_df = grouped[grouped['Train_Type'] == train_type]
                sub_df = sub_df.sort_values(by='Average_Delay', ascending=True).head(top_n)
                if not sub_df.empty:
                    plot_top_trains(sub_df, train_type, top_n, time_period_display)

        if __name__ == "__main__":
            main()

    with sub_tabs[3]:
        st.subheader("Worst Trains")

        @st.cache_data
        def load_all_data(main_folder_path):
            all_data = []
            for folder_name in os.listdir(main_folder_path):
                folder_path = os.path.join(main_folder_path, folder_name)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(folder_path, file))
                            df = df.dropna(subset=["Delay"])
                            df = df[df["Delay"] != 0]
                            if not df.empty:
                                df['Train_Type'] = folder_name.replace('List_of_All_', '').replace('_Trains', '')
                                df['Time_Period'] = file.replace('Category_', '').replace('_Delay.csv', '')
                                all_data.append(df)
            return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        def plot_worst_trains(df, train_type, top_n, time_period_display):
            fig = px.bar(
                df,
                x="Train_Label",
                y="Average_Delay",
                color="Average_Delay",
                color_continuous_scale="Reds",
                title=f"Top {top_n} {train_type} Trains with Worst Delay ({time_period_display})",
                labels={"Average_Delay": "Avg Delay (min)", "Train_Label": "Train"}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis_tickfont=dict(size=10)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{train_type}")

        def main():
            st.title("Top Trains with Worst Delay")

            main_folder_path = "RailViz_Data_Scraping-main"
            df = load_all_data(main_folder_path)

            if df.empty:
                st.error("No data found. Check your folder path.")
                return

            time_mapping = {
                "Weekly": "Weekly",
                "Monthly": "Monthly",
                "3M": "3 Months",
                "6M": "6 Months",
                "Yearly": "Yearly"
            }
            time_display = st.selectbox("Select Time Period", list(time_mapping.values()), key="worst_time_period")
            reverse_map = {v: k for k, v in time_mapping.items()}
            selected_period = reverse_map[time_display]

            top_n = st.selectbox("Select Number of Top Trains", [10, 20, 50, 100], key="worst_top_n")

            df = df[df['Time_Period'] == selected_period]

            if df.empty:
                st.warning(f"No data for {time_display}.")
                return

            df['Train_Label'] = df['Train_No'].astype(str) + " – " + df['Train_Name']
            grouped = df.groupby(['Train_Label', 'Train_Type']).agg(Average_Delay=('Delay', 'mean')).reset_index()

            for ttype in grouped['Train_Type'].unique():
                sub = grouped[grouped['Train_Type'] == ttype]
                sub = sub.sort_values('Average_Delay', ascending=False).head(top_n)
                if not sub.empty:
                    plot_worst_trains(sub, ttype, top_n, time_display)

        if __name__ == "__main__":
            main()

with main_tabs[7]:

    st.title("Indian Railway Station Ratings")

    @st.cache_data
    def load_data():
        return pd.read_csv("all_city_station_ratings_reviews_no_first_col.csv")  # make sure this file is in your folder

    df = load_data()

    df = df.dropna(subset=["Rating"])

    st.sidebar.header("Filters")
    min_reviews = st.sidebar.slider("Minimum number of reviews", 0, int(df["Number of Reviews"].max()), 50)
    filtered_df = df[df["Number of Reviews"] >= min_reviews]

    st.subheader("Station Ratings (Bar Chart)")
    bar_df = filtered_df.sort_values(by="Rating", ascending=True)
    fig_bar = px.bar(
        bar_df,
        x="Rating",
        y="Station Name",
        orientation="h",
        color="Rating",
        text="Number of Reviews",
        color_continuous_scale="Plasma"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(height=600, xaxis_title="Rating", yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Rating Distribution Histogram")
    fig_hist = px.histogram(
        filtered_df,
        x="Rating",
        nbins=10,
        title="How are station ratings distributed?",
        color_discrete_sequence=["#636EFA"]
    )
    fig_hist.update_layout(height=400, xaxis_title="Rating", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Rating vs. Number of Reviews")
    fig_scatter = px.scatter(
        filtered_df,
        x="Number of Reviews",
        y="Rating",
        size="Number of Reviews",
        hover_name="Station Name",
        color="Rating",
        color_continuous_scale="Viridis",
        title="Are higher-rated stations also highly reviewed?"
    )
    fig_scatter.update_layout(height=500, xaxis_title="Number of Reviews", yaxis_title="Rating")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Station Ratings Table (Sorted by Rating)")
    st.dataframe(filtered_df.sort_values(by="Rating", ascending=False), use_container_width=True)





