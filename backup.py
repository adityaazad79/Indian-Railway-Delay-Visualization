import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib


# Load train schedule CSV
@st.cache_data

def load_schedule():
    return pd.read_csv("train_detail.csv")  # Replace with your actual dataset

schedule_df = load_schedule()


# Sidebar for navigation
st.sidebar.title("Indian Railways Visualization")
menu = st.sidebar.radio("Select a Page", [
    "India Heat Map",
    "Train-wise Delay",
    "Station Cleanliness Ranking",
    "Train Schedule Gantt"
])

# =================== Page 1: India Heat Map ===================
if menu == "India Heat Map":
    st.title("📍 India Train Delay Heat Map")
    df = pd.read_csv("Merged_Output.csv")
    df_clean = df.dropna(subset=['Long', 'Lat', 'Delay'])

    fig = px.scatter_mapbox(df_clean,
                            lat="Lat", lon="Long",
                            color="Delay",
                            size="Delay",
                            color_continuous_scale=["yellow", "orange", "red"],
                            mapbox_style="carto-positron",
                            zoom=4,
                            height=700,
                            title="Train Delay Across India")
    st.plotly_chart(fig)

# =================== Page 2: Train-wise Delay ===================
elif menu == "Train-wise Delay":
    st.title("📊 Train-wise Delay Analysis")
    df = pd.read_csv("Merged_Output.csv")
    search_query = st.text_input("Search Train (Number or Name)", "")

    filtered_trains = df[df["Train_No"].astype(str).str.contains(search_query, case=False, na=False) |
                         df["Train_Name"].str.contains(search_query, case=False, na=False)]

    train_options = filtered_trains[["Train_No", "Train_Name"]].drop_duplicates()

    if not train_options.empty:
        train_selection = st.selectbox(
            "Select Train",
            train_options.apply(lambda x: f"{x.Train_No} - {x.Train_Name}", axis=1)
        )

        selected_train_no = train_selection.split(" - ")[0]
        train_data = df[df["Train_No"].astype(str) == selected_train_no].sort_values(by="Station_Code")

        fig = px.line(train_data, x="Station_Name", y="Delay", markers=True, title="Train Delay at Each Station")
        fig.update_xaxes(title="Station Name", tickangle=45)
        fig.update_yaxes(title="Delay (Minutes)")

        st.plotly_chart(fig)
    else:
        st.warning("No trains found. Please search again.")

# =================== Page 3: Station Cleanliness Ranking (Dummy Data) ===================
elif menu == "Station Cleanliness Ranking":
    st.title("🧼 Station Cleanliness Ranking")

    dummy_cleanliness_data = {
        "Station_Name": ["New Delhi", "Mumbai Central", "Chennai Central", "Kolkata Howrah", "Bangalore City"],
        "Cleanliness_Score": [8.5, 7.8, 8.2, 7.5, 8.0]
    }
    cleanliness_df = pd.DataFrame(dummy_cleanliness_data)
    cleanliness_df = cleanliness_df.sort_values(by="Cleanliness_Score", ascending=False)

    fig = px.bar(cleanliness_df, x="Station_Name", y="Cleanliness_Score",
                 title="Station Cleanliness Ranking (Dummy Data)",
                 labels={"Cleanliness_Score": "Average Cleanliness Score"},
                 color="Cleanliness_Score")

    st.plotly_chart(fig)

# =================== Page 4: Train Schedule Gantt Chart ================elif menu == "Train Schedule Gantt":
elif menu == "Train Schedule Gantt":
    with st.container():
        st.markdown("<h1 style='text-align: center;'>\u23f1\ufe0f Train Schedule Timeline Between Stations</h1>", unsafe_allow_html=True)

        station_list = sorted(schedule_df["Station Name"].dropna().unique())
        print(station_list)

        col1, col2 = st.columns([1, 1])
        source_default = station_list.index("KANPUR CENTRAL ") if "KANPUR CENTRAL "  in station_list else 0
        print(f"Source Default : {source_default}")
        destination_default = station_list.index("NEW DELHI      ") if "NEW DELHI      " in station_list else 1
        print(f"Source Default : {source_default}  | destination Default : {destination_default}")
        with col1:
            source_station = st.selectbox("Select Source Station", station_list, index=source_default)
        with col2:
            destination_station = st.selectbox("Select Destination Station", station_list, index=destination_default)

        st.markdown("---")

        if source_station and destination_station and source_station != destination_station:
            trains_with_stations = schedule_df.groupby("Train No.").filter(
                lambda x: source_station in x["Station Name"].values and destination_station in x["Station Name"].values)

            delay_df = pd.read_csv("Master_Monthly_Delay.csv")
            delay_df.columns = delay_df.columns.str.strip()
            delay_df["Train_No"] = delay_df["Train_No"].astype(str).str.strip()
            delay_df["Station_Name"] = delay_df["Station_Name"].str.upper().str.strip()

            train_segments = []

            for train_no, group in trains_with_stations.groupby("Train No."):
                group_sorted = group.sort_values("islno")
                try:
                    src_row = group_sorted[group_sorted["Station Name"] == source_station].iloc[0]
                    dst_row = group_sorted[group_sorted["Station Name"] == destination_station].iloc[0]

                    if src_row["islno"] < dst_row["islno"]:
                        start_time = pd.to_datetime("2023-01-01 " + str(src_row["Departure time"]), errors='coerce')
                        end_time = pd.to_datetime("2023-01-01 " + str(dst_row["Arrival time"]), errors='coerce')

                        if pd.notna(start_time) and pd.notna(end_time) and end_time <= start_time:
                            end_time += pd.Timedelta(days=1)

                        if pd.notna(start_time) and pd.notna(end_time):
                            train_name = src_row["train Name"]

                            train_no_clean = str(train_no).strip().replace("'", "")
                            src_station = source_station.upper().strip()
                            dst_station = destination_station.upper().strip()

                            delay_src = delay_df[(delay_df["Train_No"] == train_no_clean) & (delay_df["Station_Name"] == src_station)]["Delay"].mean()
                            delay_dst = delay_df[(delay_df["Train_No"] == train_no_clean) & (delay_df["Station_Name"] == dst_station)]["Delay"].mean()

                            if pd.isna(delay_src) or pd.isna(delay_dst):
                                continue

                            train_segments.append({
                                "Train_Name": train_name,
                                "Train_No": train_no,
                                "Start_Time": start_time,
                                "End_Time": end_time,
                                "Delay_Source": round(delay_src, 1),
                                "Delay_Destination": round(delay_dst, 1)
                            })

                except:
                    continue

            if train_segments:
                chart_df = pd.DataFrame(train_segments)
                chart_df["Label"] = chart_df["Train_No"].astype(str) + " - " + chart_df["Train_Name"]
                chart_df["Duration (min)"] = (chart_df["End_Time"] - chart_df["Start_Time"]).dt.total_seconds() / 60
                chart_df = chart_df.sort_values(by="Start_Time")


                # Normalize durations for color mapping
                

                chart_df["Color"] = "#faff91"

                fig = px.timeline(
                    chart_df,
                    x_start="Start_Time",
                    x_end="End_Time",
                    y="Label",
                    color="Color",
                    color_discrete_map="identity",  # Use custom color hex values directly
                    hover_data=[
                        "Train_No", "Train_Name", "Start_Time", "End_Time", "Duration (min)",
                        "Delay_Source", "Delay_Destination"
                    ],
                )

                # 🔴 Add pink overlay bar for delay
                # Map labels to y-axis positions
                label_to_y = {label: i for i, label in enumerate(chart_df["Label"])}

                # 🔴 Add pink overlay bar for delay
                for i, row in chart_df.iterrows():
                    if row["Delay_Source"] != "N/A" and row["Delay_Destination"] != "N/A":
                        delay_start = row["Start_Time"] + pd.Timedelta(minutes=row["Delay_Source"])
                        delay_end = row["End_Time"] + pd.Timedelta(minutes=row["Delay_Destination"])
                        y_pos = label_to_y[row["Label"]]

                        fig.add_shape(
                            type="rect",
                            x0=delay_start,
                            x1=delay_end,
                            y0=y_pos - 0.35,
                            y1=y_pos + 0.35,
                            xref="x",
                            yref="y",
                            fillcolor="red",
                            opacity=0.3,
                            layer="above",
                            line_width=0,
                        )

                fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), categoryorder="array", categoryarray=chart_df["Label"])
                fig.update_traces(width=0.5)

                chart_height = max(300, len(chart_df) * 35)

                fig.update_layout(
                    height=chart_height,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=70, b=20),
                    title={
                        "text": f"<b>Train Schedules from {source_station.upper()} to {destination_station.upper()}</b>",
                        "x": 0.5,
                        "xanchor": "center"
                    },
                    xaxis_title="",
                    xaxis=dict(
                    tickformat="%H:%M",
                    dtick=14400000,  # 4 hours in milliseconds
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=1,
                    ticks="outside"
                )
                )

                # 🟨 Custom Legend (aligned horizontally on top right)
                legend_y_top = 1.05
                box_height = 0.03
                box_width = 0.025
                x_start = 0.85

                # Scheduled Duration (Yellow)
                fig.add_shape(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=x_start, x1=x_start + box_width,
                    y0=legend_y_top, y1=legend_y_top + box_height,
                    fillcolor="#FFFF00",  # Bright yellow
                    line=dict(width=1, color="white"),
                    layer="above"
                )
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=x_start + box_width + 0.01,
                    y=legend_y_top + box_height / 2,
                    text="Scheduled Duration",
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    align="left",
                    valign="middle"
                )

                # Average Delay (Red with Opacity)
                fig.add_shape(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=x_start, x1=x_start + box_width,
                    y0=legend_y_top - 0.05, y1=legend_y_top - 0.05 + box_height,
                    fillcolor="red",
                    opacity=0.3,
                    line=dict(width=1, color="white"),
                    layer="above"
                )
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=x_start + box_width + 0.01,
                    y=legend_y_top - 0.05 + box_height / 2,
                    text="Average Delay",
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    align="left",
                    valign="middle"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No trains available between the selected stations in that direction.")
        else:
            st.info("Please select two distinct stations to view schedules.")
