import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Merged_Output.csv")  # Replace with your actual dataset

df = load_data()

# Sidebar for navigation
st.sidebar.title("Indian Railways Visualization")
menu = st.sidebar.radio("Select a Page", ["India Heat Map", "Train-wise Delay", "Station Cleanliness Ranking"])

# =================== Page 1: India Heat Map (Using Plotly Mapbox) ===================
if menu == "India Heat Map":
    st.title("📍 India Train Delay Heat Map")
    
    # Drop NaN values
    df_clean = df.dropna(subset=['Long', 'Lat', 'Delay'])
    
    # Create Mapbox Scatter Plot
    fig = px.scatter_mapbox(df_clean, 
                             lat="Lat", lon="Long", 
                             color="Delay", 
                             size="Delay", 
                             color_continuous_scale=["green","yellow", "orange", "red"],
                             mapbox_style="carto-positron",
                             zoom=4,
                             height=900,
                             title="Train Delay Across India")
    st.plotly_chart(fig)

# =================== Page 2: Train-wise Delay ===================
elif menu == "Train-wise Delay":
    st.title("📊 Train-wise Delay Analysis")
    
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
        # train_data = df[df["Train_No"].astype(str) == selected_train_no].sort_values(by="Station_Code")
        train_data = df[df["Train_No"].astype(str) == selected_train_no]

        fig = px.line(train_data, x="Station_Name", y="Delay", markers=True, title="Train Delay at Each Station")
        fig.update_xaxes(title="Station Name", tickangle=45)
        fig.update_yaxes(title="Delay (Minutes)")

        st.plotly_chart(fig)
    else:
        st.warning("No trains found. Please search again.")

# =================== Page 3: Station Cleanliness Ranking (Dummy Data) ===================
elif menu == "Station Cleanliness Ranking":
    st.title("🧼 Station Cleanliness Ranking")

    # Dummy cleanliness data
    dummy_cleanliness_data = {
        "Station_Name": ["New Delhi", "Mumbai Central", "Chennai Central", "Kolkata Howrah", "Bangalore City"],
        "Cleanliness_Score": [8.5, 7.8, 8.2, 7.5, 8.0]
    }
    cleanliness_df = pd.DataFrame(dummy_cleanliness_data)
    # cleanliness_df = cleanliness_df.sort_values(by="Cleanliness_Score", ascending=False)

    fig = px.bar(cleanliness_df, x="Station_Name", y="Cleanliness_Score", 
                 title="Station Cleanliness Ranking (Dummy Data)",
                 labels={"Cleanliness_Score": "Average Cleanliness Score"},
                 color="Cleanliness_Score")

    st.plotly_chart(fig)
