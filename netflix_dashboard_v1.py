#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 16:07:22 2025

@author: prabhanvishnu
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Netflix Content Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #E50914, #FF6B6B);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the Netflix dataset"""
    try:
        # Try to load from uploaded file first, then from URL
        data = pd.read_csv("netflix_titles.csv")
        return data
    except:
        st.error("Please upload the netflix_titles.csv file or ensure it's in the same directory as this script.")
        return None

def clean_data(df):
    """Clean and preprocess the data"""
    if df is None:
        return None
    
    # Convert date_added to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    # Clean duration column
    df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)
    
    # Create duration type
    df['duration_type'] = df['duration'].apply(lambda x: 'Minutes' if 'min' in str(x) else 'Seasons')
    
    # Clean genres
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    
    # Clean countries
    df['country'] = df['country'].fillna('Unknown')
    df['primary_country'] = df['country'].apply(lambda x: str(x).split(',')[0].strip())
    
    return df

def create_genre_data(df):
    """Process genre data for analysis"""
    all_genres = []
    for genres in df['listed_in'].dropna():
        genre_list = [genre.strip() for genre in str(genres).split(',')]
        all_genres.extend(genre_list)
    return Counter(all_genres)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Netflix Content Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Netflix Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data()
    
    if df is None:
        st.warning("Please upload the Netflix dataset to continue.")
        st.info("Download the dataset from: https://www.kaggle.com/datasets/shivamb/netflix-shows")
        return
    
    # Clean the data
    df = clean_data(df)
    
    # Sidebar filters
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Content type filter
    content_types = ['All'] + list(df['type'].unique())
    selected_type = st.sidebar.selectbox("Content Type", content_types)
    
    # Year filter
    year_range = st.sidebar.slider(
        "Release Year Range",
        min_value=int(df['release_year'].min()),
        max_value=int(df['release_year'].max()),
        value=(int(df['release_year'].min()), int(df['release_year'].max()))
    )
    
    # Country filter
    top_countries = df['primary_country'].value_counts().head(20).index.tolist()
    selected_countries = st.sidebar.multiselect(
        "Select Countries (Top 20)",
        options=top_countries,
        default=top_countries[:5]
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    
    filtered_df = filtered_df[
        (filtered_df['release_year'] >= year_range[0]) &
        (filtered_df['release_year'] <= year_range[1])
    ]
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['primary_country'].isin(selected_countries)]
    
    # Key Metrics
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(filtered_df):,}</h3>
            <p>Total Content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{movies_count:,}</h3>
            <p>Movies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        tv_count = len(filtered_df[filtered_df['type'] == 'TV Show'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{tv_count:,}</h3>
            <p>TV Shows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        countries_count = filtered_df['primary_country'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{countries_count}</h3>
            <p>Countries</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main visualizations
    st.header("📈 Content Analysis")
    
    # Row 1: Content distribution and trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Type Distribution")
        type_counts = filtered_df['type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Movies vs TV Shows",
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#FF6B6B'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Content Added Over Time")
        yearly_data = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
        yearly_data = yearly_data.dropna()
        
        fig_line = px.line(
            yearly_data,
            x='year_added',
            y='count',
            color='type',
            title="Content Addition Trends",
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#FF6B6B'}
        )
        fig_line.update_layout(xaxis_title="Year", yaxis_title="Number of Titles")
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Row 2: Geographic and rating analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Content Producing Countries")
        country_counts = filtered_df['primary_country'].value_counts().head(15)
        fig_bar = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="Content by Country",
            labels={'x': 'Number of Titles', 'y': 'Country'},
            color=country_counts.values,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Content Ratings Distribution")
        rating_counts = filtered_df['rating'].value_counts().head(10)
        fig_rating = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Most Common Ratings",
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.values,
            color_continuous_scale='Reds'
        )
        fig_rating.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    # Row 3: Genre analysis
    st.subheader("🎭 Genre Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Genres")
        genre_counter = create_genre_data(filtered_df)
        top_genres = dict(genre_counter.most_common(15))
        
        fig_genres = px.bar(
            x=list(top_genres.values()),
            y=list(top_genres.keys()),
            orientation='h',
            title="Most Popular Genres",
            labels={'x': 'Number of Titles', 'y': 'Genre'},
            color=list(top_genres.values()),
            color_continuous_scale='Reds'
        )
        fig_genres.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_genres, use_container_width=True)
    
    with col2:
        st.subheader("Genre Distribution by Content Type")
        # Create genre-type matrix
        genre_type_data = []
        for _, row in filtered_df.iterrows():
            if pd.notna(row['listed_in']):
                genres = [g.strip() for g in str(row['listed_in']).split(',')]
                for genre in genres:
                    genre_type_data.append({'genre': genre, 'type': row['type']})
        
        genre_type_df = pd.DataFrame(genre_type_data)
        genre_type_counts = genre_type_df.groupby(['genre', 'type']).size().reset_index(name='count')
        top_genres_list = list(dict(genre_counter.most_common(10)).keys())
        genre_type_filtered = genre_type_counts[genre_type_counts['genre'].isin(top_genres_list)]
        
        fig_stacked = px.bar(
            genre_type_filtered,
            x='genre',
            y='count',
            color='type',
            title="Top Genres by Content Type",
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#FF6B6B'}
        )
        fig_stacked.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_stacked, use_container_width=True)
    
    # Row 4: Duration analysis
    st.subheader("⏱️ Duration Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Movie duration distribution
        movies_df = filtered_df[filtered_df['type'] == 'Movie']
        if len(movies_df) > 0:
            fig_duration = px.histogram(
                movies_df,
                x='duration_num',
                title="Movie Duration Distribution",
                labels={'duration_num': 'Duration (Minutes)', 'count': 'Frequency'},
                nbins=30,
                color_discrete_sequence=['#E50914']
            )
            st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # TV Show seasons distribution
        tv_shows_df = filtered_df[filtered_df['type'] == 'TV Show']
        if len(tv_shows_df) > 0:
            fig_seasons = px.histogram(
                tv_shows_df,
                x='duration_num',
                title="TV Show Seasons Distribution",
                labels={'duration_num': 'Number of Seasons', 'count': 'Frequency'},
                nbins=20,
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig_seasons, use_container_width=True)
    
    # Advanced Analysis Section
    st.header("🔍 Advanced Analysis")
    
    # Heatmap of content addition by month and year
    st.subheader("Content Addition Heatmap")
    heatmap_data = filtered_df.groupby(['year_added', 'month_added']).size().reset_index(name='count')
    heatmap_data = heatmap_data.dropna()
    
    if len(heatmap_data) > 0:
        heatmap_pivot = heatmap_data.pivot(index='year_added', columns='month_added', values='count').fillna(0)
        
        fig_heatmap = px.imshow(
            heatmap_pivot,
            title="Content Addition Patterns (Month vs Year)",
            labels=dict(x="Month", y="Year", color="Number of Titles"),
            aspect="auto",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Data Quality Insights
    st.header("📋 Data Quality Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Data Analysis")
        missing_data = filtered_df.isnull().sum().sort_values(ascending=False)
        missing_percentage = (missing_data / len(filtered_df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percentage.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            fig_missing = px.bar(
                missing_df,
                x='Missing %',
                y='Column',
                orientation='h',
                title="Missing Data by Column",
                color='Missing %',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_missing, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Summary Statistics")
        summary_stats = {
            'Total Records': len(filtered_df),
            'Date Range': f"{filtered_df['release_year'].min()} - {filtered_df['release_year'].max()}",
            'Unique Directors': filtered_df['director'].nunique(),
            'Unique Genres': len(create_genre_data(filtered_df)),
            'Average Movie Duration': f"{filtered_df[filtered_df['type'] == 'Movie']['duration_num'].mean():.1f} min",
            'Average TV Seasons': f"{filtered_df[filtered_df['type'] == 'TV Show']['duration_num'].mean():.1f}"
        }
        
        for key, value in summary_stats.items():
            st.metric(key, value)
    
    # Raw data view
    if st.checkbox("Show Raw Data"):
        st.subheader("📝 Raw Dataset")
        st.dataframe(filtered_df)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='netflix_filtered_data.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()