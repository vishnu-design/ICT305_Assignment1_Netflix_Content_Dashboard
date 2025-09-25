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

# Modern color schemes
COLOR_SCHEMES = {
    'Netflix Classic': {'primary': '#E50914', 'secondary': '#FF6B6B', 'accent': '#FFB3B3', 'gradient': ['#E50914', '#FF6B6B', '#FFB3B3']},
    'Dark Mode': {'primary': '#1E1E1E', 'secondary': '#333333', 'accent': '#666666', 'gradient': ['#1E1E1E', '#333333', '#666666']},
    'Ocean Blue': {'primary': '#0077BE', 'secondary': '#4A9EFF', 'accent': '#87CEEB', 'gradient': ['#0077BE', '#4A9EFF', '#87CEEB']},
    'Sunset': {'primary': '#FF6B35', 'secondary': '#F7931E', 'accent': '#FFD23F', 'gradient': ['#FF6B35', '#F7931E', '#FFD23F']},
    'Purple Dreams': {'primary': '#6A4C93', 'secondary': '#9B59B6', 'accent': '#C39BD3', 'gradient': ['#6A4C93', '#9B59B6', '#C39BD3']},
    'Forest Green': {'primary': '#228B22', 'secondary': '#32CD32', 'accent': '#90EE90', 'gradient': ['#228B22', '#32CD32', '#90EE90']},
    'Cyberpunk': {'primary': '#FF1493', 'secondary': '#00FFFF', 'accent': '#FFFF00', 'gradient': ['#FF1493', '#00FFFF', '#FFFF00']}
}

# Custom CSS for modern styling
def apply_custom_css(color_scheme, theme_mode):
    theme_colors = COLOR_SCHEMES[color_scheme]
    
    if theme_mode == 'Dark':
        bg_color = '#0E1117'
        text_color = '#FFFFFF'
        sidebar_bg = '#262730'
        card_bg = 'rgba(255, 255, 255, 0.05)'
    else:
        bg_color = '#FFFFFF'
        text_color = '#000000'
        sidebar_bg = '#F0F2F6'
        card_bg = 'rgba(255, 255, 255, 0.9)'
    
    st.markdown(f"""
    <style>
        /* Main styling */
        .stApp {{
            background: linear-gradient(135deg, {theme_colors['primary']}15, {theme_colors['secondary']}10);
        }}
        
        .main-header {{
            font-size: 3.5rem;
            background: linear-gradient(45deg, {theme_colors['primary']}, {theme_colors['secondary']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 900;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {theme_colors['primary']}, {theme_colors['secondary']});
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        }}
        
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .sidebar .sidebar-content {{
            background: {sidebar_bg};
            border-radius: 15px;
        }}
        
        .stSelectbox > div > div {{
            background: {card_bg};
            border-radius: 10px;
            border: 1px solid {theme_colors['primary']};
        }}
        
        .stMultiSelect > div > div {{
            background: {card_bg};
            border-radius: 10px;
            border: 1px solid {theme_colors['primary']};
        }}
        
        .stSlider > div > div > div {{
            color: {theme_colors['primary']};
        }}
        
        /* Custom buttons */
        .filter-button {{
            background: linear-gradient(45deg, {theme_colors['primary']}, {theme_colors['secondary']});
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            margin: 0.2rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .filter-button:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        /* Animated elements */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.6s ease-out;
        }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the Netflix dataset"""
    try:
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
    df['day_of_week'] = df['date_added'].dt.day_name()
    
    # Clean duration column
    df['duration_num'] = df['duration'].str.extract(r'(\d+)', expand=False).astype(float)
    df['duration_type'] = df['duration'].apply(lambda x: 'Minutes' if 'min' in str(x) else 'Seasons')
    
    # Create decade column
    df['decade'] = (df['release_year'] // 10) * 10
    
    # Clean genres and create genre categories
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    df['genre_count'] = df['listed_in'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    
    # Clean countries
    df['country'] = df['country'].fillna('Unknown')
    df['primary_country'] = df['country'].apply(lambda x: str(x).split(',')[0].strip())
    df['is_international'] = df['country'].apply(lambda x: ',' in str(x))
    
    # Add content age category
    df['content_age'] = 2024 - df['release_year']
    df['age_category'] = pd.cut(df['content_age'], 
                                bins=[0, 5, 10, 20, 50, 100], 
                                labels=['Recent (0-5y)', 'New (5-10y)', 'Mature (10-20y)', 'Classic (20-50y)', 'Vintage (50y+)'])
    
    return df

def create_genre_data(df):
    """Process genre data for analysis"""
    all_genres = []
    for genres in df['listed_in'].dropna():
        genre_list = [genre.strip() for genre in str(genres).split(',')]
        all_genres.extend(genre_list)
    return Counter(all_genres)

def create_enhanced_filters(df, color_scheme):
    """Create enhanced sidebar filters with modern styling"""
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Theme selection
    st.sidebar.markdown("### 🎨 Appearance")
    theme_mode = st.sidebar.radio("Theme Mode", ['Light', 'Dark'], index=1)
    selected_scheme = st.sidebar.selectbox("Color Scheme", list(COLOR_SCHEMES.keys()), index=0)
    
    st.sidebar.markdown("### 🎬 Content Filters")
    
    # Content type filter with visual indicators
    content_types = ['All'] + list(df['type'].unique())
    selected_type = st.sidebar.selectbox("🎭 Content Type", content_types)
    
    # Enhanced year filter with decade option
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_year = st.sidebar.number_input("From Year", min_value=int(df['release_year'].min()), 
                                             max_value=int(df['release_year'].max()), value=int(df['release_year'].min()))
    with col2:
        max_year = st.sidebar.number_input("To Year", min_value=int(df['release_year'].min()), 
                                             max_value=int(df['release_year'].max()), value=int(df['release_year'].max()))
    
    # Decade filter
    decades = sorted(df['decade'].dropna().unique())
    selected_decades = st.sidebar.multiselect("📅 Decades", options=decades, default=decades[-3:] if len(decades) >= 3 else decades)
    
    # Geographic filters
    st.sidebar.markdown("### 🌍 Geographic Filters")
    top_countries = df['primary_country'].value_counts().head(30).index.tolist()
    selected_countries = st.sidebar.multiselect("🏳️ Select Countries", options=top_countries, default=top_countries[:10])
    
    international_only = st.sidebar.checkbox("🌐 International Co-productions Only")
    
    # Content characteristics
    st.sidebar.markdown("### 📊 Content Characteristics")
    
    # Rating filter
    ratings = df['rating'].dropna().unique().tolist()
    selected_ratings = st.sidebar.multiselect("🔞 Content Ratings", options=ratings)
    
    # Duration filters
    if selected_type == 'Movie' or selected_type == 'All':
        duration_range = st.sidebar.slider("🎬 Movie Duration (minutes)", 
                                             min_value=int(df[df['type'] == 'Movie']['duration_num'].min()),
                                             max_value=int(df[df['type'] == 'Movie']['duration_num'].max()),
                                             value=(60, 180))
    
    if selected_type == 'TV Show' or selected_type == 'All':
        season_range = st.sidebar.slider("📺 TV Show Seasons", 
                                         min_value=int(df[df['type'] == 'TV Show']['duration_num'].min()),
                                         max_value=int(df[df['type'] == 'TV Show']['duration_num'].max()),
                                         value=(1, 5))
    
    # Genre filters
    all_genres = create_genre_data(df)
    top_genres = [genre for genre, count in all_genres.most_common(20)]
    selected_genres = st.sidebar.multiselect("🎭 Genres", options=top_genres)
    
    # Advanced filters
    st.sidebar.markdown("### 🔍 Advanced Filters")
    
    # Content age category
    age_categories = df['age_category'].dropna().unique().tolist()
    selected_age_cats = st.sidebar.multiselect("⏳ Content Age", options=age_categories)
    
    # Added date filters
    if df['date_added'].notna().any():
        added_year_range = st.sidebar.slider("📅 Year Added to Netflix", 
                                             min_value=int(df['year_added'].min()),
                                             max_value=int(df['year_added'].max()),
                                             value=(int(df['year_added'].min()), int(df['year_added'].max())))
    
    return {
        'theme_mode': theme_mode,
        'color_scheme': selected_scheme,
        'content_type': selected_type,
        'year_range': (min_year, max_year),
        'decades': selected_decades,
        'countries': selected_countries,
        'international_only': international_only,
        'ratings': selected_ratings,
        'duration_range': duration_range if selected_type == 'Movie' or selected_type == 'All' else None,
        'season_range': season_range if selected_type == 'TV Show' or selected_type == 'All' else None,
        'genres': selected_genres,
        'age_categories': selected_age_cats,
        'added_year_range': added_year_range if df['date_added'].notna().any() else None
    }

def apply_filters(df, filters):
    """Apply all selected filters to the dataframe"""
    filtered_df = df.copy()
    
    # Basic filters
    if filters['content_type'] != 'All':
        filtered_df = filtered_df[filtered_df['type'] == filters['content_type']]
    
    filtered_df = filtered_df[
        (filtered_df['release_year'] >= filters['year_range'][0]) &
        (filtered_df['release_year'] <= filters['year_range'][1])
    ]
    
    if filters['decades']:
        filtered_df = filtered_df[filtered_df['decade'].isin(filters['decades'])]
    
    if filters['countries']:
        filtered_df = filtered_df[filtered_df['primary_country'].isin(filters['countries'])]
    
    if filters['international_only']:
        filtered_df = filtered_df[filtered_df['is_international'] == True]
    
    if filters['ratings']:
        filtered_df = filtered_df[filtered_df['rating'].isin(filters['ratings'])]
    
    # Duration filters
    if filters['duration_range'] and filters['content_type'] in ['Movie', 'All']:
        movie_mask = (filtered_df['type'] == 'Movie') & \
                     (filtered_df['duration_num'] >= filters['duration_range'][0]) & \
                     (filtered_df['duration_num'] <= filters['duration_range'][1])
        tv_mask = filtered_df['type'] == 'TV Show'
        filtered_df = filtered_df[movie_mask | tv_mask]
    
    if filters['season_range'] and filters['content_type'] in ['TV Show', 'All']:
        tv_mask = (filtered_df['type'] == 'TV Show') & \
                  (filtered_df['duration_num'] >= filters['season_range'][0]) & \
                  (filtered_df['duration_num'] <= filters['season_range'][1])
        movie_mask = filtered_df['type'] == 'Movie'
        filtered_df = filtered_df[tv_mask | movie_mask]
    
    # Genre filter
    if filters['genres']:
        genre_mask = filtered_df['listed_in'].apply(
            lambda x: any(genre in str(x) for genre in filters['genres'])
        )
        filtered_df = filtered_df[genre_mask]
    
    # Age category filter
    if filters['age_categories']:
        filtered_df = filtered_df[filtered_df['age_category'].isin(filters['age_categories'])]
    
    # Added date filter
    if filters['added_year_range']:
        filtered_df = filtered_df[
            (filtered_df['year_added'] >= filters['added_year_range'][0]) &
            (filtered_df['year_added'] <= filters['added_year_range'][1])
        ]
    
    return filtered_df

def create_modern_visualizations(filtered_df, color_scheme):
    """Create modern visualizations with enhanced styling"""
    colors = COLOR_SCHEMES[color_scheme]
    
    # Enhanced metrics with animations
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        (len(filtered_df), "Total Content", "🎬"),
        (len(filtered_df[filtered_df['type'] == 'Movie']), "Movies", "🎥"),
        (len(filtered_df[filtered_df['type'] == 'TV Show']), "TV Shows", "📺"),
        (filtered_df['primary_country'].nunique(), "Countries", "🌍"),
        (len(create_genre_data(filtered_df)), "Unique Genres", "🎭")
    ]
    
    for i, (value, label, icon) in enumerate(metrics):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{icon}</h2>
                <h3>{value:,}</h3>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced visualizations
    st.header("📈 Content Analysis")
    
    # Row 1: Distribution and trends with enhanced colors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Content Type Distribution")
        type_counts = filtered_df['type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Movies vs TV Shows",
            color_discrete_sequence=colors['gradient']
        )
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(line=dict(color='white', width=2))
        )
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Content Added Over Time")
        yearly_data = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
        yearly_data = yearly_data.dropna()
        
        fig_line = px.area(
            yearly_data,
            x='year_added',
            y='count',
            color='type',
            title="Content Addition Trends",
            color_discrete_sequence=colors['gradient']
        )
        fig_line.update_layout(
            xaxis_title="Year", 
            yaxis_title="Number of Titles",
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Row 2: Geographic analysis with modern styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Global Content Distribution")
        country_counts = filtered_df['primary_country'].value_counts().head(15)
        fig_bar = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="Top Content Producing Countries",
            labels={'x': 'Number of Titles', 'y': 'Country'},
            color=country_counts.values,
            color_continuous_scale=px.colors.sequential.Reds
        )
        fig_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("⏳ Content by Age Category")
        age_counts = filtered_df['age_category'].value_counts()
        fig_age = px.bar(
            x=age_counts.index,
            y=age_counts.values,
            title="Content Age Distribution",
            labels={'x': 'Age Category', 'y': 'Count'},
            color=age_counts.values,
            color_continuous_scale=colors['gradient']
        )
        fig_age.update_layout(xaxis_tickangle=45, coloraxis_showscale=False)
        st.plotly_chart(fig_age, use_container_width=True)

    # --- New charts added ---
    st.header("🎨 Advanced Analytics")
    
    # Row 3: New heatmap and stacked bar chart
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🔥 Ratings by Country Heatmap")
        ratings_country_df = filtered_df.groupby(['primary_country', 'rating']).size().reset_index(name='count')
        top_countries_for_heatmap = filtered_df['primary_country'].value_counts().head(10).index.tolist()
        ratings_country_df = ratings_country_df[ratings_country_df['primary_country'].isin(top_countries_for_heatmap)]
    
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=ratings_country_df['count'],
            x=ratings_country_df['primary_country'],
            y=ratings_country_df['rating'],
            colorscale='YlOrRd'
        ))
    
        fig_heatmap.update_layout(
            title='Content Ratings by Country',
            xaxis_title='Country',
            yaxis_title='Content Rating',
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col4:
        st.subheader("⏳ Age Categories by Content Type")
        age_type_df = filtered_df.groupby(['age_category', 'type']).size().reset_index(name='count')
        
        fig_age_type_stacked = px.bar(
            age_type_df,
            x='age_category',
            y='count',
            color='type',
            title='Content Age Distribution by Type',
            labels={'age_category': 'Content Age Category', 'count': 'Number of Titles', 'type': 'Content Type'},
            barmode='stack',
            color_discrete_sequence=colors['gradient']
        )
        fig_age_type_stacked.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_age_type_stacked, use_container_width=True)

    # Multi-dimensional analysis (Existing Treemap & Stacked Bar Chart)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎭 Genre Popularity Matrix")
        genre_counter = create_genre_data(filtered_df)
        top_genres = dict(genre_counter.most_common(12))
        
        fig_treemap = px.treemap(
            names=list(top_genres.keys()),
            values=list(top_genres.values()),
            title="Genre Distribution (Treemap)",
            color=list(top_genres.values()),
            color_continuous_scale=colors['gradient']
        )
        fig_treemap.update_traces(textposition="middle center")
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    with col2:
        st.subheader("📊 Content Release Patterns")
        decade_type = filtered_df.groupby(['decade', 'type']).size().reset_index(name='count')
        
        fig_decade = px.bar(
            decade_type,
            x='decade',
            y='count',
            color='type',
            title="Content by Decade and Type",
            barmode='stack',
            color_discrete_sequence=colors['gradient']
        )
        st.plotly_chart(fig_decade, use_container_width=True)

    return colors

def main():
    # Initialize session state for theme
    if 'color_scheme' not in st.session_state:
        st.session_state.color_scheme = 'Netflix Classic'
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'Dark'
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Netflix Content Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # File upload with modern styling
    uploaded_file = st.file_uploader("📁 Upload Netflix Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data()
    
    if df is None:
        st.markdown("""
        <div class="glass-card">
            <h3>⚠️ Dataset Required</h3>
            <p>Please upload the Netflix dataset to continue.</p>
            <p><strong>Download from:</strong> <a href="https://www.kaggle.com/datasets/shivamb/netflix-shows" target="_blank">Kaggle Netflix Dataset</a></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Clean the data
    df = clean_data(df)
    
    # Create enhanced filters
    filters = create_enhanced_filters(df, st.session_state.color_scheme)
    
    # Update session state
    st.session_state.color_scheme = filters['color_scheme']
    st.session_state.theme_mode = filters['theme_mode']
    
    # Apply custom CSS with selected theme
    apply_custom_css(filters['color_scheme'], filters['theme_mode'])
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches your current filters. Please adjust your selection.")
        return
    
    # Create visualizations
    colors = create_modern_visualizations(filtered_df, filters['color_scheme'])
    
    # Performance metrics
    with st.expander("📈 Dashboard Performance & Data Quality"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filter Performance", f"{len(filtered_df):,} / {len(df):,}", 
                      f"{(len(filtered_df)/len(df)*100):.1f}% of data")
        
        with col2:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with col3:
            unique_ratio = df.nunique().sum() / len(df)
            st.metric("Data Diversity", f"{unique_ratio:.2f}", "Uniqueness ratio")
    
    # Raw data with modern styling
    if st.checkbox("📋 Show Filtered Dataset"):
        st.markdown("### 📊 Current Dataset View")
        st.dataframe(
            filtered_df.style.format({'release_year': '{:.0f}', 'duration_num': '{:.0f}'}),
            use_container_width=True
        )
        
        # Enhanced download options
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f'netflix_filtered_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name=f'netflix_filtered_{datetime.now().strftime("%Y%m%d")}.json',
                mime='application/json'
            )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Dashboard Error: {e}")
        st.info("💡 Try running with: streamlit run netflix_dashboard.py")