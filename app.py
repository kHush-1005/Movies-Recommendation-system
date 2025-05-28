import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Movie Recommender 🎥", layout="wide")

# Sidebar: Theme toggle
with st.sidebar:
    st.title("⚙️ Settings")
    dark_mode = st.checkbox("🌙 Enable Dark Mode")

# Theme definitions
if dark_mode:
    main_bg = "#0E1117"
    main_text = "#FAFAFA"
    card_bg = "#1E1E1E"
    sidebar_bg = "#181C24"
    sidebar_text = "#E6EDF3"
else:
    main_bg = "#FFFFFF"
    main_text = "#000000"
    card_bg = "#F7F7F7"
    sidebar_bg = "#F0F2F6"
    sidebar_text = "#000000"

# Custom CSS to isolate sidebar and main styles
st.markdown(f"""
    <style>
        /* Main app background */
        .stApp {{
            background-color: {main_bg};
            color: {main_text};
        }}

        /* Override header, paragraphs, etc */
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: {main_text} !important;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            color: {sidebar_text} !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {{
            color: {sidebar_text} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# Load data
movies_data = pd.read_csv("movies.csv")

# Preprocessing
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data[selected_features].agg(' '.join, axis=1)

# Vectorization
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Header
st.markdown(f"<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>Find movies similar to your favorite film!</p>", unsafe_allow_html=True)

# User input
user_movie = st.text_input("📽️ Enter your favorite movie:")

if user_movie:
    with st.spinner("🔍 Searching for similar movies..."):
        movie_list = movies_data['title'].tolist()
        match = difflib.get_close_matches(user_movie, movie_list, n=4)

        if match:
            close_match = match[0]
            index = movies_data[movies_data.title == close_match].index[0]
            similarity_scores = list(enumerate(similarity[index]))
            sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

            st.success(f"🎯 Top Recommendations based on **{close_match}**:")

            for i, (movie_index, score) in enumerate(sorted_similar_movies):
                movie = movies_data.iloc[movie_index]
                title = movie['title']

                with st.container():
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.markdown(f"🎞️ {i+1}")
                        st.markdown("\n")
                        st.markdown("\n")

                    with col2:
                        st.markdown(f"**{title}**")
                        st.markdown(f"🎭 **Genres:** {movie['genres']}")
                        st.markdown(f"🎙️ **Cast:** {movie['cast']}")
                        st.markdown(f"🎬 **Director:** {movie['director']}")
                        st.markdown(f"🧮 Similarity Score: `{score:.2f}`")
                        st.markdown("\n")
                        st.markdown("\n")

        else:
            st.error("🚫 Sorry! No close match found. Try another movie.")

# Sidebar content continued
with st.sidebar:
    st.markdown("---")
    st.header("🎥 About")
    st.markdown("This is a **Content-Based Movie Recommendation System** built using:")
    st.markdown("- Streamlit 🖥️")
    st.markdown("- Scikit-learn 🧠")
    st.markdown("- TF-IDF & Cosine Similarity")
    st.markdown("📁 Data Source: `movies.csv`")
    st.markdown("👨‍💻 Made with ❤️ by Khushal Prajapati")
