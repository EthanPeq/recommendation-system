import streamlit as st
import numpy as np
import random
from utils import get_model, get_faiss_index, get_dataframe

df = get_dataframe()

if "liked_pins" not in st.session_state:
    st.session_state["liked_pins"] = []
if "disliked_pins" not in st.session_state:
    st.session_state["disliked_pins"] = []
if "random_index" not in st.session_state:
    st.session_state["random_index"] = random.randint(0, len(df) - 1)

model = get_model()
index = get_faiss_index()

# Show the current pin
pin = df.iloc[st.session_state["random_index"]]
col_title, col_button = st.columns([0.8, 0.2])
with col_title:
    st.title("📌 Smart Pin Feedback Recommender")

with col_button:
    if st.button("🗑 Clear History"):
        st.session_state["liked_pins"] = []
        st.session_state["disliked_pins"] = []
        st.success("Feedback history cleared!")
st.subheader("💡 Current Pin")

st.markdown(f"**Title:** {pin['title']}")
st.markdown(f"**Description:** {pin['description']}")
st.markdown(f"**Repin Count:** {int(pin['repin_count'])}")

# Button functionality like/dislike, and next pin
def like_pin():
    if pin['full_text'] not in st.session_state["liked_pins"]:
        st.session_state["liked_pins"].append(pin['full_text'])
    st.success("You liked this pin!")
    # Remove from disliked pins if this pin was previously disliked
    if st.session_state["disliked_pins"] and st.session_state["disliked_pins"][-1] == pin['full_text']:
        st.session_state["disliked_pins"].remove(pin['full_text'])
    st.rerun()

def dislike_pin():
    if pin['full_text'] not in st.session_state["disliked_pins"]:
        st.session_state["disliked_pins"].append(pin['full_text'])
        st.error("You disliked this pin!")
        # Remove from liked pins if this pin was previously liked
        if st.session_state["liked_pins"] and st.session_state["liked_pins"][-1] == pin['full_text']:
            st.session_state["liked_pins"].remove(pin['full_text'])
        st.rerun()

def show_new_pin():
    st.session_state["random_index"] = random.randint(0, len(df) - 1)
    st.rerun()

##
# Columns for like/dislike and next buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("👍 Like"):
        like_pin()
with col2:
    if st.button("👎 Dislike"):
        dislike_pin()
with col3:
    if st.button("➡️ Next"):
        show_new_pin()

# =========== Recommendations section ===========
# Combine liked and disliked pins for recommendations
if st.session_state["liked_pins"] or st.session_state["disliked_pins"]:
    st.subheader("🔍 Recommendations Based on Your Feedback")

    # Step 1: Encode liked and disliked texts separately
    liked_embeddings = model.encode(
        st.session_state["liked_pins"],
        normalize_embeddings=True
    ).astype(np.float32) if st.session_state["liked_pins"] else np.zeros((0, 384))

    disliked_embeddings = model.encode(
        st.session_state["disliked_pins"],
        normalize_embeddings=True
    ).astype(np.float32) if st.session_state["disliked_pins"] else np.zeros((0, 384))

    # Step 2: Apply weights
    # Likes = +1, Dislikes = -0.5 (or whatever you'd like)
    if liked_embeddings.shape[0] + disliked_embeddings.shape[0] == 0:
        st.warning("No feedback given yet.")
    else:
        if liked_embeddings.shape[0] > 0:
            avg_liked = np.mean(liked_embeddings, axis=0)
        else:
            avg_liked = np.zeros((liked_embeddings.shape[1],))

        if disliked_embeddings.shape[0] > 0:
            avg_disliked = np.mean(disliked_embeddings, axis=0)
        else:
            avg_disliked = np.zeros((disliked_embeddings.shape[1],))

        weighted_profile = (avg_liked - 0.5 * avg_disliked).reshape(1, -1)

        # Normalize weighted_profile for cosine similarity
        norm = np.linalg.norm(weighted_profile)
        if norm > 0:
            weighted_profile = weighted_profile / norm

        # Search FAISS index with the weighted profile
        distances, indices = index.search(weighted_profile.astype(np.float32), 5)
        results = df.iloc[indices[0]].copy()

        st.write(results[['title', 'description']])

# =========== Feedback summary section ===========
if st.session_state["liked_pins"] or st.session_state["disliked_pins"]:
    st.markdown("---")
    st.subheader("🧠 Your Feedback Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👍 Liked Pins")
        if st.session_state["liked_pins"]:
            for text in st.session_state["liked_pins"]:
                st.write(text)
        else:
            st.info("No liked pins yet.")

    with col2:
        st.markdown("### 👎 Disliked Pins")
        if st.session_state["disliked_pins"]:
            for text in st.session_state["disliked_pins"]:
                st.write(text)
        else:
            st.info("No disliked pins yet.")
