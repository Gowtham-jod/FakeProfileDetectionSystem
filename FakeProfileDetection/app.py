import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("Fake Profile Detection System")

st.write("Enter profile details below:")

# Inputs
followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)
posts = st.number_input("Posts", min_value=0)
bio = st.number_input("Bio Length", min_value=0)

profile_pic = st.selectbox("Profile Picture", [0, 1])
private = st.selectbox("Private Account", [0, 1])

# Prediction
if st.button("Predict"):
    features = np.array([[followers, following, posts, bio, profile_pic, private]])
    result = model.predict(features)

    if result[0] == 1:
        st.error("🚨 Fake Profile Detected")
    else:
        st.success("✅ Genuine Profile")