print("hi")
import streamlit as st
import datetime
from datetime import date
st.write("hi")
st.write("🎧 Transcribing audio...")
st.text("🚀 Starting pipeline...")

st.markdown("## 📚 Semantic Search Results")
st.markdown("- ✅ Step 1 complete\n- 🤖 Embedding text chunks")

d = st.date_input("select a date",datetime.date(2023,1,2))
st.write ("Date is:" ,d)
