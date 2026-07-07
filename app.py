import streamlit as st
from predict import translate

st.set_page_config(
    page_title="English → Telugu Translator",
    page_icon="🌍",
    layout="wide"
)

# ---------------------- CSS ----------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#1E3C72,#2A5298);
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title{
font-size:48px;
font-weight:bold;
color:white;
text-align:center;
}

.subtitle{
font-size:20px;
color:#dddddd;
text-align:center;
margin-bottom:30px;
}

.box{
background:rgba(255,255,255,0.12);
padding:30px;
border-radius:20px;
backdrop-filter:blur(15px);
box-shadow:0px 10px 30px rgba(0,0,0,0.3);
}

.result{
background:#ffffff;
padding:20px;
border-radius:15px;
font-size:28px;
font-weight:bold;
text-align:center;
color:#1565C0;
margin-top:20px;
}

div.stButton > button{
width:100%;
height:55px;
font-size:22px;
font-weight:bold;
border-radius:12px;
background:#00C853;
color:white;
border:none;
}

div.stButton > button:hover{
background:#00E676;
color:black;
}

textarea{
font-size:20px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------

st.sidebar.title("📌 Project Information")

st.sidebar.success("Many-to-Many RNN")

st.sidebar.write("### Model")
st.sidebar.write("Encoder-Decoder LSTM")

st.sidebar.write("### Input")
st.sidebar.write("English Sentence")

st.sidebar.write("### Output")
st.sidebar.write("Telugu Sentence")

st.sidebar.write("### Developed Using")
st.sidebar.write("- TensorFlow")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Python")

# ---------------- Main ----------------

st.markdown('<p class="title">🌍 English → Telugu Translator</p>',
unsafe_allow_html=True)

st.markdown('<p class="subtitle">Many-to-Many RNN </p>',
unsafe_allow_html=True)

left,right=st.columns([2,1])

with left:

    st.markdown('<div class="box">',unsafe_allow_html=True)

    text=st.text_area(
        "Enter English Sentence",
        height=180,
        placeholder="Example: How are you?"
    )

    if st.button("🚀 Translate"):

        if text.strip()=="":

            st.warning("Please enter a sentence.")

        else:

            result=translate(text)

            st.markdown(
                f'<div class="result">🇮🇳 {result}</div>',
                unsafe_allow_html=True
            )

    st.markdown("</div>",unsafe_allow_html=True)

with right:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3898/3898082.png",
        width=280
    )

    st.info("""
### Features

✅ Many-to-Many RNN

✅ Encoder-Decoder

✅ English → Telugu

✅ Streamlit Deployment

✅ TensorFlow LSTM
""")

st.markdown("---")

st.markdown(
"<center><h4 style='color:white'>Developed using ❤️ TensorFlow & Streamlit</h4></center>",
unsafe_allow_html=True
)