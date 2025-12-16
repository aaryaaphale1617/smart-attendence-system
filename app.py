import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import pandas as pd
import os
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

from face_module import start_recognition_session, train_recognizer, get_names_list

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Smart Attendance System", layout="wide")
st.title("📸 Smart Attendance System")

# ================== INIT DATA FILES ==================
def ensure_data_files():
    os.makedirs("data", exist_ok=True)
    os.makedirs("trainingimages", exist_ok=True)

    if not os.path.exists("data/students.csv"):
        pd.DataFrame(columns=["ID", "Name", "RollNo", "Department"]).to_csv("data/students.csv", index=False)

    if not os.path.exists("data/staff.csv"):
        pd.DataFrame(columns=["ID", "Name", "Designation", "Department"]).to_csv("data/staff.csv", index=False)

    if not os.path.exists("data/attendance.csv"):
        pd.DataFrame(columns=["Name", "Role", "Date", "Time"]).to_csv("data/attendance.csv", index=False)


def save_uploaded_image(uploaded_file, person_name):
    folder = os.path.join("trainingimages", person_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def dataframe_to_pdf_bytes(df, title="Attendance Report"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, title)

    c.setFont("Helvetica", 10)
    y = height - 70

    for i, col in enumerate(df.columns):
        c.drawString(40 + i * 120, y, str(col))

    y -= 20
    for _, row in df.iterrows():
        for i, item in enumerate(row):
            c.drawString(40 + i * 120, y, str(item))
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 40

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


ensure_data_files()

# ================== AUTHENTICATION ==================
with open("credentials.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    cookie_key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"],
    preauthorized=config.get("preauthorized"),
)

# ---- LOGIN FORM ----
authenticator.login(location="main")

auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")

if auth_status is None:
    st.warning("🔐 Please enter your username and password")
    st.stop()

if auth_status is False:
    st.error("❌ Username or password incorrect")
    st.stop()

# ================== LOGGED IN AREA ==================
st.success(f"Welcome {name} 👋")
authenticator.logout("Logout", "sidebar")

menu = st.sidebar.selectbox(
    "📌 Menu",
    ["Live Attendance", "Manage Students", "Manage Staff", "Attendance", "Reports", "Settings"],
)

# ================== LIVE ATTENDANCE ==================
if menu == "Live Attendance":
    st.header("🎥 Live Attendance")
    st.info("Camera will run for a few seconds. Press Q to stop.")

    if st.button("📸 Start Face Recognition"):
        frame, detected = start_recognition_session()
        if frame is not None:
            st.image(frame, channels="BGR")
        if detected:
            st.success(f"Attendance marked for: {', '.join(detected)}")

# ================== MANAGE STUDENTS ==================
elif menu == "Manage Students":
    st.header("👨‍🎓 Manage Students")
    df = pd.read_csv("data/students.csv")
    st.dataframe(df, use_container_width=True)

    st.subheader("Add New Student")
    with st.form("add_student"):
        sid = st.number_input("ID", min_value=1, step=1)
        sname = st.text_input("Full Name")
        sroll = st.text_input("Roll No")
        sdept = st.text_input("Department")
        spic = st.file_uploader("Upload Face Image", type=["jpg", "png"])
        submit = st.form_submit_button("Add Student")

        if submit and sname:
            new = pd.DataFrame([[sid, sname, sroll, sdept]], columns=df.columns)
            df = pd.concat([df, new], ignore_index=True)
            df.to_csv("data/students.csv", index=False)
            if spic:
                save_uploaded_image(spic, sname)
            st.success("Student added successfully")

# ================== MANAGE STAFF ==================
elif menu == "Manage Staff":
    st.header("👩‍🏫 Manage Staff")
    df = pd.read_csv("data/staff.csv")
    st.dataframe(df, use_container_width=True)

    st.subheader("Add New Staff")
    with st.form("add_staff"):
        sid = st.number_input("ID", min_value=1, step=1)
        sname = st.text_input("Full Name")
        sdes = st.text_input("Designation")
        sdept = st.text_input("Department")
        spic = st.file_uploader("Upload Face Image", type=["jpg", "png"])
        submit = st.form_submit_button("Add Staff")

        if submit and sname:
            new = pd.DataFrame([[sid, sname, sdes, sdept]], columns=df.columns)
            df = pd.concat([df, new], ignore_index=True)
            df.to_csv("data/staff.csv", index=False)
            if spic:
                save_uploaded_image(spic, sname)
            st.success("Staff added successfully")

# ================== ATTENDANCE ==================
elif menu == "Attendance":
    st.header("📝 Attendance Records")
    att = pd.read_csv("data/attendance.csv")
    st.dataframe(att, use_container_width=True)

    if not att.empty:
        if st.button("⬇️ Download CSV"):
            st.download_button("Download", att.to_csv(index=False), "attendance.csv", "text/csv")

        if st.button("⬇️ Download PDF"):
            pdf = dataframe_to_pdf_bytes(att)
            st.download_button("Download", pdf, "attendance.pdf", "application/pdf")

# ================== REPORTS ==================
elif menu == "Reports":
    st.header("📊 Attendance Reports")
    att = pd.read_csv("data/attendance.csv")

    if not att.empty:
        summary = att.groupby("Name").size().reset_index(name="Count")
        st.dataframe(summary)

        fig, ax = plt.subplots()
        summary.set_index("Name").plot(kind="bar", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No attendance data available")

# ================== SETTINGS ==================
elif menu == "Settings":
    st.header("⚙️ Settings")

    if st.button("🔄 Retrain Face Model"):
        with st.spinner("Training model..."):
            ok, msg = train_recognizer()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.subheader("Known Persons")
    st.write(get_names_list())
