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

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Smart Attendance System", layout="wide")

# ------------------ HELPER FUNCTIONS ------------------
def ensure_data_files():
    os.makedirs("data", exist_ok=True)
    os.makedirs("TrainingImages", exist_ok=True)

    if not os.path.exists("data/students.csv"):
        pd.DataFrame(columns=["ID", "Name", "RollNo", "Department"]).to_csv(
            "data/students.csv", index=False
        )

    if not os.path.exists("data/staff.csv"):
        pd.DataFrame(columns=["ID", "Name", "Designation", "Department"]).to_csv(
            "data/staff.csv", index=False
        )

    if not os.path.exists("data/attendance.csv"):
        pd.DataFrame(columns=["Name", "Role", "Date", "Time"]).to_csv(
            "data/attendance.csv", index=False
        )


def save_uploaded_image(uploaded_file, person_name):
    folder = os.path.join("TrainingImages", person_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


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

# ------------------ AUTHENTICATION ------------------
with open("credentials.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)
# ✅ SAFE LOGIN - Handles None return
name, authentication_status, username = login_result

authenticator.login(location="main")

authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if authentication_status is False:
    st.error("❌ Username/password incorrect")
    st.stop()

elif authentication_status is None:
    st.warning("🔐 Please enter username and password")
    st.stop()

elif authentication_status:
    st.success(f"Welcome {name} 👋")
    authenticator.logout("Logout", "sidebar")

    # ------------------ MANAGE STUDENTS ------------------
    if menu == "Manage Students":
        st.title("👨‍🎓 Manage Students")
        df = pd.read_csv("data/students.csv")
        st.dataframe(df)

        st.subheader("Add New Student")
        with st.form("add_student"):
            sid = st.number_input("ID", min_value=1, step=1)
            sname = st.text_input("Full Name")
            sroll = st.text_input("Roll No")
            sdept = st.text_input("Department")
            spic = st.file_uploader("Upload Face Image", type=["jpg", "png"])
            submit = st.form_submit_button("Add Student")

            if submit:
                new = pd.DataFrame(
                    [[sid, sname, sroll, sdept]],
                    columns=["ID", "Name", "RollNo", "Department"],
                )
                df = pd.concat([df, new], ignore_index=True)
                df.to_csv("data/students.csv", index=False)
                if spic:
                    save_uploaded_image(spic, sname)
                st.success("Student added successfully")

    # ------------------ MANAGE STAFF ------------------
    if menu == "Manage Staff":
        st.title("👩‍🏫 Manage Staff")
        df = pd.read_csv("data/staff.csv")
        st.dataframe(df)

        st.subheader("Add New Staff")
        with st.form("add_staff"):
            sid = st.number_input("ID", min_value=1, step=1)
            sname = st.text_input("Full Name")
            sdes = st.text_input("Designation")
            sdept = st.text_input("Department")
            spic = st.file_uploader("Upload Face Image", type=["jpg", "png"])
            submit = st.form_submit_button("Add Staff")

            if submit:
                new = pd.DataFrame(
                    [[sid, sname, sdes, sdept]],
                    columns=["ID", "Name", "Designation", "Department"],
                )
                df = pd.concat([df, new], ignore_index=True)
                df.to_csv("data/staff.csv", index=False)
                if spic:
                    save_uploaded_image(spic, sname)
                st.success("Staff added successfully")

    # ------------------ ATTENDANCE ------------------
    if menu == "Attendance":
        st.title("📝 Attendance Records")
        att = pd.read_csv("data/attendance.csv")
        st.dataframe(att)

        if not att.empty:
            if st.button("Export CSV"):
                st.download_button(
                    "Download CSV",
                    att.to_csv(index=False),
                    "attendance.csv",
                    "text/csv",
                )

            if st.button("Export PDF"):
                pdf = dataframe_to_pdf_bytes(att)
                st.download_button(
                    "Download PDF",
                    pdf,
                    "attendance.pdf",
                    "application/pdf",
                )

    # ------------------ REPORTS ------------------
    if menu == "Reports":
        st.title("📊 Attendance Reports")
        att = pd.read_csv("data/attendance.csv")

        if not att.empty:
            summary = att.groupby("Name").size().reset_index(name="Count")
            st.dataframe(summary)

            fig, ax = plt.subplots()
            summary.set_index("Name").plot(kind="bar", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No data available")

    # ------------------ SETTINGS ------------------
    if menu == "Settings":
        st.title("⚙️ Settings")

        if st.button("🔄 Retrain Face Model"):
            with st.spinner("Training..."):
                ok, msg = train_recognizer()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.write("Known Persons:")
        st.write(get_names_list())

  # ------------------ LIVE ATTENDANCE ------------------
if menu == "Live Attendance":
    st.title("🎥 Live Attendance")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("📸 Start Face Recognition"):
            start_recognition_session()
    
    with col2:
        img_file = st.camera_input("Take a photo for attendance")
        if img_file:
            # Save image and process with face recognition
            person_name = "Unknown"  # Replace with your face recognition result
            now = datetime.now()
            
            # Mark attendance
            att_df = pd.read_csv("data/attendance.csv")
            new_entry = pd.DataFrame({
                "Name": [person_name],
                "Role": ["Student"],  # or detect from students/staff CSV
                "Date": [now.strftime("%Y-%m-%d")],
                "Time": [now.strftime("%H:%M:%S")]
            })
            att_df = pd.concat([att_df, new_entry], ignore_index=True)
            att_df.to_csv("data/attendance.csv", index=False)
            
            st.success(f"✅ Attendance marked for {person_name}")

