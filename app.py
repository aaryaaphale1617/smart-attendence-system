# app.py
import streamlit as st
import yaml
from yaml import SafeLoader
import streamlit_authenticator as stauth
import pandas as pd
import os
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from face_module import start_recognition_session, train_recognizer, get_names_list

st.set_page_config(page_title="Smart Attendance System", layout="wide")

# ------------------ helper functions -------------------
def ensure_data_files():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/students.csv"):
        pd.DataFrame(columns=["ID", "Name", "RollNo", "Department"]).to_csv("data/students.csv", index=False)
    if not os.path.exists("data/staff.csv"):
        pd.DataFrame(columns=["ID", "Name", "Designation", "Department"]).to_csv("data/staff.csv", index=False)
    if not os.path.exists("data/attendance.csv"):
        pd.DataFrame(columns=["Name","Role","Date","Time"]).to_csv("data/attendance.csv", index=False)

def save_uploaded_image(uploaded_file, person_name):
    folder = os.path.join("TrainingImages", person_name)
    os.makedirs(folder, exist_ok=True)
    # save multiple frames if needed
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
        c.drawString(40 + i*120, y, str(col))
    y -= 20
    for idx, row in df.iterrows():
        for i, item in enumerate(row):
            c.drawString(40 + i*120, y, str(item))
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

ensure_data_files()

# ------------------ Authentication -----------------------
with open('credentials.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# NEW CORRECT LOGIN CALL
authentication_status, username = authenticator.login(location="main")

if authentication_status is None:
    st.warning("Please enter your username and password.")
    st.stop()

elif authentication_status is False:
    st.error("❌ Username or Password Incorrect")
    st.stop()

elif authentication_status:
    st.success(f"Welcome {username}!")

    # --------------- Manage Students -----------------------
    if menu == "Manage Students":
        st.title("👨‍🎓 Manage Students")
        df = pd.read_csv("data/students.csv")
        st.dataframe(df)
        st.subheader("Add new student (with photo)")
        with st.form("add_student"):
            sid = st.number_input("ID", min_value=1, step=1)
            sname = st.text_input("Full Name")
            sroll = st.text_input("Roll No")
            sdept = st.text_input("Department")
            spic = st.file_uploader("Upload student face image (jpg/png)", type=["jpg","png"])
            submitted = st.form_submit_button("Add Student")
            if submitted:
                new = pd.DataFrame([[sid, sname, sroll, sdept]], columns=["ID","Name","RollNo","Department"])
                df = pd.concat([df, new], ignore_index=True)
                df.to_csv("data/students.csv", index=False)
                if spic is not None:
                    save_uploaded_image(spic, sname)
                st.success("Student added. After adding images, click Retrain Model (Settings).")

        st.subheader("Edit / Delete Student")
        sel = st.selectbox("Choose student ID", df["ID"].astype(int))
        selrow = df[df["ID"]==int(sel)].iloc[0]
        with st.form("edit_student"):
            ename = st.text_input("Name", value=selrow["Name"])
            eroll = st.text_input("Roll No", value=selrow["RollNo"])
            edept = st.text_input("Department", value=selrow["Department"])
            efile = st.file_uploader("Add another face image for this student", type=["jpg","png"], key="add_img_student")
            up = st.form_submit_button("Update Student")
            if up:
                df.loc[df["ID"]==int(sel), ["Name","RollNo","Department"]] = [ename, eroll, edept]
                df.to_csv("data/students.csv", index=False)
                if efile is not None:
                    save_uploaded_image(efile, ename)
                st.success("Updated student. Retrain model in Settings if you added images.")

        if st.button("Delete Selected Student"):
            df = df[df["ID"]!=int(sel)]
            df.to_csv("data/students.csv", index=False)
            st.warning("Deleted student. Note: images remain in TrainingImages; remove manually if desired.")

    # --------------- Manage Staff -----------------------
    if menu == "Manage Staff":
        st.title("👩‍🏫 Manage Staff")
        df = pd.read_csv("data/staff.csv")
        st.dataframe(df)
        st.subheader("Add new staff (with photo)")
        with st.form("add_staff"):
            sid = st.number_input("ID", min_value=1, step=1, key="staff_id")
            sname = st.text_input("Full Name", key="staff_name")
            sdes = st.text_input("Designation", key="staff_des")
            sdept = st.text_input("Department", key="staff_dept")
            spic = st.file_uploader("Upload staff face image (jpg/png)", type=["jpg","png"], key="staff_pic")
            submitted = st.form_submit_button("Add Staff")
            if submitted:
                new = pd.DataFrame([[sid, sname, sdes, sdept]], columns=["ID","Name","Designation","Department"])
                df = pd.concat([df, new], ignore_index=True)
                df.to_csv("data/staff.csv", index=False)
                if spic is not None:
                    save_uploaded_image(spic, sname)
                st.success("Staff added. Retrain model in Settings if you added images.")

        st.subheader("Edit / Delete Staff")
        if len(df) > 0:
            sel = st.selectbox("Choose staff ID", df["ID"].astype(int))
            selrow = df[df["ID"]==int(sel)].iloc[0]
            with st.form("edit_staff"):
                ename = st.text_input("Name", value=selrow["Name"], key="edit_staff_name")
                edes = st.text_input("Designation", value=selrow["Designation"], key="edit_staff_des")
                edept = st.text_input("Department", value=selrow["Department"], key="edit_staff_dept")
                efile = st.file_uploader("Add another face image for this staff", type=["jpg","png"], key="add_img_staff")
                up = st.form_submit_button("Update Staff")
                if up:
                    df.loc[df["ID"]==int(sel), ["Name","Designation","Department"]] = [ename, edes, edept]
                    df.to_csv("data/staff.csv", index=False)
                    if efile is not None:
                        save_uploaded_image(efile, ename)
                    st.success("Updated staff. Retrain model in Settings if you added images.")
            if st.button("Delete Selected Staff"):
                df = df[df["ID"]!=int(sel)]
                df.to_csv("data/staff.csv", index=False)
                st.warning("Deleted staff. Images remain; remove manually if desired.")
        else:
            st.info("No staff records yet")

    # --------------- Attendance -----------------------
    if menu == "Attendance":
        st.title("📝 Attendance Records")
        att = pd.read_csv("data/attendance.csv")
        st.dataframe(att)
        st.markdown("*Filter / Export*")
        names = sorted(att["Name"].unique().tolist()) if not att.empty else []
        selname = st.selectbox("Filter by Name (optional)", ["All"] + names)
        if selname != "All":
            att = att[att["Name"]==selname]
            st.dataframe(att)
        if st.button("Export CSV"):
            tmp = att.to_csv(index=False).encode()
            st.download_button("Download CSV", data=tmp, file_name=f"attendance_{datetime.now().date()}.csv", mime="text/csv")
        if st.button("Export PDF"):
            pdfbytes = dataframe_to_pdf_bytes(att, title="Attendance Report")
            st.download_button("Download PDF", data=pdfbytes, file_name=f"attendance_{datetime.now().date()}.pdf", mime="application/pdf")

    # --------------- Reports -----------------------
    if menu == "Reports":
        st.title("📈 Reports")
        att = pd.read_csv("data/attendance.csv")
        if att.empty:
            st.info("No attendance data yet")
        else:
            st.subheader("Attendance count by person")
            summary = att.groupby(["Name","Role"]).size().reset_index(name="Count")
            st.dataframe(summary.sort_values("Count", ascending=False))
            fig, ax = plt.subplots()
            summary.groupby("Name")["Count"].sum().nlargest(10).plot(kind="bar", ax=ax)
            ax.set_ylabel("Attendance count")
            st.pyplot(fig)

    # --------------- Settings -----------------------
    if menu == "Settings":
        st.title("⚙ Settings & Model")
        st.info("If you added new face images (via upload), click Retrain to rebuild the recognizer.")
        if st.button("Retrain Model"):
            with st.spinner("Training..."):
                ok, msg = train_recognizer()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        st.write("Known persons:", get_names_list())
        st.markdown("---")
        st.write("System maintenance: remove training images manually if you want to delete a person's training data.")