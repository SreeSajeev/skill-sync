import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Constants
skills = ['Food Sorting', 'Inventory Management', 'Community Outreach', 'Logistics', 'Food Distribution']
locations = ['Warehouse', 'Distribution Center', 'Community Center']
times = ['Morning', 'Afternoon', 'Evening']

# Load or generate sample data
try:
    volunteers = pd.read_csv('volunteers.csv')
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    volunteers_data = {
        'name': [f'Volunteer_{i}' for i in range(1, 101)],
        'skills': np.random.choice(skills, 100),
        'location': np.random.choice(locations, 100),
        'availability': np.random.choice(times, 100),
        'points': np.random.randint(0, 100, size=100),
        'badges': [np.random.choice(['Top Sorter', 'Best Organizer', 'Community Champion'], 1)[0] for _ in range(100)]
    }
    tasks_data = {
        'task_name': [f'Task_{i}' for i in range(1, 31)],
        'required_skills': np.random.choice(skills, 30),
        'location': np.random.choice(locations, 30),
        'time': np.random.choice(times, 30),
        'points': np.random.randint(10, 50, size=30)
    }
    volunteers = pd.DataFrame(volunteers_data)
    tasks = pd.DataFrame(tasks_data)
    volunteers.to_csv('volunteers.csv', index=False)
    tasks.to_csv('tasks.csv', index=False)

# Initialize LabelEncoders
le_skills = LabelEncoder()
le_location = LabelEncoder()
le_availability = LabelEncoder()

# Fit encoders on the data
le_skills.fit(volunteers['skills'].unique())
le_location.fit(volunteers['location'].unique())
le_availability.fit(volunteers['availability'].unique())

# Encode data
volunteers['skills_encoded'] = le_skills.transform(volunteers['skills'])
volunteers['location_encoded'] = le_location.transform(volunteers['location'])
volunteers['availability_encoded'] = le_availability.transform(volunteers['availability'])
tasks['skills_encoded'] = le_skills.transform(tasks['required_skills'])
tasks['location_encoded'] = le_location.transform(tasks['location'])
tasks['time_encoded'] = le_availability.transform(tasks['time'])

# Prepare data for training
X = volunteers[['skills_encoded', 'location_encoded', 'availability_encoded']]
y = np.random.choice(tasks['task_name'], size=len(volunteers))

# Load or train machine learning models
try:
    with open('task_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open('task_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Define advanced task matching function
def advanced_match_task(volunteer):
    volunteer_encoded = {
        'skills_encoded': le_skills.transform([volunteer['skills']])[0],
        'location_encoded': le_location.transform([volunteer['location']])[0],
        'availability_encoded': le_availability.transform([volunteer['availability']])[0]
    }
    X_volunteer = pd.DataFrame([volunteer_encoded])
    recommended_task = model.predict(X_volunteer)
    return recommended_task[0]

# Define clustering function
def cluster_volunteers():
    features = volunteers[['skills_encoded', 'location_encoded', 'availability_encoded']]
    kmeans = KMeans(n_clusters=5, random_state=42)
    volunteers['cluster'] = kmeans.fit_predict(features)
    return kmeans

def advanced_match_task_cluster(volunteer):
    cluster_model = cluster_volunteers()
    volunteer_encoded = {
        'skills_encoded': le_skills.transform([volunteer['skills']])[0],
        'location_encoded': le_location.transform([volunteer['location']])[0],
        'availability_encoded': le_availability.transform([volunteer['availability']])[0]
    }
    volunteer_df = pd.DataFrame([volunteer_encoded])
    cluster_label = cluster_model.predict(volunteer_df)[0]
    cluster_volunteers_df = volunteers[volunteers['cluster'] == cluster_label]
    most_common_skills = cluster_volunteers_df['skills'].mode()
    if not most_common_skills.empty:
        most_common_skill = most_common_skills[0]
        best_task = tasks[tasks['required_skills'] == most_common_skill]
        if not best_task.empty:
            return best_task.loc[best_task['points'].idxmax()]['task_name']
    return 'No suitable task found'

# Streamlit UI
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #FFFFFF;
        font-size: 2.5em;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">SkillSync: AI-Powered Task Matching for Food Banks</div>', unsafe_allow_html=True)

option = st.sidebar.selectbox("Are you a:", ("Volunteer", "Administrator"))

if option == "Volunteer":
    st.sidebar.header('Volunteer Registration')
    volunteer_name = st.sidebar.text_input('Name')
    volunteer_skills = st.sidebar.selectbox('Skills', skills)
    volunteer_location = st.sidebar.selectbox('Location', locations)
    volunteer_availability = st.sidebar.selectbox('Availability', times)

    if st.sidebar.button('Submit'):
        new_volunteer = {
            'name': volunteer_name,
            'skills': volunteer_skills,
            'location': volunteer_location,
            'availability': volunteer_availability,
            'points': 0,
            'badges': []
        }
        task_match = advanced_match_task_cluster(new_volunteer)
        st.write(f'Task Assigned: {task_match}')
        new_volunteer['task'] = task_match
        volunteers = pd.concat([volunteers, pd.DataFrame([new_volunteer])], ignore_index=True)
        volunteers.to_csv('volunteers.csv', index=False)
        st.write('Volunteer registered successfully!')
        st.write(f'Recommended Task: {task_match}')

if option == "Administrator":
    st.sidebar.header('Post a Task')
    task_name = st.sidebar.text_input('Task Name')
    task_skills = st.sidebar.selectbox('Required Skills', skills)
    task_location = st.sidebar.selectbox('Location', locations)
    task_time = st.sidebar.selectbox('Time', times)
    task_points = st.sidebar.number_input('Points', min_value=0, step=10)

    if st.sidebar.button('Post Task'):
        new_task = {
            'task_name': task_name,
            'required_skills': task_skills,
            'location': task_location,
            'time': task_time,
            'points': task_points
        }
        tasks = pd.concat([tasks, pd.DataFrame([new_task])], ignore_index=True)
        tasks.to_csv('tasks.csv', index=False)
        st.sidebar.success('Task posted successfully!')

    st.header('Assigned Tasks')
    for task in tasks.itertuples():
        st.subheader(task.task_name)
        st.write(f"Required Skills: {task.required_skills}")
        st.write(f"Location: {task.location}")
        st.write(f"Time: {task.time}")
        st.write(f"Points: {task.points}")

        assigned_volunteers = volunteers[volunteers['task'] == task.task_name]
        for volunteer in assigned_volunteers.itertuples():
            st.write(f"- {volunteer.name} ({volunteer.skills}, {volunteer.location}, {volunteer.availability})")

# Gamification Dashboard

# Top 10 Volunteers by Points
top_volunteers = volunteers.nlargest(10, 'points')
st.subheader('Top 10 Volunteers')
top_volunteers_chart = top_volunteers[['name', 'points']].set_index('name')
st.bar_chart(top_volunteers_chart, use_container_width=True)

# Leaderboard by Skill
st.header('Leaderboard by Skill')
skill_leaderboard = volunteers.groupby('skills')['points'].sum().reset_index()
skill_leaderboard = skill_leaderboard.sort_values(by='points', ascending=False)
st.subheader('Points by Skill')
skill_leaderboard_chart = skill_leaderboard.set_index('skills')
st.bar_chart(skill_leaderboard_chart, use_container_width=True)

# Volunteer Network and Collaboration
st.header('Volunteer Network')
for volunteer in volunteers.itertuples():
    st.subheader(volunteer.name)
    st.write(f"Skills: {volunteer.skills}")
    st.write(f"Location: {volunteer.location}")
    st.write(f"Availability: {volunteer.availability}")

    # Create a unique key for the text_area widget
    unique_key = f"{volunteer.name}_{volunteer.Index}"
    st.text_area("Share your experience", key=unique_key)
