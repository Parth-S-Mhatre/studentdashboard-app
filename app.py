import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .tips-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models and transformers"""
    try:
        model = joblib.load("model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        column_transformer = joblib.load("column_transformer.pkl")
        
        # Get categories from the trained model
        onehot_encoder = column_transformer.named_transformers_['onehot']
        school_type_categories = onehot_encoder.categories_[0].tolist()
        location_categories = onehot_encoder.categories_[1].tolist()
        
        return model, label_encoder, column_transformer, school_type_categories, location_categories
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, [], []

def get_performance_insights(prediction):
    """Get insights and recommendations based on prediction"""
    if prediction >= 85:
        return {
            "level": "Outstanding",
            "color": "#28a745",
            "icon": "🌟",
            "message": "Exceptional performance! You're on track for academic excellence.",
            "tips": [
                "Maintain your current study routine",
                "Consider helping peers with their studies",
                "Explore advanced learning opportunities",
                "Set higher academic goals"
            ]
        }
    elif prediction >= 75:
        return {
            "level": "Excellent",
            "color": "#17a2b8",
            "icon": "🎯",
            "message": "Great job! You're performing very well academically.",
            "tips": [
                "Keep up the consistent study habits",
                "Focus on areas that need improvement",
                "Participate actively in class discussions",
                "Seek additional challenges"
            ]
        }
    elif prediction >= 65:
        return {
            "level": "Good",
            "color": "#ffc107",
            "icon": "👍",
            "message": "You're doing well! There's room for improvement.",
            "tips": [
                "Increase daily study hours gradually",
                "Improve attendance if possible",
                "Create a structured study schedule",
                "Ask teachers for extra help"
            ]
        }
    else:
        return {
            "level": "Needs Improvement",
            "color": "#dc3545",
            "icon": "📈",
            "message": "Let's work together to improve your performance!",
            "tips": [
                "Significantly increase study time",
                "Improve attendance regularly",
                "Seek help from teachers and tutors",
                "Create a daily study routine",
                "Limit distractions during study time"
            ]
        }

def create_gauge_chart(score):
    """Create a gauge chart for the predicted score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Score"},
        delta = {'reference': 75},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 90], 'color': "lightgreen"},
                {'range': [90, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Load models
    model, label_encoder, column_transformer, school_type_categories, location_categories = load_models()
    
    if model is None:
        st.error("Unable to load models. Please ensure model files are available.")
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 School Student Performance Dashboard</h1>
        <p>Predict and improve your academic performance with AI-powered insights</p>
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; display: inline-block; margin-top: 1rem;">
            <small>🚀 BETA VERSION v1.0 | Designed for School Students (Grades 6-12)</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("📝 Enter Your Information")
    st.sidebar.markdown("Fill in your details to get personalized predictions and recommendations.")
    
    # Student Information Section
    with st.sidebar.expander("👤 Personal Information", expanded=True):
        student_name = st.text_input("Your Name (Optional)", placeholder="Enter your name...")
        grade_level = st.selectbox("📚 Your Grade Level", 
                                 ["Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"],
                                 help="Select your current grade/class")
        parent_edu = st.selectbox("👨‍🎓 Parent's Education Level", 
                                 ["High School", "Bachelor", "Master", "PhD"],
                                 help="Highest education level of your parents")
        
        # Additional school context (for display purposes, not used in prediction)
        family_support = st.selectbox("👪 Family Study Support", 
                                    ["Very Supportive", "Supportive", "Somewhat Supportive", "Limited Support"],
                                    help="How much does your family help with your studies?")
    
    # Academic Information Section
    with st.sidebar.expander("📚 Academic Information", expanded=True):
        hours_studied = st.slider("📖 Daily Study Hours (After School)", 
                                min_value=0.0, max_value=8.0, value=3.0, step=0.5,
                                help="Hours you spend studying after school hours")
        
        attendance = st.slider("📅 School Attendance Percentage", 
                             min_value=0.0, max_value=100.0, value=85.0, step=1.0,
                             help="Your average school attendance percentage")
        
        pe_level = st.slider("📊 Last Report Card Performance", 
                           min_value=1, max_value=10, value=6,
                           help="Rate your last report card/exam performance (1=Poor, 10=Excellent)")
        
        favorite_subject = st.selectbox("❤️ Favorite Subject", 
                                      ["Mathematics", "Science", "English", "History", "Geography", 
                                       "Art", "Music", "Physical Education", "Computer Science", "Languages"],
                                      help="Your favorite school subject")
    
    # School Information Section
    with st.sidebar.expander("🏫 School Information", expanded=True):
        if school_type_categories:
            school_type = st.selectbox("🏢 School Type", school_type_categories,
                                     help="Type of school you attend")
        else:
            school_type = st.selectbox("🏢 School Type", ["Public School", "Private School", "Charter School"])
            
        if location_categories:
            location = st.selectbox("📍 School Location", location_categories,
                                  help="Your school's location area")
        else:
            location = st.selectbox("📍 School Location", ["Urban", "Suburban", "Rural"])
        
        transportation = st.selectbox("🚌 How do you get to school?", 
                                    ["School Bus", "Walk", "Bike", "Parent Drop-off", "Public Transport"],
                                    help="Your usual way of getting to school")
    
    # Lifestyle Information Section
    with st.sidebar.expander("⚡ Daily Life & Activities", expanded=True):
        internet_hours = st.slider("💻 Internet Hours (Study + Recreation)", 
                                 min_value=0.0, max_value=10.0, value=3.0, step=0.5,
                                 help="Total hours spent on internet daily")
        
        sleep_hours = st.slider("😴 Sleep Hours per Night", 
                              min_value=6.0, max_value=12.0, value=8.0, step=0.5,
                              help="Average hours of sleep per night")
        
        extracurricular = st.multiselect("🎯 Extracurricular Activities", 
                                       ["Sports", "Music", "Art", "Drama", "Debate", "Science Club", 
                                        "Math Club", "Student Council", "Volunteer Work", "Part-time Job"],
                                       help="Select activities you participate in")
        
        study_environment = st.selectbox("📚 Where do you usually study?", 
                                       ["Own Room", "Living Room", "Library", "Study Hall", "Kitchen Table"],
                                       help="Your primary study location")
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Predict My Performance", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            try:
                # Use the original parent education categories that the model was trained on
                parent_edu_encoded = label_encoder.transform([parent_edu])[0]
                
                # Prepare input data
                input_data = {
                    'School_Type': [school_type],
                    'Location': [location],
                    'Hours_Studied': [hours_studied],
                    'Attendance': [attendance],
                    'Previous_Exam_Level': [pe_level],
                    'Parental_Education': [parent_edu_encoded],
                    'Internet_Hours': [internet_hours],
                    'Sleep_Hours': [sleep_hours]
                }
                
                # Try prediction with all features first
                try:
                    input_df = pd.DataFrame(input_data)
                    input_transformed = column_transformer.transform(input_df)
                except ValueError:
                    # Fallback to core features only
                    input_data = {
                        'School_Type': [school_type],
                        'Location': [location],
                        'Hours_Studied': [hours_studied],
                        'Attendance': [attendance],
                        'Previous_Exam_Level': [pe_level],
                        'Parental_Education': [parent_edu_encoded]
                    }
                    input_df = pd.DataFrame(input_data)
                    input_transformed = column_transformer.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_transformed)[0]
                insights = get_performance_insights(prediction)
                
                # Display additional context information
                st.subheader("📋 Your Profile Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Student Info:**
                    - Grade: {grade_level}
                    - Favorite Subject: {favorite_subject}
                    - Study Hours: {hours_studied} hrs/day
                    - Attendance: {attendance}%
                    """)
                
                with col2:
                    st.info(f"""
                    **Environment:**
                    - School: {school_type}
                    - Location: {location}
                    - Family Support: {family_support}
                    - Sleep: {sleep_hours} hours
                    """)
                
                if extracurricular:
                    st.success(f"🎯 **Activities:** {', '.join(extracurricular)}")
                else:
                    st.warning("💡 **Tip:** Consider joining extracurricular activities for well-rounded development!")
                
                # Display prediction with styling
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>{insights['icon']} Your Predicted Score</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prediction:.1f}/100</h1>
                    <h3>{insights['level']} Performance</h3>
                    <p style="font-size: 1.1rem;">{insights['message']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display gauge chart
                st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
                
                # Enhanced performance analysis with school context
                st.subheader("📊 Your Performance Analysis")
                
                # Create comprehensive comparison chart
                performance_factors = {
                    'Study Hours': min(100, (hours_studied / 4) * 100),  # 4 hours = 100%
                    'School Attendance': attendance,
                    'Report Card Level': pe_level * 10,
                    'Sleep Quality': min(100, (sleep_hours / 9) * 100),  # 9 hours = 100% for teens
                    'Predicted Score': prediction
                }
                
                # Add bonus for extracurriculars
                if extracurricular:
                    performance_factors['Activity Participation'] = min(100, len(extracurricular) * 20)
                
                fig_bar = px.bar(
                    x=list(performance_factors.keys()),
                    y=list(performance_factors.values()),
                    title="Your Academic Performance Factors",
                    color=list(performance_factors.values()),
                    color_continuous_scale="viridis",
                    text=[f"{v:.0f}%" for v in performance_factors.values()]
                )
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # School-specific insights
                st.subheader("🎯 Your Academic Strengths & Areas for Growth")
                
                strengths = []
                improvements = []
                
                if hours_studied >= 3:
                    strengths.append("📚 Good study habits")
                else:
                    improvements.append("📖 Increase daily study time")
                
                if attendance >= 90:
                    strengths.append("📅 Excellent attendance")
                elif attendance >= 80:
                    strengths.append("📅 Good attendance")
                else:
                    improvements.append("🏫 Improve school attendance")
                
                if sleep_hours >= 8:
                    strengths.append("😴 Healthy sleep schedule")
                else:
                    improvements.append("💤 Get more sleep for better focus")
                
                if extracurricular:
                    strengths.append("🎯 Well-rounded with activities")
                else:
                    improvements.append("🏃‍♂️ Consider joining school activities")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success("**🌟 Your Strengths:**")
                    for strength in strengths:
                        st.write(f"✅ {strength}")
                
                with col2:
                    st.info("**📈 Growth Areas:**")
                    for improvement in improvements:
                        st.write(f"🎯 {improvement}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
        
        else:
            # Welcome message when no prediction is made
            st.markdown("""
            ### 👋 Welcome to Your School Performance Dashboard!
            
            This smart dashboard is specifically designed for **school students (Grades 6-12)** to help you understand and improve your academic performance using AI technology.
            
            **🚀 BETA VERSION v1.0 Features:**
            1. 📝 Fill in your school information in the sidebar
            2. 🔮 Get your predicted performance score
            3. 📊 See visual analysis of your study habits
            4. 💡 Get personalized tips to improve your grades
            5. 📅 Receive a customized study schedule
            
            **What makes this special for school students:**
            - 🎯 Designed specifically for Grades 6-12
            - 📚 Focuses on school subjects and report cards
            - 🏫 Considers school-specific factors
            - 👨‍👩‍👧‍👦 Includes parent education influence
            - 🎮 Accounts for internet and social media usage
            - 🏃‍♂️ Considers extracurricular activities
            
            **Beta Version Notice:**
            This is a beta version designed to help school students. We're continuously improving based on student feedback!
            """)
            
            # Sample visualization
            sample_data = pd.DataFrame({
                'Factor': ['Study Hours', 'Attendance', 'Sleep', 'Previous Performance'],
                'Impact': [85, 92, 78, 88],
                'Your Level': [70, 85, 75, 60]
            })
            
            fig_sample = px.bar(sample_data, x='Factor', y=['Impact', 'Your Level'], 
                              title="Sample: Performance Factors Impact",
                              barmode='group')
            st.plotly_chart(fig_sample, use_container_width=True)
    
    with col2:
        # Display recommendations if prediction is made
        if predict_button and 'insights' in locals():
            st.markdown(f"""
            <div class="tips-card">
                <h3>💡 Personalized Recommendations</h3>
                <p><strong>Performance Level:</strong> {insights['level']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("🚀 Action Items")
            for i, tip in enumerate(insights['tips'], 1):
                st.markdown(f"**{i}.** {tip}")
            
            # Study schedule recommendation for school students
            st.subheader("📅 After-School Study Schedule")
            if hours_studied < 2:
                st.warning("⏰ Consider increasing after-school study time to 2-3 hours daily")
                recommended_hours = 2.5
            elif hours_studied > 5:
                st.info("📚 Great dedication! Make sure to balance with rest and activities")
                recommended_hours = hours_studied
            else:
                st.success("✅ Your study hours look balanced!")
                recommended_hours = hours_studied
            
            # Create school-appropriate study schedule
            schedule = pd.DataFrame({
                'Time Slot': ['Right After School', 'Early Evening', 'Before Bed'],
                'Duration (hrs)': [recommended_hours * 0.5, recommended_hours * 0.4, recommended_hours * 0.1],
                'Focus Area': ['Homework/Assignments', 'Review & Practice', 'Next Day Prep'],
                'Tips': ['When mind is fresh', 'Deeper understanding', 'Quick review']
            })
            
            st.dataframe(schedule, use_container_width=True)
            
            # Show grade-specific advice
            if 'grade_level' in locals():
                st.subheader(f"🎯 Specific Tips for {grade_level}")
                grade_num = int(grade_level.split()[-1])
                
                if grade_num <= 8:
                    st.info("📚 **Middle School Focus**: Build strong study habits, explore interests, ask questions!")
                elif grade_num <= 10:
                    st.info("🎯 **High School Focus**: Balance academics with activities, start thinking about future goals!")
                else:
                    st.info("🚀 **Senior Years**: Prepare for college/career, maintain grades, leadership opportunities!")
            
        else:
            # Default recommendations
            st.markdown("""
            ### 📚 School Study Tips for Better Grades
            
            **⭐ Top Performance Boosters for School Students:**
            - 📖 Study 2-4 hours daily after school
            - 📅 Maintain 95%+ school attendance
            - 😴 Get 8-9 hours of sleep (growing minds need rest!)
            - 💻 Limit social media during homework time
            - 🎯 Set specific grade goals for each subject
            - 📝 Take good notes in class
            
            **🔥 School Success Pro Tips:**
            - Use the Pomodoro Technique (25min study + 5min break)
            - Create a homework schedule right after school
            - Form study groups with classmates
            - Ask teachers questions during office hours
            - Review notes the same day you take them
            - Prepare for tests at least 3 days ahead
            
            **🏆 Grade-Specific Advice:**
            - **Grades 6-8**: Focus on building good study habits
            - **Grades 9-10**: Balance academics with activities
            - **Grades 11-12**: Prepare for college applications
            """)
            
            # Quick stats for school students
            st.subheader("📈 School Success Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ideal Study Time", "2-4 hrs/day", help="After school study time")
            with col2:
                st.metric("Target Attendance", "95%+", help="School attendance goal")
            with col3:
                st.metric("Sleep for Growth", "8-9 hours", help="Growing minds need rest")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 School Student Performance Dashboard v1.0 BETA | Designed for Grades 6-12</p>
        <p>Powered by Machine Learning and Developed by Parth Mhatre | Remember: Consistent effort leads to better grades! 🌟</p>
         <p>&copy; 2025 Parth Mhatre | Studentdashboard-app. All rights reserved.</p>
        <p><small>⚠️ Beta Version: Please report any issues or suggestions for improvement</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()