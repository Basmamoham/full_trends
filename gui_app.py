import streamlit as st
import pandas as pd
import requests
import numpy as np

API_ENDPOINT = "http://127.0.0.1:8000/analyze_trends"

st.set_page_config(page_title="Content Trends", layout="wide")
st.title("Content Trends")


# gui_app.py

# gui_app.py

def prepare(df):

    if "timestamp" not in df.columns:
        cols = df.columns
        time_col = None
        for c in cols:
            if "time" in c or "date" in c:
                time_col = c
                break
        if time_col:
            df.rename(columns={time_col: "timestamp"}, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_posted"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    # إصلاح خطأ Timestamp: التحويل إلى نص قبل الإرسال
    df['timestamp'] = df['timestamp'].astype(str)


    for col in ['content', 'audience_target', 'us_region', 'event_type']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)


    int_cols = ["likes", "num_comments", "num_shares", "num_hashtags", "content_length", "word_count", "page_followers"]
    for col in int_cols:
        if col not in df.columns:
            df[col] = 0
       
        df[col] = pd.to_numeric(df[col], errors='coerce')
       
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0).astype(int)


    float_cols = [
        "likes_rate", "comments_rate", "shares_rate", "audience_size",
        "target_engagement", "fan_loyalty", "sentiment_score", "new_positive_count",
        "new_negative_count", "tfidf_svd_0", "tfidf_svd_1", "tfidf_svd_2",
        "tfidf_svd_3", "tfidf_svd_4", "tfidf_svd_5", "tfidf_svd_6", "tfidf_svd_7"
        # أضف هنا أي أعمدة أخرى من نوع float
    ]
    for col in float_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        # Replace any inf values with 0.0 as well
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).astype(float)


    if "media_type" not in df.columns:
        df["media_type"] = "Unknown"
    df["media_type"] = df["media_type"].fillna("Unknown").astype(str)


    # Replace NaN and inf in all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].fillna(0)
    
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].fillna('')

    return df





if "report" not in st.session_state:
    st.session_state.report = None

file = st.file_uploader("Upload CSV", type="csv")

# Add a button to check API health
if st.button("Check API Connection"):
    try:
        response = requests.get(API_ENDPOINT.replace("/analyze_trends", ""))
        if response.status_code == 200:
            st.success("✅ API is running and reachable!")
            st.json(response.json())
        else:
            st.warning(f"API responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please start the API server by running: `uvicorn api_service:app --reload`")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if file:
    df = pd.read_csv(file)
    df = prepare(df)
    st.write(df.head())

    if st.button("Analyze"):
        df['timestamp'] = df['timestamp'].astype(str)


        df = df.replace([np.nan, np.inf, -np.inf], [None, None, None])
        
        # Also ensure no object columns have NaN
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')

        payload = {"data": df.to_dict("records")}
        
        try:
            with st.spinner("Analyzing trends..."):
                r = requests.post(API_ENDPOINT, json=payload)
            
            if r.status_code == 200:
                st.session_state.report = r.json()
                st.success("Analysis complete!")
            else:
                st.error(f"API Error: {r.status_code} - {r.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the API server is running at http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if st.session_state.report:
    st.divider()
    st.header("Analysis Results")
    
    rep = st.session_state.report.get("report", st.session_state.report)
    
    if "trending_topics" in rep:
        st.subheader("📈 Trending Topics (Top 5)")
        st.json(rep["trending_topics"][:5])
    
    if "recommendations" in rep:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(rep["recommendations"], 1):
            st.write(f"{i}. {rec}")
    
    if "optimal_times" in rep:
        st.subheader("⏰ Best Posting Times")
        if "best_hours" in rep["optimal_times"]:
            hours = rep['optimal_times']['best_hours']
            def hour_label(h):
                h = int(h)
                if h == 0:
                    return "12 AM"
                elif h == 12:
                    return "12 PM"
                elif h < 12:
                    return f"{h} AM"
                else:
                    return f"{h - 12} PM"
            hour_strings = [hour_label(h) for h in hours]
            unique_hours = list(dict.fromkeys(hour_strings))  # Remove duplicates, preserve order
            if len(unique_hours) == 1:
                st.write(f"Only one hour of posting activity found: {unique_hours[0]}")
            elif len(unique_hours) == 2:
                st.write(f"Top 2 hours: {', '.join(unique_hours)}")
            else:
                st.write(f"Top 3 hours: {', '.join(unique_hours[:3])}")
        
        if "best_days" in rep["optimal_times"]:
            days = rep['optimal_times']['best_days']
            # Map day numbers to day names (0=Monday, 1=Tuesday, etc.)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_strings = [day_names[d] if isinstance(d, int) and 0 <= d < 7 else str(d) for d in days]
            st.write(f"Best days: {', '.join(day_strings)}")
    
    if "top_performing_content" in rep:
        st.subheader("🏆 Top Performing Content")
        for i, content in enumerate(rep["top_performing_content"][:5], 1):
            st.write(f"{i}. {content}")
