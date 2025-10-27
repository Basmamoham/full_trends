import numpy as np
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ContentTrendsAnalyzer:
    def __init__(self, data):
        self.df = data
        self.trending_topics = []
        self.topic_sentiments = {}

    def preprocess_text(self, text):
        """Clean and prepare the text for analysis"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Remove very short words
        words = text.split()
        words = [word for word in words if len(word) > 2]

        return ' '.join(words)

    def extract_keywords(self, text, n=10):
        """Extract keywords from the text"""
        words = text.split()
        # Remove common words
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'are', 'was', 'were', 'has', 'have', 'had'}
        keywords = [word for word in words if word not in stop_words]

        return Counter(keywords).most_common(n)

    def analyze_sentiment(self, text):
        """Sentiment analysis for the text"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def detect_trending_topics_lda(self, n_topics=8, n_top_words=10):
        """Discover topics using LDA"""

        # Check if 'content' column exists
        if 'content' not in self.df.columns:
            print("Warning: 'content' column not found, using fallback analysis")
            return self.fallback_topic_analysis()

        # Clean the texts
        contents = self.df['content'].apply(self.preprocess_text)

        # Create word matrix
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=1000
        )

        try:
            X = vectorizer.fit_transform(contents)

            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )

            lda.fit(X)

            # Extract distinctive words for each topic
            feature_names = vectorizer.get_feature_names_out()

            trending_topics = []

            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]

                # Calculate topic strength
                topic_strength = topic.sum()

                # Sentiment analysis for the topic
                topic_text = ' '.join(top_words)
                sentiment = self.analyze_sentiment(topic_text)

                topic_info = {
                    'topic_id': topic_idx,
                    'keywords': top_words,
                    'strength': topic_strength,
                    'sentiment': sentiment,
                    'trend_score': self.calculate_trend_score(topic_idx, lda, X)
                }

                trending_topics.append(topic_info)

            # Sort topics by trend strength
            trending_topics.sort(key=lambda x: x['trend_score'], reverse=True)
            self.trending_topics = trending_topics

            return trending_topics

        except Exception as e:
            print(f"Error in LDA analysis: {e}")
            return self.fallback_topic_analysis()

    def fallback_topic_analysis(self):
        """Alternative method for topic analysis in case LDA fails"""
        print("Using fallback topic analysis...")

        all_contents = ' '.join(self.df['content'].apply(self.preprocess_text))
        keywords = self.extract_keywords(all_contents, 50)

        # Group keywords into topics
        sports_keywords = ['tennis', 'basketball', 'football', 'hockey', 'baseball', 'soccer', 'game', 'championship']
        news_keywords = ['news', 'update', 'breaking', 'report', 'announcement']
        event_keywords = ['trade', 'sign', 'winner', 'champion', 'cup', 'award']

        topics = []

        # Analyze sports topic
        sport_words = [word for word, count in keywords if any(sport in word for sport in sports_keywords)]
        if sport_words:
            topics.append({
                'topic_id': 0,
                'keywords': sport_words[:10],
                'strength': len(sport_words),
                'sentiment': 0.1,
                'trend_score': len(sport_words) * 0.8
            })

        # Analyze events topic
        event_words = [word for word, count in keywords if any(event in word for event in event_keywords)]
        if event_words:
            topics.append({
                'topic_id': 1,
                'keywords': event_words[:10],
                'strength': len(event_words),
                'sentiment': 0.2,
                'trend_score': len(event_words) * 0.7
            })

        topics.sort(key=lambda x: x['trend_score'], reverse=True)
        self.trending_topics = topics
        return topics

    def calculate_trend_score(self, topic_idx, lda_model, X):
        """Calculate the trend score for the topic"""
        # Topic distribution across documents
        doc_topic_dist = lda_model.transform(X)
        topic_prevalence = doc_topic_dist[:, topic_idx].mean()

        # Calculate engagement for the topic
        topic_engagement = self.calculate_topic_engagement(topic_idx, doc_topic_dist)

        return topic_prevalence * topic_engagement

    def calculate_topic_engagement(self, topic_idx, doc_topic_dist):
        """Calculate average engagement for the topic"""
        topic_docs = doc_topic_dist[:, topic_idx]
        top_doc_indices = topic_docs.argsort()[-10:][::-1]  # Top 10 documents

        total_engagement = 0
        count = 0

        for idx in top_doc_indices:
            if idx < len(self.df):
                engagement = (self.df.iloc[idx]['likes'] +
                           self.df.iloc[idx]['num_comments'] * 2 +
                           self.df.iloc[idx]['num_shares'] * 3)
                total_engagement += engagement
                count += 1

        return total_engagement / count if count > 0 else 0

    def analyze_content_patterns(self):
        """Analyze content patterns"""
        try:
            patterns = {
                'emoji_usage': 0,
                'hashtag_usage': 0,
                'content_length': 0,
                'video_content_ratio': 0,
                'image_content_ratio': 0
            }
            
            if 'content' in self.df.columns:
                patterns['emoji_usage'] = self.df['content'].apply(lambda x: len(re.findall(r'[^\w\s]', str(x)))).mean()
                patterns['content_length'] = self.df['content_length'].mean() if 'content_length' in self.df.columns else 0
            
            if 'num_hashtags' in self.df.columns:
                patterns['hashtag_usage'] = self.df['num_hashtags'].mean()
            
            if 'media_type' in self.df.columns:
                patterns['video_content_ratio'] = (self.df['media_type'] == 'Video').mean()
                patterns['image_content_ratio'] = (self.df['media_type'] == 'Image').mean()

            return patterns
        except Exception as e:
            print(f"Error in analyze_content_patterns: {e}")
            return {
                'emoji_usage': 0,
                'hashtag_usage': 0,
                'content_length': 0,
                'video_content_ratio': 0,
                'image_content_ratio': 0
            }

    def get_trending_analysis_report(self):
        """Create a trending analysis report"""

        # Discover rising topics
        trending_topics = self.detect_trending_topics_lda()

        # Analyze content patterns
        content_patterns = self.analyze_content_patterns()

        # Analyze best times to post
        time_analysis = self.analyze_optimal_times()

        report = {
            'trending_topics': trending_topics,
            'content_patterns': content_patterns,
            'optimal_times': time_analysis,
            'top_performing_content': self.get_top_performing_content(),
            'recommendations': self.generate_recommendations()
        }

        return report

    def analyze_optimal_times(self):
        """Analyze optimal posting times based on engagement"""
        try:
            if 'hour_posted' in self.df.columns:
                engagement_by_hour = self.df.groupby('hour_posted').agg({
                    'likes': 'mean',
                    'num_comments': 'mean',
                    'num_shares': 'mean'
                }).mean(axis=1)
                best_hours = engagement_by_hour.sort_values(ascending=False).head(3).index.tolist()
                while len(best_hours) < 3 and best_hours:
                    best_hours.append(best_hours[-1])  # Repeat last value
                if not best_hours:
                    best_hours = [12, 12, 12]  # All missing, use noon as safe default
                hourly_engagement = engagement_by_hour.to_dict()
            else:
                best_hours = [12, 12, 12]
                hourly_engagement = {}
            
            if 'day_of_week' in self.df.columns:
                engagement_by_day = self.df.groupby('day_of_week').agg({
                    'likes': 'mean',
                    'num_comments': 'mean',
                    'num_shares': 'mean'
                }).mean(axis=1)
                best_days = engagement_by_day.sort_values(ascending=False).head(3).index.tolist()
                daily_engagement = engagement_by_day.to_dict()
            else:
                best_days = [0, 1, 2]
                daily_engagement = {}
            
            return {
                'best_hours': best_hours[:3],
                'best_days': best_days,
                'hourly_engagement': hourly_engagement,
                'daily_engagement': daily_engagement
            }
        except Exception as e:
            print(f"Error in analyze_optimal_times: {e}")
            return {
                'best_hours': [12, 12, 12],
                'best_days': [0, 1, 2],
                'hourly_engagement': {},
                'daily_engagement': {}
            }

    def get_top_performing_content(self, n=5):
        """Get top performing content without near-duplicate sentences, ignoring emojis/case/symbols."""
        import re
        try:
            # Require essential columns
            needed = ['likes', 'num_comments', 'num_shares', 'content']
            if all(col in self.df.columns for col in needed):
                self.df['total_engagement'] = (
                    self.df['likes'] + self.df['num_comments'] * 2 + self.df['num_shares'] * 3
                )
                # Normalize content: remove emojis/symbols, lowercase, strip
                def normalize_content(text):
                    return re.sub(r'[^\w\s]', '', str(text)).lower().strip()
                self.df['normalized_content'] = self.df['content'].apply(normalize_content)
                sorted_df = self.df.sort_values('total_engagement', ascending=False)
                deduped_df = sorted_df.drop_duplicates('normalized_content')
                available_cols = ['content', 'likes', 'num_comments', 'num_shares', 'total_engagement']
                cols_to_return = [col for col in available_cols if col in deduped_df.columns]
                top_content = deduped_df[cols_to_return].head(n)
                return top_content.to_dict('records')
            else:
                return []
        except Exception as e:
            print(f"Error in get_top_performing_content: {e}")
            return []

    def generate_recommendations(self):
        """Generate recommendations based on the analysis"""
        recommendations = []

        try:
            # Analyze rising topics
            top_topic = self.trending_topics[0] if self.trending_topics else None

            if top_topic:
                keywords = top_topic.get('keywords', [])
                if keywords:
                    recommendations.append(
                        f"Trending Topic: Focus on content related to {', '.join(keywords[:3])}"
                    )

                sentiment = top_topic.get('sentiment', 0)
                if sentiment > 0.3:
                    recommendations.append("Positive content gets higher engagement - use positive language")
                elif sentiment < -0.3:
                    recommendations.append("Try to improve the tone - negative content may reduce engagement")

            # Analyze media patterns
            if 'has_video' in self.df.columns and self.df['has_video'].mean() > 0.3:
                recommendations.append("Visual content gets good engagement - continue producing videos")
            elif 'has_video' in self.df.columns:
                recommendations.append("Try increasing the proportion of visual content to improve engagement")

            # Analyze timing
            optimal_times = self.analyze_optimal_times()
            best_hours = optimal_times.get('best_hours', [])
            best_days = optimal_times.get('best_days', [])
            
            
        
        except Exception as e:
            print(f"Error in generate_recommendations: {e}")
            recommendations.append("Analysis completed with partial data.")

        return recommendations

    def visualize_trends(self):
        """Visualize trends and results"""
        if not self.trending_topics:
            self.detect_trending_topics_lda()

        # Plot rising topics
        topics_data = []
        for topic in self.trending_topics[:5]:
            topics_data.append({
                'Topic': f"Topic {topic['topic_id']}",
                'Trend Score': topic['trend_score'],
                'Sentiment': topic['sentiment']
            })

        topics_df = pd.DataFrame(topics_data)

        plt.figure(figsize=(12, 8))

        # Plot 1: Trend scores
        plt.subplot(2, 2, 1)
        sns.barplot(data=topics_df, x='Trend Score', y='Topic', palette='viridis')
        plt.title('Top Trending Topics by Score')

        # Plot 2: Sentiments
        plt.subplot(2, 2, 2)
        sns.barplot(data=topics_df, x='Sentiment', y='Topic', palette='coolwarm')
        plt.title('Topic Sentiment Analysis')

        # Plot 3: Engagement by time
        plt.subplot(2, 2, 3)
        time_analysis = self.analyze_optimal_times()
        hours = list(time_analysis['hourly_engagement'].keys())
        engagement = list(time_analysis['hourly_engagement'].values())
        plt.plot(hours, engagement, marker='o')
        plt.title('Engagement by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Engagement')

        # Plot 4: Media type distribution
        plt.subplot(2, 2, 4)
        media_counts = self.df['media_type'].value_counts()
        plt.pie(media_counts.values, labels=media_counts.index, autopct='%1.1f%%')
        plt.title('Content Media Type Distribution')

        plt.tight_layout()
        plt.show()

# Usage of the code

def run_analysis(data):
    analyzer = ContentTrendsAnalyzer(data)
    return analyzer.get_trending_analysis_report()
