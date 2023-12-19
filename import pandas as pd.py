import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample Social Media Dataset
data = {
    'Username': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Post': ['I love the new iPhone! Best purchase ever. #happy', 'Feeling sad today. 😞', 'Excited for the weekend! #FridayFeeling', 'Not sure about this product. #confused', 'Amazing experience at the concert! #awesome'],
    'Likes': [100, 20, 150, 10, 200]
}

df = pd.DataFrame(data)

# Sentiment Analysis using TextBlob
df['Sentiment'] = df['Post'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Visualize Sentiment
plt.figure(figsize=(10, 6))
plt.bar(df['Username'], df['Sentiment'], color=['green' if s >= 0 else 'red' for s in df['Sentiment']])
plt.title('Sentiment Analysis of Social Media Posts')
plt.xlabel('Usernames')
plt.ylabel('Sentiment Score')
plt.show()
