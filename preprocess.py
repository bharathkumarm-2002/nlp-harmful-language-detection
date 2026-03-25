import pandas as pd

# Load dataset
df = pd.read_csv("datasets/train.csv")

# Create a binary label: 1 = any toxic class, 0 = clean
df['label'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

# Keep only required columns
final_df = df[['comment_text', 'label']]
final_df.to_csv("datasets/text_dataset.csv", index=False)

print("✅ Final dataset saved as 'text_dataset.csv'")
