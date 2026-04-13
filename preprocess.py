import pandas as pd
import ast
import re
import ftfy
from sklearn.model_selection import train_test_split

INPUT_PATH         = r"C:\Users\your_path"
OUTPUT_TRAIN       = "train.csv"
OUTPUT_VAL         = "val.csv"
OUTPUT_TEST        = "test.csv"

MIN_WORD_COUNT     = 5
MIN_ASPECT_COUNT   = 250
TRAIN_RATIO        = 0.8
VAL_RATIO          = 0.1
TEST_RATIO         = 0.1
RANDOM_SEED        = 42
 

ASPECT_MAP = {
    "app.performance"        : "app performance",
    "brand.satisfaction"     : "brand satisfaction",
    "experience.ease-of-use" : "ease of use",
    "pricing.value"          : "pricing value",
    "delivery.speed"         : "delivery speed",
    "support.attitude"       : "support attitude",
    "food.quality"           : "food quality",
    "brand.competitor"       : "competitor comparison",
    "pricing.discounts"      : "pricing discounts",
    "account.access"         : "account access",
    "service.response-time"  : "service response time",
    "product.design"         : "product design",
}
 

df = pd.read_csv(INPUT_PATH)
print(f"[1] Loaded: {df.shape}")
 

text_label_groups = df.groupby('text')['labels'].apply(set)
conflicting_texts = text_label_groups[text_label_groups.apply(len) > 1].index
df = df[~df['text'].isin(conflicting_texts)]
print(f"[2] After dropping conflicting duplicates: {df.shape}")
 

df = df.drop_duplicates(subset=['text'], keep=False)
print(f"[3] After dropping non-conflicting duplicates: {df.shape}")
 

df = df[~df['text'].str.contains(r'ORG\d+', regex=True)]
print(f"[4] After dropping brand mention texts: {df.shape}")
 


df['text'] = df['text'].apply(ftfy.fix_text)
print(f"[5] Mojibake fixed")
 

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F9FF"
    u"\U00002700-\U000027BF"
    u"\U0001FA00-\U0001FA6F"
    u"\U00002500-\U00002BEF"
    u"\U00010000-\U0010FFFF"
    "]+", flags=re.UNICODE)
df['text'] = df['text'].apply(lambda x: emoji_pattern.sub('', x))
print(f"[6] Emojis stripped")
 

df['text'] = df['text'].apply(lambda x: re.sub(r' +', ' ', x).strip())
print(f"[7] Whitespace normalized")
 

df = df[df['text'].apply(lambda x: len(x.split()) >= MIN_WORD_COUNT)]
print(f"[8] After dropping texts < {MIN_WORD_COUNT} words: {df.shape}")
 

df = df.drop(columns=['industry', 'data_source'])
print(f"[9] Dropped unused columns")
 

unique_texts = df['text'].unique()
train_texts, temp_texts = train_test_split(
    unique_texts, test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED
)
val_texts, test_texts = train_test_split(
    temp_texts, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED
)
train_df = df[df['text'].isin(train_texts)].copy()
val_df   = df[df['text'].isin(val_texts)].copy()
test_df  = df[df['text'].isin(test_texts)].copy()
print(f"[10] Split (sentence-level): train={len(train_df)} val={len(val_df)} test={len(test_df)} rows (pre-explode)")
 

def explode_labels(df):
    df = df.copy()
    df['labels_parsed'] = df['labels'].apply(ast.literal_eval)
    df = df.explode('labels_parsed').reset_index(drop=True)
    df[['aspect', 'sentiment']] = df['labels_parsed'].str.rsplit('.', n=1).apply(pd.Series)
    df = df.drop(columns=['labels', 'labels_parsed'])
    return df
 
train_df = explode_labels(train_df)
val_df   = explode_labels(val_df)
test_df  = explode_labels(test_df)
print(f"[11] After exploding: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
 

aspect_counts = train_df['aspect'].value_counts()
valid_aspects = aspect_counts[aspect_counts >= MIN_ASPECT_COUNT].index
dropped_aspects = aspect_counts[aspect_counts < MIN_ASPECT_COUNT].index.tolist()
print(f"[12] Dropping aspects: {dropped_aspects}")
 
train_df = train_df[train_df['aspect'].isin(valid_aspects)].reset_index(drop=True)
val_df   = val_df[val_df['aspect'].isin(valid_aspects)].reset_index(drop=True)
test_df  = test_df[test_df['aspect'].isin(valid_aspects)].reset_index(drop=True)
print(f"     After: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
 

train_df['aspect'] = train_df['aspect'].map(ASPECT_MAP)
val_df['aspect']   = val_df['aspect'].map(ASPECT_MAP)
test_df['aspect']  = test_df['aspect'].map(ASPECT_MAP)
print(f"[13] Aspects mapped to natural language")
 

sentiment_map = {-1: 0, 1: 1}
train_df['sentiment'] = train_df['sentiment'].astype(int).map(sentiment_map)
val_df['sentiment']   = val_df['sentiment'].astype(int).map(sentiment_map)
test_df['sentiment']  = test_df['sentiment'].astype(int).map(sentiment_map)
print(f"[14] Sentiment remapped: -1→0, 1→1")
 

print(f"\n=== FINAL SHAPES ===")
print(f"Train : {train_df.shape}")
print(f"Val   : {val_df.shape}")
print(f"Test  : {test_df.shape}")
 
print(f"\n=== TRAIN ASPECT DISTRIBUTION ===")
print(train_df['aspect'].value_counts().to_string())
 
print(f"\n=== TRAIN SENTIMENT DISTRIBUTION ===")
print(train_df['sentiment'].value_counts())
 
print(f"\n=== TRAIN PER-ASPECT SENTIMENT IMBALANCE ===")
ct = train_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
ct.columns = ['neg', 'pos']
ct['ratio_pos_neg'] = (ct['pos'] / ct['neg']).round(2)
print(ct.sort_values('ratio_pos_neg').to_string())
 
print(f"\n=== NULL CHECK ===")
print("Train:", train_df.isnull().sum().to_dict())
print("Val  :", val_df.isnull().sum().to_dict())
print("Test :", test_df.isnull().sum().to_dict())
 
print(f"\n=== LEAKAGE CHECK (no sentence should appear in more than one split) ===")
train_set = set(train_df['text'])
val_set   = set(val_df['text'])
test_set  = set(test_df['text'])
print(f"Train ∩ Val  : {len(train_set & val_set)}")
print(f"Train ∩ Test : {len(train_set & test_set)}")
print(f"Val   ∩ Test : {len(val_set & test_set)}")
 

train_df.to_csv(OUTPUT_TRAIN, index=False)
val_df.to_csv(OUTPUT_VAL,     index=False)
test_df.to_csv(OUTPUT_TEST,   index=False)
print(f"\n[16] Saved: {OUTPUT_TRAIN}, {OUTPUT_VAL}, {OUTPUT_TEST}")
