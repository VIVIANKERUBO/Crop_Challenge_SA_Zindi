import utils
from utils import load_df

for c in collections:
    download(c)

train_df = load_df('ref_south_africa_crops_competition_v1_train_labels')
test_df = load_df('ref_south_africa_crops_competition_v1_test_labels')
