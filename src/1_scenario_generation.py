import pandas as pd
import numpy as np
# import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import Counter

# Path
path_gitrepo = r'C:/Users/20190285/Documents/GitHub/energycommunityplanning/'
path_out = path_gitrepo + 'data/in/'
version = 'demo'

file_RCmodels = 'https://github.com/JulienLeprince/greybrickbuildings/blob/main/data/calibrated_models.csv?raw=true'
df_RC = pd.read_csv(file_RCmodels, index_col='identification_number')
df_RC = df_RC[df_RC['nCPBES'] < 0.003]  # filtering RC models based on nCPBES values - here covering good and close model fits

# Random inputs
buildings = list(df_RC.index)
horizon = 24*365*2

random_data = np.random.rand(horizon, 2)
tidx = pd.date_range('2022-01-01', periods=horizon, freq='H')
df_weather = pd.DataFrame(random_data, columns=['T_a', 'Q_sol'], index=tidx)
random_data = np.random.rand(horizon, 2)
df_price = pd.DataFrame(random_data, columns=['Day-ahead Price [EUR/kWh]', 'gas_price [EUR/kWh]'], index=tidx)
df_building = dict()
for blg in buildings:
    random_data = np.random.rand(horizon, 2)
    df_building[blg] = pd.DataFrame(random_data, columns=['E_elec', 'T_blg_set'], index=tidx)

# Join everything under one dataframe
df_all = df_weather.join(df_price)
for blg in df_building:
  df_all = df_all.join(df_building[blg].rename(columns={'E_elec':str(blg)+'_E_elec', 'T_blg_set':str(blg)+'_T_blg_set'}))

# Bootstrapping
nb_boot = 3           # number of scenarios that will be bootstrapped (here corresponding to the number of years)
nb_of_representative_scenarios = 2          # final number of representative scenarios kept

def bootstrap(df_input,
              n_boot: int):
    """A boostrap sampling function.
    Bootstrap aggregation (also known as bagging) was proposed by Leo Breiman in 1994; it is a model aggregation
    technique to reduce model variance. The training data is split into multiple samples with a replacement called
    bootstrap samples. Bootstrap sample size will be the same as the original sample size, with 3/4 of the original
    values and replacement resulting in repetition of values"""

    bootstrap_samples = []
    horizon = df_input.shape[0]
    for i in range(n_boot):
        bootsample_index = np.random.choice(df_input.index.values, size=horizon, replace=True)
        bootsample = df_input.loc[bootsample_index]
        bootstrap_samples.append(bootsample.T.values)
    return np.array(bootstrap_samples)


def seasonal_selection_function(block: pd.DataFrame,
                                blocks_to_sample_from: pd.DataFrame) -> np.array:
    """Function to identify index values with similar seasonal characteristics as the input block DataFrame.
    The conditions include (1) a selection of similar weeks of the year +- 4 weeks,
                           (2) a week day / week-end selection."""

    # Week conditional - selecting blocks with similar seasonal conditions
    week = block.index.isocalendar().week[0]
    week_conditions1 = np.array([blocks_to_sample_from.index.isocalendar().week >= (week - 4),
                                 blocks_to_sample_from.index.isocalendar().week <= (week + 4)])
    week_conditions2 = np.array([blocks_to_sample_from.index.isocalendar().week >= (week - 4 + 53),
                                 blocks_to_sample_from.index.isocalendar().week <= (week + 4 - 53)])
    week_conditions = np.array([week_conditions1.min(axis=0),
                                week_conditions2.max(axis=0)]).max(axis=0)
    # Day of week conditional - selecting blocks with similar week-day / week-end conditions
    dayofweek = block.index.dayofweek[0]
    weekday, weekend = [0, 1, 2, 3, 4], [5, 6]
    weekdaytype = weekday if dayofweek in weekday else weekend
    # Combining conditions
    seasonal_conditions = np.array([[day in weekdaytype for day in blocks_to_sample_from.index.dayofweek],
                                    week_conditions])
    seasonal_selection = seasonal_conditions.min(axis=0)
    return seasonal_selection


def seasonal_block_bootstrap(df: pd.DataFrame,
                             n_boot: int,
                             block_length: int = 24):
    """A boostrap sampling function with seasonal subgroup block sampling"""

    horizon = df.shape[0]
    for i in range(n_boot):

        j = 0
        while j < horizon:
            # Block identifications
            block = df.iloc[j:(j + block_length)]
            blocks_to_sample_from = df[~df.index.isin(block.index)]

            # Identification of subsample from seasonal characteristics
            seasonal_selection = seasonal_selection_function(block, blocks_to_sample_from)
            seasonal_block_subsample = blocks_to_sample_from[seasonal_selection]
            dates_to_sample_from = list(set(seasonal_block_subsample.index.date))

            # Block sampling from identified subsample
            random_date_selection = np.random.choice(dates_to_sample_from, size=1)[0]
            bootsample = df.loc[df.index.date == random_date_selection]

            # Adding block to bootstrap sample
            bootstrap_samples = bootsample.T.values if j == 0 \
                else np.append(bootstrap_samples, bootsample.T.values, axis=1)
            j += block_length

        n_bootstrap_samples = [bootstrap_samples] if i == 0 \
            else np.append(n_bootstrap_samples, [bootstrap_samples], axis=0)
    return n_bootstrap_samples


# Generating scenarios from seasonal block bootstrap
bootstrap_scenarios = seasonal_block_bootstrap(df_all, n_boot=nb_boot, block_length=24)    # bootstrap_scenarios shape = [N_boot, n_cols , horizon]

# Reducing scenarios to fewer representative ones with K-medoid clustering
X = StandardScaler().fit_transform(bootstrap_scenarios[:, 1, :])
diss = euclidean_distances(X)
medoids = KMedoids(n_clusters=nb_of_representative_scenarios, metric='precomputed')
medoids.fit(diss)
medoids.labels_

# Extracting scenario probabilities
counts = Counter(medoids.labels_)
prob = dict()
for key in counts:
  prob[key] = counts[key]/len(medoids.labels_)   #np.shape(diss_per_bootsample)[0]

# Save information
representative_scenarios = bootstrap_scenarios[medoids.medoid_indices_,:,:]
scenario_probabilities = pd.DataFrame(data=prob.values(), index=prob.keys(), columns=['probabilities'])

# Save information
for i, scenario in enumerate(representative_scenarios):
  data = pd.DataFrame(scenario.transpose(), columns=df_all.columns)
  data.to_csv(path_out + 'scenario_'+ str(i) +'.csv')
scenario_probabilities.to_csv(path_out + 'scenario_probabilities.csv')


# Distance matrix - climate-occ - remove eco
list_to_loop_over = [x for x in range(len(buildings)*2+4)]
list_to_loop_over.remove(2)
list_to_loop_over.remove(3)
diss_weather = np.array([cdist(StandardScaler().fit_transform(representative_scenarios[:,x,:]),
                               StandardScaler().fit_transform(representative_scenarios[:,y,:]), 'euclidean')
                         for (x, y) in zip(list_to_loop_over, list_to_loop_over)])
diss = sum(diss_weather)
medoids = KMedoids(n_clusters=1, metric='precomputed')
medoids.fit(diss)
print('The cluster center of the climate & occupant uncertainty scenarios is: ', medoids.medoid_indices_[0])

# Distance matrix - eco-climate - remove occ
list_to_loop_over = [0,1,2,3]
diss_eco = np.array([cdist(StandardScaler().fit_transform(representative_scenarios[:,x,:]),
                           StandardScaler().fit_transform(representative_scenarios[:,y,:]), 'euclidean')
                     for (x, y) in zip(list_to_loop_over, list_to_loop_over)])
diss = sum(diss_eco)
medoids = KMedoids(n_clusters=1, metric='precomputed')
medoids.fit(diss)
print('The cluster center of the climate & economic uncertainty scenarios is: ', medoids.medoid_indices_[0])

# Distance matrix - eco-occ - remove climate
list_to_loop_over = [x for x in range(len(buildings)*2+4)]
list_to_loop_over.remove(0)
list_to_loop_over.remove(1)
diss_occ = np.array([cdist(StandardScaler().fit_transform(representative_scenarios[:,x,:]),
                           StandardScaler().fit_transform(representative_scenarios[:,y,:]), 'euclidean')
                     for (x, y) in zip(list_to_loop_over, list_to_loop_over)])
diss = sum(diss_occ)
medoids = KMedoids(n_clusters=1, metric='precomputed')
medoids.fit(diss)
print('The cluster center of the occupant & economic uncertainty scenarios is: ', medoids.medoid_indices_[0])
