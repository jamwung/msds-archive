import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def z_per_element(x, y):
    return np.exp(np.square(x) + np.cos(y)) + 2

def row_dot(x, y):
    return np.sum(np.matmul(x, y), axis=1)

def shrink(x):
    return np.transpose(x[::2, 1::2])

def multiplier(x, y):
    return np.multiply(x, y.reshape(-1, 1))

def double_quadrant(x):
    z = x.copy()
    rows, cols = z.shape
    z[:rows//2, :cols//2] *= 2
    return z

def init_series():
    s1 = pd.Series(np.linspace(start=1,stop=100,num=10), name='foo', 
                   index=list('abcdefghij'))
    s2 = pd.Series(np.arange(start=0,stop=10), 
                   index=list('fghijklmno'))
    s3 = pd.Series(np.linspace(start=5,stop=5, num=3, dtype=int), 
                   index=list('xyz'))
    return s1, s2, s3


def init_df(s1, s2):
    df1 = pd.DataFrame(s1, columns=[s1.name])
    df2 = pd.DataFrame({'series2': s2, 'series1': s1})
    return df1, df2

def top_plus_bottom():
    df = pd.read_csv('/mnt/data/public/retaildata/Online Retail.csv',
                     usecols = ('Quantity', 'UnitPrice'))
    
    return df.head(5).to_numpy() + df.tail(5).to_numpy()

def count_alone():
    filepath = ('/mnt/data/public/elections/comelec/voters_profile/'
                + 'philippine_2016_voter_profile_by_age_group.csv')
    df = pd.read_csv(filepath)
    
    df['alone'] = df['single'] + df['widow'] + df['legally_seperated']
    return df.sort_values(by=['alone'] ,ascending=False)

def scores_stats():
    df = pd.read_csv('/mnt/data/public/movielens/20m/'
                     'ml-20m/genome-scores.csv')
    return df['relevance'].describe()

def top_cat():
    df = pd.read_csv('/mnt/data/public/agora/Agora.csv', encoding='latin1')
    df.columns = df.columns.str.strip().str.replace(' ','_')
    return df['Category'].value_counts().head(10)

def listing_info():
    df = pd.read_csv('/mnt/data/public/insideairbnb/data.insideairbnb.com/'
                     'united-kingdom/england/london/2015-04-06/data/'
                     'listings.csv.gz', 
                     usecols=['id', 'name', 'summary', 
                              'space', 'description'], 
                     index_col='id')
    
    return df.sort_index().loc[11076:15400]

def aisle_dep():
    df = pd.read_csv('/mnt/data/public/instacart/'
                     'instacart_2017_05_01/products.csv',
                    index_col='product_id')
    
    df.loc[df['aisle_id'] == 5, 
           'product_name'] += (' (' + df['aisle_id'].astype(str) + '-'
                               + df['department_id'].astype(str)) + ')'
    return df

def camsur_reps():
    df = pd.read_csv('/mnt/data/public/elections/comelec/'
                     'congress_results/congressional_results_2013.csv')#,
                     #usecols=['name', 'votes_obtained'])
    df = df.loc[df['province_or_city'] == 'Camarines Sur']
    df = df.filter(['name', 'votes_obtained'])
    surname = df['name'].str.split(',', n=1, expand=True)[0]
    df.insert(0, 'surname', surname)
    return df[['surname', 'votes_obtained']]

def no_pop():
    df = pd.read_csv('/mnt/data/public/millionsong/AdditionalFiles/'
                     'tracks_per_year.txt',
                     delimiter='<SEP>', 
                     names=['year', 'track_id', 'artist', 'title'],
                     engine='python')
    return df.query('year < 2000 and (artist != "Britney Spears")' 
                    'and (artist != "Backstreet Boys")')

def read_trips():
    df = pd.read_csv('/mnt/data/public/nyctaxi/trip_data/trip_data_1.csv',
                     nrows=100, 
                     parse_dates=['pickup_datetime', 'dropoff_datetime'])
    df['rate_code'] = df['rate_code'].astype('string') 
    return df
    
def write_trips(df_trips):
    columns = ['pickup_longitude', 'pickup_latitude',
               'dropoff_longitude', 'dropoff_latitude']
    df_trips[columns].to_csv('trip_coords.csv')
    return df_trips[columns]

def largest_invoice():
    s = pd.read_csv('/mnt/data/public/retaildata/Online Retail.csv',
                    usecols=['InvoiceNo'], dtype={'InvoiceNo' : str})
    s = s.pivot_table(index=['InvoiceNo'], aggfunc='size')
    s.name = 'InvoiceNo'
    return s.sort_values(ascending=False).head(10)

def most_daily_tagged_artists():
    df = pd.read_csv('/mnt/data/public/hetrec2011/lastfm/'
                     'user_taggedartists.dat', delimiter='\t',
                     parse_dates={'date': ['month', 'day', 'year']})
    df = df.groupby(['userID', 'date'])['artistID'].nunique().nlargest(10)
    return df.index.get_level_values(0).tolist()

def bin_names():
    df = pd.read_csv('/mnt/data/public/brazilian-ecommerce'
                     '/olist_products_dataset.csv')
    
    df['bins'] = pd.cut(df['product_name_lenght'], 10, precision=4)
    series = df.groupby('bins')['product_description_lenght'].median()
    
    return series

def charge_per_state():
    df = pd.read_csv('/mnt/data/public/cms-gov'
                     '/Medicare_Provider_Util_Payment_PUF_CY2013'
                     '/Medicare_Provider_Util_Payment_PUF_CY2013.txt', 
                     nrows=1_000_000, sep='\t', skiprows=[1],
                     usecols=['NPPES_PROVIDER_STATE', 
                              'AVERAGE_SUBMITTED_CHRG_AMT'],
                     engine='python')
    
    df = (df.groupby('NPPES_PROVIDER_STATE').mean())
    
    mean_per_state = df.plot.bar(figsize=(12, 8), legend=None, 
                                 ylabel='Average submitted charge amount ($)', 
                                 yticks=np.arange(0, 801, 100))
    mean_per_state.set_yticklabels(np.arange(0, 801, 100).astype(str))
    
    return mean_per_state

def voters_profile():
    df = pd.read_csv('/mnt/data/public/elections/comelec/voters_profile'
                     '/philippine_2016_voter_profile_by_provinces_and_'
                     'cities_or_municipalities_including_districts.csv')
    
    df = df.iloc[:, 3:28].drop('literacy', axis=1)
    col_order = sorted(df.columns)
    
    figure, axis = plt.subplots(5, 4, figsize=(10, 12), dpi=300, 
                                gridspec_kw={'hspace':.5, 'wspace':.4})
    
    df[col_order].hist(ax=axis, bins=10)
    figure.canvas.draw()
    plt.close()

    return figure

def standardize_ratings():
    df = pd.read_csv('/mnt/data/public/movielens/20m/ml-20m/ratings.csv',
                     nrows=1_000_000)
    z = lambda x: (x - x.mean()) / x.std()
    df['rating'] = df.groupby(['userId'])['rating'].transform(z)
                   
    return df

def user_songcount():
    df = pd.read_csv('/mnt/data/public/millionsong/taste/train_triplets.txt',
                     nrows=1_000_000, header=None, 
                     names=['userID', 'songID', 'No'], delimiter='\t')
    df['userID'] = df['userID'].str.slice(0,5)
    return df['userID'].value_counts().sort_index()

def at_least_10():
    df = pd.read_csv('/mnt/data/public/nowplaying-rs/nowplaying_rs_dataset/'
                     'user_track_hashtag_timestamp.csv',
                     nrows=1000)
    unique = df.groupby(['user_id']).nunique()
    unique = unique[unique['track_id'] >= 10].index
    return df[df['user_id'].isin(unique)]

def source_dest():
    df = pd.read_csv('/mnt/data/public/wikipedia/clickstream/clickstream/'
                     '2017-11/clickstream-enwiki-2017-11.tsv.gz', 
                     delimiter='\t', nrows=1000, header=None,
                     names=['source', 'destination'],
                     usecols=['source', 'destination'],
                     compression='gzip')
    
    series = df.groupby('source')['destination'].unique()
    
    return series

def mean_std_votes():
    df = pd.read_csv('/mnt/data/public/elections/comelec/congress_results'
                     '/congressional_results_2013.csv')
    
    stat = (df.groupby('province_or_city')['votes_obtained']
            .agg(['mean', 'std']))

    return stat

def double_work(df):
    df.loc['2021-01-01 09:00:00':'2021-01-01 17:00:00', 'a'] = 2 * df['b']
    return df

def hourly_mem_usage():
    df = pd.read_csv('mem.csv')
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    
    return (df.groupby(pd.Grouper(key='Time', freq='H', 
                                  closed='left'))['accesslab']
            .mean()
            .tz_convert('Asia/Manila'))

def daily_mem_usage():
    df = pd.read_csv('mem.csv', parse_dates=['Time'])
    
    return (df.groupby(pd.Grouper(key='Time', 
                                  freq='D', 
                                  closed='left'))['accesslab']
            .mean()
            .to_period())

def longest_distances():
    df = pd.read_csv('/mnt/data/public/nyctaxi/trip_data/trip_data_1.csv',
                     nrows=1_000_000, parse_dates=['pickup_datetime'])
    
    return df.groupby([df['pickup_datetime'].dt.hour, 
                       'passenger_count'])['trip_distance'].max()

def mean_ratings():
    df = pd.read_csv('/mnt/data/public/insideairbnb/data.insideairbnb.com'
                     '/united-kingdom/england/london/2015-04-06'
                     '/data/listings.csv.gz', 
                     usecols=['host_since', 'review_scores_rating'],
                     parse_dates=['host_since'])
    
    return df.groupby(pd.Grouper(key='host_since', 
                                 freq='M'))['review_scores_rating'].mean()

def product_aisles():
    df = pd.read_csv('/mnt/data/public/instacart/'
                     'instacart_2017_05_01/products.csv')
    
    aisle = pd.read_csv('/mnt/data/public/instacart/'
                        'instacart_2017_05_01/aisles.csv')
    
    return df.merge(aisle, how='left').set_index('product_id')

def tracks_with_loc():
    df_tracks = pd.read_csv('/mnt/data/public/millionsong/AdditionalFiles'
                            '/unique_tracks.txt',
                            names=['track_id', 'song_id', 'artist', 'title'],
                            sep='<SEP>', engine='python')
    df_artists = pd.read_csv('/mnt/data/public/millionsong/AdditionalFiles'
                             '/artist_location.txt',
                             names=['lat', 'lon', 'artist', 'location'],
                             sep='<SEP>', engine='python')
    
    df_artists.index.rename('artist_id', inplace=True)
    
    df_artists = df_artists.reset_index().drop_duplicates(subset='artist')

    return (df_tracks.merge(df_artists, on='artist', how='left')
            .sort_values('track_id'))

def party_votes():
    df = pd.read_csv('/mnt/data/public/elections/comelec/congress_results'
                     '/congressional_results_2013.csv')
    
    df = (df.groupby(['province_or_city', 'party_affiliation'])
          .sum().unstack().fillna(0).astype(int))
    df.columns = df.columns.droplevel(0)
    
    return df

def naia_traffic():
    df = pd.read_csv('/mnt/data/public/opendata/transport/caap-aircraft'
                     '/airdata_aircraft_movement_2016.csv')
    df = df[df['airport'] == 'NAIA'].drop(columns=['region', 'airport', 
                                                   'total', 'Unnamed: 16'])
    
    df = df.set_index('airline_operator').stack().reset_index()
    
    df.columns = ['airline_operator', 'month', 'passengers']
    df['passengers'] = df['passengers'].astype(int)
    df['month'] = df['month'].str.title()
    
    return df

def pudo():
    df = pd.read_csv('/mnt/data/public/nyctaxi/all/yellow_tripdata_2017-12.csv', 
                     nrows=1_000_000, usecols=['PULocationID', 'DOLocationID'])

    df['count'] = (df['PULocationID'] == (df['DOLocationID'])).astype(int)
    df = df.groupby(['PULocationID', 'DOLocationID']).count().unstack()
    df = df.fillna(0).astype(int)
    
    df.columns = df.columns.droplevel(0)
    
    return df