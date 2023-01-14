import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter

def digit_sum(number):
    list_of_char = [int(char) for char in str(number)]
    return sum(list_of_char)

def count_vowels(text):
    count = 0
    vowels = ['a', 'e', 'i', 'o', 'u']
    for char in text.lower():
        if char in vowels:
            count += 1
    return count

def is_interlock(word_list, word1, word2):
    i_locked_word1, i_locked_word2 = '', ''
    list1, list2 = list(word1), list(word2)

    for i in range(len(list1)):
        i_locked_word1 += list1[i] + list2[i]

    for j in range(len(list2)):
        i_locked_word2 += list2[j] + list1[j]
    
    interlockness = ((i_locked_word1 in word_list) 
                     or (i_locked_word2 in word_list))
    return interlockness

def count_types(a_string):
    alpha_dict = {}
    
    whitespace = list(string.whitespace)
    punctuation = list(string.punctuation)
    numeric = list(string.digits)
    uppercase = list(string.ascii_uppercase)
    lowercase = list(string.ascii_lowercase)

    char_types = [('lowercase', lowercase), ('uppercase', uppercase), 
                  ('numeric', numeric), ('punctuation', punctuation),
                  ('whitespace', whitespace)]
    
    sorted_list = sorted(a_string, reverse=True)
    
    for dict_key, char_type_ in char_types:
        count = 0
        for i in range(len(sorted_list)):
            if sorted_list[i] not in char_type_:
                sorted_list[:] = sorted_list[i:]
                break
            else:
                count += 1
            
        alpha_dict[dict_key] = count
                
    return alpha_dict

def matmul(A, B):
    AB = []
    
    for i in range(len(A)):
        AB_row = []
        for j in range(len(B[0])):
            ij = 0
            for k in range(len(B)):
                ij += A[i][k] * B[k][j]
                
            AB_row.append(ij)
            
        AB.append(AB_row)
    return AB

def encode(text):
    no_space_text = text.replace(' ', '')
    num_rows = round(len(no_space_text) ** 0.5)
    num_cols = num_rows
    
    if num_rows * num_cols < len(no_space_text):
        num_cols += 1
        
    code_list = []
    for i in range(num_rows):
        code_list.append(no_space_text[:num_cols])
        no_space_text = no_space_text[num_cols:]
    
    code = ''
    for col in range(num_cols):
        for row in range(num_rows):
            try:
                code += code_list[row][col]
            except:
                break
        code += ' '
    
    return code.strip()

def check_brackets(str_with_brackets):
    bracket_list = ['(', ')', '[', ']', '{', '}', '<', '>']
    bracket_dict = {'(':')', '[':']', '{':'}', '<':'>'}
    
    bracket_string = ''
    for char in str_with_brackets:
        if char not in bracket_list:
            continue
        else:
            bracket_string += char

    if bracket_string[0] not in bracket_dict.keys():
        return False
    
    tracking_bracket = bracket_string[0]
    for i in range(1, len(bracket_string)):
        if bracket_string[i] in bracket_dict.keys():
            tracking_bracket += bracket_string[i]
        elif bracket_dict[tracking_bracket[-1]] == bracket_string[i]:
            tracking_bracket = tracking_bracket[:-1]
        else:
            return False        
        
    return True

def nested_sum(list_of_lists):
    sum_int = 0
    for list_ in list_of_lists:
        sum_int += sum(list_)
    return sum_int

def count_people(log):
    tracking_count = 0
    log = log.replace('\n','\t').split('\t')
    log = log[:-1]
    
    num_logs = int(len(log)/2)
    for i in range(num_logs):
        if log[2 * i] == 'IN':
            tracking_count += int(log[2 * i + 1])
        else:
            tracking_count -= int(log[2 * i + 1])
    
    return tracking_count

def next_word(text, word=None):
    text_list = text.split()
    
    word_dict = {}
    
    for a_word in set(text_list):
        
        next_words = []
        for i, item in enumerate(text_list[:-1]):
            if item is a_word:
                next_words.append(text_list[i + 1])

        next_word_dict = dict(Counter(next_words))
        
        most_freq = max(next_word_dict.values())
        for most_likely_next_word, freq in next_word_dict.items():
            if most_freq == freq:
                
                word_dict[a_word] = most_likely_next_word
              
    if word is None:
        most_likely_next_word = [(k, v) for k, v in word_dict.items()]
    elif word not in word_dict.keys():
        most_likely_next_word = ''
    else:
        most_likely_next_word = (word, word_dict[word])
    
    return most_likely_next_word

def div(a, b):
    if b == 0:
        return np.nan
    return a / b

def gen_array():
    arr = np.arange(1, 101)
    arr[arr % 3 == 0] = 0
    return arr.reshape((10,10))

def dot(arr1, arr2):
    if (type(arr1) != np.ndarray) or (type(arr2) != np.ndarray):
        raise ValueError
    arr1, arr2 = arr1.reshape(3, 3), arr2.reshape(3, 3)
    return np.dot(arr1, arr2)

def mult3d(a, b):
    return a * b[..., np.newaxis]

class ABM:
    def __init__(self):
        self.timestep = 0

    def step(self):
        self.timestep += 1

    def status(self):
        return 'step'


def step_model(model, steps):
    statuses = []
    for _ in range(steps):
        statuses.append(model.status())
        model.step()
    return statuses

class Tracker:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def get_position(self):
        return (self.lon, self.lat)

class FlightTracker(Tracker):
    def __init__(self, lon, lat, height):
        super().__init__(lon, lat)
        self.height = height

    def get_height(self):
        return self.height

class Polygon:
    def __init__(self, *sides):
        self.sides = sides

    def compute_perimeter(self):
        return sum(self.sides)

class TransitionMatrix:
    def __init__(self, arr):
        if type(arr) != np.ndarray:
            raise TypeError
        elif np.any((arr < 0) | (arr > 1)):
            raise ValueError
        self.probabilities = arr
        self.transition_matrix = arr
        
    def step(self):
        new_self = TransitionMatrix(self.transition_matrix)
        new_self.probabilities = self.probabilities * self.transition_matrix
        return new_self

def zero_crossings(readings):
    readings = np.array(readings)
    count = 0
    for i in range(1, len(readings)):
        if (readings[i - 1]) == 0 and (readings[i] == 0):
            continue
        if abs(sum(readings[i - 1:i + 1])) <= max(abs(readings[i - 1:i + 1])):
            count += 1
    return count

def outer_sum(x, y):
    x, y = np.array(x), np.array(y)
    z = x.reshape(-1, 1) + y.reshape(1, -1)
    z = z.tolist()
    return z

def peel(df):
    return df.iloc[1:-1, 1:-1]

def patch(df, upper_left, lst):

    lst = np.array(lst)
    i, j = lst.shape
    idx = dict(zip(df.index, range(df.index.size)))
    cols = dict(zip(df.columns, range(df.columns.size)))

    slice_ = df.iloc[idx[upper_left[0]]:idx[upper_left[0]] + i,
                     idx[upper_left[1]]: idx[upper_left[1]] + j].shape

    # lst = lst[:slice_[0], :slice_[1]]

    df.iloc[idx[upper_left[0]]:idx[upper_left[0]] + i,
            idx[upper_left[1]]: idx[upper_left[1]] + j] = lst

    return df

def pop_stats(province, municipality=None, census_year=2015):

    census = {1960: 'Feb-60', 1970: 'May-70', 1975: 'May-75', 1980: 'May-80',
              1990: 'May-90', 1995: 'Sep-95', 2000: 'May-00', 2007: 'Aug-07',
              2010: 'May-10', 2015: 'Aug-15'}

    df = pd.read_csv('Municipality Data - PSA.csv')

    if ((province.title() not in df['province'].to_numpy())
        or (census_year not in census.keys())):
        return None

    if municipality is not None:
        if municipality.upper() not in df['municipality'].to_numpy():
            return None
        df.set_index('municipality', inplace=True)

        return df.at[municipality.upper(), census[census_year]]

    df = df[df['is_total'] != 1]
    df = (df.groupby('province')[census[census_year]]
          .agg(['sum', 'mean', 'std']))

    return tuple(item for item in df.loc[province.title()].to_numpy())

def plot_pop(municipality):

    df = pd.read_csv('Municipality Data - PSA.csv')
    df.drop(columns=['province', 'is_total'], inplace=True)

    df.set_index('municipality', inplace=True)
    
    fig, ax = plt.subplots()
    plt.close()
    ax = df.T[[mun.upper() for mun in municipality]].plot()
    ax.legend(labels=municipality)
    plt.xticks(rotation=45)
    
    return ax

def find_max(province):
    province = province.title()

    df = pd.read_csv('Municipality Data - PSA.csv')

    if province not in df['province'].to_numpy():
        raise ValueError

    df = df[df['is_total'] != 1]
    df.drop(columns='is_total', inplace=True)

    date_dict = {}
    for i in range(2, 11):
        df.iloc[:, i] = df.iloc[:, i + 1] - df.iloc[:, i]
        date_dict[df.iloc[:, i].name] = df.iloc[:, i + 1].name
        
    df['Aug-15'] = 0
    df['max_change'] = df.iloc[:, 2:].max(axis=1)
    df['date_start'] = df.iloc[:, 2:].idxmax(axis=1)
    df['date_end'] = df['date_start']
    df.replace({'date_end': date_dict}, inplace=True)
    df = df[['province', 'municipality', 'max_change',
             'date_start', 'date_end']]

    df = (df.sort_values('max_change', ascending=False)
          .groupby('province')
          .head(1)
          .drop(columns=['max_change'])
          .set_index('province'))
    return tuple(x for x in df.loc[province].to_numpy())

def most_populous():
    df = pd.read_csv('Municipality Data - PSA.csv')
    df = df[df['is_total'] != 1]

    return df.groupby('province')['Aug-15'].mean().nlargest(10)

def hourly_hashtag():
    df = pd.read_csv('/mnt/data/public/nowplaying-rs/nowplaying_rs_dataset'
                     '/user_track_hashtag_timestamp.csv', nrows=1_000_000)
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df.drop(columns=['user_id'], inplace=True)
    df.rename(columns={'track_id': 'count'}, inplace=True)

    return (df.groupby(['hashtag', pd.Grouper(key='created_at', freq='H')])
            .count().reset_index())

def aisle_counts():
    df = pd.read_csv('/mnt/data/public/instacart/instacart_2017_05_01'
                     '/order_products__prior.csv', nrows=1_000_000)
    aisle = pd.read_csv('/mnt/data/public/instacart'
                        '/instacart_2017_05_01/aisles.csv')
    products = pd.read_csv('/mnt/data/public/instacart'
                           '/instacart_2017_05_01/products.csv')

    aisle = (products.merge(aisle, on='aisle_id', how='right')
             .drop(columns=['product_name', 'aisle_id', 'department_id']))
    df = df.merge(aisle, on='product_id')
    return (df.groupby('aisle')['add_to_cart_order']
            .count().sort_values(ascending=False))

def from_to():
    df = pd.read_csv('/mnt/data/public/wikipedia/clickstream'
                     '/clickstream/2017-11/clickstream-enwiki-2017-11.tsv.gz',
                     nrows=1000, header=None,
                     sep='\t', compression='gzip')
    df = df.groupby([0, 1]).sum().unstack().fillna(0)
    return df.droplevel(0, axis=1)