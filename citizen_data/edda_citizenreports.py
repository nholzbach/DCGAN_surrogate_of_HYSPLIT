import os
import pandas as pd
import numpy as np
import pytz
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir
from copy import deepcopy
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from pandas.api.indexers import FixedForwardWindowIndexer
from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, confusion_matrix

from tqdm.notebook import tqdm 
tqdm.pandas()

from xml.sax import saxutils as su

path = "raw/"
#  set plot style
sns.set_style("whitegrid")
# set params like fontstyle, font size, etc.
sns.set_context("talk", font_scale=1.5)

#  FUNCTIONS FOR SMELL VALUE ANALYSIS    
def preprocess_smell(df):
    """
    This function is the answer of task 4.
    Preprocess smell data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The raw smell reports data.
         
    Returns
    -------
    pandas.DataFrame
        The preprocessed smell data.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # Drop the columns that we do not need.
    df = df.drop(columns=["symptoms", "smell description", "zipcode", "date & time", "skewed latitude", "skewed longitude", "additional comments"])
    
    # Select only the reports within the range of 3 and 5.
    # df = df[(df["smell value"]>=3)&(df["smell value"]<=5)]
    
    # Convert the timestamp to datetime.
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    # Resample the timestamps by hour and sum up all the future values.
    # Because we want data from the future, so label need to be "left".
    # df = df.resample("60Min", label="left").sum()
    
    # Fill in the missing data with value 0.
    df = df.fillna(0)
    return df        

# Now plotting average number of smell reports distrubted by the day of the week and the hour of the day
def is_datetime_obj_tz_aware(dt):
    """
    Find if the datetime object is timezone aware.
    
    Parameters
    ----------
    dt : pandas.DatetimeIndex
        A datatime index object.
    """
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def plot_smell_by_day_and_hour(df, figname):
    """
    Plot the average number of smell reports by the day of week and the hour of day.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed smell data.
    """
    # Copy the data frame to prevent editing the original one.
    df = df.copy(deep=True)
    
    # Convert timestamps to the local time in Pittsburgh.
    if is_datetime_obj_tz_aware(df.index):
        df.index = df.index.tz_convert(pytz.timezone("US/Eastern"))
    else:
        df.index = df.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    
    # Compute the day of week and the hour of day.
    df["day_of_week"] = df.index.dayofweek
    df["hour_of_day"] = df.index.hour
    
    # Plot the graph.
    y_l = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    df_pivot = pd.pivot_table(df, values="smell value", index=["day_of_week"], columns=["hour_of_day"], aggfunc=np.sum)
    f, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df_pivot, annot=False, cmap="Reds", fmt="g", linewidths=0.1, yticklabels=y_l, ax=ax)
    # y label
    ax.set(ylabel="Day of Week", xlabel="Hour of Day")
    plt.title("Sum of reported smell values per hour of day for each day of week in 2019")
    f.savefig(f"results/{figname}", dpi=300, bbox_inches="tight")


def plot_smell_by_season_heatmap(df, figname):
    """
    Plot the average number of smell reports by the day during each season.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed smell data.
        
    figname : str
        the name of the figure to be saved
    """
    # Copy the data frame to prevent editing the original one.
    df = df.copy(deep=True)
    
    # Convert timestamps to the local time in Pittsburgh.
    if is_datetime_obj_tz_aware(df.index):
        df.index = df.index.tz_convert(pytz.timezone("US/Eastern"))
    else:
        df.index = df.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    
    # Compute the season of the year
    df['month'] = df.index.month
    df['season'] = df['month'].apply(lambda x: 'winter' if x in [12,1,2] else ('spring' if x in [3,4,5] else ('summer' if x in [6,7,8] else 'autumn')))
    df['day'] = df.index.day
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    
    # Plot the graph.
    df_pivot = pd.pivot_table(df, values="smell value", columns=["day"], index=["season"], aggfunc=np.sum)
    # plot per month and hour of day
    df_pivot2 = pd.pivot_table(df, values="smell value", columns=["hour_of_day"], index=["month"], aggfunc=np.sum)

    # Plot the graph.
    y_l = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    df_pivot3 = pd.pivot_table(df, values="smell value", index=["day_of_week"], columns=["hour_of_day"], aggfunc=np.sum) 
    
    f, ax = plt.subplots(3,1,figsize=(16, 16), tight_layout=True)
    sns.heatmap(df_pivot, annot=False, cmap="Reds", fmt="g", linewidths=0.1, ax=ax[0])
    sns.heatmap(df_pivot2, annot=False, cmap="Reds", fmt="g", linewidths=0.1, ax=ax[1], vmin=0, vmax=1750)
    sns.heatmap(df_pivot3, annot=False, cmap="Reds", fmt="g", linewidths=0.1, yticklabels=y_l, ax=ax[2])

    
    ax[0].set(xlabel="Day of Month", ylabel="Season")
    ax[1].set(ylabel="Month", xlabel="")
    ax[2].set(xlabel="Hour of Day", ylabel="Day of Week")
    # f.tight_layout(pad=1.0)
    # f.suptitle("Sum of reported smell values per day of month for each season in 2019")
    f.savefig(f"results/{figname}")
    
def plot_smell_by_season_linegraph(df, figname, agg=np.mean):
    """
    Plot the average number of smell reports by the day during each season.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed smell data.
        
    figname : str
        the name of the figure to be saved
    """
    # Copy the data frame to prevent editing the original one.
    df = df.copy(deep=True)
    
    # Convert timestamps to the local time in Pittsburgh.
    if is_datetime_obj_tz_aware(df.index):
        df.index = df.index.tz_convert(pytz.timezone("US/Eastern"))
    else:
        df.index = df.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    
    # Compute the season of the year
    df['month'] = df.index.month
    df['season'] = df['month'].apply(lambda x: 'winter' if x in [12,1,2] else ('spring' if x in [3,4,5] else ('summer' if x in [6,7,8] else 'autumn')))
    df['day'] = df.index.day
    df["hour_of_day"] = df.index.hour
    
    # Plot the graph.
    
    fig, ax = plt.subplots(figsize=(15, 6))
    for season in ['winter', 'spring', 'summer', 'autumn']:
        df_season = df[df['season'] == season]
        df_pivot = pd.pivot_table(df_season, values="smell value", index=["hour_of_day"], aggfunc=agg)
        # df_pivot.plot(ax=ax)
        plt.plot(df_pivot, label=season)
    # f.legend(['winter', 'spring', 'summer', 'autumn'])
    plt.legend(['Winter', 'Spring', 'Summer', 'Autumn'])
    plt.xlabel("Hour of Day")
    plt.ylabel("Smell value")
    # if agg == np.mean:
    #     plt.title("Average smell value per hour of day for each season for 2019")
    # else:
    #     plt.title("Sum of all smell value per hour of day for each season for 2019") 
    plt.tight_layout()
    
    fig.savefig(f"results/{figname}", bbox_inches="tight")

def plot_count_by_season_linegraph(df, figname):
    df = df.copy(deep=True)
    
    # Convert timestamps to the local time in Pittsburgh.
    if is_datetime_obj_tz_aware(df.index):
        df.index = df.index.tz_convert(pytz.timezone("US/Eastern"))
    else:
        df.index = df.index.tz_localize(pytz.utc, ambiguous="infer").tz_convert(pytz.timezone("US/Eastern"))
    
    # Compute the season of the year
    df['month'] = df.index.month
    df['season'] = df['month'].apply(lambda x: 'winter' if x in [12,1,2] else ('spring' if x in [3,4,5] else ('summer' if x in [6,7,8] else 'autumn')))
    df['day'] = df.index.day
    df["hour_of_day"] = df.index.hour    
    fig, ax = plt.subplots(figsize=(14, 6))

    # Loop through each season
    for season in ['winter', 'spring', 'summer', 'autumn']:
        df_season = df[df['season'] == season]
        print(df_season)
        df_grouped = df_season.groupby('hour_of_day').size()
        print(df_grouped)
        plt.plot(df_grouped.index, df_grouped.values, label=season)  # Plot the count of entries for the current season

    # Customize the plot
    # plt.title("Count of Entries by Hour of Day and Season for 2019")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count of Entries")
    plt.legend(['Winter', 'Spring', 'Summer', 'Autumn'])
    plt.tight_layout()
    fig.savefig(f"results/{figname}", bbox_inches='tight')
    
# TEXT DATA PROCESSING FOR SMELL REPORTS
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def wordnet_pos(nltk_pos):
    """
    Function to map POS tags to wordnet tags for lemmatizer.
    """
    if nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN
  
def preprocess_text_data(df):
    """
    This function collects data from 
    the smell reports in order to process the text data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The raw smell reports data.
         
    Returns
    -------
    pandas.DataFrame
        The preprocessed smell text data.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)
    
    # Drop the columns that we do not need.
    df = df.drop(columns=["zipcode", "date & time","skewed latitude", "skewed longitude"])
    df["smell description"] = df["smell description"].str.lower()
    df["symptoms"] = df["symptoms"].str.lower()
    df["additional comments"] = df["additional comments"].str.lower()
    
    
    # df = df.dropna(thresh=3)
    # Select only the reports within the range of 3 and 5.
    # df = df[(df["smell_value"]>=3)&(df["smell_value"]<=5)]
    
    # Convert the timestamp to datetime.
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    # Fill in the missing data with value 0.
    # df = df.fillna(0)
    return df 

def tokenize_and_lemmatize(df, col, remove_stopwords=False):
    """
    Tokenize and lemmatize the text in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the text column.
    col : str
        Name of the column containing the text, with quotation marks
    Returns
    -------
    pandas.DataFrame
        The dataframe with the added tokens column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    df[col] = df[col].astype(str)   
    # Apply the tokenizer to create the tokens column.
    df['tokens'] = df[col].progress_apply(lambda x: word_tokenize(x))

    # Apply the lemmatizer on every word in the tokens list.
    df['tokens'] = df['tokens'].progress_apply(lambda tokens: [lemmatizer.lemmatize(token, wordnet_pos(tag))
                                                               for token, tag in nltk.pos_tag(tokens)])
   
    # remove non-words
    df['tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token.isalpha()])
    
    if remove_stopwords == True:
        # remove stopwords
        df['tokens'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stopwords.words('english')])
    
    # Create bigrams
    df["bigrams"] = df['tokens'].apply(lambda tokens: list(nltk.bigrams(tokens)))
    
    
    return df

def most_used_words(df, token_col, extent = 5):
    """
    Generate a dataframe with the 5 most used words per smell rating, and their count.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the smell rating and tokens columns.
    extent : int
        the number of most used words to be displayed (default 5)
    Returns
    -------
    pandas.DataFrame
        The dataframe with 5 rows per class, and an added 'count' column.
        The dataframe is sorted in ascending order on the class and in descending order on the count.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Explode the tokens so that every token gets its own row.
    df = df.explode(token_col)

    # Option 1: groupby on smell_value and token, get the size of how many rows per item,
    # add that as a column.
    counts = df.groupby(['smell value', token_col]).size().reset_index(name=f'count')

    # Option 2: make a pivot table based on the smell value and token based on how many
    # rows per combination there are, add counts as a column.
    # counts = counts.pivot_table(index=['smell_value', 'tokens'], aggfunc='size').reset_index(name='smell_value')

    # Sort the values on the smell value and count, get only the first 5 rows per smell value.
    counts = counts.sort_values(['smell value', 'count'], ascending=[True, False]).groupby('smell value').head(extent)

    return counts


def remove_stopwords(df, token_col):
    """
    Remove stopwords from the tokens.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing at least the tokens column,
        where the value in each row is a list of tokens.

    Returns
    -------
    pandas.DataFrame
        The dataframe with stopwords removed from the tokens column.
    """
    # Copy the dataframe to avoid editing the original one.
    df = df.copy(deep=True)

    # Using a set for quicker lookups.
    stopwords_list = stopwords.words('english')
    stopwords_set = set(stopwords_list)
    stopwords_set.add('nan')

    # Filter stopwords from tokens.
    df[token_col] = df[token_col].apply(lambda tokens: [token for token in tokens
                                                      if token.lower() not in stopwords_set])

    return df

def process_text(df, extent, filename):
    """Find the top most words and bigrams for each text column in the smell reports

    Args:
        df (_type_): year of smell reports
        extent (_type_): number of top words to be displayed
        filename (_type_): name of the file to be saved
    """
    text_data = preprocess_text_data(df)
    for col in ['smell description', 'symptoms', 'additional comments']:
        tokenized = tokenize_and_lemmatize(text_data, col)
        for tokencol in ['tokens', 'bigrams']:
            top_words = most_used_words(tokenized, tokencol, extent)

            tokenized_again = tokenize_and_lemmatize(text_data, col, remove_stopwords = True)
            top_sans_stopwords = most_used_words(tokenized_again, tokencol, extent)
            # top_bigrams_sans_stopwords = most_bigrams(tokenized, extent)
            
            print(f'top {tokencol} for {col}: {top_words}')

            print(f'top {tokencol} for {col} WITHOUT STOPWORDS: {top_sans_stopwords}')

        # write this all to a txt file
    with open(filename, 'w') as f:
        for col in ['smell description', 'symptoms', 'additional comments']:
            for tokencol in ['tokens', 'bigrams']:
                f.write(f'top {tokencol} for {col}: {top_words}')
                f.write('\n')
                f.write(f'top {tokencol} for {col} WITHOUT STOPWORDS: {top_sans_stopwords}')
                f.write('\n')

 

def main():
    """ Main function to run the script. 
    The script plots the smell reports data by day and hour, and by season, in order
    to visually find temporal patterns in the data."""

    years = [16,17,18,19, 20, 21, 22]
    for year in years:
        print("working on temporal patterns: ", year)
        df_smell = pd.read_csv(f'raw/smell_reports{year}.csv').set_index("epoch time")
        df_smell = preprocess_smell(df_smell)
        plot_smell_by_day_and_hour(df_smell, f"daily_smell_report_{year}.png")
        plot_smell_by_season_heatmap(df_smell, f"seasonal_smell_report_{year}.pdf")
        plot_smell_by_season_linegraph(df_smell, f"seasonal_smell_report_linegraph_{year}.png")
        
        # text processing
        print("working on text patterns: ", year)
        process_text(df_smell, 5, f'text_results_{year}.txt')


if __name__ == "__main__":
    main()
