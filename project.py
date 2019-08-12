# Author: Yunkyu Song, Rahul Chandra
# Below are codes for CSE 163 sp19 final project that investigates the
# relationship between the player's peak age and player's various factors,
# and also the relationship between the player's peak PER and the various
# factors
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


sns.set()
pd.set_option('display.max_columns', None)


def join_csv(left_file, right_file, left_key, right_key):
    """
    Takes dataframes path(path to the files), left_file(name of the file),
    right_file(name of the file), left_key(attribute on the left file to join
    on), and right_key(attribute on the left file to join on) as parameters.
    Returns the joined file.
    """
    joined = left_file.merge(right_file, left_on=left_key, right_on=right_key)
    return joined


def filter_after_year(data, year, column_name):
    """
    Takes a dataframe data and int year as parameters.
    Returns a dataframe that contains the data after specified year.
    """
    return data[data[column_name] >= year]


def avg_attribute(data, attr):
    """
    Takes a dataframe data and column(quantitative) name String attr as
    parameters.
    Returns an average value for the attribute.
    """
    return data[attr].mean()


def check_null(attrs, data):
    """
    Takes a dataframe data and list of Strings attrs as parameters.
    Returns a tuple of a dataframe, which is a filtered data that drops all
    rows having NaN values of columns in attrs, and a float that shows how much
    data is left in percentage compared to the original data.
    """
    total = len(data)
    data = data.dropna(subset=attrs)
    return (data, 100 * len(data) / total)


def plot_position_to_peak(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between position and peak age of the players in the data
    using scatter plot.
    """
    sns.catplot(x='position', y='peak age', hue='position', data=data)
    plt.savefig('plot_position_to_peak.png')
    plt.show()


def plot_height_to_peak(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between height and peak age of the players in the data
    using line plot.
    """
    sns.relplot(x='height', y='peak age', data=data, kind="line")
    plt.xticks(rotation=-45)
    plt.savefig('plot_height_to_peak.png')
    plt.show()


def plot_weight_to_peak(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between weight and peak age of the players in the data
    using line plot.
    """
    sns.relplot(x='weight', y='peak age', data=data, kind="line")
    plt.savefig('plot_weight_to_peak.png')
    plt.show()


def plot_position_to_PER(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between position and peak PER of the players in the data
    using scatter plot.
    """
    sns.catplot(x='position', y='peak PER', hue='position', data=data)
    plt.savefig('plot_position_to_PER.png')
    plt.show()


def plot_height_to_PER(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between weight and peak PER of the players in the data
    using line plot.
    """
    sns.relplot(x='height', y='peak PER', data=data, kind="line")
    plt.xticks(rotation=-45)
    plt.savefig('plot_height_to_PER.png')
    plt.show()


def plot_weight_to_PER(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between weight and peak PER of the players in the data
    using line plot.
    """
    sns.relplot(x='weight', y='peak PER', data=data, kind="line")
    plt.savefig('plot_weight_to_PER.png')
    plt.show()


def plot_weight_to_decline(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between weight and decline of the players in the data
    using line plot.
    """
    sns.relplot(x='weight', y='decline measure', data=data, kind="line")
    plt.savefig('plot_weight_to_decline.png')
    plt.show()


def plot_height_to_decline(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between weight and decline of the players in the data
    using line plot.
    """
    sns.relplot(x='height', y='decline measure', data=data, kind="line")
    plt.xticks(rotation=-45)
    plt.savefig('plot_height_to_decline.png')
    plt.show()


def plot_position_to_decline(data):
    """
    Takes a dataframe data as a parameter.
    Plots a relation between position and decline of the players in the data
    using scatter plot.
    """
    sns.catplot(x='position', y='decline measure', hue='position', data=data)
    plt.savefig('plot_position_to_decline.png')
    plt.show()


def main():
    # Reads csv file necessary to compute average all star age, and csv file
    # of players, which is required throughout the project.
    all_star_data = pd.read_csv(
       'mens-professional-basketball/basketball_player_allstar.csv')
    player_data = pd.read_csv('nba-players-stats/player_data.csv')

    # Creates a new column 'real_name' in order to join it with player_data.
    all_star_data['real_name'] = all_star_data[
       'first_name'] + ' ' + all_star_data['last_name']
    all_star_data = join_csv(all_star_data, player_data, 'real_name', 'name')

    # Filters player's data before 1980.
    all_star_data = filter_after_year(all_star_data, 1980, 'season_id')

    # Creates a new column 'birth_year' to compute age for each player, and
    # computes the average all star age.
    all_star_data['birth_year'] = all_star_data['birth_date'].str[-4:]
    all_star_data['all_star_age'] = all_star_data[
       'season_id'].astype(int) - all_star_data['birth_year'].astype(int)
    all_star_average = avg_attribute(all_star_data, 'all_star_age')
    print("Average age for All-Star Selection:" + str(all_star_average))

    # Reads a csv file necessary to compute average MVP age.
    awards_data = pd.read_csv(
       'mens-professional-basketball/basketball_awards_players.csv')

    # Filters irrelevant awards.
    mvp_data = awards_data[awards_data['award'] == 'Most Valuable Player']
    mvp_data = join_csv(all_star_data, mvp_data, 'player_id', 'playerID')
    mvp_data = filter_after_year(mvp_data, 1980, 'season_id')

    # Creates a new column 'birth_year' to compute age for each player, and
    # computes the average average MVP age.
    mvp_data['birth_year'] = mvp_data['birth_date'].str[-4:]
    mvp_data['mvp_age'] = mvp_data['year'].astype(int) - mvp_data[
       'birth_year'].astype(int)
    mvp_average = avg_attribute(mvp_data, 'mvp_age')
    print("Average age for Most Valuable Player:" + str(mvp_average))

    # Reads a csv file necessary to access players' statistics throughout the
    # basketball history.
    season_data = pd.read_csv('nba-players-stats/Seasons_Stats.csv')
    season_data = season_data[season_data['MP'] >= 1500]
    player_data = join_csv(player_data, season_data, 'name', 'Player')

    # Filters player's data before 1980.
    player_data = filter_after_year(player_data, 1980, 'Year')

    # Finds the peak PER and the year and the age when a player achieved it for
    # each player.
    player_peak_PER = player_data.loc[
       player_data.groupby('name')['PER'].idxmax()]
    player_peak_PER = player_peak_PER[['name', 'Year', 'Age', 'PER']]
    player_peak_PER = player_peak_PER.rename(
       columns={'Year': 'peak year', 'Age': 'peak age', 'PER': 'peak PER'})
    player_data = join_csv(player_data, player_peak_PER, 'name', 'name')

    # Filters the players whose data doesn't have PERs until 4 years after
    # their peak years.
    player_data = player_data[((
       player_data['peak year']+4) < (player_data['year_end']))]
    player_data['birth_year'] = player_data['birth_date'].str[-4:]

    # Creates a new column 'beginning_age' that stores the age when each player
    # began his career.
    player_data['beginning_age'] = player_data[
       'year_start'].astype(int) - player_data['birth_year'].astype(int)

    # Uses decision tree regressor and factors contained in attrs
    # (excluding peak age) in order to make a model that predicts player's peak
    # age and a model that predicts player's decline.
    model = DecisionTreeRegressor()

    # Filters the data to only contain players' statistics when they began
    # their careers, and remove duplicates.
    player_data_predict_peak = player_data[
       player_data['Year'] == player_data['year_start']]
    player_data_predict_peak = player_data_predict_peak.drop_duplicates([
       'name', 'year_start', 'year_end'], keep='first')
    player_data_predict_peak = player_data_predict_peak[
       ['height', 'weight', 'position', 'beginning_age', 'peak age']]
    attrs = ['height', 'weight', 'position', 'beginning_age', 'peak age']

    player_data_predict_peak = check_null(attrs, player_data_predict_peak)[0]

    X = player_data_predict_peak.drop('peak age', axis=1)
    X = pd.get_dummies(X)
    Y = player_data_predict_peak[['peak age']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Prints MSE of the model that predicts peak age.
    print('Training MSE for Peak Age:', mean_squared_error(
       y_train, y_train_pred))
    print('Test MSE for Peak Age:', mean_squared_error(y_test, y_test_pred))

    # Creates new columns 'peak_n PER's, which represents PER of a player
    # n years after the player's peak. 1 <= n <= 4.
    a1 = player_data[player_data['Year'] == player_data['peak year'] + 1][
       ['name', 'PER']]
    a1 = a1.rename(columns={'PER': 'peak_1 PER'})
    a2 = player_data[player_data['Year'] == player_data['peak year'] + 2][
       ['name', 'PER']]
    a2 = a2.rename(columns={'PER': 'peak_2 PER'})
    a3 = player_data[player_data['Year'] == player_data['peak year'] + 3][
       ['name', 'PER']]
    a3 = a3.rename(columns={'PER': 'peak_3 PER'})
    a4 = player_data[player_data['Year'] == player_data['peak year'] + 4][
       ['name', 'PER']]
    a4 = a4.rename(columns={'PER': 'peak_4 PER'})
    player_data = join_csv(player_data, a1, 'name', 'name')
    player_data = join_csv(player_data, a2, 'name', 'name')
    player_data = join_csv(player_data, a3, 'name', 'name')
    player_data = join_csv(player_data, a4, 'name', 'name')

    # Creates a new column 'min_peak_n_PER,' which stores a minimum of peak_n
    # PER values.
    m = player_data[['peak_1 PER', 'peak_2 PER', 'peak_3 PER', 'peak_4 PER']]
    player_data['min_peak_n_PER'] = m.min(axis=1)

    # Creates a new column 'decline measure,' which estimates players' decline.
    player_data['decline measure'] = 4 * player_data['peak PER'] - (
                                          player_data['peak PER'] / 2 +
                                          player_data['peak_1 PER'] +
                                          player_data['peak_2 PER'] +
                                          player_data['peak_3 PER'] +
                                          player_data['peak_4 PER'] / 2)

    # Filters the data to only contain players' statistics when they peaked
    # their careers, and remove duplicates.
    player_data_predict_decline = player_data[
       player_data['Year'] == player_data['peak year']]
    player_data_predict_decline = player_data_predict_decline.drop_duplicates(
       ['name', 'year_start', 'year_end'], keep='first')
    player_data_predict_decline = player_data_predict_decline[
       ['height', 'weight',
        'position', 'beginning_age', 'peak age', 'peak PER',
        'decline measure',
        'USG%', 'FG%', 'FG', '3P', 'AST', 'STL', 'BLK', 'VORP', '3PA',
        'WS', 'BPM']]
    attrs = ['height', 'weight',
             'position', 'beginning_age', 'peak age', 'peak PER',
             'decline measure',
             'USG%', 'FG%', 'FG', '3P', 'AST', 'STL', 'BLK', 'VORP', '3PA',
             'WS', 'BPM']
    player_data_predict_decline = check_null(attrs,
                                             player_data_predict_decline)[0]

    X = player_data_predict_decline.drop('decline measure', axis=1)
    X = pd.get_dummies(X)
    Y = player_data_predict_decline[['decline measure']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Prints MSE of the model that predicts decline.
    print('Train MSE for Decline Measure:', mean_squared_error(y_train,
                                                               y_train_pred))
    print('Test MSE for Decline Measure:', mean_squared_error(
       y_test, y_test_pred))

    # Plots graphs that shows relationships between peak year and factors
    # (position, height, weight).
    plot_position_to_peak(player_data)
    plot_height_to_peak(player_data)
    plot_weight_to_peak(player_data)

    # Plots graphs that shows relationships between peak PER and factors
    # (position, height, weight).
    plot_position_to_PER(player_data)
    plot_height_to_PER(player_data)
    plot_weight_to_PER(player_data)

    # Plots graphs that shows relationships between decline and factors
    # (position, height, weight).
    plot_position_to_decline(player_data)
    plot_height_to_decline(player_data)
    plot_weight_to_decline(player_data)


if __name__ == "__main__":
    main()
