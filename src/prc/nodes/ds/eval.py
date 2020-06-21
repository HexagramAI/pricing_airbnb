import numpy as np
import pandas as pd
from scipy import stats
import datetime
import logging
logger = logging.getLogger(__name__)
from kedro.framework.context import load_context
context = load_context('/Users/neil/Desktop/Hui_Yuan/Projects/pricing_airbnb-master', env='base')
catalog = context.catalog
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
from functools import partial
from scipy.optimize import minimize_scalar
from prc.nodes.ds import embedding_model

### step 0 : clean data
def load_data():
    df = catalog.load("fea_output")
    df = df.set_index(["listing_id", "date"])
    return df

### step 1: process data
def process_raw_data(df=None,
                     continuous_col = ["price", "minimum_nights", "lag", "number_of_reviews","calculated_host_listings_count"],
                     embedding_cols = ["neighbourhood_group", "room_type", "month"]):
    """



    :param df:
    :param continuous_col: ["price", "minimum_nights", "lag",...], the first element must be price.
    :param embedding_cols: the categorical data column
    :return:
    """
    df_select = df[continuous_col + embedding_cols + ["available"]]
    df_select = df_select.dropna(subset=embedding_cols + continuous_col, how="any")
    df_true = df_select[df_select["available"] == "t"]
    df_false = df_select[df_select["available"] != "t"]
    if len(df_true) < len(df_false):
        df_sampled = pd.concat([df_false.iloc(len(df_true)), df_true], axis=0)
    else:
        df_sampled = pd.concat([df_true.iloc(len(df_false)), df_false], axis=0)

    embed_lookup, x_embed_raw = embedding_model.build_embedding_dic(df_sampled, embedding_cols)
    return df_sampled, continuous_col, embedding_cols, embed_lookup, x_embed_raw

### step 2: train_model
def train_model(df, continuous_col, embed_lookup, x_embed_raw, epochs):
    num_dimensions = len(continuous_col)
    num_samples = len(df)
    model = embedding_model.build_model(num_dimensions, num_samples, embed_lookup, lr=0.001)
    x_val = df[continuous_col].values
    X = [x_val, x_embed_raw[0], x_embed_raw[1], x_embed_raw[2]]
    y = (df["available"] == "t").astype(int)
    model.fit(X, y, epochs=epochs)
    # the parameter of w and bias
    candidate_w_bs = [(model.layers[-1].kernel_posterior.sample().numpy(),
                       model.layers[-1].bias_posterior.sample().numpy())]

    return model

### estimate best rental price
### step3.1 deal with data
def series_row(row, continuous_col, embed_lookup):
    continuous = np.array(row[continuous_col].values.tolist()).reshape(-1, len(continuous_col))
    input_row = [continuous]
    for key, val in embed_lookup.items():
        input_row.append(np.array([val[row[key]]]))
    return input_row


### step3.2 calc revenue
def cal_neg_revnue(price, input_model, row, continuous_col):
    """ only change price to figure out the accept rate """
    continuous = np.append(np.array(price), row[0][0][1:]).reshape(-1, len(continuous_col))
    new_price_row = [continuous]
    for i in range(1, len(row)):
        new_price_row.append(row[i])
    prob = input_model.predict(new_price_row)
    return -1 * price * prob[0][0]


### step3.3 estimate price and prob
def optimal_sellable_capacity(model=None, row=None, method="Bounded",
                              min_search_bound=1, max_search_bound=10,
                              embed_lookup=None, continuous_col=None):
    func = partial(
        cal_neg_revnue,
        input_model=model,
        row=row,
        embed_lookup=embed_lookup,
        continuous_col=continuous_col

    )

    min_ = minimize_scalar(
        func, bounds=(min_search_bound, max_search_bound), method=method
    )

    best_price = min_.x

    x0 = np.append(np.array(best_price), row[0][0][1:]).reshape(-1, len(continuous_col))
    new_price_row = [x0]
    for i in range(1, len(row)):
        new_price_row.append(np.array(row[i]))
    prob = model.predict(new_price_row)
    return best_price, prob[0][0]


### step3.4 combined
def wrapper_price_func(model=None, row=None, embedding_cols = None, continuous_col=None, embed_lookup=None,
                       min_search_bound=1, max_search_bound=10):
    input_row = series_row(row, embedding_cols, continuous_col, embed_lookup)
    est_price, prob = optimal_sellable_capacity(model=model, row=input_row,
                                                min_search_bound=min_search_bound,
                                                max_search_bound=max_search_bound,
                                                continuous_col=continuous_col,
                                                embed_lookup=embed_lookup)
    return est_price, prob


### step 3.5 apply to dataframe
def run_estimate_price(df, model, continuous_col, embedding_cols, embed_lookup):
    df_sampled_new = df.reset_index(level=['listing_id', 'date'])
    index_series = df_sampled_new[['listing_id', 'date']]
    df_sampled_new = df_sampled_new[continuous_col + embedding_cols + ['listing_id', 'date']]

    ### select sample to test
    ### date:'2019-05-01' to '2019-6-01' inclusive
    ### test first 20 rows
    sample_num = 20
    df_sampled_new = df_sampled_new[df_sampled_new['date'].between('2019-05-01', '2019-6-01', inclusive=True)]
    df_new = df_sampled_new.iloc[:sample_num, :-2].reset_index(drop=True)
    # estimate best rental price
    df_new['price_prob'] = df_test.apply(lambda x: wrapper_price_func(min_search_bound=1, max_search_bound=10,
                                                                      continuous_col=continuous_col,
                                                                      embed_lookup=embed_lookup,
                                                                      model=model, row=x)
                                         , axis=1)
    df_new['estimate_price'] = df_new['price_prob'].apply(lambda x: '$' + str(round(np.exp(x[0]), 2)))
    df_new['prob'] = df_new['price_prob'].apply(lambda x: str(round(x[1] * 100, 2)) + '%')
    df_new = df_new.drop("price_prob", axis=1)

    # get back log columns
    log_cols = ["price", "minimum_nights", "lag"]
    df_new[log_cols] = np.exp(df_new[log_cols])

    # change price, minimum_nights, lag format
    df_new['price'] = df_new['price'].apply(lambda x: '$' + str(round(x, 2)))
    df_new['minimum_nights'] = df_new['minimum_nights'].apply(lambda x: int(x))
    df_new['lag'] = df_new['lag'].apply(lambda x: int(x))

    # reset index
    index_new_listId = index_series.iloc[:sample_num, 0]
    index_new_date = index_series.iloc[:sample_num, 1]
    df_new = df_new.set_index([index_new_listId, index_new_date])
    return df_new


class pricing_airbnb():
    def __init__(self,
                 continuous_col=["price", "minimum_nights", "lag", "number_of_reviews", "calculated_host_listings_count"],
                 embedding_cols=["neighbourhood_group", "room_type", "month"],
                 epochs=10):
        self.continuous_col = continuous_col
        self.embedding_cols = embedding_cols
        self.epochs = epochs

    def load_data(self):
        df = catalog.load("fea_output")
        df = df.set_index(["listing_id", "date"])
        return df

    def data_process(self,df):
        df_select = df[self.continuous_col + self.embedding_cols + ["available"]]
        df_select = df_select.dropna(subset=self.embedding_cols + self.continuous_col, how="any")
        df_true = df_select[df_select["available"] == "t"]
        df_false = df_select[df_select["available"] != "t"]
        if len(df_true) < len(df_false):
            df_sampled = pd.concat([df_false.iloc(len(df_true)), df_true], axis=0)
        else:
            df_sampled = pd.concat([df_true.iloc(len(df_false)), df_false], axis=0)

        embed_lookup, x_embed_raw = embedding_model.build_embedding_dic(df_sampled,  self.embedding_cols)
        return df_sampled, embed_lookup, x_embed_raw

    def train_model(self, df, embed_lookup, x_embed_raw):
        num_dimensions = len(self.continuous_col)
        num_samples = len(df)
        model = embedding_model.build_model(num_dimensions, num_samples, embed_lookup, lr=0.001)
        x_val = df[self.continuous_col].values
        X = [x_val]
        for i in range(len(self.embedding_cols)):
            X.append(x_embed_raw[i])
        y = (df["available"] == "t").astype(int)
        model.fit(X, y, epochs=self.epochs)
        # the parameter of w and bias
        candidate_w_bs = [(model.layers[-1].kernel_posterior.sample().numpy(),
                           model.layers[-1].bias_posterior.sample().numpy())]

        return model

    ### estimate best rental price
    ### step3.1 deal with data
    def series_row(self, row, embed_lookup):
        continuous = np.array(row[self.continuous_col].values.tolist()).reshape(-1, len(self.continuous_col))
        input_row = [continuous]
        for key, val in embed_lookup.items():
            input_row.append(np.array([val[row[key]]]))
        return input_row

    ### step3.2 calc revenue
    def cal_neg_revnue(self, price, input_model, test_row):
        """ only change price to figure out the accept rate """
        continuous = np.append(np.array(price), test_row[0][0][1:]).reshape(-1, len(self.continuous_col))
        new_price_row = [continuous]
        for i in range(1, len(test_row)):
            new_price_row.append(test_row[i])
        prob = input_model.predict(new_price_row)
        return -1 * price * prob[0][0]

    ### step3.3 estimate price and prob
    def optimal_sellable_capacity(self, model=None, row=None, method="Bounded",
                                  min_search_bound=1, max_search_bound=10,
                                  embed_lookup=None):
        func = partial(
            cal_neg_revnue,
            input_model=model,
            row=row,
            embed_lookup=embed_lookup,
        )

        min_ = minimize_scalar(
            func, bounds=(min_search_bound, max_search_bound), method=method
        )

        best_price = min_.x

        x0 = np.append(np.array(best_price), row[0][0][1:]).reshape(-1, len(self.continuous_col))
        new_price_row = [x0]
        for i in range(1, len(row)):
            new_price_row.append(np.array(row[i]))
        prob = model.predict(new_price_row)
        return best_price, prob[0][0]

    ### step3.4 combined
    def wrapper_price_func(self, model=None, row=None,  embed_lookup=None,
                           min_search_bound=1, max_search_bound=10):
        input_row = series_row(row, self.embedding_cols, self.continuous_col, embed_lookup)
        est_price, prob = optimal_sellable_capacity(model=model, row=input_row,
                                                    min_search_bound=min_search_bound,
                                                    max_search_bound=max_search_bound,
                                                    embed_lookup=embed_lookup)
        return est_price, prob

    ### step 3.5 apply to dataframe
    def run_estimate_price(self, df, model, embed_lookup):
        """


        :param df: run self.load_data():
        :param model:  run self.data_process(df)and self.train_model(df, embed_lookup, x_embed_raw) to get model
        :param embed_lookup: run self.data_process(df) get embed_lookup
        :return: df with estimated price and probability
        """
        df_sampled_new = df.reset_index(level=['listing_id', 'date'])
        index_series = df_sampled_new[['listing_id', 'date']]
        embed_lookup, x_embed_raw = embedding_model.build_embedding_dic(df_sampled_new, self.embedding_cols)
        df_sampled_new = df_sampled_new[self.continuous_col + self.embedding_cols + ['listing_id', 'date']]

        ### select sample to test
        ### date:'2019-05-01' to '2019-6-01' inclusive
        ### test first 20 rows
        sample_num = 20
        df_sampled_new = df_sampled_new[df_sampled_new['date'].between('2019-05-01', '2019-6-01', inclusive=True)]
        df_new = df_sampled_new.iloc[:sample_num, :-2].reset_index(drop=True)
        # estimate best rental price
        df_new['price_prob'] = df_new.apply(lambda x: wrapper_price_func(min_search_bound=1, max_search_bound=10,
                                                                          embed_lookup=embed_lookup,
                                                                          model=model, row=x)
                                             , axis=1)
        df_new['estimate_price'] = df_new['price_prob'].apply(lambda x: '$' + str(round(np.exp(x[0]), 2)))
        df_new['prob'] = df_new['price_prob'].apply(lambda x: str(round(x[1] * 100, 2)) + '%')
        df_new = df_new.drop("price_prob", axis=1)

        # get back log columns

        log_cols = ["price", "minimum_nights", "lag"]
        df_new[log_cols] = np.exp(df_new[log_cols])

        # change price, minimum_nights, lag format
        df_new['price'] = df_new['price'].apply(lambda x: '$' + str(round(x, 2)))
        df_new['minimum_nights'] = df_new['minimum_nights'].apply(lambda x: int(x))
        df_new['lag'] = df_new['lag'].apply(lambda x: int(x))

        # reset index
        index_new_listId = index_series.iloc[:sample_num, 0]
        index_new_date = index_series.iloc[:sample_num, 1]
        df_new = df_new.set_index([index_new_listId, index_new_date])
        return df_new




