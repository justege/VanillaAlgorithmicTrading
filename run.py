from model.models import *
import os


def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate', 'tic']).reset_index(drop=True)
    return df


def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets

    df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0] * start
    # turbulence_index = [0]
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index]]
        cov_temp = hist_price.cov()
        current_temp = (current_price - np.mean(hist_price, axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame({'datadate': df_price_pivot.index,
                                     'turbulence': turbulence_index})
    return turbulence_index


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "data/0003_test.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        data = add_turbulence(data)

    print(data.info)
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20210101) & (data.datadate <= 20221001)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    ## Ensemble Strategy
    run_ensemble_strategy(df=data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    # _logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    run_model()
