# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import os
from io import BytesIO

import pandas as pd
import requests


class DatasetCovidDeathsUS:
    """
    Class to load COVID-19 deaths data for the US.
    Source: https://github.com/CSSEGISandData/COVID-19
    Note: in this dataset, reporting deaths only started on the 22th of January 2020,

    Attributes:
        df: Timeseries dataframe of confirmed COVID deaths for all the US states
    """

    def __init__(self, data_dir="", download_latest_data=True):
        if not os.path.exists(data_dir):
            print(
                "Creating a dynamic data directory to store "
                "COVID-19 deaths data: {}".format(data_dir)
            )
            os.makedirs(data_dir)

        filename = "daily_us_deaths.csv"
        if download_latest_data or filename not in os.listdir(data_dir):
            print(
                "Fetching latest U.S. COVID-19 deaths data from John Hopkins, "
                "and saving it in {}".format(data_dir)
            )

            req = requests.get(
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
                "csse_covid_19_data/csse_covid_19_time_series/"
                "time_series_covid19_deaths_US.csv"
            )
            self.df = pd.read_csv(BytesIO(req.content))
            self.df.to_csv(
                os.path.join(data_dir, filename)
            )  # Note: performs an overwrite
        else:
            print(
                "Not fetching the latest U.S. COVID-19 deaths data from John Hopkins."
                " Using whatever was saved earlier in {}!!".format(data_dir)
            )
            assert filename in os.listdir(data_dir)
            self.df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False)
