# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import os
from io import BytesIO

import pandas as pd
import requests


class DatasetCovidVaccinationsUS:
    """
    Class to load COVID-19 vaccination data for the US.
    Source: https://ourworldindata.org/covid-vaccinations

    Attributes:
        df: Timeseries dataframe of COVID vaccinations for all the US states
    """

    def __init__(self, data_dir="", download_latest_data=True):
        if not os.path.exists(data_dir):
            print(
                "Creating a dynamic data directory to store COVID-19 "
                "vaccination data: {}".format(data_dir)
            )
            os.makedirs(data_dir)

        filename = "daily_us_vaccinations.csv"
        if download_latest_data or filename not in os.listdir(data_dir):
            print(
                "Fetching latest U.S. COVID-19 vaccination data from "
                "Our World in Data, and saving it in {}".format(data_dir)
            )

            req = requests.get(
                "https://raw.githubusercontent.com/owid/covid-19-data/master/"
                "public/data/vaccinations/us_state_vaccinations.csv"
            )
            self.df = pd.read_csv(BytesIO(req.content))

            # Rename New York State to New York for consistency with other datasets
            self.df = self.df.replace("New York State", "New York")

            # Interpolate missing values
            self.df = self.df.interpolate(method="linear")

            self.df.to_csv(
                os.path.join(data_dir, filename)
            )  # Note: performs an overwrite
        else:
            print(
                "Not fetching the latest U.S. COVID-19 deaths data from "
                "Our World in Data. Using whatever was saved earlier in {}!!".format(
                    data_dir
                )
            )
            assert filename in os.listdir(data_dir)
            self.df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False)
