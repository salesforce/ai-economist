# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests


class DatasetCovidPoliciesUS:
    """
    Class to load COVID-19 government policies for the US states.
    Source: https://github.com/OxCGRT/USA-covid-policy

    Other references:
    - Codebook: https://github.com/OxCGRT/covid-policy-tracker/blob/master/
    documentation/codebook.md
    - Index computation methodology: https://github.com/OxCGRT/covid-policy-tracker/
    blob/master/documentation/index_methodology.md

    Attributes:
        df: Timeseries dataframe of state-wide policies
    """

    def __init__(self, data_dir="", download_latest_data=True):
        if not os.path.exists(data_dir):
            print(
                "Creating a dynamic data directory to store COVID-19 "
                "policy tracking data: {}".format(data_dir)
            )
            os.makedirs(data_dir)

        filename = "daily_us_policies.csv"
        if download_latest_data or filename not in os.listdir(data_dir):
            print(
                "Fetching latest U.S. COVID-19 policies data from OxCGRT, "
                "and saving it in {}".format(data_dir)
            )
            req = requests.get(
                "https://raw.githubusercontent.com/OxCGRT/USA-covid-policy/master/"
                "data/OxCGRT_US_latest.csv"
            )
            self.df = pd.read_csv(BytesIO(req.content), low_memory=False)
            self.df["Date"] = self.df["Date"].apply(
                lambda x: datetime.strptime(str(x), "%Y%m%d")
            )

            # Fetch only the state-wide policies
            self.df = self.df.loc[self.df["Jurisdiction"] != "NAT_GOV"]

            self.df.to_csv(
                os.path.join(data_dir, filename)
            )  # Note: performs an overwrite
        else:
            print(
                "Not fetching the latest U.S. COVID-19 policies data from OxCGRT. "
                "Using whatever was saved earlier in {}!!".format(data_dir)
            )
            assert filename in os.listdir(data_dir)
            self.df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False)

    def process_policy_data(
        self,
        stringency_policy_key="StringencyIndex",
        num_stringency_levels=10,
    ):
        """
        Gather the relevant policy indicator frm the dataframe,
        fill in the null values (if any),
        and discretize/quantize the policy into num_stringency_levels.
        Note: Possible values for stringency_policy_key are
        ["StringencyIndex", "Government response index",
        "Containment and health index", "Economic Support index".]
        Reference: https://github.com/OxCGRT/covid-policy-tracker/blob/master/
        documentation/index_methodology.md
        """

        def discretize(policies, num_indicator_levels=10):
            """
            Discretize the policies (a Pandas series) into num_indicator_levels
            """
            # Indices are normalized to be in [0, 100]
            bins = np.linspace(0, 100, num_indicator_levels)
            # Find left and right values of bin and find the nearer edge
            bin_index = np.digitize(policies, bins, right=True)
            bin_left_edges = bins[bin_index - 1]
            bin_right_edges = bins[bin_index]
            discretized_policies = bin_index + np.argmin(
                np.stack(
                    (
                        np.abs(policies.values - bin_left_edges),
                        np.abs(policies.values - bin_right_edges),
                    )
                ),
                axis=0,
            )
            return discretized_policies

        # Gather just the relevant columns
        policy_df = self.df[["RegionName", "Date", stringency_policy_key]].copy()

        # Fill in null values via a "forward fill"
        policy_df[stringency_policy_key].fillna(method="ffill", inplace=True)

        # Discretize the stringency indices
        discretized_stringency_policies = discretize(
            policy_df[stringency_policy_key],
            num_indicator_levels=num_stringency_levels,
        )
        policy_df.loc[:, stringency_policy_key] = discretized_stringency_policies

        # Replace Washington DC by District of Columbia to keep consistent
        # (with the other data sources)
        policy_df = policy_df.replace("Washington DC", "District of Columbia")

        policy_df = policy_df.sort_values(by=["RegionName", "Date"])

        return policy_df
