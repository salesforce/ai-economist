# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import bz2
import os
import pickle
import queue
import threading
import urllib.request as urllib2

import pandas as pd
from bs4 import BeautifulSoup


class DatasetCovidUnemploymentUS:
    """
    Class to load COVID-19 unemployment data for the US states.
    Source: https://www.bls.gov/lau/
    """

    def __init__(self, data_dir="", download_latest_data=True):
        if not os.path.exists(data_dir):
            print(
                "Creating a dynamic data directory to store COVID-19 "
                "unemployment data: {}".format(data_dir)
            )
            os.makedirs(data_dir)

        filename = "monthly_us_unemployment.bz2"
        if download_latest_data or filename not in os.listdir(data_dir):
            # Construct the U.S. state to FIPS code mapping
            state_fips_df = pd.read_excel(
                "https://www2.census.gov/programs-surveys/popest/geographies/2017/"
                "state-geocodes-v2017.xlsx",
                header=5,
            )
            # remove all statistical areas and cities
            state_fips_df = state_fips_df.loc[state_fips_df["State (FIPS)"] != 0]
            self.us_state_to_fips_dict = pd.Series(
                state_fips_df["State (FIPS)"].values, index=state_fips_df.Name
            ).to_dict()

            print(
                "Fetching the U.S. unemployment data from "
                "Bureau of Labor and Statistics, and saving it in {}".format(data_dir)
            )
            self.data = self.scrape_bls_data()
            fp = bz2.BZ2File(os.path.join(data_dir, filename), "wb")
            pickle.dump(self.data, fp)
            fp.close()

        else:
            print(
                "Not fetching the U.S. unemployment data from Bureau of Labor and"
                " Statistics. Using whatever was saved earlier in {}!!".format(data_dir)
            )
            assert filename in os.listdir(data_dir)
            with bz2.BZ2File(os.path.join(data_dir, filename), "rb") as fp:
                self.data = pickle.load(fp)
            fp.close()

    # Scrape monthly unemployment from the Bureau of Labor Statistics website
    def get_monthly_bls_unemployment_rates(self, state_fips):
        with urllib2.urlopen(
            "https://data.bls.gov/timeseries/LASST{:02d}0000000000003".format(
                state_fips
            )
        ) as response:
            html_doc = response.read()

        soup = BeautifulSoup(html_doc, "html.parser")
        table = soup.find_all("table")[1]
        table_rows = table.find_all("tr")

        unemployment_dict = {}

        mth2idx = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }

        for tr in table_rows[1:-1]:
            td = tr.find_all("td")[-1]
            unemp = float("".join([c for c in td.text if c.isdigit() or c == "."]))
            th = tr.find_all("th")
            year = int(th[0].text)
            month = mth2idx[th[1].text]
            if year not in unemployment_dict:
                unemployment_dict[year] = {}
            unemployment_dict[year][month] = unemp

        return unemployment_dict

    def scrape_bls_data(self):
        def do_scrape(us_state, fips, queue_obj):
            out = self.get_monthly_bls_unemployment_rates(fips)
            queue_obj.put([us_state, out])

        print("Getting BLS Data. This might take a minute...")
        result = queue.Queue()
        threads = [
            threading.Thread(target=do_scrape, args=(us_state, fips, result))
            for us_state, fips in self.us_state_to_fips_dict.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        monthly_unemployment = {}
        while not result.empty():
            us_state, data = result.get()
            monthly_unemployment[us_state] = data

        return monthly_unemployment
