
import sys
import datetime as dt
import glob
import pandas as pd
import numpy as np
import json
import os
from math import sqrt
import pyspark.sql.functions as F
import pyspark.sql.window as W
from pyspark.sql import DataFrame
from delta.tables import DeltaTable
import pyspark.sql.types as T
from numpy.random import normal
from collections import defaultdict

# from kvutils.util import AzureKeyVaultUtil
import configparser
import matplotlib.pyplot as plt
import scipy.stats

# kvutil = AzureKeyVaultUtil(dbutils)
# ENV_WORKSPACE_READ = kvutil.get_secret('ENV-WORKSPACE-READ').decode('utf-8')
ENV_WORKSPACE_READ = "Prod"

config_file = "/dbfs/deepbrew/drivethru/config_paths.ini"
configpar = configparser.ConfigParser()
configpar.read(config_file)


config = {
 "paths": {
 "input_paths": {
 "variant_config": (configpar["variant_bandit"]["variant_config"]),
 "variant_kpis": (
 configpar["default"]["out_root"]
 + configpar["variant_bandit"]["variant_kpis"]
 ).format(ENV_WORKSPACE_READ),
 "blackout_config": (configpar["variant_bandit"]["blackout_config"]),
 }
 },
 "days_pull_history_l": [7 * 52, 7 * 26, 7 * 13, 7 * 8], # days to pull KPI history
 "round_precision": 12,
 "look_back_window": -400,
 # minimum number of historical data for variant to be considered for optimization
 "min_days_of_history": 7 * 8,
 "maall_rewards_lotal_variant_weight": 1,
 "iqr_coefficient": 1.0,
 "tau_0": 0.0000000001,
 "tardigrade6_exclusion_date": "20210205",
 "target_efficiency_l": [0.90, 0.92, 0.94, 0.96, 0.98],
 "time_steps": 10 ** 5,
 "cost_of_playing_arm": 1,
}


def extract_variant_kpis(start_date, end_date, exclusion_ls, dbfs_prefix="/dbfs"):
 """
 extracts data for KPIs
 """
 print("DEBUG: extracting kpi from {} to {}".format(start_date, end_date))
 path = config["paths"]["input_paths"]["variant_kpis"]
 file_list = glob.glob(dbfs_prefix + path + "BusinessDate" + "=*")
 file_list_nondbfs = [_path.split("/dbfs")[1] for _path in file_list]

 df = (
 spark.read.option("basePath", path[:-1])
 .parquet((*file_list_nondbfs[config["look_back_window"] :]))
 .withColumn("BusinessDate", F.col("BusinessDate").cast("string"))
 .where(
 (F.col("BusinessDate") >= start_date) & (F.col("BusinessDate") <= end_date)
 )
 ).select("BusinessDate", "StoreNumber", "hour", "VariantID", "avg_ticket")
 if len(exclusion_ls) > 0:
 # remove blackout date from kpis in case data is there
 df = df.where(~F.col("BusinessDate").isin(exclusion_ls))
 print(
 "DEBUG: excluded {} from kpi due to blackout overlap".format(exclusion_ls)
 )
 df = df.withColumn("avg_ticket", F.col("avg_ticket").cast("float"))

 # filter out tardigrade6 before v1.11.1 release
 df = df.where(
 ~(
 (F.col("VariantID") == "tardigrade6")
 & (F.col("BusinessDate") < config["tardigrade6_exclusion_date"])
 )
 )

 print("SUCCESS: extract_variant_kpis")
 return df


def get_variant_id(df):
 """
 identify distinct variants in kpi
 """
 kpi_var_ls = []
 for ele in df.select("variantID").distinct().collect():
 kpi_var_ls.append(ele[0])

 print("SUCCESS: get_variant_id")
 return kpi_var_ls


def validate_kpi_history(df, kpi_var_ls, variant_config_d):
 """
 drops variants from the list of kpis if number of businessdate is smaller than min_days_of_history
 """
 variant_days_df = df.groupBy("VariantID").agg(
 F.countDistinct("BusinessDate").alias("days")
 )

 for var in kpi_var_ls[:]:
 num_days = variant_days_df.filter(F.col("VariantID") == var).collect()[0][
 "days"
 ]
 print("DEBUG: var {}; days history {}".format(var, num_days))
 if num_days < config["min_days_of_history"]:
 kpi_var_ls.remove(var)
 print(
 "DEBUG: less than {} days history detected. dropping {} from optimization process. initial weight will be used!".format(
 config["min_days_of_history"], var
 )
 )

 # drop variants from kpis when variant is deprecated
 for var in kpi_var_ls[:]:
 if var not in variant_config_d["variant_id"].keys():
 print(
 "DEBUG: {} not in config anymore. dropping it from variant list".format(
 var
 )
 )
 kpi_var_ls.remove(var)

 print("SUCCESS: validate_kpi_history")
 return kpi_var_ls


def validate_kpi_value(df):
 """
 this should be taken care of earlier in variant_kpis.py
 double check just in case
 """
 primary_cnt = df.count()
 # drop null/zero/negative avg_tickets
 df = df.where(F.col("avg_ticket").isNotNull()).where(F.col("avg_ticket") > 0)
 print("DEBUG: dropped {} rows from kpi".format(primary_cnt - df.count()))

 print("SUCCESS: sanity_check_kpi")
 return df


def transform_kpi(df, kpi_var_ls):
 """
 keep valid variants kpis
 log transform and remove outliers
 """
 # keep varainats with more than min days history
 df = df.where(F.col("VariantID").isin(kpi_var_ls))
 # log transform average ticket
 df = (
 df.withColumn("avg_ticket", F.log(F.col("avg_ticket")))
 .withColumn(
 "q1",
 F.expr("percentile_approx(avg_ticket, 0.25)").over(
 W.Window.partitionBy("VariantID")
 ),
 )
 .withColumn(
 "q3",
 F.expr("percentile_approx(avg_ticket, 0.75)").over(
 W.Window.partitionBy("VariantID")
 ),
 )
 .withColumn("iqr", F.col("q3") - F.col("q1"))
 )
 # drop outside quantile range
 df = df.where(
 (F.col("avg_ticket") > (F.col("q1") - config["iqr_coefficient"] * F.col("iqr")))
 & (
 F.col("avg_ticket")
 < (F.col("q3") + config["iqr_coefficient"] * F.col("iqr"))
 )
 )
 print("SUCCESS: transform_kpi")
 return df.toPandas()


def extract_variant_config():
 """
 read variant_config_json
 """
 with open(config["paths"]["input_paths"]["variant_config"]) as f:
 variant_config_d = json.load(f)

 if (len(variant_config_d["variant_id"].keys())) == 0:
 raise ValueError("No variant found in config")

 print("SUCCESS: extract_variant_config")
 return variant_config_d


def sanity_check_config(variant_config_d, end_date):
 """
 check variant config
 """
 # check variant keys
 if (len(variant_config_d["variant_id"].keys())) == 0:
 raise ValueError("No variant found in config")

 sum_min_weight = 0
 for var in variant_config_d["variant_id"].keys():
 # check min weight
 if (
 variant_config_d["variant_id"][var]["initial_weight"]
 < variant_config_d["variant_id"][var]["min_weight"]
 ):
 raise ValueError("{} initial_weigh is smaller than min_weight".format(var))

 # check config keys
 default_keys = [
 "description",
 "start_date",
 "end_date",
 "initial_weight",
 "min_weight",
 "max_weight",
 ]
 if len(variant_config_d["variant_id"][var].keys()) != len(default_keys):
 raise ValueError("config keys does not match")

 # check max weight
 if (
 variant_config_d["variant_id"][var]["initial_weight"]
 > variant_config_d["variant_id"][var]["max_weight"]
 ):
 raise ValueError("{} initial_weigh is greater than max_weight".format(var))

 for var in list(variant_config_d["variant_id"].keys()):
 # check end dates
 if dt.datetime.strptime(
 variant_config_d["variant_id"][var]["end_date"], "%Y-%m-%d"
 ) <= dt.datetime.strptime(end_date, "%Y%m%d"):
 # if varinat expired
 variant_config_d["variant_id"].pop(var)
 print("DEBUG: {} expired and was removed from config".format(var))

 for var in list(variant_config_d["variant_id"].keys()):
 # check start date
 # TODO review logic align with scheduler
 # it checks for the variants 8 days in the future and keep them otherwise it will be dropped
 if (
 dt.datetime.strptime(
 variant_config_d["variant_id"][var]["start_date"], "%Y-%m-%d"
 )
 - dt.timedelta(days=7)
 ) > dt.datetime.strptime(end_date, "%Y%m%d"):
 # if varinat expired
 variant_config_d["variant_id"].pop(var)
 print(
 "DEBUG: {} start date is not reached and was removed from config".format(
 var
 )
 )

 for var in variant_config_d["variant_id"].keys():
 # get the mins after all the drops
 sum_min_weight += float(variant_config_d["variant_id"][var]["min_weight"])

 if sum_min_weight > config["maall_rewards_lotal_variant_weight"]:
 raise ValueError(
 "Found sum of mininum variants weights > {} in config".format(
 config["maall_rewards_lotal_variant_weight"]
 )
 )

 print("SUCCESS: sanity_check_config")
 return variant_config_d


def generate_variant_stats(df, kpi_var_ls, variant_config_d):
 """
 get variant gaussian stats
 """
 mu_ls = list()
 tau_ls = list()
 n_ls = list()

 # dictionary to keep index of variants
 index_d = dict()
 for i, var in enumerate(kpi_var_ls):
 if var in variant_config_d["variant_id"].keys():
 index_d[i] = var
 df_var = df[df["VariantID"] == var]["avg_ticket"]
 avg_ticket_l = [
 float(i) for i in (df[df["VariantID"] == var]["avg_ticket"].values)
 ]

 # normal conjugate with fixed variance assumed, estimating unknown mean

 # prior assumptions
 mu_0 = np.round(np.mean(df["avg_ticket"].values), config["round_precision"])
 tau_0 = config["tau_0"]

 # assume fixed population variance within arm, measured independently over time of look back
 tau = 1 / np.var(avg_ticket_l)

 # get look back observations
 # TODO can loop through weeks here for update
 n = df_var.shape[0]
 x = np.round(np.sum(avg_ticket_l), config["round_precision"])

 # posterior hyperparam update
 mu_0 = ((tau_0 * mu_0) + (tau * x)) / (tau_0 + (n * tau))
 tau_0 = n * tau

 # appends
 mu_ls.append(np.round(mu_0, config["round_precision"]))
 tau_ls.append(np.round(tau_0, config["round_precision"]))
 n_ls.append(n)

 print(
 "DEBUG - OBS:\nmean:\t {}\nprecisions:\t {}\nsample size:\t {}\nvariant index:\t {}".format(
 mu_ls, tau_ls, n_ls, index_d
 )
 )
 return mu_ls, tau_ls, index_d, n_ls


def read_blackout_config():
 # read the config
 with open(config["paths"]["input_paths"]["blackout_config"]) as f:
 blackout_d = json.load(f)

 print("SUCCESS: read_blackout_config")
 return blackout_d


def get_blackout_date():
 # generate list of dates
 blackout_d = read_blackout_config()
 blackout_ls = []
 for ele in blackout_d["date_blackout_ranges"]:
 blackout_start_date = dt.datetime.strptime(ele["begin_date"], "%Y-%m-%d")
 blackout_end_date = dt.datetime.strptime(ele["end_date"], "%Y-%m-%d")
 blackout_ls.append((blackout_start_date, blackout_end_date))
 print("SUCCESS: get_blackout_date")
 return blackout_ls


def get_start_end_date(date_s, days_pull_history):
 # checks for all the overlaps and extend pull history period based on the number of blackout days
 blackout_ls = get_blackout_date()
 start_date = dt.datetime.strptime(date_s, "%Y-%m-%d") - dt.timedelta(
 days=days_pull_history - 1
 )
 end_date = dt.datetime.strptime(date_s, "%Y-%m-%d")
 print("DEBUG: start_date {} and end_date: {}".format(start_date, end_date))
 blackout_period = 0
 exclusion_ls = []
 # find all overlapping periods
 for ele in blackout_ls:
 if ele[0] >= start_date and ele[0] <= end_date:
 if ele[1] <= end_date:
 # days is not inclusive so adding one
 blackout_period += (ele[1] - ele[0]).days + 1
 print(
 "DEBUG: {}-{} overlap with run. adding {} days for padding. total padding days {}".format(
 ele[0], ele[1], (ele[1] - ele[0]).days + 1, blackout_period
 )
 )
 for i in range((ele[1] - ele[0]).days + 1):
 # add all the days in the blackout range to exclusion list
 date_e = (ele[0] + dt.timedelta(days=i)).strftime("%Y%m%d")
 exclusion_ls.append(date_e)
 elif ele[1] > end_date:
 # when run date is in the middle of blackout period
 blackout_period += (end_date - ele[0]).days + 1
 print(
 "DEBUG: {}-{} overlap with run. adding {} days for padding. total padding days {}".format(
 ele[0], ele[1], (end_date - ele[0]).days + 1, blackout_period
 )
 )
 for i in range((end_date - ele[0]).days + 1):
 # add all the days in the blackout range to exclusion list
 date_e = (ele[0] + dt.timedelta(days=i)).strftime("%Y%m%d")
 exclusion_ls.append(date_e)

 # add padding days
 if blackout_period > 0:
 start_date = start_date - dt.timedelta(days=(blackout_period))
 print("DEBUG: New start date is {}".format(start_date))
 return (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), exclusion_ls)


def calc_variant_stats(date_s, days_pull_history):
 """
 get KPI date range, extract KPIs, transform and get stats.
 """

 start_date, end_date, exclusion_ls = get_start_end_date(date_s, days_pull_history)
 variant_config_d = extract_variant_config()
 variant_config_d = sanity_check_config(variant_config_d, end_date)
 df = extract_variant_kpis(start_date, end_date, exclusion_ls).cache()
 kpi_var_ls = get_variant_id(df)
 kpi_var_ls = validate_kpi_history(df, kpi_var_ls, variant_config_d)
 df = validate_kpi_value(df)
 df = transform_kpi(df, kpi_var_ls)
 mu_ls, tau_ls, index_d, n_ls = generate_variant_stats(
 df, kpi_var_ls, variant_config_d
 )

 print("====== Original Variant stats =====")
 print(mu_ls, tau_ls, index_d, n_ls)

 print("SUCCESS: calc_variant_stats")

 return index_d, mu_ls, tau_ls, n_ls


def get_control_stats(index_d, mu_ls, tau_ls):
 """
 get control stats
 """

 control_idx = [k for k, v in index_d.items() if v == "control"][0]
 control_mean = mu_ls[control_idx]
 control_tau = tau_ls[control_idx]

 print("===== Control Variant stats =====")
 print(control_idx, control_mean, control_tau)

 print("SUCCESS: get_control_stats")
 return control_idx, control_mean, control_tau


def get_sorted_stats(mu_ls, tau_ls, index_d, control_idx):
 """
 get sorted stats, Variant with highest mean at the top
 """
 mu_l = [mu_ls[i] for i in range(len(mu_ls)) if i != control_idx]
 tau_l = [tau_ls[i] for i in range(len(mu_ls)) if i != control_idx]
 trt_name = [v for k, v in index_d.items() if k != control_idx]

 sort_mu_index = np.argsort(mu_l)[::-1]
 sorted_mu_l = list()
 sorted_tau_l = list()
 sorted_treatment_name_l = list()

 for i in sort_mu_index:
 sorted_mu_l.append(mu_l[i])
 sorted_tau_l.append(tau_l[i])
 sorted_treatment_name_l.append(trt_name[i])

 print("<<<< AFTER SORTING & No CONTROL >>>>")
 print(sort_mu_index, sorted_mu_l, sorted_tau_l, sorted_treatment_name_l)

 print("SUCCESS: get_sorted_stats")

 return sort_mu_index, sorted_mu_l, sorted_tau_l, sorted_treatment_name_l


def get_bandits_df(sort_mu_index, sorted_mu_l, sorted_tau_l, sorted_treatment_name_l):
 """
 get bandits stats as pnadas df
 """

 pd_bandit_stats = pd.DataFrame(
 columns=["index", "Variant", "means_log", "stddev_log", "means", "stddev"]
 )
 pd_bandit_stats["index"] = sort_mu_index
 pd_bandit_stats["Variant"] = sorted_treatment_name_l
 pd_bandit_stats["means_log"] = sorted_mu_l
 pd_bandit_stats["means"] = np.exp(sorted_mu_l)
 pd_bandit_stats["stddev_log"] = 1 / np.sqrt(sorted_tau_l)
 pd_bandit_stats["stddev"] = np.exp(1 / np.sqrt(sorted_tau_l))
 # pd_bandit_stats.sort_values(by='means', ascending=False)

 print("SUCCESS: get_bandits_df")
 return pd_bandit_stats


def plot_fitted_norm_pdfs(mu_ls, tau_ls, index_d, endDate, zoom=1.0, fig_size=(20, 5)):
 """
 plot probablity distribution for all variants
 """

 means = mu_ls
 precisions = tau_ls
 variants = [v for _, v in index_d.items()]

 x_min = np.min(means)
 x_max = np.max(means)
 x_diff = zoom * (x_max - x_min)
 x = np.linspace(x_min - x_diff, x_max + x_diff, 100)

 fig, ax = plt.subplots(figsize=fig_size)

 for i in range(len(means)):
 y = scipy.stats.norm.pdf(x, means[i], 1 / np.sqrt(precisions[i]))
 ax.plot(x, y, label=variants[i])
 ax.set_xlim(x_min - x_diff, x_max + x_diff)

 ax.set_title("Tardigrade Variants for " + str(endDate))
 ax.set_xlabel("log scale avg_ticket")
 ax.set_ylabel("Probablity Density")
 plt.legend(bbox_to_anchor=(1.05, 1))
 display(fig)
 plt.close("all")


def multiplay_sample(
 num_arms_to_pull_l, sorted_mu_l, sorted_tau_l, sort_mu_index, win_threshold
):
 """
 given variants average ticket performance, select top L variants to play
 """

 # sample from sorted variant ordering
 arm_samples = [
 np.random.normal(loc=i[0], scale=1 / np.sqrt(i[1]), size=1)[0]
 for i in zip(sorted_mu_l, sorted_tau_l)
 ]

 # observe reward when sample is higher than control
 variant_reward_t_l = [
 1 if arm_samples[a] > win_threshold else 0 for a in range(num_arms_to_pull_l)
 ]

 # pick top L variants to run
 top_arms_l = sort_mu_index[:num_arms_to_pull_l]

 print("SUCCESS: multiplay_sample")

 return variant_reward_t_l, top_arms_l


def KL_divergence_gaussian(x, d, sig2x=0.25, precision=0.0):
 """KL-UCB index computation for Gaussian distributions.
 - Note that it does not require any search.
 .. warning:: it works only if the good variance constant is given.
 """

 return x + np.sqrt(2 * sig2x * d)


def KL_Scaling(L, variant_cost_l, variant_wins_l, t, no_of_arms, target_eff):
 """
 Scale up/down in order to meet target efficiency
 * when current efficieny <= target efficiency then
 - scale down as we can't explore much so exploit
 * when current effiency > target efficiency then
 - if we are confident that adding annother arm will lead to lower or same eff. thr KL-UCB index then scale up.
 """

 # ratio of rewards/total pulls
 reward_ratio_l = [
 variant_wins_l[i] / variant_cost_l[i] if variant_cost_l[i] > 0 else 1
 for i in range(no_of_arms)
 ]

 # calculate efficiency for top L variants
 efficiency_t = np.sum(reward_ratio_l[:L]) / L

 if efficiency_t <= target_eff:
 # scale down to select top performing variants
 arms_to_chose = max(L - 1, 1)
 else:
 Lt_1_arm = L + 1
 if Lt_1_arm > no_of_arms:
 Lt_1_arm = no_of_arms

 # get KL-UCB divergence
 KL_l = [
 KL_divergence_gaussian(reward_ratio_l[i], np.log(t + 1 / variant_cost_l[i]))
 for i in range(no_of_arms)
 ]

 # find L+1th largest index
 b_t = np.sort(KL_l)[::-1][Lt_1_arm - 1]

 # calculate upper confidence bound for L+1 efficiency
 B_hat = (L / (L + 1)) * efficiency_t + (1 / (L + 1)) * b_t

 if B_hat > target_eff:
 # scale up variants
 arms_to_chose = min(L + 1, no_of_arms)
 else:
 arms_to_chose = L

 print("SUCCESS: KL_Scaling")
 return arms_to_chose, efficiency_t


def scalable_mab_variants(
 sorted_mu_l, sorted_tau_l, sort_mu_index, win_threshold, target_eff
):
 """
 iterate optimized arms for a given timesteps
 """

 no_of_arms = len(sorted_mu_l)
 variant_cost_l = [0] * no_of_arms
 variant_wins_l = [0] * no_of_arms
 num_arms_to_pull_l = [no_of_arms]
 all_arms_l = list()
 all_rewards_l = list()
 efficiency_l = list()

 for t_ in range(config["time_steps"]):

 variant_reward_t_l, top_arms_l = multiplay_sample(
 num_arms_to_pull_l[t_],
 sorted_mu_l,
 sorted_tau_l,
 sort_mu_index,
 win_threshold,
 )
 all_arms_l.append(top_arms_l)
 all_rewards_l.append(variant_reward_t_l)

 for i in range(num_arms_to_pull_l[t_]):

 variant_cost_l[i] += config["cost_of_playing_arm"]
 variant_wins_l[i] += variant_reward_t_l[i]

 arms_to_chose, efficiency_t = KL_Scaling(
 num_arms_to_pull_l[t_],
 variant_cost_l,
 variant_wins_l,
 t_,
 no_of_arms,
 target_eff,
 )
 num_arms_to_pull_l.append(arms_to_chose)
 efficiency_l.append(efficiency_t)

 print("scalable_mab_variants")
 return variant_cost_l, variant_wins_l, efficiency_l, num_arms_to_pull_l


def plot_efficiency(efficiency_l, target_eff, pull_days):
 # efficinecy
 plt.plot(efficiency_l)
 plt.xlabel("Time")
 plt.ylabel("Efficiency")
 plt.title(
 "Efficiency vs Time target eff:{} pull days:{}".format(target_eff, pull_days)
 )
 plt.show()
 display()
 plt.close()


def plot_num_of_variants(num_arms_to_pull_l, target_eff, pull_days):
 plt.plot(num_arms_to_pull_l)
 plt.xlabel("Time")
 plt.ylabel("Number of top Variants")
 plt.title(
 "Top Variants by Time target eff:{} pull days:{}".format(target_eff, pull_days)
 )
 plt.show()
 display()
 plt.close()


def plot_draws_per_variant(
 sort_mu_index, variant_cost_l, target_eff, pull_days, variants_label
):
 fig, ax = plt.subplots() # figsize = (15, 10))
 ax.bar(sort_mu_index, variant_cost_l)

 ax.set_xlabel("Variants")
 ax.set_ylabel("Draws")
 ax.set_title(
 "Draws per Variant target eff:{} pull days:{}".format(target_eff, pull_days)
 )

 ax.set_xticklabels(variants_label, rotation=90)
 tck_l = [i + 1 for i in range(len(variants_label))]
 ax.set_xticks(ticks=tck_l)

 plt.show()
 display()
 plt.close()


def plot_wins_per_variant(
 sort_mu_index, variant_wins_l, target_eff, pull_days, variants_label
):

 fig, ax = plt.subplots() # figsize = (15, 10))
 ax.bar(sort_mu_index, variant_wins_l)

 ax.set_xlabel("Variants")
 ax.set_ylabel("Wins")
 ax.set_title(
 "Wins per Variant target eff:{} pull days:{}".format(target_eff, pull_days)
 )

 ax.set_xticklabels(variants_label, rotation=90)
 tck_l = [i + 1 for i in range(len(variants_label))]
 ax.set_xticks(ticks=tck_l)
 plt.show()
 display()
 plt.close()


def variant_diagnostic(variants_idx, index_d_l, mu_ls_l, tau_ls_l, date_s, config_idx):
 # diagnostic for bandit stats

 control_idx, control_mean, control_tau = get_control_stats(
 index_d_l[variants_idx], mu_ls_l[variants_idx], tau_ls_l[variants_idx]
 )
 (
 sort_mu_index,
 sorted_mu_l,
 sorted_tau_l,
 sorted_treatment_name_l,
 ) = get_sorted_stats(
 mu_ls_l[variants_idx],
 tau_ls_l[variants_idx],
 index_d_l[variants_idx],
 control_idx,
 )

 pd_bandits_stats = get_bandits_df(
 sort_mu_index, sorted_mu_l, sorted_tau_l, sorted_treatment_name_l
 )
 pd_bandits_stats["pull_period_days"] = [
 config["days_pull_history_l"][config_idx]
 ] * pd_bandits_stats.shape[0]

 print("Days pull history:", config["days_pull_history_l"][config_idx])
 plot_fitted_norm_pdfs(
 mu_ls_l[variants_idx],
 tau_ls_l[variants_idx],
 index_d_l[variants_idx],
 endDate=date_s,
 zoom=1.0,
 )

 return pd_bandits_stats


def plot_arms_pull_days(idx, config, pull_days_l, optimize_l, target_eff_l):
 pull_days = config["days_pull_history_l"][idx]
 days_pull_idx = [i for i in range(len(pull_days_l)) if pull_days_l[i] == pull_days]
 x = [optimize_l[i] for i in range(len(optimize_l)) if i in days_pull_idx]
 y = [target_eff_l[i] for i in range(len(target_eff_l)) if i in days_pull_idx]
 plt.plot(x, y)
 plt.xlabel("Top arms")
 plt.ylabel("Target Efficiency")
 plt.title("Target Efficiency vs Top arms for KPI_days_pull:{}".format(pull_days))
 plt.show()
 display()
 plt.close()

 return x


def plot_idx(
 plt_idx,
 all_sort_mu_index,
 all_variant_wins_l,
 all_variant_draws_l,
 all_num_arms_to_pull_l,
 target_eff_l,
 pull_days_l,
 all_efficiency_l,
 index_d_l,
):
 variants_label = [str(v) for _, v in index_d_l[plt_idx].items() if v != "control"]
 wins_val_l = [0] * len(all_sort_mu_index[plt_idx])
 val_l = all_variant_wins_l[plt_idx]

 idx_val = 0
 for i in all_sort_mu_index[plt_idx]:
 wins_val_l[i] = val_l[idx_val]
 idx_val += 1

 draws_val_l = [0] * len(all_sort_mu_index[plt_idx])
 val_l = all_variant_draws_l[plt_idx]

 idx_val = 0
 for i in all_sort_mu_index[plt_idx]:
 draws_val_l[i] = val_l[idx_val]
 idx_val += 1

 plot_num_of_variants(
 all_num_arms_to_pull_l[plt_idx], target_eff_l[plt_idx], pull_days_l[plt_idx]
 )
 plot_efficiency(
 all_efficiency_l[plt_idx], target_eff_l[plt_idx], pull_days_l[plt_idx]
 )
 plot_wins_per_variant(
 all_sort_mu_index[plt_idx],
 wins_val_l,
 target_eff_l[plt_idx],
 pull_days_l[plt_idx],
 variants_label,
 )
 plot_draws_per_variant(
 all_sort_mu_index[plt_idx],
 draws_val_l,
 target_eff_l[plt_idx],
 pull_days_l[plt_idx],
 variants_label,
 )


def main(date_s):

 optimize_l = list()
 target_eff_l = list()
 pull_days_l = list()
 index_d_l = list()
 mu_ls_l = list()
 tau_ls_l = list()

 # for diagnostic
 all_num_arms_to_pull_l = list()
 all_efficiency_l = list()
 all_sort_mu_index = list()
 all_variant_wins_l = list()
 all_variant_draws_l = list()

 for days_pull_history in config["days_pull_history_l"]:

 # varinats KPIS
 index_d, mu_ls, tau_ls, n_ls = calc_variant_stats(date_s, days_pull_history)

 print("Length of mu_ls:", len(mu_ls))
 if len(mu_ls) < 2:
 # when no variants (atleast ine variant and control should exist to continue further)

 continue

 control_idx, control_mean, control_tau = get_control_stats(
 index_d, mu_ls, tau_ls
 )
 (
 sort_mu_index,
 sorted_mu_l,
 sorted_tau_l,
 sorted_treatment_name_l,
 ) = get_sorted_stats(mu_ls, tau_ls, index_d, control_idx)

 for target_eff in config["target_efficiency_l"]:
 (
 variant_cost_l,
 variant_wins_l,
 efficiency_l,
 num_arms_to_pull_l,
 ) = scalable_mab_variants(
 sorted_mu_l, sorted_tau_l, sort_mu_index, control_mean, target_eff
 )

 index_d_l.append(index_d)
 mu_ls_l.append(mu_ls)
 tau_ls_l.append(tau_ls)

 # take maximum from last 10 optimized arms to pull. This can be generalized to last 20 etc. as well
 optimize_l.append(np.max(num_arms_to_pull_l[-10:]))
 target_eff_l.append(target_eff)
 pull_days_l.append(days_pull_history)

 # diagnostic
 all_num_arms_to_pull_l.append(num_arms_to_pull_l)
 all_efficiency_l.append(efficiency_l)
 all_sort_mu_index.append(sort_mu_index)
 all_variant_wins_l.append(variant_wins_l)
 all_variant_draws_l.append(variant_cost_l)

 return (
 optimize_l,
 target_eff_l,
 pull_days_l,
 index_d_l,
 mu_ls_l,
 tau_ls_l,
 all_num_arms_to_pull_l,
 all_efficiency_l,
 all_sort_mu_index,
 all_variant_wins_l,
 all_variant_draws_l,
 )


# # date_s = '2021-08-31'
# date_s = '2021-10-03'

# import datetime as dt

# dbutils.widgets.text("EndDate", "")
# dbutils.widgets.text("target_eff", "")

# end_date = dbutils.widgets.get("EndDate")
# target_eff = dbutils.widgets.get("target_eff")

# if len(end_date) == 0:
# end_date = (dt.datetime.now() - dt.timedelta(days=2)).strftime('%Y%m%d')

# if len(target_eff) == 0:
# target_eff = 0.9

# date_s = end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:]
# date_s
# optimize_l, target_eff_l, pull_days_l, index_d_l, mu_ls_l, tau_ls_l, all_num_arms_to_pull_l, all_efficiency_l, all_sort_mu_index, all_variant_wins_l, all_variant_draws_l = main(date_s)

