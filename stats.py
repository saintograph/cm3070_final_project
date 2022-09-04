import pandas as pd
from scipy.stats import ttest_ind

# generate CSV of training output of best model
def ppo_results():
    results_df = pd.read_csv('./testing_results/ppo_results.csv')
    df_baseline = pd.read_csv('./logs/log_deadly_corridor_ppo_baseline/PPO_train/progress.csv')
    df = pd.read_csv('./logs/log_deadly_corridor_ppo_optimized_05/PPO_train/progress.csv')
    results_df.loc[0, 'value'] = df_baseline['rollout/ep_rew_mean'].mean()
    results_df.loc[1, 'value'] = df['rollout/ep_rew_mean'].mean()
    results_df.loc[2, 'value'] = ttest_ind(df['rollout/ep_rew_mean'], df_baseline['rollout/ep_rew_mean'], equal_var = False)[1]
    results_df.to_csv('./testing_results/ppo_results.csv', index = False) 

# generate CSV of training output of best model
def a2c_results():
    results_df = pd.read_csv('./testing_results/a2c_results.csv')
    df_baseline = pd.read_csv('./logs/log_deadly_corridor_a2c_baseline/A2C_train/progress.csv')
    df = pd.read_csv('./logs/log_deadly_corridor_a2c_optimized_03/A2C_train/progress.csv')
    results_df.loc[0, 'value'] = df_baseline['rollout/ep_rew_mean'].mean()
    results_df.loc[1, 'value'] = df['rollout/ep_rew_mean'].mean()
    results_df.loc[2, 'value'] = ttest_ind(df['rollout/ep_rew_mean'], df_baseline['rollout/ep_rew_mean'], equal_var = False)[1]
    results_df.to_csv('./testing_results/a2c_results.csv', index = False) 

if __name__ == "__main__":
    ppo_results()
    a2c_results()