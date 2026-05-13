# Plot Commands and Figure Placement

Run these commands from:

```bash
cd /Users/sreeramranga/Documents/GitHub/cs185-hw-sp26/final_project_offline_online/problem
```

## Figures Already Generated

Put `plots/report_part1_naive_cube_single.png` in the "Naive Baselines" section:

```bash
uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s1_sacbc/cube-single \
  --run_dir exp/s1_fql/cube-single \
  --label "SAC+BC" \
  --label "FQL" \
  --title "Naive offline-to-online on cube-single" \
  --output plots/report_part1_naive_cube_single.png
```

Put `plots/report_offline_data_cube_double.png` and `plots/report_offline_data_antsoccer.png` in the "Retaining Offline Data" section:

```bash
uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s2_offline/sd0_20260422_184639_fql_cube-double-play-singletask-task1-v0_a100.0_od0.1_online_offline \
  --run_dir exp/s2_offline/sd0_20260424_005045_fql_cube-double-play-singletask-task1-v0_a100.0_od0.25_online_offline \
  --label "10% offline data" \
  --label "25% offline data" \
  --title "Retaining offline data on cube-double" \
  --output plots/report_offline_data_cube_double.png

uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s2_offline/sd0_20260422_184643_fql_antsoccer-arena-navigate-singletask-task1-v0_a10.0_od0.1_online_offline \
  --run_dir exp/s2_offline/sd0_20260424_005059_fql_antsoccer-arena-navigate-singletask-task1-v0_a10.0_od0.25_online_offline \
  --label "10% offline data" \
  --label "25% offline data" \
  --title "Retaining offline data on antsoccer" \
  --output plots/report_offline_data_antsoccer.png
```

Put `plots/report_wsrl_cube_double.png` and `plots/report_wsrl_antsoccer.png` in the "Warm-Start RL" section:

```bash
uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s2_wsrl/sd0_20260422_184653_fql_cube-double-play-singletask-task1-v0_a100.0_ws5000_online_offline \
  --run_dir exp/s2_wsrl/sd0_20260424_005110_fql_cube-double-play-singletask-task1-v0_a100.0_ws10000_online_offline \
  --run_dir exp/s2_wsrl/sd0_20260425_134417_fql_cube-double-play-singletask-task1-v0_a100.0_ws20000_online_offline \
  --label "5k WSRL" \
  --label "10k WSRL" \
  --label "20k WSRL" \
  --title "WSRL warm-start on cube-double" \
  --output plots/report_wsrl_cube_double.png

uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s2_wsrl/sd0_20260422_184700_fql_antsoccer-arena-navigate-singletask-task1-v0_a10.0_ws5000_online_offline \
  --run_dir exp/s2_wsrl/sd0_20260424_005118_fql_antsoccer-arena-navigate-singletask-task1-v0_a10.0_ws10000_online_offline \
  --run_dir exp/s2_wsrl/sd0_20260425_134412_fql_antsoccer-arena-navigate-singletask-task1-v0_a10.0_ws20000_online_offline \
  --label "5k WSRL" \
  --label "10k WSRL" \
  --label "20k WSRL" \
  --title "WSRL warm-start on antsoccer" \
  --output plots/report_wsrl_antsoccer.png
```

Put `plots/report_comparison_cube_double.png` and `plots/report_comparison_antsoccer.png` in the "Comparison to My Method" section:

```bash
uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s2_offline/cube-double \
  --run_dir exp/s2_wsrl/cube-double \
  --run_dir exp/s2_ifql/sd0_20260430_204146_ifql_cube-double-play-singletask-task1-v0_e0.85_online_offline \
  --run_dir exp/s2_dsrl/sd0_20260507_101945_dsrl_cube-double-play-singletask-task1-v0_n1.0_online_offline \
  --run_dir exp/s2_qsm/sd0_20260508_091905_qsm_cube-double-play-singletask-task1-v0_a30.0_i30.0_online_offline \
  --run_dir exp/s3_custom/sd0_20260509_195533_custom_cube-double-play-singletask-task1-v0_a100.0_ns64_tns8_od0.25_online_offline \
  --run_dir exp/s3_custom/sd1_20260510_072228_custom_cube-double-play-singletask-task1-v0_a100.0_ns64_tns8_od0.25_online_offline \
  --label "offline-data FQL" \
  --label "WSRL" \
  --label "IFQL" \
  --label "DSRL" \
  --label "QSM" \
  --label "custom seed 0" \
  --label "custom seed 1" \
  --title "Method comparison on cube-double" \
  --output plots/report_comparison_cube_double.png

uv run src/scripts/plot_offline_online_curves.py \
  --run_dir exp/s2_offline/antsoccer-arena \
  --run_dir exp/s2_wsrl/antsoccer-arena \
  --run_dir exp/s2_ifql/sd0_20260430_204147_ifql_antsoccer-arena-navigate-singletask-task1-v0_e0.9_online_offline \
  --run_dir exp/s2_dsrl/sd0_20260507_102032_dsrl_antsoccer-arena-navigate-singletask-task1-v0_n1.0_online_offline \
  --run_dir exp/s2_qsm/sd0_20260508_091959_qsm_antsoccer-arena-navigate-singletask-task1-v0_a30.0_i100.0_online_offline \
  --run_dir exp/s3_custom/sd0_20260510_184948_custom_antsoccer-arena-navigate-singletask-task1-v0_a30.0_ns64_tns8_od0.25_online_offline \
  --run_dir exp/s3_custom/sd1_20260511_062314_custom_antsoccer-arena-navigate-singletask-task1-v0_a30.0_ns64_tns8_od0.25_online_offline \
  --label "offline-data FQL" \
  --label "WSRL" \
  --label "IFQL" \
  --label "DSRL" \
  --label "QSM" \
  --label "custom seed 0" \
  --label "custom seed 1" \
  --title "Method comparison on antsoccer" \
  --output plots/report_comparison_antsoccer.png
```

## Missing Strict-Rubric Runs

The assignment asks for 3 offline-data amounts on cube-double and antsoccer. The current full logs have 10% and 25%. To make the plot strict, run a third amount, for example 50%:

```bash
uv run src/scripts/train_offline_online.py --run_group=s2_offline --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=0 --alpha=100 --offline_data=0.5
uv run src/scripts/train_offline_online.py --run_group=s2_offline --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --alpha=10 --offline_data=0.5
```

The final project comparison asks for results over 2 seeds for the baselines and custom method. The custom method already has 2 seeds. The baseline comparison plots in this draft use seed 0 baseline logs, so for the strict version you should run seed 1 for the baselines you want to compare:

```bash
uv run src/scripts/train_offline_online.py --run_group=s1_fql --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=0 --alpha=100
uv run src/scripts/train_offline_online.py --run_group=s1_fql --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --alpha=30
uv run src/scripts/train_offline_online.py --run_group=s1_fql --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --alpha=100
uv run src/scripts/train_offline_online.py --run_group=s1_fql --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --alpha=30

uv run src/scripts/train_offline_online.py --run_group=s2_offline --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --alpha=100 --offline_data=0.25
uv run src/scripts/train_offline_online.py --run_group=s2_offline --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --alpha=10 --offline_data=0.25

uv run src/scripts/train_offline_online.py --run_group=s2_wsrl --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --alpha=300 --wsrl_steps=20000
uv run src/scripts/train_offline_online.py --run_group=s2_wsrl --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --alpha=10 --wsrl_steps=20000

uv run src/scripts/train_offline_online.py --run_group=s2_ifql --base_config=ifql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --expectile=0.85
uv run src/scripts/train_offline_online.py --run_group=s2_ifql --base_config=ifql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --expectile=0.9

uv run src/scripts/train_offline_online.py --run_group=s2_dsrl --base_config=dsrl --env_name=cube-double-play-singletask-task1-v0 --seed=1 --noise_scale=1.0
uv run src/scripts/train_offline_online.py --run_group=s2_dsrl --base_config=dsrl --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --noise_scale=1.0

uv run src/scripts/train_offline_online.py --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=1 --alpha=30 --inv_temp=30
uv run src/scripts/train_offline_online.py --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --alpha=30 --inv_temp=100
```
