import sys
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData

import yaml
import argparse
import os
import pprint as pp
import numpy as np

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_backtest", action="store_true", help="whether only backtest or not")
    parser.add_argument("--only_stock", action="store_true", help="whether use news_data or not")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    provider_uri = os.path.expanduser("~/.qlib/qlib_data/my_data")
    print(os.path.exists(provider_uri))
    GetData().qlib_data(target_dir=provider_uri, region=REG_US, exists_skip=True)
    qlib.init(provider_uri = provider_uri, region=REG_US)
    with open("./workflow_config_informant.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    p_conf = config["task"]["preprocess"]
    p_path = os.path.expanduser(config["task"]["preprocess"]['kwargs']['output_file_path'])
    if not (os.path.exists(p_path) and config["task"]["preprocess"]["preprocessed"]):
        try:
            os.makedirs(p_path)
        except:
            p = init_instance_by_config(p_conf)
            p.preprocess()
            config["task"]["preprocess"]["preprocessed"] = True
    
    # h_conf = config["task"]["dataset"]["kwargs"]["handler"]
    # h_path = DIRNAME / f'handler_{config["task"]["dataset"]["kwargs"]["segments"]["train"][0].strftime("%Y%m%d")}' \
    #                    f'_{config["task"]["dataset"]["kwargs"]["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    # if not h_path.exists():
    #     h = init_instance_by_config(h_conf)
    #     h.to_pickle(h_path, dump_all=True)
    #     print('Save preprocessed data to', h_path)
    # config["task"]["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
    
    dataset = init_instance_by_config(config['task']["dataset"])
    
    
    

    ###################################
    # train model
    ###################################

    if not os.path.exists('./model'):
        
        
        os.mkdir("./model")

    all_metrics = {
        k: []
        for k in [
            "IC",
            "ICIR",
            "Rank IC",
            "Rank ICIR",
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.information_ratio",
        ]
    }

    for seed in range(0, 3):
        print("------------------------")
        print(f"seed: {seed}")

        config['task']["model"]['kwargs']["seed"] = seed
        model = init_instance_by_config(config['task']["model"])

        # start exp
        if not args.only_backtest:
            model.fit(dataset=dataset)
        else:
            model.load_model(f"./model/{config['market']}master_{seed}.pkl")

        with R.start(experiment_name=f"workflow_seed{seed}"):
            # prediction
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # Signal Analysis
            sar = SigAnaRecord(recorder)
            sar.generate()

            # backtest. If users want to use backtest based on their own prediction,
            # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
            par = PortAnaRecord(recorder, config['port_analysis_config'], "day")
            par.generate()

            metrics = recorder.list_metrics()
            print(metrics)
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pp.pprint(all_metrics)
    
    for k in all_metrics.keys():
        print(f"{k}: {np.mean(all_metrics[k])} +- {np.std(all_metrics[k])}")
