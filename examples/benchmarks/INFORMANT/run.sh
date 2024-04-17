if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi

# set the config
universe=nasdaq100
only_backtest=false

sed -i "s/csi.../$universe/g" workflow_config_informant.yaml
if [ $universe == 'csi300' ]; then
    sed -i "s/SH....../SH000300/g" workflow_config_informant.yaml
elif [ $universe == 'csi500' ]; then
    sed -i "s/SH....../SH000905/g" workflow_config_informant.yaml
fi
if $only_backtest; then
    nohup python -u main.py --only_backtest > ./backtest/${universe}.log 2>&1 &
else
    nohup python -u main.py > ./logs/${universe}.log 2>&1 &
fi
echo $!