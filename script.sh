python run_dense_net.py --depth=40 --train --test --logs -- dataset=C10 --sdr --exp_name="sdr"
python run_dense_net.py --depth=40 --train --test --logs -- dataset=C10 --slr --exp_name="slr"
python run_dense_net.py --depth=40 --train --test --logs -- dataset=C10 --keep_prob=0.8 --exp_name="dropout"
python run_dense_net.py --depth=40 --train --test --logs -- dataset=C10 --stochastic --exp_name="stochastic"

