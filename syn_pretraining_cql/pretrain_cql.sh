# Test MDP pretraining
python pretrain_cql.py --exp_prefix test --device cpu -it 5 -n 2 -nS 10 -nA 10 -pt 1 -tt 1 -dS 11 -dA 3

# Generate 1M synthetic MDP data with states/actions 100 and both temperatures 1 (CQL_MDP variant) and pre-train for Hopper environment
python pretrain_cql.py --exp_prefix cql_hopper_mdp --device cpu -it 100000 -n 1000 -nS 100 -nA 100 -pt 1 -tt 1 -dS 11 -dA 3

# Generate 1M synthetic MDP data with i.i.d. states/actions 100 (CQL_IID variant) and pre-train for Hopper environment
python pretrain_cql.py --exp_prefix cql_hopper_iid --device cpu -it 100000 -n 1000 -nS 100 -nA 100 -pt inf -tt inf -dS 11 -dA 3
