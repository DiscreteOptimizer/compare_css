### Author Florian Roesel 08.09.2023

# First: PREPARE an instance, e.g., type

python3 instance_prep.py -i Persistent_Random -o Prep_Random random_100_200_100_100_200

(type "python3 instance_prep.py --help" for further information)

# Then, Solve the prepared instance, e.g., type

python3 compare_css.py --instancepath Prep_Random -m 0 random_100_200_100_100_200

(type "python3 compare_css.py --help" for further information)

## Furthermore...
The program "generate_grid.py" generates NWD instances. Type "python3 generate_grid.py --help" for further information.
The program "create_MIP.py" generates Random MIP instances. Type "python3 create_MIP.py --help" for further information.

### Questions? Contact florian.roesel@fau.de / florian.roesel@outlook.com

