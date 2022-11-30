RELATIVE_DIR=`dirname "$0"`
cd $RELATIVE_DIR

python3 build_FBA.py --GPU 1
python3 build_PBA.py --GPU 1
python3 build_WGA.py --GPU 1
python3 build_WDA.py --GPU 1

python3 display_FBA.py --GPU 1 --fig_size 2
python3 display_PBA.py --GPU 1 --fig_size 2 
python3 display_WGA.py --GPU 1 --fig_size 2
python3 display_WDA.py --GPU 1 --fig_size 2
