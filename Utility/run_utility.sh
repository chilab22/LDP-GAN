RELATIVE_DIR=`dirname "$0"`
cd $RELATIVE_DIR

python3 build_DWS.py --GPU 0
python3 build_DWA.py --GPU 0
python3 build_DWP.py --GPU 0
python3 build_Corr.py --GPU 0

python3 display_DWS.py --fig_size=3
python3 display_DWA.py --fig_size=3
python3 display_DWP.py --fig_size=3
python3 display_Corr.py --fig_size=3


