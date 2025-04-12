# Cool example to show status bar
import tqdm, time
from joblib import Parallel, delayed

show_status = 1
# Using only 1 cpu
for idx in tqdm.tqdm(range(1000), desc = 'doing cool stuff', disable = not show_status):
    time.sleep(0.001)

# Using multiple (10) cpus
def Sleep(index):
    time.sleep(0.001)

DerivedSamples = Parallel(n_jobs=10)(delayed(Sleep)(idx) for idx in tqdm.tqdm(range(1000), desc = 'doing cool stuff', disable = not show_status))
