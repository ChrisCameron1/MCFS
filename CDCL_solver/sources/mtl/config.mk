##
##  This file is for system specific configurations. For instance, on
##  some systems the path to zlib needs to be added. Example:
##
 CFLAGS += -I/usr/local/include
 LFLAGS += -L/usr/local/lib

# Linux (Python 3)
#CFLAGS += -I/global/scratch/chrisc/anaconda3/include/python3.7m -I/global/scratch/chrisc/anaconda3/include/python3.7m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe  -fdebug-prefix-map=/tmp/build/80754af9/python_1565725737370/work=/usr/local/src/conda/python-3.7.4 -fdebug-prefix-map=/global/scratch/chrisc/anaconda3=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall

CFLAGS += -I/global/scratch/wildebeest/libtorch/include -D_GLIBCXX_USE_CXX11_ABI=1

#-I/global/scratch/chrisc/anaconda3/include/python3.7m -Wno-unused-result -Wsign-compare -march=nocona -ftree-vectorize -fPIC -O3 -pipe  -fdebug-prefix-map==/usr/local/src/conda/- -fdebug-prefix-map==/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall -std=c++11 -Wno-reorder -Wno-unused-but-set-variable

#LFLAGS += -L/global/scratch/chrisc/anaconda3/lib/python3.7/config-3.7m-x86_64-linux-gnu -L/global/scratch/chrisc/anaconda3/lib -lpython3.6m -lcrypt -lpthread -ldl  -lutil -lrt -lm

# Add flags for linking to libtorch
LFLAGS += -L/global/scratch/wildebeest/libtorch/lib -ltorch -lc10 -lX11 -ltorch_cpu -ltorch_cuda #-Wl,--no-as-needed

#-L/global/scratch/chrisc/anaconda3/lib/python3.7/config-3.7m-x86_64-linux-gnu -L/global/scratch/chrisc/anaconda3/lib -lpthread -ldl  -lutil -lrt -lm  -Xlinker -export-dynamic -lpython3.7m #-flto=no

#-lpython3.7m
# Linux (Python 2)
#CFLAGS += -I/global/scratch/rex/Python/miniconda3/envs/py27/include/python2.7 -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++11 -Wno-reorder -Wno-unused-but-set-variable

#LFLAGS += -L/global/scratch/rex/Python/miniconda3/envs/py27/python2.7/config -L/global/scratch/rex/Python/miniconda3/envs/py27/lib -L -lpython2.7 -lpthread -ldl -lutil -lm -Xlinker -export-dynamic

# Windows
#CFLAGS += -I../../../../../../../../cygwin64/usr/include/python3.6m -I../../../../../../../../cygwin64/lib/python3.6/site-packages/numpy/core/include/numpy -Wno-unused-result -Wsign-compare -march=nocona -ftree-vectorize -fPIC -O3 -pipe  -fdebug-prefix-map==/usr/local/src/conda/- -fdebug-prefix-map==/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall -Wno-reorder -Wno-unused-but-set-variable

#LFLAGS += -L:../../../../../../../../cygwin64/lib/python3.6/config-3.6m-x86_64-cygwin -L../../../../../../../../cygwin64/lib/python3.6 -l python3.6m.dll -lpthread -ldl -lutil -lrt -lm  -Xlinker

# export LD_LIBRARY_PATH=/global/scratch/chrisc/experiments/DeepLearningForSAT/libtorch/lib

