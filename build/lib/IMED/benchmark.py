from numpy.random import default_rng
import standardizingTrans_ndim
import spatial_ST
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import glob, os

def ndim_benchmark(dim_size, dims, dtype, num_resolutions, repetitions):
    ## Generate random dims-dimensional
    ## Data with length dim_size
    rng = default_rng()
    signal = rng.standard_normal(size=(dim_size,)*dims,dtype=dtype)
    # each step increases number of elements by int(dim_size/num_resolutions)**dims
    # alternative is
    """
        signal = rng.standard_normal(size=(int(np.ceil((array_size)**(1/dims))),)*dims,dtype=dtype)
        sizes = (np.ceil((np.linspace(4**dims,array_size,num_resolutions,dtype=int))**(1/dims)))
        resolutions   = [(int(i),)*dims for i in sizes]
        #but equidistant spacing in n dimensions is not ensured, and sometimes there are repetitions
    """
    sizes = np.linspace(4,dim_size,num_resolutions,dtype=int)
    resolutions   = [(i,)*dims for i in sizes] 
    print(resolutions)
    if dims == 2:
        #matrix methods are compatible with 2D input
        methods = ('DCT','FFT','DCT_by_FFT','sepMat','fullMat')
    else:
        methods = ('DCT','FFT','DCT_by_FFT')
    
    first_method = methods[0]
    # Make unique filename
    r = 0
    if not os.path.exists('Benchmark_saves'):
        os.makedirs('Benchmark_saves')
    while os.path.exists(f"Benchmark_saves/{first_method}_runtime{r:02d}.npy"):
        r += 1
    
    # Prepare arrays for saving runtime and resolution    
    nbytes = np.zeros([1],dtype=dtype).nbytes
    filenames     = ()
    mem_filenames = ()
    runtimes      = ()
    memory        = ()
    functions     = ()
    for method in methods:
        filenames     += (f"Benchmark_saves/{method}_runtime{r:02d}.npy",)
        mem_filenames += (f"Benchmark_saves/{method}_mem{r:02d}.npy",)

        runtimes  += (np.zeros([repetitions,num_resolutions]),)
        memory    += (np.zeros([num_resolutions]),)
        # Import relevant methods from module standardizingTrans_ndim
        try:
            functions += (getattr(standardizingTrans_ndim, f'ST_ndim_{method}') ,)
        except: 
            functions += (getattr(spatial_ST, f'ST_{method}') ,)

    
    # Do benchmark and save results 
    for i in range(len(resolutions)):
        signal_crop = np.resize(signal,resolutions[i])
        sizes[i] = signal_crop.size
        

        for j in range(repetitions):
            
            for method_num in range(len(functions)):
                print(f'(i,j)=({i},{j}) with {resolutions[i]} and method {methods[method_num]}')
                if methods[method_num] == 'fullMat':
                    if resolutions[i][0]>100: #and dtype==np.float32:
                        #should be about 25 GB memory
                        print(f'Skipping fullMat at resolution {resolutions[i]}')
                        continue
                    #elif resolutions[i][0]>100 and dtype==np.float64:
                    #    print(f'Skipping fullMat at resolution {resolutions[i]}')
                    #    continue
                        
                    #matrix method
                starttime = time.time()
                functions[method_num](signal_crop,sigma=10.,eps=0.)
                endtime = time.time()
                runtimes[method_num][j,i] = endtime-starttime
                np.save(filenames[method_num],runtimes[method_num])

                # Memory estimate:
                if j==0:
                    if methods[method_num] == 'DCT':
                        #need kernel and signal transform
                        res=np.array(resolutions[i])
                        memory[method_num][i] = 2*np.prod(res)*nbytes
                        
                    elif methods[method_num] == 'FFT':
                        #need real kernel and complex signal transform
                        #whose last axis has shape n//2 + 1
                        fft_res     = np.array(resolutions[i])
                        fft_res[-1] = fft_res[-1]//2+1 
                        memory[method_num][i] = 3*np.prod(fft_res)*nbytes
                        
                    elif methods[method_num] == 'DCT_by_FFT':
                        #kernel and complex transform have shape 
                        fft_res     = np.array(resolutions[i]) 
                        fft_res[0]=fft_res[0]*2-2
                        fft_res[-1] = fft_res[-1]//2+1 
                        memory[method_num][i] = 3*np.prod(fft_res)*nbytes
                        
                    elif methods[method_num] == 'sepMat':
                        # must store G_x,G_y with dims N^2, M^2
                        res=np.array(resolutions[i])
                        memory[method_num][i] = np.sum(res**2)*nbytes
                        
                    elif methods[method_num] == 'fullMat':
                        #must store _at least_ one N^2M^2 matrix
                        res=np.array(resolutions[i])
                        memory[method_num][i] = np.prod(res**2)*nbytes
                    
                    # save in MB
                    np.save(mem_filenames[method_num],memory[method_num]*10**(-6))

        
    np.save(f'Benchmark_saves/sizes{r:02d}_{dims:d}D.npy',sizes)


    """   # Plotting
        colors  = ['gray','blue','darkred', 'orange', 'purple']
        ecolors = colors # ['black','darkblue']
        fig, ax = plt.subplots()
        for num,method in enumerate(methods):
            ax.scatter(sizes,runtimes[num][1,:],color=colors[num],label=methods[num])

        for rep in range(2,repetitions):
            for num in range(len(methods)):
                ax.scatter(sizes,runtimes[num][rep,:],color=colors[num])

        if repetitions>1:
            for num in range(len(methods)):
                ax.errorbar(sizes, runtimes[num][1:].mean(axis=0), yerr=runtimes[num][1:].std(axis=0,ddof=1), ls='none',capsize=5, ecolor = ecolors[num], label = f'{methods[num]} errorbar')
        ax.set_title(f'{dims:d}D Strandardizing Transform Runtime')
        ax.set_xlabel('Number of elements')
        ax.set_ylabel('Runtime [s]')
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.legend()

        fig.savefig(f'Benchmark_saves/RuntimeFig{r:02d}.pdf',bbox_inches='tight')
     """   
        
    file_number = r
    #repetitions = 10

    # Find files with this run number
    all_files = glob.glob(f"Benchmark_saves/*runtime{file_number:02d}*.npy")
    num_files = len(all_files)

    #find correct size file (has # dims suffix)
    for size_file in glob.glob(f"Benchmark_saves/sizes{file_number:02d}*.npy"):
        sizes = np.load(size_file)

    # infer number of dimensions
    dims = int(glob.glob(f"Benchmark_saves/sizes{file_number:02d}*.npy")[0][-6])


    methods = ['DCT','FFT','DCT_by_FFT','sepMat','fullMat']
    methods = methods[:num_files]


    """
    #memory usage estimates:

    import pandas as pd

    for num, method in enumerate(methods):
        mem = np.load(f"{method}_mem{file_number:02d}.npy")
        size_and_mem = np.concatenate((sizes[:,np.newaxis],mem[:,np.newaxis]),axis=1)
        df = pd.DataFrame(data=size_and_mem, columns=["Elements",f"Memory Usage [MB]"],index=None)
        df.style.set_caption(f'{methods[num]}')
        print(f'{methods[num]}')
        print(df)
        print()"""


    runtimes = ()
    for i in range(len(methods)):
        method_runtime = np.load(f'Benchmark_saves/{methods[i]}_runtime{file_number:02d}.npy')
        runtimes += (method_runtime,)


    # Plotting
    colors  = ['black','blue','darkred', 'orange', 'purple']
    ecolors = colors # ['black','darkblue']
    fig, ax = plt.subplots()
    if len(methods)>3:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    methods[2] = 'DCT by FFT'

    for num,method in enumerate(methods):
        if num== 4:
            #ax2.scatter(sizes[:10],runtimes[num][1,:10],color=colors[num],label=methods[num])
            pass
        else:
            #ax.scatter(sizes,runtimes[num][1,:],color=colors[num],label=methods[num])
            pass

    for rep in range(2,repetitions):
        for num in range(len(methods)):
            if num == 4:
                #ax2.scatter(sizes[:10],runtimes[num][rep,:10],color=colors[num])
                pass    
            else:
                #ax.scatter(sizes,runtimes[num][rep,:],color=colors[num])
                pass

    if repetitions>1:
        for num in range(len(methods)):
            if num == 4:
                max_index = runtimes[num].argmin()

                color = colors[num]
                ax2.set_ylabel('fullMat ($G^{1/2}z$) Runtime [s]', color=color)  # we already handled the x-label with ax1
                ax2.errorbar(sizes[:max_index], runtimes[num][1:,:max_index].mean(axis=0), yerr=runtimes[num][1:,:max_index].std(axis=0,ddof=1),capsize=3, color =colors[num], ecolor = colors[num], label = f'{methods[num]}',linestyle='dashed')
                ax2.tick_params(axis='y', labelcolor=color)
                #remove line using ls='none'
                ax2.legend(loc='upper right')
                ax2.set_ylim(ax.get_ylim()[0],ax2.get_ylim()[1])


            else:
                ax.errorbar(sizes, runtimes[num][1:].mean(axis=0), yerr=runtimes[num][1:].std(axis=0,ddof=1), capsize=5, ecolor = colors[num], color = colors[num], label = f'{methods[num]}')

    ax.set_title(f'Runtime of {dims}D Strandardizing Transforms')
    ax.set_xlabel('Elements')#('Length of Each Dimension')
    ax.set_ylabel('Runtime [s]')
    ax.set_ylim([0,ax.get_ylim()[1]])
    ax.legend()

    fig.savefig(f'"Benchmark_saves/{dims}D_runtime_{file_number:d}.png',dpi=150,bbox_inches='tight')



    
#Run in 1D, 2D, 3D    
ndim_benchmark(dim_size=1000000,dims=1,dtype=np.float32, num_resolutions = 50, repetitions=10)
ndim_benchmark(dim_size=1000,dims=2,dtype=np.float32, num_resolutions = 50, repetitions=10)
ndim_benchmark(dim_size=100,dims=3,dtype=np.float32, num_resolutions = 20, repetitions=10)



