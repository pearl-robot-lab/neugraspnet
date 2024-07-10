import torch
import numpy as np
import os
import time

def search(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False, threshold=5., max_load_iters=100):
    
    '''
    threshold(float): Threshold for measuring the new optimum, to only focus on significant changes. unit is second.
    '''
    
    max_num_workers = os.cpu_count() #refer to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py    
    init_num_workers = max_num_workers // 2

    num_workers_list = [init_num_workers, 0, 1]
    num_workers_list += list(np.arange(start= 2, stop=init_num_workers + 1)[::-1]) 
    num_workers_list += list(np.arange(init_num_workers + 1, max_num_workers + 1))

    num_workers_list = np.array(num_workers_list)
    
    # input [1, 0, 1, 2, 3], output [1, 0, 2, 3]
    _, order_preserved_indexes = np.unique(num_workers_list, return_index=True) 
    num_workers_list = num_workers_list[np.sort(order_preserved_indexes)]

    optimal_num_worker = 0
    min_total_time = np.finfo(np.float).max
    
    skip = np.zeros(len(num_workers_list))
    
    for i, num_workers in enumerate(num_workers_list): # [0, max_num_workers]
        if skip[i]:
            continue
        
        loader = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            # sampler=sampler,
                                            # batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            # collate_fn=collate_fn,
                                            pin_memory=pin_memory,
                                            drop_last=drop_last,
                                            # timeout=timeout,
                                            # worker_init_fn=worker_init_fn,
                                            # prefetch_factor=prefetch_factor,
                                            persistent_workers=persistent_workers)
        
        t1 = time.time()
        total_time = 0
        iters = 0
        for _ in loader:
            iters += 1
            t2 = time.time()
            total_time += t2 - t1
            if total_time > min_total_time:
                break
            t1 = time.time()

            if iters > max_load_iters:
                print(f"batch_size{batch_size}, num_workers{num_workers}, total_time(sec): ", total_time)
                break
        
        if total_time < min_total_time:
            optimal_num_worker = num_workers
            if min_total_time - total_time < threshold:
                break
            min_total_time = total_time
        else: # total_time >= min_total_time
            if num_workers == 0:
                skip[num_workers_list == 1] = 1
            elif num_workers >= 2 and num_workers < optimal_num_worker:
                skip[num_workers_list < optimal_num_worker] = 1
            else:
                break
            
    return optimal_num_worker
