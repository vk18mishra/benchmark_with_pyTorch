import matplotlib.pyplot as plt
import psutil
from datetime import datetime
import pandas as pd
import time
import os
def get_processes_info():
    # the list the contain all process dictionaries
    processes = []
    for process in psutil.process_iter():
        # get all process info in one shot
        with process.oneshot():
            # get the process id
            pid = process.pid
            name = process.name()
            cpu_usage = process.cpu_percent()
            try:
            # get the memory usage in bytes
                memory_usage = process.memory_full_info().uss
            except psutil.AccessDenied:
                memory_usage = 0
            processes.append({
                'pid': pid, 'cpu_usage': cpu_usage, 'name': name, 'memory_usage': memory_usage,
            })

    return processes

def construct_dataframe(processes, columns):
    # convert to pandas dataframe
    df = pd.DataFrame(processes)
    # set the process id as index of a process
    df = df.sort_values(by=['pid'])
    #df.set_index('pid', inplace=True)
    df = df[columns.split(",")]
    return df


if __name__ == "__main__":
    columns = "pid,name,cpu_usage,memory_usage"
    cnt = 0
    cpu_use = []
    mem_use = []
    try:
        while 1:
            # get all process info
            processes = get_processes_info()
            os.system("cls") if "nt" in os.name else os.system("clear")
            df = construct_dataframe(processes, columns)
            print(df.to_string())

            #try:
            s = df[df.eq('mprof').any(1)]
            if s.empty:
                print('DataFrame is empty!')
            else:
                #print("pid:   ",s.iloc[0]['pid'])
                #print(s)
                mprof_pid = s.iloc[0]['pid']
                all_pids = df["pid"].tolist()
                #print("all pids: ",all_pids)
                mprof_ind = all_pids.index(mprof_pid)
                mprof_ind = mprof_ind+1
                main_pid = all_pids[mprof_ind]
                print("main pid: ",main_pid)
                s1 = df[df.eq(main_pid).any(1)]
                print(s1)
                #except:
                #    print("main not started")
                cpu_use_t = s1.iloc[0]['cpu_usage']
                mem_use_t = s1.iloc[0]['memory_usage']
                cnt = cnt+1
                #print("CPU Usage: ",cpu_use_t)
                mem_use.append(mem_use_t)
                cpu_use.append(cpu_use_t)
            time.sleep(4)
    except KeyboardInterrupt:
        len_cnt = len(cpu_use)
        len_cnt = len_cnt*4
        x_time = list(range(0, len_cnt, 4))
        print("Timestamps: ",x_time)
        print("CPU Consumption(%): ",cpu_use)
        print("Memory Usages: ",mem_use)
        plt.plot(x_time, cpu_use, '-o')
        plt.xlabel('Time(in secs)')
        plt.ylabel('CPU Consumption(%)')
        plt.title('Tracking CPU Consumption Overtime using second process')
        plt.savefig('cpu_consumption_overtime.png')
        plt.clf()
        plt.plot(x_time, mem_use, '-o')
        plt.xlabel('Time(in secs)')
        plt.ylabel('Memory Consumption(bytes)')
        plt.title('Tracking Memory Consumption Overtime using second process')
        plt.savefig('memory_consumption_with_secondPRO.png')


