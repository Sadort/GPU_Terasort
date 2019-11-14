#!/usr/bin/env
import os

powerdict = {};
with open('powerresults.txt') as fp:
    for line in fp:
        t_idx_start = line.find("Time") + 7;
        p_idx_start = line.find("Power") + 8;
        t_idx_stop = line.find("Power") - 3;
        p_idx_stop = line.find("Mem") - 2;
        time = int(line[t_idx_start:t_idx_stop]);
        power = int(line[p_idx_start:p_idx_stop]);
        if time not in powerdict:
            powerdict[time] = power;
        else:
            if powerdict[time] != power:
                print "Conflict\n";
                break;
sorted_powertuple = sorted(powerdict.iteritems());
sorted_powerlist = [list(elem) for elem in sorted_powertuple];
length = len(sorted_powerlist);
#print length;
    
print type(sorted_powerlist);
#for i in sorted_powerdict:
#    print i;

elapsed_time = sorted_powerlist[length-1][0] - sorted_powerlist[0][0];
max_power = sorted_powerlist[0][1];
for a, b in sorted_powerlist:
    if b > max_power:
        max_power = b;
#print elapsed_time, " ", max_power;

for i in range(1, length):
    sorted_powerlist[i][0] = sorted_powerlist[i][0] - sorted_powerlist[0][0];
sorted_powerlist[0][0] = 0;

#for i in sorted_powerlist:
#    print i;

with open('power.txt', 'w') as fp:
    for item in sorted_powerlist:
        fp.write(str(item[0]) + " " + str(item[1]) + "\n");

os.system('rm -rf powerresults.txt');
