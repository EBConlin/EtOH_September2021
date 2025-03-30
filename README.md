# EtOH_September2021
Code for analysis of BLA power spectrum during binge/withdrawal periods in rodents.
This code applies MRI techniques (Threshold Free Cluster Enhancement) to search for significant bands of activitiy in frequency space without having to bin data into gamma, delta, etc.. This would
allow researchers to find competing frequency activity within frequency bands. Additionally, unlike other methods, TFCE makes no assumptions about how significant a cluster should be before it is counted.

off_proc_TFCE contains code for analyzing the correlation of average power over sessions in ~1,000 frequency bins to the amount of alcohol ingested over session,
as well as the correlation of power ratios between frequencies within 20 Hz of one another. Also included is code to perform wilcoxon signed rank comparisontests, across individual
frequencies and neighboring frequency power ratios, between different sessions (binge and break for example). Correction for multiple comparisons and adjustment 
for the correlation of close frequency binnings is accomplished by Threshold-Free Cluster Enhancement. 

TestGraphs contains synthetic data run through the code provided for off_proc_TFCE. Graphs labeled "Reg" were convolved in multiple frequency ranges with the drinking sequence.
Graphs labeled "Inv" were convolved in multiple frequency ranges with (1 - drinking sequence). Graphs labeled "Shuffle" were convolved 
in multiple frequency ranges with a shuffled version of the drinking sequence. "Pos" graphs correspond to positive correlation, while "Neg" graphs correspond
to negative correlation, as TFCE is unable to deal with negative values.


![image (3)](https://github.com/user-attachments/assets/b147cda6-bd01-4190-ae78-ef722da1e01a)
