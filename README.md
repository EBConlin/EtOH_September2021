# EtOH_September2021
Code for analysis of BLA power spectrum during binge/withdrawal periods in rodents.

off_proc_TFCE contains code for analyzing the correlation of average power over sessions in ~1,000 frequency bins to the amount of alcohol ingested over session,
as well as the correlation of power ratios between frequencies within 20 Hz of one another. Also included is code to perform wilcoxon signed rank comparisontests, across individual
frequencies and neighboring frequency power ratios, between different sessions (binge and break for example). Correction for multiple comparisons and adjustment 
for the correlation of close frequency binnings is accomplished by Threshold-Free Cluster Enhancement. 

TestGraphs contains synthetic data run through the code provided for off_proc_TFCE. Graphs labeled "Reg" were convolved in multiple frequency ranges with the drinking sequence.
Graphs labeled "Inv" were convolved in multiple frequency ranges with (1 - drinking sequence). Graphs labeled "Shuffle" were convolved 
in multiple frequency ranges with a shuffled version of the drinking sequence. "Pos" graphs correspond to positive correlation, while "Neg" graphs correspond
to negative correlation, as TFCE is unable to deal with negative values.
