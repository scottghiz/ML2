#!/usr/bin/perl

$spath = "/home/scripts/PREDICT/TRAINING_DATA";
$cpath = "/home/scripts/PREDICT/COMMON";
$logpath = "/home/scripts/PREDICT/TRAINING_DATA/LOGS";

#system("time $spath/trino_10.pl >  $logpath/trino_10.log");

system("time $spath/joined_0a.py > $logpath/joined_0a.log");
system("time $spath/joined_0b.py > $logpath/joined_0b.log");
system("time $spath/joined_1.py >  $logpath/joined_1.log");
system("time $spath/joined_2.py >  $logpath/joined_2.log");
system("time $spath/joined_3.py >  $logpath/joined_3.log");
system("time $spath/joined_4.py >  $logpath/joined_4.log");
system("time $spath/joined_5.py >  $logpath/joined_5.py.log");
system("time $spath/joined_5.pl >  $logpath/joined_5.pl.log");
system("time $spath/joined_6.py >  $logpath/joined_6.log");

#system("/ML/XGB.FEAT_IMP.HPARAM_TUNING.0.py > /ML/XGB.FEAT_IMP.HPARAM_TUNING.0.log");
#system("/ML/XGB_scikit-learn_TRAINING_0.py > /ML/XGB_scikit-learn_TRAINING_0.log");
#system("/ML/XGB_learning_API_TRAINING_0.py > /ML/XGB_learning_API_TRAINING_0.log");
#system("/ML/XGB_scikit-learn_PREDICT_0.py > /ML/XGB_scikit-learn_PREDICT_0.log");


exit;
