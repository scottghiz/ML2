#!/usr/bin/perl

$count = 0;
open(WO,"/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_5.csv") || die("cannot open /home/scripts/PREDICT/TRAINING_DATA/CSV/joined_5.csv\n");
while($row=<WO>){
  chomp($row);
  if($row !~ m/wo_details_LL/){
    $row =~ s/\"//g;
    $row =~ s/\[/,/g;
    $row =~ s/\]/,/g;
    $row =~ s/\{/,/g;
    $row =~ s/\}/,/g;
    @row = split(/,/,$row);

    $index = @row[0];
    $isWorkorderCustomerFacing = "";
    $isDropAndGoJob = "";
    $jobTypeDescription = "";
    $hasGram = "";
    $isX1Account = "";
    $isAllWirelessIPVideo = "";
    $isXfinityFlexAccount = "";
    $isMultiTenantGatewayAccount = "";
    $proactiveSwapSubtype = "";
    $proactiveSwapMsg = "";

    foreach $thing (@row){
      if($thing =~ m/isWorkorderCustomerFacing/){
        @t = split(/:/,$thing);
        $isWorkorderCustomerFacing = @t[1];
      }
      if($thing =~ m/isDropAndGoJob/){
        @t = split(/:/,$thing);
        $isDropAndGoJob = @t[1];
      }
      if($thing =~ m/jobTypeDescription/){
        @t = split(/:/,$thing);
        $jobTypeDescription = @t[1];
      }
      if($thing =~ m/hasGram/){
        @t = split(/:/,$thing);
        $hasGram = @t[1];
      }
      if($thing =~ m/isX1Account/){
        @t = split(/:/,$thing);
        $isX1Account = @t[1];
      }
      if($thing =~ m/isAllWirelessIPVideo/){
        @t = split(/:/,$thing);
        $isAllWirelessIPVideo = @t[1];
      }
      if($thing =~ m/isXfinityFlexAccount/){
        @t = split(/:/,$thing);
        $isXfinityFlexAccount = @t[1];
      }
      if($thing =~ m/isMultiTenantGatewayAccount/){
        @t = split(/:/,$thing);
        $isMultiTenantGatewayAccount = @t[1];
      }
      if($thing =~ m/proactiveSwapSubtype/){
        @t = split(/:/,$thing);
        $proactiveSwapSubtype = @t[1];
      }
      if($thing =~ m/proactiveSwapMsg/){
        @t = split(/:/,$thing);
        $proactiveSwapMsg = @t[1];
      }
    }

    $record = $index.",".$isWorkorderCustomerFacing.",".$isDropAndGoJob.",".$jobTypeDescription.",".$hasGram.",".$isX1Account.",".$isAllWirelessIPVideo.",".$isXfinityFlexAccount.",".$isMultiTenantGatewayAccount.",".$proactiveSwapSubtype.",".$proactiveSwapMsg;
    push @wo_all,$record;
    print "$count\r";
    $count++;
  }
}
close(WO);

print "what is going on?\n";

open(WW,">","/home/scripts/PREDICT/TRAINING_DATA/CSV/wo_0.csv") || die("cannot open /home/scripts/PREDICT/TRAINING_DATA/CSV/wo_0.csv\n");
print WW "index,isWorkorderCustomerFacing,isDropAndGoJob,jobTypeDescription,hasGram,isX1Account,isAllWirelessIPVideo,isXfinityFlexAccount,isMultiTenantGatewayAccount,proactiveSwapSubtype,proactiveSwapMsg\n";
foreach $ww (@wo_all){
  print WW "$ww\n";
}
close(WW);


exit;
