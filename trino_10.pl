#!/usr/bin/perl

### COMMON ###
$spath = "/home/scripts/PREDICT/TRAINING_DATA";
$cpath = "/home/scripts/PREDICT/COMMON";

### SET DATES FOR DATA PULL ###

$startdate = `date +\"%Y-%m-%d\"`; ### START TODAY ###
#$startdate = "2022-07-15"; ### or SET DATE HERE ###
$daysback_start = 85;  # Needs to be a bigger integer than $daysback_end 
$daysback_end = 75;    # Needs to be a smaller integer than $daysback_start
$daysback_diff = $daysback_start - $daysback_end;
print "$daysback_diff\n";

if($daysback_diff < 1){
  print "--- Time window zero or less than zero.  Check \$daysback_start and \$daysback_end ---\n";
  exit;
}

open(DATE,">","$cpath/training_dates.txt") || die("cannot open $cpath/training_dates.txt\n");
for ($i = 0; $i < $daysback_diff; $i++){
  $db = $daysback_end + $i;
  $dd = `date -d \"$startdate - $db days\" +\"%Y-%m-%d\"`;
  chomp($dd);
  print DATE "$dd\n";
}
close(DATE);

### READ IN DATES FOR DATA PULL ###

open(DATE,"$cpath/training_dates.txt") || die("cannot open $cpath/training_dates.txt\n");
while($row=<DATE>){
  chomp($row);
  push @dates, $row;
}
close(DATE);

### READ DATA TABLES TO PULL FROM IN PREDICT ###

### format: catalog.schema.datatable,datestamp_column,list-of-space-separated-column-names ###
open(IN,"$cpath/joinfiles.txt") || die("cannot open $cpath/joinfiles.txt\n");
while($row=<IN>){
  chomp($row);
  push @datatable_info, $row;
}
close(IN);

### LOOP THROUGH DATA TABLE NAMES, PULL DATA BASED ON DATES ###

foreach $dt (@datatable_info){
  @dt = split(/,/,$dt);
  $data_table = @dt[1];
  $date_column = @dt[2];
  $cols = @dt[3];
  $cols =~ s/ /,/g;
  $cmd = "";
  $count_days = 0;
  foreach $date (@dates){
    if($count_days == 0){
      $cmd = "select $cols from $data_table where cast($date_column as varchar) like \'%$date%\' ";
    }
    if($count_days == 1){
      $cmd = "$cmd or cast($date_column as varchar) like \'%$date%\' "; 
    }
    $count_days = 1;
  }
  print "\n\n$cmd;\n";

  open(CMD,">","$cpath/sql_10.txt") || die("cannot open $cpath/sql_10.txt\n");
  print CMD "$cmd;";
  close(CMD);

  system("java -jar $cpath/trino-cli-391-executable.jar --server https://query.comcast.com:9443 --truststore-password changeit --user sghiz001c --password --file=$cpath/sql_10.txt --output-format=CSV_HEADER > $spath/CSV/trino_out-$data_table.csv ;");


}

exit;

########################################################################
