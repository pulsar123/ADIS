#!/bin/bash

# The script to do destarring for FITS files.

if test $# -eq 0
  then
  echo "Script for destarring."
  echo "Arguments: master.fits  list_of_FITS_files"
  exit
  fi


MASTER="$1"
shift

for RAW in $*
  do
  echo "+++ Processing $RAW +++"
 OUTPUT="ds_$(echo $RAW | rev | cut -d. -f2- | rev)"
 \rm ${OUTPUT}.fit &>/dev/null
 reconv -v -m "$MASTER" -i $RAW -o ${OUTPUT}.fit -R 8 -bias 0.03 -bmax 50
 
 echo
done

