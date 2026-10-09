#!/bin/bash
# Double-click this file in Finder to install NoteTaker on a Mac.
#
# The first time, macOS may say it "cannot be opened because it is from an
# unidentified developer". Right-click (or Control-click) the file, choose
# Open, then click Open again. That is only needed once.

cd "$(dirname "$0")" || exit 1
clear
echo "Installing NoteTaker. This window will ask a few questions; press Return"
echo "to accept the suggested answer."
echo
bash ./install.sh
echo
read -r -p "All done. Press Return to close this window. " _
