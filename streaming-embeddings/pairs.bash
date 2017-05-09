#!/bin/bash

set -e
[ $# -eq 3 ]

join <(sed 's/^/1 /' $1) <(sed 's/^/1 /' $2) | sed 's/^1 //;s/ /\t/' > $3
