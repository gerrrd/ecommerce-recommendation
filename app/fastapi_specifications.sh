#!/bin/bash
lines=$(git status | grep modified | grep -c app/schemas.py)
dir=$(dirname "$0")

if [ "$lines" -ge "1" ] || [ ! -e "$dir/api-specifications/openapi.yaml" ];
then
  echo 'Generating API specifications';
  echo "Running $dir/generate_specifications.py"
  python "$dir/generate_specifications.py"
else
  echo "$dir/api-specifications/openapi.yaml already up-to-date";
fi;
