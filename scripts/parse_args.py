"""
python scripts/parse_args.py 1,2,3
or 
python scripts/parse_args.py 1-6
"""
import sys

a = sys.argv[1]

if '-' in a:
  ll = a.split('-')
  start = int(ll[0])
  end = int(ll[1])
  results = range(start, end+1)
  results = [str(x) for x in results]
else:
  ll = a.split(",")
  results = [x for x in ll]

print(" ".join(results))
