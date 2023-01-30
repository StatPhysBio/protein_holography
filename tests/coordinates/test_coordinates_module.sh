echo "Testing coordinate-gathering routines"
pytest test_coordinates.py -ls

echo "Testing parallel implementation of routines"
pytest test_parallel_coordinates.py -ls
