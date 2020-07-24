# 3CLSTM

In order to get performance stats, run:

```
$ docker build -t 3clstm .
$ docker run -it --rm 3clstm bash
# chmod 777 3clstm/run_tests.sh
# cd 3clstm/
# ./run_tests.sh
```

Results will be inside the docker itself in the `results/` folder.
