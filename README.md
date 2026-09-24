# Solution attempt at ROADEF26

## Checking a solution
To check a solution you have to build the checker first. For instructions read
the README.md of the checker subdirectory.

If you have a built checker executable then you can optionally create a symlink
to the executable, for example with:
```
ln -sf checker/bin/<executable_name> check
```

With this you can check the solution of any instance with
```
./check --instance path/to/instance/<instance_name>
```

For example there is a toy instance under `data/toy` with the instance prefix
`toy-*`. You can check this with
```
./check --instance data/toy/toy
```

Optionally, if you want more information from the checker output you can add
the `--verbose` flag. You can also pipe the checker's output into `jq` (install
`jq` if you dont already have it) with
```
./check --instance data/toy/toy | jq .

```

This way the json output that the checker produces will be nicely formatted,
and you can query this json with `jq` if you wish to.

NOTE: It is recommended to build the checker with FTXUI enable and running the
checker with the `--pretty` flag for nice outputs.

