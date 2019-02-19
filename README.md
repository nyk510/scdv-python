# SCDV Python

SCDV python Implementation

## Requirements

```console
python==3.6
and so on
```

## Setup

### Recommended

```bash
cp project.env .env
docker-compose build

docker-compose up -d

docker exec -it scdv-jupyter bash
# or access to localhost:7001
```

### note

jupyter's default password is written on Dockerfile arg.
The default is `"dolphin"`.

### Trouble Shooting

If you catch error as below

```bash
ERROR: for scdv-jupyter  Cannot start service jupyter: driver failed programming external connectivity on endpoint scdv-jupyter 
(a16e504598f6081390b47fe6809aaba1a8b52672956e65feb11d3c00773363ba): Bind for 0.0.0.0:7011 failed: port is already allocated
```

, change .env `JUPYTER_PORT` number.
