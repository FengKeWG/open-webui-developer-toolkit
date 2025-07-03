# redis.py

`backend/open_webui/utils/redis.py` collects helper functions for working with Redis.
It supports both direct connections and [Sentinel](https://redis.io/docs/interact/sentinel/) clusters and
is pulled in by `AppConfig` when initialising the global cache.  All processes
can therefore share configuration through a single Redis instance.

## `parse_redis_service_url`

This function extracts connection info from a `redis://` URL. The return value
includes the optional username/password, the service name (or host), port and
selected database.

```python
config = parse_redis_service_url("redis://user:pass@redis-host:6380/0")
# {
#     "username": "user",
#     "password": "pass",
#     "service": "redis-host",
#     "port": 6380,
#     "db": 0,
# }
```

The service portion doubles as the Sentinel *master name* when Sentinel is used.

## `get_redis_connection`

Creates a `redis.Redis` connection.  When a list of sentinels is supplied it
first constructs a `redis.sentinel.Sentinel` object and obtains the master
connection.  Otherwise `redis.Redis.from_url` is used directly.

Example connecting to a Sentinel cluster:

```python
sentinel = redis.sentinel.Sentinel(
    redis_sentinels,
    port=cfg["port"],
    db=cfg["db"],
    username=cfg["username"],
    password=cfg["password"],
    decode_responses=True,
)
return sentinel.master_for(cfg["service"])
```

Direct connection example:

```python
cache = get_redis_connection("redis://localhost:6379/0", [])
```

## Environment helpers

`get_sentinels_from_env` turns comma separated hostnames and a port number into
the list of `(host, port)` tuples expected by `Sentinel`.  `get_sentinel_url_from_env`
constructs a usable `redis+sentinel://` URL from the same data.  They are
typically fed values from the `REDIS_SENTINEL_HOSTS` and `REDIS_SENTINEL_PORT`
environment variables.

```python
import os
hosts = get_sentinels_from_env(
    os.getenv("REDIS_SENTINEL_HOSTS"),
    os.getenv("REDIS_SENTINEL_PORT"),
)
url = get_sentinel_url_from_env(
    os.getenv("REDIS_URL"),
    os.getenv("REDIS_SENTINEL_HOSTS"),
    os.getenv("REDIS_SENTINEL_PORT"),
)
```

These helpers normalise environment configuration before it reaches `redis-py`.

### Using in `AppConfig`

The application configuration object stores values in Redis when a URL is
provided.  This allows multiple worker processes to keep their settings in sync.

```python
app.state.config = AppConfig(
    redis_url=REDIS_URL,
    redis_sentinels=get_sentinels_from_env(
        REDIS_SENTINEL_HOSTS,
        REDIS_SENTINEL_PORT,
    ),
)
```

With this setup one instance changing a value will cause the others to pick it
up the next time `AppConfig.__getattr__` fetches the key from Redis.

## Redis backed helper classes

Two small utilities under `backend/open_webui/socket/utils.py` rely on the
connection helpers above to coordinate websocket state across processes.

### `RedisDict`

`RedisDict` exposes a dictionary-like interface backed by a Redis hash.  Each
operation simply delegates to `hset`, `hget` and related commands while
serialising values to JSON.

```python
from open_webui.socket.utils import RedisDict
from open_webui.utils.redis import get_sentinels_from_env

pool = RedisDict(
    "open-webui:session_pool",
    redis_url=WEBSOCKET_REDIS_URL,
    redis_sentinels=get_sentinels_from_env(
        WEBSOCKET_SENTINEL_HOSTS,
        WEBSOCKET_SENTINEL_PORT,
    ),
)

pool["abc"] = {"id": 1}
assert "abc" in pool
```

### `RedisLock`

`RedisLock` provides a simple distributed mutex using `SET` with `nx=True` and
an expiry.  The websocket cleanup task uses it so that only one worker removes
stale usage entries.

```python
from open_webui.socket.utils import RedisLock
from open_webui.utils.redis import get_sentinels_from_env

cleanup_lock = RedisLock(
    redis_url=WEBSOCKET_REDIS_URL,
    lock_name="usage_cleanup_lock",
    timeout_secs=WEBSOCKET_REDIS_LOCK_TIMEOUT,
    redis_sentinels=get_sentinels_from_env(
        WEBSOCKET_SENTINEL_HOSTS,
        WEBSOCKET_SENTINEL_PORT,
    ),
)

if cleanup_lock.aquire_lock():
    try:
        perform_cleanup()
    finally:
        cleanup_lock.release_lock()
```

