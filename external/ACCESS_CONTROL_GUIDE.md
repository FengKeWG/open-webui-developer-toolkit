# access_control.py

`backend/open_webui/utils/access_control.py` contains helper functions for permission checks and group based access control. These utilities are used across the routers and models to verify whether a user may read or write a resource.

The module revolves around nested permission dictionaries and "access_control" mappings stored in database objects.

## Filling default permissions

`fill_missing_permissions(permissions, default_permissions)` recursively merges missing keys from the defaults. It is used by `get_permissions` and `has_permission` to ensure every level in the hierarchy has a value.

```python
from open_webui.config import DEFAULT_USER_PERMISSIONS

perms = {"files": {"read": True}}
filled = fill_missing_permissions(perms, DEFAULT_USER_PERMISSIONS)
```

## Computing effective permissions

`get_permissions(user_id, default_permissions)` collects every group the user belongs to and combines the stored permission structures. Boolean values are merged by taking the most permissive option:

```python
user_perms = get_permissions(user.id, DEFAULT_USER_PERMISSIONS)
if user_perms["files"]["write"]:
    ...
```

Each group record has a `permissions` JSON field. When multiple groups specify the same key, `True` wins over `False`.

## Checking a specific permission

`has_permission(user_id, permission_key, default_permissions={})` looks up a dotted key like `"files.read"` and returns a boolean. If none of the user's groups provide the permission the function falls back to the supplied defaults.

```python
if has_permission(user.id, "models.publish"):
    publish_model()
```

## Access control blocks

Data models such as chats and tools store an `access_control` dictionary with `read` and `write` entries. Each entry lists `group_ids` and `user_ids` allowed to perform the action. `has_access(user_id, type="write", access_control=None)` verifies membership:

```python
acl = {
    "read": {"group_ids": ["public"], "user_ids": []},
    "write": {"group_ids": ["editors"], "user_ids": [owner_id]},
}

if not has_access(current_user.id, "write", acl):
    raise HTTPException(403)
```

The helper returns `True` for read access when no ACL is present, matching the behaviour of the upstream models.

`get_users_with_access(type="write", access_control=None)` returns a list of `UserModel` objects for all users who match the ACL. It resolves group membership to individual user ids.

## Usage example

```python
from open_webui.utils import access_control

# combine group permissions for display
perms = access_control.get_permissions(user.id, DEFAULT_USER_PERMISSIONS)
print(perms["knowledge"]["create"])  # bool

# enforce write access on a channel
if not access_control.has_access(user.id, "write", channel.access_control):
    raise HTTPException(403)
```

These primitives underpin the role and group system throughout Open WebUI.
